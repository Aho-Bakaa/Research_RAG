"""markdown_chunker.py — chunker aware of equations, tables, and figures.

The legacy `chunk_text` in ingestion.py is character-count based. It
will happily split in the middle of an equation, tearing LaTeX apart.
For VLM-extracted markdown with LaTeX math and structured figure
blocks, we need a smarter chunker that respects semantic boundaries.

Rules
─────
1. Start a new chunk at a heading (## or ###) when the current
   chunk is already past the minimum size.
2. Never split inside a display equation block `$$…$$`.
3. Never split inside a markdown table (consecutive lines starting
   with `|`).
4. Never split inside a Figure block (a section that opens with
   `### Figure N` — keep the whole figure description together until
   the next heading).
5. If a chunk exceeds the max size, split at paragraph breaks
   (double newlines) outside of the above protected regions.

The output chunks are strings; the ingestion pipeline attaches
metadata (paper_id, page_number, chunk index) in the normal way.
"""

from __future__ import annotations

import re


# A paragraph break: at least two consecutive newlines.
_PARA_BREAK = re.compile(r"\n\s*\n")

# A heading line: starts with one or more '#'.
_HEADING = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)

# Start/end markers for protected blocks.
_DISPLAY_MATH = "$$"


def chunk_markdown(
    markdown: str,
    target_size: int = 1200,
    max_size: int = 1800,
    min_size: int = 400,
) -> list[str]:
    """Split markdown into semantically-bounded chunks.

    Args:
        markdown: input markdown (usually one page of VLM output).
        target_size: preferred chunk length in characters.
        max_size: hard cap — chunks above this are forcibly split.
        min_size: minimum chunk length before we consider closing.
            Prevents one-sentence chunks at heading-dense pages.

    Returns:
        list of chunk strings, in order.
    """
    if not markdown or not markdown.strip():
        return []

    # Split into "segments" — runs of text separated by paragraph
    # breaks. Segments are the atomic units we merge into chunks.
    # We preserve segments that are display-math or table blocks
    # atomically (no internal splitting).
    segments = _split_into_segments(markdown)

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    def flush() -> None:
        nonlocal buffer, buffer_len
        if buffer:
            text = "\n\n".join(buffer).strip()
            if text:
                chunks.append(text)
        buffer = []
        buffer_len = 0

    for seg in segments:
        seg_len = len(seg)
        is_heading = _starts_with_heading(seg)

        # Heading that's also the start of a Figure block — keep the
        # heading merged with the following segments until next heading.
        # Simpler rule: any heading closes the current chunk if we're
        # past min_size.
        if is_heading and buffer_len >= min_size:
            flush()

        # If adding this segment would overflow max_size AND the
        # buffer already has content, flush first.
        if buffer_len + seg_len + 2 > max_size and buffer:
            flush()

        # Oversized single segment (e.g., a long display-math block or
        # a big table): we accept it as its own chunk even if it's
        # larger than max_size — atomicity wins over size.
        if seg_len > max_size and not buffer:
            chunks.append(seg.strip())
            continue

        buffer.append(seg)
        buffer_len += seg_len + 2

        # Target-size close: if we're past target and the next segment
        # would be a heading or we're at the last segment, flush.
        if buffer_len >= target_size and seg == segments[-1]:
            flush()

    # Flush whatever's left
    flush()

    # Post-pass: merge any "orphan equation / table / figure" chunks
    # into an adjacent prose chunk. Without this, a display-math
    # segment that the chunker emitted by itself (because the
    # surrounding prose already closed a chunk) becomes a tiny,
    # context-less chunk whose English embedding is so weak that RAG
    # can't find it. See post-mortem on Mayle Eq. 9.
    chunks = _merge_orphan_chunks(chunks, min_prose_chars=120)

    return chunks


# ── Orphan coalescing ────────────────────────────────────────────────


def _merge_orphan_chunks(
    chunks: list[str],
    min_prose_chars: int = 120,
) -> list[str]:
    """Second pass: attach orphan equation/table chunks to neighbors.

    Two triggers for merging a chunk with its predecessor:

      (1) LEADING-MATH trigger — the chunk STARTS with display math
          (first non-whitespace chars are `$$`). This almost always
          means the "setup sentence" is the tail of the previous
          chunk, e.g.:
             prev:  "...A correlation of the present data provides"
             cur:   "$$Re_theta,t = 400 Tu^{-5/8}$$"
          The user's English query will match the setup sentence,
          not the bare formula. Joining them is always correct.

      (2) LOW-PROSE trigger — the chunk has display math or a table
          AND the surrounding English prose is below `min_prose_chars`
          (chunk is mostly math). This catches orphan table rows and
          dense equation stacks that don't start with `$$` (e.g. they
          start with a heading or a fragment of sentence continuation).

    Both triggers merge into the predecessor. If there's no
    predecessor (first chunk), we leave the orphan alone — there's no
    upstream context to attach to.
    """
    if len(chunks) <= 1:
        return chunks

    out: list[str] = []
    for cur in chunks:
        should_merge = False
        reason = ""
        if out:
            # Trigger 1: leading math
            if _starts_with_display_math(cur):
                should_merge = True
                reason = "leading-math"
            # Trigger 2: low prose around math/table
            elif _is_orphan_math_or_table(cur, min_prose_chars):
                should_merge = True
                reason = "low-prose"

        if should_merge:
            out[-1] = out[-1] + "\n\n" + cur
            # Reason kept in case future debug logging wants it — the
            # retrieval metadata is derived from the MERGED chunk text
            # downstream, so the merge itself doesn't need a tag here.
            _ = reason
        else:
            out.append(cur)

    return out


def _starts_with_display_math(chunk: str) -> bool:
    """True if the first non-whitespace content is a `$$...$$` block.

    Allowed leading content: blank lines. Anything else means the
    chunk has its own natural opener (heading, prose, table row).
    """
    if not chunk:
        return False
    stripped = chunk.lstrip()
    return stripped.startswith("$$")


def _is_orphan_math_or_table(chunk: str, min_prose_chars: int) -> bool:
    """Return True if `chunk` is mostly math/table with too little prose.

    We measure "prose" as characters that aren't inside $...$, $$...$$,
    or a markdown table line (starting with |). If prose char count is
    below `min_prose_chars` AND the chunk contains at least one display
    math block or table, call it orphan.
    """
    if not chunk:
        return False
    has_display_math = "$$" in chunk
    has_table = any(line.strip().startswith("|") for line in chunk.splitlines())
    if not (has_display_math or has_table):
        return False

    # Strip display math and table rows to measure remaining prose
    stripped = re.sub(r"\$\$.+?\$\$", " ", chunk, flags=re.DOTALL)
    stripped = re.sub(r"(?<!\$)\$[^$\n]+?\$", " ", stripped)
    stripped = "\n".join(
        ln for ln in stripped.splitlines()
        if not ln.strip().startswith("|")
    )
    # Also drop heading lines — they don't count as prose context
    stripped = "\n".join(
        ln for ln in stripped.splitlines()
        if not ln.strip().startswith("#")
    )
    prose_len = len(stripped.strip())
    return prose_len < min_prose_chars


# ── Segment splitter ─────────────────────────────────────────────────

def _split_into_segments(markdown: str) -> list[str]:
    """Split markdown into atomic segments.

    A segment is either:
      - a display-math block ($$...$$) — atomic
      - a markdown table (consecutive | lines) — atomic
      - a ``` fenced code block — atomic
      - a paragraph of prose (separated by blank lines) — atomic
    """
    segments: list[str] = []
    lines = markdown.split("\n")
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Display math $$...$$ (may span multiple lines)
        if stripped.startswith(_DISPLAY_MATH):
            block_lines = [line]
            # Single-line $$...$$
            if stripped.count(_DISPLAY_MATH) >= 2:
                segments.append(line)
                i += 1
                continue
            i += 1
            while i < n and _DISPLAY_MATH not in lines[i]:
                block_lines.append(lines[i])
                i += 1
            if i < n:
                block_lines.append(lines[i])
                i += 1
            segments.append("\n".join(block_lines))
            continue

        # Markdown table — consecutive lines starting with |
        if stripped.startswith("|"):
            table_lines = [line]
            i += 1
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            segments.append("\n".join(table_lines))
            continue

        # Fenced code block
        if stripped.startswith("```"):
            block_lines = [line]
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                block_lines.append(lines[i])
                i += 1
            if i < n:
                block_lines.append(lines[i])
                i += 1
            segments.append("\n".join(block_lines))
            continue

        # Ordinary paragraph — gather until a blank line or another
        # protected-block opener.
        para_lines: list[str] = []
        while i < n:
            ln = lines[i]
            stripped_ln = ln.strip()
            if not stripped_ln:
                i += 1
                break
            if stripped_ln.startswith(_DISPLAY_MATH) and para_lines:
                break
            if stripped_ln.startswith("|") and para_lines:
                break
            if stripped_ln.startswith("```") and para_lines:
                break
            para_lines.append(ln)
            i += 1
        if para_lines:
            segments.append("\n".join(para_lines))

    return segments


def _starts_with_heading(segment: str) -> bool:
    first_line = segment.split("\n", 1)[0].strip()
    return first_line.startswith("#")
