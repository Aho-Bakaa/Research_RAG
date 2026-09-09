"""4-layer symbol normalization pipeline.

Layer 1: Canonical YAML dictionary (canonical_symbols.yaml)
Layer 2: Ingestion normalizer — replaces aliases with canonical forms during PDF ingestion
Layer 3: Query normalizer — normalizes user queries before retrieval
Layer 4: Display formatter — converts canonical symbols back to pretty display forms

This ensures that whether a paper writes "FSTI", "Tu", "Ti", or
"turbulence intensity", it all maps to the canonical "Tu" for retrieval.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


class SymbolNormalizer:
    """Four-layer symbol normalization for cross-paper consistency."""

    def __init__(self, yaml_path: str | Path | None = None) -> None:
        if yaml_path is None:
            yaml_path = Path(__file__).parent / "canonical_symbols.yaml"
        self._yaml_path = Path(yaml_path)
        self._symbols: dict[str, dict[str, Any]] = {}
        self._alias_to_canonical: dict[str, str] = {}
        self._canonical_to_display: dict[str, str] = {}
        # NEW: paper-specific alias map.  Keyed by paper_id, then by alias.
        # When a paper_id is supplied to normalize_text_with_log(), the
        # paper-specific aliases for that paper_id are applied FIRST
        # (they win over global aliases of the same string).  This is
        # how Mayle's τ_0 → τ_w mapping fires only in mayle_1991 chunks
        # without mangling Suzen-Huang's τ_0 elsewhere.
        self._paper_aliases: dict[str, dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        """Load the canonical symbol dictionary."""
        with open(self._yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        for key, entry in data.items():
            canonical = entry.get("canonical", key)
            display = entry.get("display", canonical)
            aliases = entry.get("aliases", [])

            self._symbols[key] = entry
            self._canonical_to_display[canonical] = display

            # Map every alias to its canonical form (GLOBAL — used when no
            # paper_id is supplied OR as fallback when the paper-specific
            # map doesn't have a hit).
            for alias in aliases:
                self._alias_to_canonical[alias] = canonical
                self._alias_to_canonical[alias.lower()] = canonical

            # Map the canonical form to itself
            self._alias_to_canonical[canonical] = canonical
            self._alias_to_canonical[canonical.lower()] = canonical

            # NEW: paper-specific aliases — only fire when normalize is
            # called with a matching paper_id.  These exist to resolve
            # symbol collisions (Mayle's τ_0 = wall shear; AGS's τ_t =
            # turbulence intensity; Narasimha's K = TKE; etc.) without
            # affecting other papers' chunks.
            paper_specific = entry.get("paper_specific_aliases", {}) or {}
            if isinstance(paper_specific, dict):
                for paper_id, paper_aliases in paper_specific.items():
                    if not paper_id or not isinstance(paper_aliases, list):
                        continue
                    by_paper = self._paper_aliases.setdefault(str(paper_id), {})
                    for alias in paper_aliases:
                        if not isinstance(alias, str):
                            continue
                        by_paper[alias] = canonical
                        by_paper[alias.lower()] = canonical

        # Sort aliases by length (longest first) to avoid partial matches
        self._sorted_aliases = sorted(
            self._alias_to_canonical.keys(), key=len, reverse=True
        )
        # Pre-sort each paper's aliases too for the paper-aware pass
        self._sorted_paper_aliases: dict[str, list[str]] = {
            pid: sorted(d.keys(), key=len, reverse=True)
            for pid, d in self._paper_aliases.items()
        }

    # ── Layer 2: Ingestion normalizer ─────────────────────────────

    def normalize_text(self, text: str, paper_id: str | None = None) -> str:
        """Replace known aliases with canonical symbols in ingested text.

        Word-boundary aware: alphanumeric aliases (e.g. "Ti", "FSTI")
        only match when surrounded by non-word chars, so "Ti" never
        matches inside "correlation". Non-alphanumeric aliases (e.g.
        "δ*", Greek letters) use direct substitution.

        When `paper_id` is supplied, the paper-specific alias map for
        that paper_id is applied FIRST (so collision-prone symbols
        like Mayle's τ_0 = wall shear vs AGS's τ_t = turbulence
        intensity each rewrite correctly).  Global aliases run after.

        After all substitutions, adjacent duplicates of the same
        canonical form are collapsed: "intermittency gamma" → both
        map to "γ", so we'd get "γ γ" — we dedupe to a single "γ".
        """
        result, _ = self.normalize_text_with_log(text, paper_id=paper_id)
        return result

    def normalize_text_with_log(
        self, text: str,
        paper_id: str | None = None,
    ) -> tuple[str, list[tuple[str, str]]]:
        """Same as `normalize_text` but ALSO returns the list of
        (alias, canonical) substitutions actually made.

        Two-pass logic:
          1. If `paper_id` is supplied and we have paper-specific
             aliases for it, apply those FIRST (they win over global
             aliases that share the same string).
          2. Apply the global alias map second.

        This is the key to handling symbol collisions across papers
        without polluting the global rewrite table.  Mayle's τ_0
        becomes τ_w only inside mayle_1991 chunks; AGS's τ_t becomes
        Tu only inside abu_ghannam_shaw_1980 chunks; etc.

        Returns:
            (normalized_text, [(alias, canonical), ...])
        """
        result = text
        substitutions: list[tuple[str, str]] = []

        # ── Pass 1: paper-specific aliases (only when paper_id given) ──
        if paper_id and paper_id in self._paper_aliases:
            paper_map = self._paper_aliases[paper_id]
            for alias in self._sorted_paper_aliases[paper_id]:
                if len(alias) < 2:
                    continue
                canonical = paper_map[alias]
                if alias == canonical:
                    continue
                try:
                    starts_word = alias[0].isalnum() or alias[0] == "_"
                    ends_word   = alias[-1].isalnum() or alias[-1] == "_"
                    left  = r"(?<!\w)" if starts_word else ""
                    right = r"(?!\w)"  if ends_word   else ""
                    pattern = re.compile(
                        left + re.escape(alias) + right, re.IGNORECASE,
                    )
                    new_result, count = pattern.subn(canonical, result)
                    if count > 0 and new_result != result:
                        substitutions.append((alias, canonical))
                        result = new_result
                except re.error:
                    continue

        # ── Pass 2: global aliases ──────────────────────────────────
        for alias in self._sorted_aliases:
            if len(alias) < 2:
                continue

            canonical = self._alias_to_canonical[alias]
            if alias == canonical:
                continue

            try:
                starts_word = alias[0].isalnum() or alias[0] == "_"
                ends_word   = alias[-1].isalnum() or alias[-1] == "_"
                left  = r"(?<!\w)" if starts_word else ""
                right = r"(?!\w)"  if ends_word   else ""
                pattern = re.compile(left + re.escape(alias) + right, re.IGNORECASE)
                # subn() returns (new_text, count); count tells us
                # whether the pattern fired.  But fired-with-no-change
                # is also possible: the case-insensitive regex matches
                # "Tu" against alias "tu" and substitutes canonical
                # "Tu" — net text unchanged.  Those should NOT be
                # logged as substitutions (they're spurious "this
                # text already canonical" hits, not actual rewrites).
                new_result, count = pattern.subn(canonical, result)
                if count > 0 and new_result != result:
                    substitutions.append((alias, canonical))
                    result = new_result
            except re.error:
                continue

        result = self._collapse_adjacent_duplicates(result)
        return result, substitutions

    def _collapse_adjacent_duplicates(self, text: str) -> str:
        """If a canonical symbol appears twice in a row with only
        whitespace between, collapse to one occurrence.

        Handles single-char canonicals (γ, ν, θ, δ, H, k) too —
        the "X X" pattern requires at least 3 chars so there's no
        false-positive risk from common English letters.
        """
        for canonical in self._canonical_to_display.keys():
            if not canonical:
                continue
            # Pattern: <canonical>\s+<canonical>  -> <canonical>
            pattern = re.compile(
                re.escape(canonical) + r"(\s+" + re.escape(canonical) + r")+"
            )
            text = pattern.sub(canonical, text)
        return text

    # ── Layer 3: Query normalizer ─────────────────────────────────

    def normalize_query(self, query: str) -> str:
        """Normalize a user query so that any symbol variant maps to canonical.

        Less aggressive than text normalization — only replaces exact matches.
        """
        words = query.split()
        normalized = []

        for word in words:
            # Strip punctuation for lookup
            clean = word.strip(".,;:!?()[]{}\"'")
            canonical = self._alias_to_canonical.get(
                clean, self._alias_to_canonical.get(clean.lower())
            )
            if canonical and canonical != clean:
                normalized.append(word.replace(clean, canonical))
            else:
                normalized.append(word)

        return " ".join(normalized)

    # ── Layer 4: Display formatter ────────────────────────────────

    def to_display(self, canonical: str) -> str:
        """Convert a canonical symbol to its pretty display form."""
        return self._canonical_to_display.get(canonical, canonical)

    def format_for_display(self, text: str) -> str:
        """Replace canonical symbols with display forms in output text."""
        result = text
        for canonical, display in self._canonical_to_display.items():
            if canonical != display and len(canonical) > 1:
                result = result.replace(canonical, display)
        return result

    # ── Lookup utilities ──────────────────────────────────────────

    def lookup(self, term: str) -> str | None:
        """Look up the canonical form of any symbol or alias."""
        return self._alias_to_canonical.get(
            term, self._alias_to_canonical.get(term.lower())
        )

    def get_all_aliases(self, canonical: str) -> list[str]:
        """Get all known aliases for a canonical symbol."""
        return [
            alias for alias, canon in self._alias_to_canonical.items()
            if canon == canonical
        ]

    def get_symbol_info(self, term: str) -> dict[str, Any] | None:
        """Get full symbol info (canonical, display, unit, description)."""
        canonical = self.lookup(term)
        if canonical is None:
            return None

        for key, entry in self._symbols.items():
            if entry.get("canonical") == canonical:
                return entry
        return None

    @property
    def all_canonical_symbols(self) -> list[str]:
        return list(self._canonical_to_display.keys())

    # ── Glossary growth (used by Node 2.5 SymbolNormalizer) ────────

    def add_alias(
        self,
        canonical: str,
        new_alias: str,
        source_paper: str | None = None,
        source_page: str | None = None,
    ) -> bool:
        """Add a newly discovered alias to the glossary and persist it.

        Used when Node 2.5's LLM pass resolves an unknown symbol to a
        canonical one that's already in the dictionary. The new alias
        is appended to the YAML file on disk so future runs pick it up
        without needing to re-resolve.

        Returns True if the alias was added, False if it was already
        present (idempotent call).
        """
        # Find the canonical's entry key (e.g. "turbulence_intensity")
        entry_key = None
        for key, entry in self._symbols.items():
            if entry.get("canonical") == canonical:
                entry_key = key
                break

        if entry_key is None:
            # Unknown canonical — nothing to attach the alias to
            return False

        # An alias identical to its canonical is a no-op, not a real addition.
        if new_alias == canonical:
            return False

        existing_aliases = self._symbols[entry_key].get("aliases", [])
        if new_alias in existing_aliases:
            return False

        # Update in-memory structures
        existing_aliases.append(new_alias)
        self._symbols[entry_key]["aliases"] = existing_aliases
        self._alias_to_canonical[new_alias] = canonical
        self._alias_to_canonical[new_alias.lower()] = canonical
        # Re-sort so longer aliases still get priority during replacement
        self._sorted_aliases = sorted(
            self._alias_to_canonical.keys(), key=len, reverse=True
        )

        # Persist to YAML. We write the full dict back out so comments
        # above each block are unfortunately lost — but the structure
        # is preserved and the file stays human-editable.
        self._persist()

        # Optional provenance trail — stored in a sibling JSON so the
        # YAML stays clean. Lets users audit where auto-added aliases
        # came from.
        if source_paper:
            self._record_alias_source(canonical, new_alias, source_paper, source_page)

        return True

    def _persist(self) -> None:
        """Write the in-memory glossary back to the YAML file."""
        with open(self._yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self._symbols,
                f,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )

    def _record_alias_source(
        self,
        canonical: str,
        alias: str,
        paper_id: str,
        page: str | None,
    ) -> None:
        """Append a row to alias_sources.json — audit trail for auto-adds."""
        import json
        src_path = self._yaml_path.parent / "alias_sources.json"
        sources: list[dict[str, Any]] = []
        if src_path.exists():
            try:
                with open(src_path, "r", encoding="utf-8") as f:
                    sources = json.load(f)
            except Exception:
                sources = []
        sources.append({
            "canonical": canonical,
            "alias": alias,
            "paper_id": paper_id,
            "page": page or "",
            "added_by": "auto",
        })
        with open(src_path, "w", encoding="utf-8") as f:
            json.dump(sources, f, indent=2, ensure_ascii=False)
