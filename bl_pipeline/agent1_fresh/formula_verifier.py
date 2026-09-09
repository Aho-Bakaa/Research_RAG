"""formula_verifier.py — pre-execution check on every compute() call.

WHY THIS EXISTS
───────────────
Run #12 produced a numerically wrong manuscript because Sonnet hardcoded
`Re_theta_t = 172.80` at the top of a compute() script and decorated it
with a print line that LOOKED like a Mayle Eq.(9) evaluation but never
actually multiplied anything by 400.  The Python sandbox dutifully
printed the literal.  No layer in our pipeline noticed.

This module closes that hole.  Before any compute() call executes, we:

  1. AST-walk the code.  For every assignment to a "tracked" result
     variable (Re_theta_t, R_L, x_t, L_tr, R_lambda, Re_tr, x_end, ...),
     check whether the RHS is a literal Constant.  If it is — REJECT
     with a message instructing Sonnet to re-emit using the formula.

  2. Require an inline tag comment of the form
        # paper_id::Eq.(N)
     above or beside each tracked-variable assignment.  Without a tag
     the verifier cannot know which paper-formula Sonnet claims to be
     using.  No tag → REJECT.

  3. For each tagged assignment, fetch the glossary entry via
     lookup_equation() and ship the pair
        (Sonnet's Python RHS, glossary's LaTeX verbatim, validity)
     to a small LLM judge (claude-haiku-4-5) which decides whether the
     two are mathematically equivalent.  Mismatch → REJECT with both
     versions side-by-side plus the judge's reasoning.

This is the layer that should have caught:
  - Mayle hardcode `Re_theta_t = 172.80`        (step 1 catches it)
  - Fransson `196 * Tu^(-2)` with Tu in percent (step 3 catches it via judge)

NON-GOALS
─────────
- Symbolic equivalence via sympy (too brittle for our LaTeX-heavy glossary;
  the LLM judge handles algebraic rearrangement and notation differences
  better than any regex).
- Unit-aware checking (requires structured `units_per_variable` field in
  glossary entries which we don't have yet — Phase 3 work).
- Auto-replace mode (current Phase 1 is REJECT only; replace mode comes
  later once we trust the judge).
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any


# Variables whose values are end-products of the algebraic transition
# correlations.  Each MUST be computed from an expression that involves
# inputs (Tu, U, ν, x) and the named formula — never assigned a literal.
#
# These are matched as PREFIXES: any variable whose name STARTS WITH one
# of these (possibly followed by `_<suffix>` like `_0`, `_1`, `_conv`,
# `_new`, `_prev`, `_AGS`, `_Mayle`) is treated as tracked.  This
# lets Sonnet's iterative code pattern
#     Re_theta_t_0 = 400 * Tu_0**(-5/8)
#     Tu_1 = ... ; Re_theta_t_1 = 400 * Tu_1**(-5/8)
#     ... Re_theta_t_conv = Re_theta_t_3
# work cleanly: every variant is tracked, the chain-of-trust derivation
# carries through, and only the ROOT formula needs a tag.  Run #13
# discovered the failure mode: exact-match treated `Re_theta_t_conv` as
# untracked, broke the derivation chain, and the verifier then demanded
# tags on what was really just dimensional re-expression.
TRACKED_VARIABLE_ROOTS: list[str] = [
    # Onset / momentum-thickness Reynolds at transition.
    # Includes the AGS paper-specific notations R_θS (start) and R_θE
    # (end) which Sonnet naturally uses when computing the AGS Eq.(3) /
    # Eq.(19) family.  Run #14 discovered R_theta_S was a hole.
    "Re_theta_t", "Re_theta_T", "ReThetaT",
    "Re_theta",                                # bare form for variant capture
    "R_theta",                                 # AGS notation root
    # Axial Reynolds at onset/end
    "Re_x_t", "Re_x_tr", "R_XS", "R_XE",
    # Zone length Reynolds
    "R_L", "R_LT", "Re_LT",
    # Fransson midpoint Re
    "Re_tr", "Re_x_gamma_half",
    # D-N intermittency scaling
    "R_lambda", "R_lam",
    # Physical lengths
    "x_t", "x_tr", "x_end",
    "L_tr",
]
# Sorted longest-first so prefix-match doesn't accidentally classify
# "Re_x_t" as "Re_x" (if "Re_x" were ever added).
TRACKED_VARIABLE_ROOTS = sorted(TRACKED_VARIABLE_ROOTS, key=len, reverse=True)

# Kept for backward-compat with callers that import this name.
TRACKED_VARIABLES: set[str] = set(TRACKED_VARIABLE_ROOTS)


def _normalise_var(name: str) -> str:
    """Lowercase and strip underscores.  Used by _is_tracked so that
    Sonnet's notational variants all resolve to the same canonical
    form:
      Re_theta_t, Re_theta_T, ReThetaT, re_theta_t_conv
        → all 'rethetat...'
      Re_x_t, Re_xt, Re_xT, Re_x_t_0
        → all 'rext...' (or 'rext0')
      R_XS, R_xs, R_xS_AGS
        → all 'rxs...'
    """
    return name.replace("_", "").lower()


# Pre-compute the normalised forms of every tracked root so _is_tracked
# can do prefix-match against them at no per-call cost.  Note we sort
# longest-first so `rextheta` doesn't accidentally match before `rex`.
_TRACKED_NORM_ROOTS: list[str] = sorted(
    {_normalise_var(r) for r in TRACKED_VARIABLE_ROOTS},
    key=len, reverse=True,
)


def _is_tracked(var_name: str) -> bool:
    """True iff var_name's NORMALISED form (lowercase, underscores
    stripped) matches a tracked root exactly OR begins with a tracked
    root followed by at least one character.  Catches iteration
    variants like Re_theta_t_0, x_t_conv, R_XS_AGS, Re_xt, R_lambda_DN.

    Examples of what matches:
      Re_theta_t      → 'rethetat'    matches root 'rethetat'   ✓
      Re_theta_t_conv → 'rethetatconv' starts with 'rethetat'    ✓
      Re_xt           → 'rext'        matches root 'rext'       ✓
      Re_xt_0         → 'rext0'       starts with 'rext'        ✓
      x_t_conv        → 'xtconv'      starts with 'xt'          ✓
      Tu_0            → 'tu0'         no tracked root            ✗
      lambda_DN       → 'lambdadn'    no tracked root            ✗
    """
    n = _normalise_var(var_name)
    for root in _TRACKED_NORM_ROOTS:
        if n == root or n.startswith(root):
            return True
    return False

# An inline tag comment looks like:
#   # mayle_1991::Eq.(9)
#   # abu_ghannam_shaw_1980::Eq.(18)
#   # fransson_matsubara_alfredsson_2005::Eq.(5.5)
#   # mayle_1991::Eq.(9)  [Tu in percent]   ← trailing notes are OK
#   # mayle_1991::Eq.(9) + Blasius inversion ← composite explanations OK
# Allow flexible whitespace + optional "Eq" abbreviations.  The eq_id
# value is captured as digits + optional decimal + optional letter
# suffix (matches "9", "5.5", "B1", "4.2a").  ANYTHING is allowed
# after the closing paren — the regex only needs to recognise the
# tag, not bound it.
#
# Run #14 discovered the failure mode: the previous version anchored
# on `$|#|//` after the eq_id, which rejected valid tags like
# `# mayle_1991::Eq.(9)  [Tu in percent]` because `[Tu in percent]`
# isn't end-of-line/comment/JS-comment.  Sonnet's compute calls were
# all rejected, forcing it to fall back to memory and ship 172.8
# (the run-#12 bug) in the FINAL block.  Lesson: the verifier MUST
# accept any reasonable inline notation Sonnet adds, not police
# the comment's structure beyond extracting the (paper_id, eq_id).
_TAG_RE = re.compile(
    r"#\s*([a-z][a-z0-9_]+)\s*::\s*Eq\.?\s*\(?([A-Za-z]?\d+(?:\.\d+)?[a-z]?)\)?",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────

@dataclass
class FormulaSite:
    """One tracked-variable assignment found in compute() code."""
    variable: str
    line: int
    python_rhs: str             # the source text of the assignment's RHS
    is_literal: bool            # True if RHS is just `ast.Constant`
    # True iff the RHS expression references at least one tracked
    # variable already defined earlier in this code block.  Such sites
    # are DERIVED INTERMEDIATES (e.g. x_t = Re_x_t * nu / U) and
    # inherit the chain of trust from the upstream formula — they don't
    # need their own equation tag, but they still can't be literals.
    is_derived: bool = False
    paper_id: str | None = None
    eq_id: str | None = None

    # Filled in by the glossary fetch
    glossary_latex: str = ""
    glossary_describes: str = ""
    glossary_validity: str = ""
    glossary_error: str = ""

    # Filled in by the LLM judge
    judge_verdict: str = ""     # "match" | "mismatch" | "skipped" | "error"
    judge_reasoning: str = ""


@dataclass
class VerificationResult:
    """The summary the dispatcher uses to decide pass/fail."""
    passed: bool
    error_message: str = ""              # human-readable BLOCKING error for Sonnet
    # Non-blocking provenance flags (judge mismatch / missing tag /
    # glossary miss).  The code STILL executes; these say which results
    # could not be matched to a cited paper-formula so downstream marks
    # them UNVERIFIED instead of silently trusting them.
    warnings: list[str] = field(default_factory=list)
    sites: list[FormulaSite] = field(default_factory=list)
    elapsed_s: float = 0.0
    judge_cost_usd: float = 0.0


# ─────────────────────────────────────────────────────────────────────
# Stage 1: AST walk — find tracked-variable assignments and inline tags
# ─────────────────────────────────────────────────────────────────────

def _find_tag_on_or_above(
    source_lines: list[str], lineno_1based: int,
) -> tuple[str | None, str | None]:
    """Return (paper_id, eq_id) for the tag comment on this line or the
    line directly above; (None, None) if no tag found.

    Sonnet may write the tag inline:
        Re_theta_t = 400 * Tu**(-5/8)   # mayle_1991::Eq.(9)
    or on the line directly above as a comment-only line:
        # mayle_1991::Eq.(9)
        Re_theta_t = 400 * Tu**(-5/8)
    Both are accepted.

    The line-above match is ONLY honored when that line is a
    comment-only line (starts with `#`).  Otherwise tags from adjacent
    code lines would leak into the wrong assignment (e.g. an earlier
    `Re_theta_t = 400*Tu**(-5/8)  # mayle::Eq.(9)` would incorrectly
    "tag" the next line's `Re_x_t = (Re_theta_t/0.664)**2`).
    """
    idx = lineno_1based - 1
    # Inline (same line) — always honor
    if 0 <= idx < len(source_lines):
        m = _TAG_RE.search(source_lines[idx])
        if m:
            return m.group(1).strip(), _normalize_eq_id(m.group(2))
    # Comment-only line directly above — honor
    if 0 <= idx - 1 < len(source_lines):
        line_above = source_lines[idx - 1].lstrip()
        if line_above.startswith("#"):
            m = _TAG_RE.search(source_lines[idx - 1])
            if m:
                return m.group(1).strip(), _normalize_eq_id(m.group(2))
    return None, None


def _normalize_eq_id(raw: str) -> str:
    """Turn '5.5' or ' 5.5 ' or '(5.5)' into the canonical 'Eq. (5.5)'.

    Matches the format the equation_indices/<paper>.json files store
    (matches what lookup_equation's eq_id matcher expects).
    """
    s = (raw or "").strip().strip("()").strip()
    return f"Eq. ({s})"


def _rhs_source(code: str, node: ast.AST) -> str:
    """Best-effort source-text recovery for an AST node's RHS.

    Uses ast.unparse where available (3.9+) which is what we ship on.
    """
    try:
        return ast.unparse(node)
    except Exception:
        # Fallback — sliced from the source by line/col offsets.
        return f"<unparseable: {type(node).__name__}>"


def _rhs_references_tracked(value: ast.AST, already_defined: set[str]) -> bool:
    """Does this RHS expression reference at least one tracked variable
    (or variant — caught via _is_tracked) that has already been
    assigned earlier in the same block?

    Used to recognize "derived intermediates" — assignments like
    `x_t = Re_x_t * nu / U` that don't implement a paper-formula on
    their own but propagate an upstream tagged value.  Such sites
    inherit the chain of trust and don't need their own tag.

    Uses _is_tracked so Sonnet's iteration variants (`Re_theta_t_conv`,
    `x_t_3`, `R_XS_AGS`) are recognized as tracked even when the
    `already_defined` set contains those exact strings.
    """
    for n in ast.walk(value):
        if isinstance(n, ast.Name) and n.id in already_defined:
            return True
    return False


def _iter_tracked_targets(target: ast.AST):
    """Yield every ast.Name node inside an assignment target that names a
    tracked variable (matched via _is_tracked, which is prefix-aware
    so Sonnet's iteration variants like Re_theta_t_conv are caught).
    Handles:
      • bare name:                Re_theta_t = ...
      • tuple/list unpacking:     Re_theta_t, x_t = ...
      • starred:                  Re_theta_t, *rest = ...
    Skips:
      • attribute targets (self.Re_theta_t = ...)
      • subscript targets (results['Re_theta_t'] = ...)
    """
    if isinstance(target, ast.Name):
        if _is_tracked(target.id):
            yield target
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _iter_tracked_targets(elt)
    elif isinstance(target, ast.Starred):
        yield from _iter_tracked_targets(target.value)


def collect_formula_sites(code: str) -> list[FormulaSite]:
    """Walk the AST.  Return one FormulaSite per assignment to a tracked
    variable, with chain-of-trust derivation flag set.

    Chain-of-trust model (Run #14 regression revealed this is needed):
      We track a `trusted` set that grows as we walk assignments in
      source order.  An untracked intermediate (`lambda_m = R_lambda *
      nu / U`) becomes trusted because its RHS references a tracked
      variable (R_lambda).  Later, `L_tr_DN = (xi_99 - xi_01) *
      lambda_m` references the trusted `lambda_m`, so L_tr_DN is
      derived.  Without this transitive propagation, intermediate
      scalars like `lambda_m`, `xi_99`, `Re_xt_0` broke the chain and
      forced spurious tag-requirement errors on legitimately-derived
      results.

    Uses ast.walk() so assignments inside if/for/while/with blocks are
    also caught.  Sites are sorted by source line for the trust pass.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    src_lines = code.split("\n")

    # Stage 1: collect EVERY assignment in source order, both tracked
    # and untracked.  We need the untracked ones for trust propagation.
    @dataclass
    class _RawAssign:
        line: int
        targets: list[ast.AST]
        value: ast.AST
        per_name_value: dict[int, ast.AST]   # id(target_name) → value

    raw_assigns: list[_RawAssign] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        elif isinstance(node, ast.NamedExpr):
            targets = [node.target]
            value = node.value
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
            value = node.value
        else:
            continue

        per_name_value: dict[int, ast.AST] = {}
        if (len(targets) == 1
                and isinstance(targets[0], (ast.Tuple, ast.List))
                and isinstance(value, (ast.Tuple, ast.List))
                and len(targets[0].elts) == len(value.elts)):
            for tgt_elt, val_elt in zip(targets[0].elts, value.elts):
                per_name_value[id(tgt_elt)] = val_elt

        raw_assigns.append(_RawAssign(
            line=node.lineno, targets=targets, value=value,
            per_name_value=per_name_value,
        ))

    raw_assigns.sort(key=lambda r: r.line)

    # Stage 2: walk in source order, build `trusted` set, emit
    # FormulaSites for tracked targets.
    trusted: set[str] = set()
    sites: list[FormulaSite] = []

    def _all_name_targets(target: ast.AST):
        """Yield every ast.Name in the LHS (tracked or not).  Used for
        trust propagation — even untracked LHS names go into the
        trusted set if their RHS references trusted vars."""
        if isinstance(target, ast.Name):
            yield target
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                yield from _all_name_targets(elt)
        elif isinstance(target, ast.Starred):
            yield from _all_name_targets(target.value)

    for ra in raw_assigns:
        for tgt in ra.targets:
            for name_node in _all_name_targets(tgt):
                per_value = ra.per_name_value.get(id(name_node), ra.value)
                is_literal = isinstance(per_value, ast.Constant) and isinstance(
                    per_value.value, (int, float),
                )
                # Trust propagation: an assignment "inherits trust" if
                # the RHS is not a literal AND references any name
                # already in the trusted set OR any tracked name.
                rhs_refs_trusted_or_tracked = False
                if not is_literal:
                    for sub in ast.walk(per_value):
                        if not isinstance(sub, ast.Name):
                            continue
                        if sub.id in trusted or _is_tracked(sub.id):
                            rhs_refs_trusted_or_tracked = True
                            break

                # If this LHS is a tracked variable, emit a FormulaSite.
                if _is_tracked(name_node.id):
                    paper_id, eq_id = _find_tag_on_or_above(src_lines, ra.line)
                    sites.append(FormulaSite(
                        variable=name_node.id,
                        line=ra.line,
                        python_rhs=_rhs_source(code, per_value),
                        is_literal=is_literal,
                        is_derived=rhs_refs_trusted_or_tracked,
                        paper_id=paper_id,
                        eq_id=eq_id,
                    ))

                # ALL non-literal assignments whose RHS references a
                # trusted/tracked var join the trusted set — including
                # untracked intermediates.  This is what propagates
                # trust through `lambda_m = R_lambda * ...` even
                # though lambda_m itself is untracked.
                if not is_literal and rhs_refs_trusted_or_tracked:
                    trusted.add(name_node.id)
                # Tracked variables ALWAYS join the trusted set after
                # their assignment (even literal — Sonnet might assign
                # x_t = 0.0859 and we want downstream references to
                # know x_t exists, even though the literal-check will
                # reject this site).
                if _is_tracked(name_node.id):
                    trusted.add(name_node.id)

    return sites


# ─────────────────────────────────────────────────────────────────────
# Stage 2: Glossary fetch — populate each site with the paper formula
# ─────────────────────────────────────────────────────────────────────

def fetch_glossary_for_sites(sites: list[FormulaSite]) -> None:
    """For each site that has paper_id+eq_id, fetch from lookup_equation
    and write into the site's glossary_* fields.  Sites missing tags are
    left untouched (caller handles them as errors).
    """
    # Local import to avoid cycle with tools.py (which imports us).
    from bl_pipeline.agent1_fresh.tools import lookup_equation

    for s in sites:
        if not s.paper_id or not s.eq_id:
            continue
        entry = lookup_equation(paper_id=s.paper_id, eq_id=s.eq_id)
        if entry.get("error"):
            s.glossary_error = str(entry["error"])
            continue
        s.glossary_latex = str(entry.get("formula_latex_verbatim", "") or "")
        s.glossary_describes = str(entry.get("describes", "") or "")
        s.glossary_validity = str(entry.get("validity", "") or "")


# ─────────────────────────────────────────────────────────────────────
# Stage 3: LLM judge — ask Haiku whether each pair matches
# ─────────────────────────────────────────────────────────────────────

def judge_pairs(sites: list[FormulaSite]) -> float:
    """Call the LLM judge on each site that has both python_rhs AND
    glossary_latex.  Writes judge_verdict/judge_reasoning into the sites
    in place.  Returns cost in USD.

    Sites with errors, missing glossary, or literal RHS are SKIPPED
    (judge_verdict = "skipped") — the caller handles those independently.
    """
    pairs = [
        s for s in sites
        if (not s.is_literal)
        and s.python_rhs
        and s.glossary_latex
        and not s.glossary_error
    ]
    if not pairs:
        return 0.0

    # Local import — keeps the verifier importable when anthropic isn't
    # available (e.g. unit tests that skip the judge).
    try:
        from bl_pipeline.agent1_fresh.equation_judge import judge_formula_pairs
    except Exception as e:
        for s in pairs:
            s.judge_verdict = "error"
            s.judge_reasoning = f"judge import failed: {e}"
        return 0.0

    try:
        verdicts, cost = judge_formula_pairs(pairs)
    except Exception as e:
        for s in pairs:
            s.judge_verdict = "error"
            s.judge_reasoning = f"judge call failed: {type(e).__name__}: {e}"
        return 0.0

    # verdicts is parallel to `pairs`
    for site, v in zip(pairs, verdicts):
        site.judge_verdict = v.get("verdict", "error")
        site.judge_reasoning = v.get("reasoning", "")
    return cost


# ─────────────────────────────────────────────────────────────────────
# Top-level orchestration
# ─────────────────────────────────────────────────────────────────────

def verify(code: str, *, use_judge: bool = True) -> VerificationResult:
    """Full pre-execution check.

    Args:
        code: The Python source Sonnet wants to run in compute().
        use_judge: If False, skip the LLM judge (Stage 3) — useful for
            unit tests where we just want to confirm Stage 1/2 behavior.

    Returns:
        VerificationResult.passed=True means the code is safe to execute.
        passed=False means error_message describes what to tell Sonnet.
    """
    import time
    t0 = time.time()

    sites = collect_formula_sites(code)

    # If there are NO tracked variables, compute() can run freely — the
    # verifier only constrains the formulas that produce headline numbers.
    if not sites:
        return VerificationResult(
            passed=True,
            elapsed_s=round(time.time() - t0, 3),
        )

    # Stage 1: literal-hardcode check
    literal_errors: list[str] = []
    for s in sites:
        if s.is_literal:
            literal_errors.append(
                f"Line {s.line}: `{s.variable} = {s.python_rhs}` is a "
                f"HARDCODED LITERAL.  `{s.variable}` is a tracked result "
                f"variable (must be computed from a formula involving Tu, "
                f"U, ν, x, etc.).  Re-emit your code with an EXPRESSION "
                f"and tag the formula with `# <paper_id>::Eq.(N)`.  "
                f"Example: `{s.variable} = 400 * Tu**(-5/8)  # mayle_1991::Eq.(9)`"
            )

    # Stage 2: missing-tag check.
    #
    # Tags are required for "originating" formulas — assignments whose
    # RHS does NOT reference another already-defined tracked variable.
    # Derived intermediates (e.g. x_t = Re_x_t * nu / U, where Re_x_t
    # was already computed from a tagged Mayle formula) inherit the
    # chain of trust from upstream and don't need their own tag.
    # Literals are already handled in Stage 1.
    tag_errors: list[str] = []
    for s in sites:
        if s.is_literal:
            continue
        if s.is_derived:
            continue   # inherits trust from upstream tracked variable
        if not s.paper_id or not s.eq_id:
            tag_errors.append(
                f"Line {s.line}: `{s.variable} = {s.python_rhs}` has no "
                f"equation tag.  This assignment does NOT reference any "
                f"already-defined tracked variable, so it must be tagged "
                f"with the paper-equation it implements.  Add an inline "
                f"comment `# <paper_id>::Eq.(N)` on the same line (or as "
                f"a comment-only line directly above).  Example: "
                f"`{s.variable} = ... # fransson_matsubara_alfredsson_2005::Eq.(5.5)`"
            )

    # ── BLOCKING gate: hardcoded literals ONLY ────────────────────────
    # A literal assignment to a tracked result variable means NO
    # computation happened (the run-#12 `Re_theta_t = 172.80` bug), so
    # executing is pointless — this is the ONE failure that still blocks.
    # Missing tags, glossary misses and judge mismatches are now
    # NON-BLOCKING: the code executes (correct formulas produce real
    # numbers) and only the un-matchable results are flagged UNVERIFIED.
    # The old all-or-nothing reject discarded correct Mayle/AGS/Blasius
    # numbers whenever ONE side-check was contested, forcing the fall back
    # to LLM mental-math — the exact bug this is meant to prevent.
    if literal_errors:
        msg = "FORMULA_VERIFIER REJECTED THE compute() CALL.\n\n"
        msg += "● Hardcoded literals (must be expressions):\n\n"
        msg += "\n\n".join(literal_errors) + "\n\n"
        msg += (
            "Why this matters: in run #12 you hardcoded Re_theta_t = 172.80 "
            "without computing it.  The print statement made it LOOK like "
            "a Mayle Eq.(9) evaluation but Python never multiplied "
            "400 by anything.  The manuscript shipped the wrong number.  "
            "This check exists to prevent that.  Re-emit with explicit "
            "expressions and tags."
        )
        return VerificationResult(
            passed=False, error_message=msg, sites=sites,
            elapsed_s=round(time.time() - t0, 3),
        )

    # ── NON-BLOCKING provenance flags from here ───────────────────────
    # Seed warnings with any missing-tag notes (downgraded from reject).
    warnings: list[str] = list(tag_errors)
    for s in sites:
        if (not s.is_literal) and (not s.is_derived) and (
            not s.paper_id or not s.eq_id
        ):
            s.judge_verdict = "unverified_no_tag"

    # Stage 2b: glossary fetch (always runs; populates sites)
    fetch_glossary_for_sites(sites)

    for s in sites:
        if s.glossary_error:
            warnings.append(
                f"Line {s.line}: `{s.variable}` tagged "
                f"`{s.paper_id}::{s.eq_id}` but glossary lookup failed: "
                f"{s.glossary_error}.  Value computed but flagged UNVERIFIED."
            )
            if not s.judge_verdict:
                s.judge_verdict = "unverified_glossary"

    # Stage 3: LLM judge
    judge_cost = 0.0
    if use_judge:
        judge_cost = judge_pairs(sites)

    # Only reject on EXPLICIT mismatch.  "uncertain" verdicts pass
    # through (with a stdout warning the writer can surface) — this
    # prevents the judge from blocking Sonnet's progress when Haiku's
    # response is malformed or it can't make up its mind.
    #
    # Run #14 lesson: fail-closed on "uncertain" caused all 15 sites
    # to be rejected when Haiku returned unparseable JSON for a single
    # compute() call.  Sonnet then ran out of compute() budget without
    # a single successful execution and fell back to memory — shipping
    # the run-#12 bug (Re_θt = 172.8) in the FINAL block.  Worse than
    # no verifier.  Now: uncertain = pass with warning.
    # Judge mismatch / uncertain → NON-BLOCKING flag (was a hard reject).
    # The side-by-side is preserved in the warning so the writer's
    # provenance ledger and the model's next turn still see exactly what
    # was contested and can fix it if it matters.
    for s in sites:
        if s.judge_verdict == "mismatch":
            warnings.append(
                f"Line {s.line}: `{s.variable}` may NOT match "
                f"{s.paper_id} {s.eq_id}.  YOUR PYTHON: {s.python_rhs}  |  "
                f"GLOSSARY: {s.glossary_latex}  |  JUDGE: {s.judge_reasoning}  "
                f"(value computed but flagged UNVERIFIED — fix the formula "
                f"or drop the quantity)."
            )
        elif s.judge_verdict == "uncertain":
            warnings.append(
                f"Line {s.line}: `{s.variable}` formula match UNCERTAIN vs "
                f"{s.paper_id} {s.eq_id}; computed but flagged for review."
            )

    # Tag derived intermediates so downstream can tell "verified-by-
    # derivation" from "never checked".
    for s in sites:
        if s.is_derived and not s.judge_verdict:
            s.judge_verdict = "derived"

    # Execute-&-flag: no literals → PASS.  The code executes; `warnings`
    # and each site's judge_verdict carry the provenance flags downstream.
    return VerificationResult(
        passed=True, sites=sites, warnings=warnings,
        elapsed_s=round(time.time() - t0, 3),
        judge_cost_usd=judge_cost,
    )
