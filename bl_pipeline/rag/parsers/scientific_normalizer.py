"""scientific_normalizer.py — Unicode, accent, and spaced text normalizer for scientific literature."""
from __future__ import annotations

import re
import unicodedata

# 1. Explicit Latin Ligature Replacements
LIGATURE_MAP: dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "ft",
    "\ufb06": "st",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}

# 2. Control Characters: strip non-printables while preserving \t (\x09), \n (\x0a), \r (\x0d)
CONTROL_CHARS_RX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# 3. Known Aerodynamic Variables and Accented Tokens
AERO_VAR_RULES: list[tuple[re.Pattern, str]] = [
    # Accented Re_thetat: R˜ e t, R~ e t, R ˜ e t, R˜e_θt
    (re.compile(r"\bR\s*[\u02dc\u0303~∼]\s*e\s*[_]?\s*(?:θ|theta)?\s*t\b", re.IGNORECASE), "Re_thetat"),
    # General accented Reynolds number Re_tilde
    (re.compile(r"\bR\s*[\u02dc\u0303~∼]\s*e\b", re.IGNORECASE), "Re_tilde"),
    # Spaced Re_thetat and Re_theta
    (re.compile(r"\bR\s*e\s*_\s*(?:θ|theta)\s*t\b", re.IGNORECASE), "Re_thetat"),
    (re.compile(r"\bR\s*e\s*_\s*(?:θ|theta)\b", re.IGNORECASE), "Re_theta"),
    # Generic spaced Re subscripts: R e _ x -> Re_x, R e _ c -> Re_c
    (re.compile(r"\bR\s*e\s*_\s*([a-z0-9]+)\b", re.IGNORECASE), r"Re_\1"),
    # Spaced Re prefix
    (re.compile(r"\bR\s+e\s*_\s*", re.IGNORECASE), "Re_"),
    # Standalone spaced Re before math operator: R e = 500
    (re.compile(r"\bR\s+e\b(?=\s*[=<>+\-*/(])", re.IGNORECASE), "Re"),
    # Aerodynamic dimensionless groups
    (re.compile(r"\bT\s+u\b"), "Tu"),
    (re.compile(r"\bF\s*S\s*T\s*I\b", re.IGNORECASE), "FSTI"),
    (re.compile(r"\bC\s+f\b"), "Cf"),
    (re.compile(r"\bC\s+p\b"), "Cp"),
    (re.compile(r"\bC\s+l\b"), "Cl"),
    (re.compile(r"\bC\s+d\b"), "Cd"),
    (re.compile(r"\bM\s+a\b"), "Ma"),
    (re.compile(r"\bP\s+r\b"), "Pr"),
    # Turbulence & transition model variables
    (re.compile(r"\bk\s*-\s*ω(?!\w)", re.IGNORECASE), "k-ω"),
    (re.compile(r"\bk\s*-\s*omega\b", re.IGNORECASE), "k-omega"),
    (re.compile(r"\bk\s*-\s*ε(?!\w)", re.IGNORECASE), "k-ε"),
    (re.compile(r"\bk\s*-\s*epsilon\b", re.IGNORECASE), "k-epsilon"),
    (re.compile(r"\bγ\s*-\s*Re\b", re.IGNORECASE), "γ-Re"),
    (re.compile(r"\bgamma\s*-\s*Re\b", re.IGNORECASE), "gamma-Re"),
    (re.compile(r"\bτ\s*_\s*w\b"), "τ_w"),
    (re.compile(r"\btau\s*_\s*w\b", re.IGNORECASE), "tau_w"),
    # Free-stream asymptotic limits (using (?!\w) to support math symbol ∞)
    (re.compile(r"ν\s*_\s*∞(?!\w)"), "ν_∞"),
    (re.compile(r"\bnu\s*_\s*∞(?!\w)", re.IGNORECASE), "nu_∞"),
    (re.compile(r"[uU]\s*_\s*∞(?!\w)"), "U_∞"),
]

# 4. Spaced Single Characters in Words (>= 4 single letters separated by 1 space)
# e.g., 'f l u c t u a t i n g' -> 'fluctuating'
SPACED_WORD_RX = re.compile(r"\b([A-Za-z](?: [A-Za-z]){3,})\b")


def normalize_scientific_text(text: str) -> str:
    """Normalize extracted PDF scientific text: ligatures, accents, spaced kerning, and control chars."""
    if not text:
        return ""

    # Step 1: Strip non-printable control characters
    text = CONTROL_CHARS_RX.sub("", text)

    # Step 2: Replace explicit ligatures
    for lig, rep in LIGATURE_MAP.items():
        if lig in text:
            text = text.replace(lig, rep)

    # Step 3: Unicode NFKC normalization (decomposes font variants, normalizes subscripts)
    text = unicodedata.normalize("NFKC", text)

    # Step 4: De-space known aerodynamic variables & mathematical accents
    for rx, rep in AERO_VAR_RULES:
        text = rx.sub(rep, text)

    # Step 5: De-space spaced single characters embedded within words (>= 4 letters)
    text = SPACED_WORD_RX.sub(lambda m: m.group(1).replace(" ", ""), text)

    # Step 6: Collapse multiple horizontal spaces while preserving newlines
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()
