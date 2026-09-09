"""plot.py — deterministic plot generation from reasoner findings.

This module is **pure post-processing** — no LLM call, no events, no
ReAct.  It reads the reasoner's `final` block + the parsed flow + the
plan and produces a fixed set of PNG plots an experimentalist needs to
plan a wind-tunnel run:

    1. probe_layout.png        — horizontal strip with regimes + stations
    2. intermittency.png       — γ(x) profile across the transition zone
    3. bl_thickness.png        — δ(x) laminar + turbulent curves with stations
    4. onset_comparison.png    — bar chart of x_t predictions across correlations
    5. tu_decay.png            — Tu(x) decay profile (only if iterative
                                  correction was applied and we can infer L_ref)

Design rules:
  • NEVER raise.  If matplotlib is missing, return [].  If a specific
    plot can't be drawn (e.g. no L_tr in findings), skip that one.
  • Output paths are absolute, so the writer can embed via `![](path)`.
  • Use Agg backend — no display needed, works headless.
  • Conservative styling — clean, label everything, no chartjunk.

The writer reads the returned list of (label, abs_path) pairs from
its user message and embeds them into Part A.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bl_pipeline.agent1_fresh.state import FlowConditions, Plan, ReasonerTrace


# ══════════════════════════════════════════════════════════════════════
# Public data type — what we return to the writer
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PlotArtifact:
    """One generated plot — label + absolute file path + 1-line caption."""
    label: str           # short stable id, e.g. "probe_layout"
    path: str            # absolute filesystem path to the PNG
    caption: str         # 1-line caption the writer can use inline


# ══════════════════════════════════════════════════════════════════════
# Number-extraction helpers — pull values out of the reasoner's FINAL
# ══════════════════════════════════════════════════════════════════════

# Aliases the reasoner sometimes uses — we normalise to canonical names.
_CANONICAL_ALIASES = {
    "x_t":       {"x_t", "xt", "x_onset", "x_start", "x_t_decay_corrected",
                  "x_t_decay", "x_trans_start"},
    "L_tr":      {"l_tr", "ltr", "l_trans", "l_tr_ags", "l_tr_full"},
    "x_end":     {"x_end", "x_e", "x_complete", "x_t_end", "x_turb"},
    "lambda_dn": {"lambda_dn", "lambda", "lam", "intermittency_lambda_dn",
                  "intermittency_lambda"},
    "re_theta_t_mayle": {"re_theta_t_mayle", "re_theta_t", "re_theta_t_mayle_eq9"},
    "re_theta_t_ags":   {"re_theta_t_ags", "re_theta_s", "re_theta_s_ags"},
    # re_theta_t_lm alias REMOVED — Langtry-Menter 2009 is a CFD closure
    # model (γ-Re_θ-SST), not a stand-alone A1 algebraic correlation;
    # A1 defers it to Agent 2.
    "re_theta_t_sh":    {"re_theta_t_sh", "re_theta_t_suzen_huang"},
}


def _normalise_finding_name(name: str) -> str:
    """Strip parenthesized qualifiers and trailing PRIMARY/secondary tags
    so rich names like 'x_t (decay-corrected, PRIMARY)' become 'x_t'.

    Bug 8 from run #3: only 1 of 5 plots generated because the rich
    quantity names the reasoner emitted couldn't match _CANONICAL_ALIASES.
    Same root cause as verify.py Bug 7 — applied here independently
    so the plot helper doesn't depend on verify.py internals.
    """
    import re
    n = (name or "").strip().lower()
    # Strip parenthesized qualifier(s)
    n = re.sub(r"\s*\([^)]*\)\s*", " ", n)
    # Strip trailing PRIMARY / secondary / commas / dots
    n = re.sub(r"\s*[*,.]+\s*$|\s*\b(primary|secondary)\b\s*$",
               "", n, flags=re.IGNORECASE)
    return " ".join(n.split()).strip()


def _pull_findings(final: dict[str, Any] | None) -> dict[str, float]:
    """Flatten `key_findings` into {canonical_name: float_value}.

    Tolerates missing/malformed entries — returns whatever we could
    parse, nothing more.

    Bug 8 fix: applies _normalise_finding_name() BEFORE alias-matching
    so rich names like 'x_t (decay-corrected, PRIMARY)' get matched to
    the 'x_t' canonical via the existing _CANONICAL_ALIASES.  Previously
    only the rare bare 'x_t' entry matched, so all 4 of {probe_layout,
    intermittency, bl_thickness, onset_comparison} plots silently
    skipped because no x_t was found.
    """
    out: dict[str, float] = {}
    if not final:
        return out
    findings = final.get("key_findings") or []
    if not isinstance(findings, list):
        return out
    # Sub-bug from the Bug 8 fix-verification: the FINAL block sometimes
    # lists the inlet-Tu (no-decay) value BEFORE the decay-corrected
    # PRIMARY value.  "First match wins" picked the wrong one.  Fix:
    # do two passes — first take entries whose name contains "primary",
    # then fill in remaining slots with non-primary entries.
    def _is_primary(raw_name: str) -> bool:
        return "primary" in raw_name.lower()

    # Sort: PRIMARY entries first, others second.  Stable order.
    sorted_findings = sorted(
        (e for e in findings if isinstance(e, dict)),
        key=lambda e: 0 if _is_primary(str(e.get("quantity", ""))) else 1,
    )
    for entry in sorted_findings:
        raw_name = str(entry.get("quantity", "")).strip()
        if not raw_name:
            continue
        # Aggressive normaliser FIRST — strips parens, PRIMARY tags
        name = _normalise_finding_name(raw_name)
        # ALSO normalize separators: spaces and hyphens → underscores
        # ('mayle vs ags onset spread' → 'mayle_vs_ags_onset_spread')
        name = name.replace(" ", "_").replace("-", "_")
        val = entry.get("value")
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue
        # Map to canonical name if it matches an alias set
        canonical = None
        for canon, aliases in _CANONICAL_ALIASES.items():
            # Accept exact alias match OR if normalised name starts
            # with the alias (handles 'x_t_decay_corrected' etc.)
            if name in aliases or any(name.startswith(a + "_") or name == a
                                       for a in aliases):
                canonical = canon
                break
        # PRIMARY-first sort above ensures PRIMARY entries register
        # their canonical key first; later non-primary duplicates are
        # skipped by the `if key not in out` guard.
        key = canonical or name
        if key not in out:
            out[key] = val
    return out


# ══════════════════════════════════════════════════════════════════════
# Public entry — generate all plots we can
# ══════════════════════════════════════════════════════════════════════

def generate_plots(
    reasoner: ReasonerTrace,
    flow: FlowConditions,
    plan: Plan | None,
    output_dir: str | Path,
) -> list[PlotArtifact]:
    """Generate the standard plot suite.  Return a list of PlotArtifacts.

    On any matplotlib-missing or per-plot failure, that plot is silently
    skipped — the rest still render.  Never raises.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []  # matplotlib unavailable; ship without plots

    out_dir = Path(output_dir).resolve()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return []

    findings = _pull_findings(reasoner.final if reasoner else None)
    plots: list[PlotArtifact] = []

    # ── 1. Probe-layout strip ──────────────────────────────────────
    try:
        artifact = _plot_probe_layout(findings, flow, out_dir, plt)
        if artifact:
            plots.append(artifact)
    except Exception:
        pass

    # ── 2. Intermittency profile γ(x) ──────────────────────────────
    try:
        artifact = _plot_intermittency(findings, flow, out_dir, plt)
        if artifact:
            plots.append(artifact)
    except Exception:
        pass

    # ── 3. Boundary-layer thickness δ(x) ───────────────────────────
    try:
        artifact = _plot_bl_thickness(findings, flow, out_dir, plt)
        if artifact:
            plots.append(artifact)
    except Exception:
        pass

    # ── 4. Onset comparison (bar) ──────────────────────────────────
    try:
        artifact = _plot_onset_comparison(findings, flow, out_dir, plt)
        if artifact:
            plots.append(artifact)
    except Exception:
        pass

    # ── 5. Tu decay curve ──────────────────────────────────────────
    try:
        artifact = _plot_tu_decay(findings, flow, out_dir, plt)
        if artifact:
            plots.append(artifact)
    except Exception:
        pass

    # ── 6. Skin-friction Cf(x) — canonical transition signature ────
    try:
        artifact = _plot_cf_x(findings, flow, out_dir, plt)
        if artifact:
            plots.append(artifact)
    except Exception:
        pass

    # ── 7. Re_theta(x) growth vs onset criterion ───────────────────
    try:
        artifact = _plot_re_theta_x(findings, flow, out_dir, plt)
        if artifact:
            plots.append(artifact)
    except Exception:
        pass

    return plots


# ══════════════════════════════════════════════════════════════════════
# Individual plot functions — each returns a PlotArtifact or None
# ══════════════════════════════════════════════════════════════════════

def _plot_probe_layout(
    findings: dict[str, float],
    flow: FlowConditions,
    out_dir: Path,
    plt,
) -> PlotArtifact | None:
    """Horizontal strip showing laminar/transition/turbulent regions +
    suggested probe stations."""
    x_t = findings.get("x_t")
    L_tr = findings.get("L_tr")
    x_end = findings.get("x_end") or (x_t + L_tr if x_t and L_tr else None)
    chord = flow.chord_m
    if not (x_t and x_end and chord):
        return None

    fig, ax = plt.subplots(figsize=(10, 2.4))
    # Regime bands
    ax.axvspan(0, x_t, color="#cce7ff", alpha=0.7, label="laminar")
    ax.axvspan(x_t, x_end, color="#fff2b3", alpha=0.7, label="transition")
    ax.axvspan(x_end, chord, color="#ffc9b3", alpha=0.7, label="turbulent")

    # Probe stations: 3 laminar, 5 transition, 3 turbulent
    laminar_probes  = [round(x_t * 0.35, 4), round(x_t * 0.65, 4), round(x_t * 0.90, 4)]
    transition_probes = [
        round(x_t + L_tr * f, 4) for f in (0.05, 0.20, 0.40, 0.60, 0.85)
    ]
    turbulent_probes = [
        round(x_end + (chord - x_end) * f, 4) for f in (0.10, 0.35, 0.70)
    ]

    for x in laminar_probes:
        ax.plot(x, 0.5, "o", color="navy", markersize=8)
    for x in transition_probes:
        ax.plot(x, 0.5, "s", color="darkgoldenrod", markersize=8)
    for x in turbulent_probes:
        ax.plot(x, 0.5, "^", color="firebrick", markersize=8)

    # Key location markers
    ax.axvline(x_t,   color="#1a5fb4", linestyle="--", linewidth=1.5)
    ax.axvline(x_end, color="#a51d2d", linestyle="--", linewidth=1.5)
    ax.text(x_t,   1.05, f"$x_t = {x_t:.3f}$ m",   ha="center", fontsize=9)
    ax.text(x_end, 1.05, f"$x_e = {x_end:.3f}$ m", ha="center", fontsize=9)

    ax.set_xlim(0, chord)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([])
    ax.set_xlabel("x  (m)  —  streamwise distance from leading edge")
    ax.set_title("Predicted transition zone & candidate stations (prior for pilot) — "
                 "laminar (○) / transition (■) / turbulent (▲)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    path = out_dir / "probe_layout.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    n_probes = len(laminar_probes) + len(transition_probes) + len(turbulent_probes)
    return PlotArtifact(
        label="probe_layout",
        path=str(path),
        caption=(f"Agent-1 PREDICTED transition zone with {n_probes} candidate "
                 f"stations (prior only): 3 laminar, 5 transition (γ≈0.05–0.85), "
                 f"3 turbulent, around predicted x_t={x_t:.3f} m, x_e={x_end:.3f} m. "
                 f"Agent 3's pilot measures the true location and sets the final plan."),
    )


def _plot_intermittency(
    findings: dict[str, float],
    flow: FlowConditions,
    out_dir: Path,
    plt,
) -> PlotArtifact | None:
    """γ(x) using Dhawan-Narasimha Eq.1 form if λ is available, else a
    smooth interpolation between x_t and x_end."""
    x_t = findings.get("x_t")
    x_end = findings.get("x_end") or (x_t + findings.get("L_tr", 0))
    lam = findings.get("lambda_dn")
    if not (x_t and x_end and x_end > x_t):
        return None

    import numpy as np
    x = np.linspace(max(0.0, x_t - 0.05), x_end + 0.05, 400)

    if lam and lam > 0:
        # Narasimha 1985 Eq. (4.8) γ = 1 − exp(−0.412 · ξ²),  ξ = (x − x_t)/λ
        # (Universal form. NOT Dhawan-Narasimha 1958 Eq.(1), which had the
        #  early ξ⁴ exponent; that form is obsolete for this profile.)
        xi = np.clip((x - x_t) / lam, 0, None)
        gamma = 1.0 - np.exp(-0.412 * xi ** 2)
        formula_label = r"$\gamma(x) = 1 - \exp(-0.412 \cdot \xi^2)$, Narasimha 1985 Eq.(4.8)"
    else:
        # Smooth interpolation as fallback — quadratic with γ=0 at x_t, γ=1 at x_end
        L = x_end - x_t
        xi = np.clip((x - x_t) / L, 0, 1)
        gamma = xi ** 2 * (3 - 2 * xi)   # smoothstep
        formula_label = r"$\gamma(x)$ interpolated (no $\lambda_{D-N}$ available)"

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, gamma, color="#1c71d8", linewidth=2)
    ax.axvline(x_t,   color="#1a5fb4", linestyle="--", linewidth=1,
               label=f"$x_t = {x_t:.3f}$ m")
    ax.axvline(x_end, color="#a51d2d", linestyle="--", linewidth=1,
               label=f"$x_e = {x_end:.3f}$ m")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
    ax.fill_between(x, 0, gamma, alpha=0.15, color="#1c71d8")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("x  (m)")
    ax.set_ylabel(r"$\gamma$  (intermittency, 0 = laminar / 1 = turbulent)")
    ax.set_title(f"Intermittency profile across transition zone — {formula_label}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "intermittency.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return PlotArtifact(
        label="intermittency",
        path=str(path),
        caption=(f"Intermittency γ(x) ramps from 0 (fully laminar) to 1 "
                 f"(fully turbulent) between x_t = {x_t:.3f} m and "
                 f"x_end = {x_end:.3f} m."),
    )


def _plot_bl_thickness(
    findings: dict[str, float],
    flow: FlowConditions,
    out_dir: Path,
    plt,
) -> PlotArtifact | None:
    """Boundary-layer thickness δ(x) — Blasius laminar curve + Schlichting
    turbulent curve, with transition stations marked."""
    x_t = findings.get("x_t")
    x_end = findings.get("x_end") or (x_t + findings.get("L_tr", 0))
    U = flow.velocity_ms
    nu = flow.kinematic_viscosity_m2s
    chord = flow.chord_m
    if not (x_t and x_end and U and nu and chord):
        return None

    import numpy as np
    x = np.linspace(0.001, chord, 400)
    Re_x = U * x / nu
    # Blasius laminar: δ = 5x / √Re_x
    delta_lam_mm = 5.0 * x / np.sqrt(Re_x) * 1000
    # Schlichting turbulent: δ = 0.37 · x / Re_x^0.2
    delta_turb_mm = 0.37 * x / (Re_x ** 0.2) * 1000

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, delta_lam_mm, color="#1c71d8", linewidth=2,
            label=r"Blasius (laminar) $\delta = 5x/\sqrt{Re_x}$")
    ax.plot(x, delta_turb_mm, color="#a51d2d", linewidth=2,
            label=r"Schlichting (turbulent) $\delta = 0.37x/Re_x^{0.2}$")

    # Vertical bands
    ax.axvspan(0, x_t, color="#cce7ff", alpha=0.3)
    ax.axvspan(x_t, x_end, color="#fff2b3", alpha=0.3)
    ax.axvspan(x_end, chord, color="#ffc9b3", alpha=0.3)

    # Mark stations
    for x_st, label in [(x_t, "$x_t$"), (x_end, "$x_e$")]:
        ax.axvline(x_st, color="black", linestyle="--", linewidth=1)
        ax.text(x_st, ax.get_ylim()[1] * 0.95, label, ha="center",
                fontsize=9, backgroundcolor="white")

    ax.set_xlim(0, chord)
    ax.set_ylim(0, max(delta_turb_mm) * 1.1)
    ax.set_xlabel("x  (m)")
    ax.set_ylabel(r"$\delta$  (boundary-layer thickness, mm)")
    ax.set_title("Boundary-layer thickness — use for hot-wire wall-distance placement (~ δ/2)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "bl_thickness.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return PlotArtifact(
        label="bl_thickness",
        path=str(path),
        caption=("Boundary-layer thickness δ vs x. Laminar (Blasius) curve "
                 "applies upstream of x_t; turbulent (Schlichting) downstream "
                 "of x_end. Place hot-wire probe near y = δ/2 at each station."),
    )


def _plot_onset_comparison(
    findings: dict[str, float],
    flow: FlowConditions,
    out_dir: Path,
    plt,
) -> PlotArtifact | None:
    """Bar chart of x_t predictions across onset correlations.

    Falls back to a single bar if only one correlation was computed.
    """
    U = flow.velocity_ms
    nu = flow.kinematic_viscosity_m2s
    if not (U and nu):
        return None

    # Try to extract Re_θ from named correlations; convert each to x_t.
    # Langtry-Menter (2009) Eq.36 row INTENTIONALLY ABSENT — γ-Re_θ-SST
    # is a CFD closure model, deferred to Agent 2; A1's per-method x_t
    # plot must not include it as if it were an algebraic correlation.
    methods = []
    for canon, label in [
        ("re_theta_t_mayle",   "Mayle (1991) Eq.9"),
        ("re_theta_t_ags",     "Abu-Ghannam & Shaw (1980) Eq.3"),
        ("re_theta_t_sh",      "Suzen-Huang (2000) Eq.9"),
    ]:
        re_th = findings.get(canon)
        if re_th and re_th > 0:
            re_x = (re_th / 0.664) ** 2
            x_t = re_x * nu / U
            methods.append((label, x_t))

    # If we only have a single x_t with no Re_θ values, still draw the
    # primary value alone.
    if not methods:
        x_t = findings.get("x_t")
        if x_t:
            methods.append(("Primary (decay-corrected)", x_t))
    if not methods:
        return None

    labels = [m[0] for m in methods]
    values_mm = [m[1] * 1000 for m in methods]   # mm for readability

    fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(methods) + 1)))
    bars = ax.barh(range(len(methods)), values_mm,
                   color=["#1c71d8", "#26a269", "#e5a50a", "#a51d2d"][:len(methods)])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("x_t  (mm)")
    ax.set_title("Transition onset x_t — comparison across correlations")
    for bar, v in zip(bars, values_mm):
        ax.text(bar.get_width() + max(values_mm) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.1f} mm", va="center", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "onset_comparison.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    spread = (max(values_mm) - min(values_mm)) if len(methods) > 1 else 0
    return PlotArtifact(
        label="onset_comparison",
        path=str(path),
        caption=(f"x_t predictions across {len(methods)} algebraic correlation(s). "
                 + (f"Spread: {spread:.1f} mm." if spread else "")),
    )


def _plot_tu_decay(
    findings: dict[str, float],
    flow: FlowConditions,
    out_dir: Path,
    plt,
) -> PlotArtifact | None:
    """Tu(x) decay curve using Fransson power-law with assumed L_ref.

    Only produces if we have inlet Tu and chord.  This is illustrative —
    the actual decay depends on the tunnel grid, which we don't know.
    """
    Tu0 = flow.turbulence_intensity_pct
    chord = flow.chord_m
    if not (Tu0 and chord):
        return None

    import numpy as np
    x = np.linspace(0.001, chord, 400)
    # Power-law decay: Tu(x)/Tu_0 = (1 + x/L_ref)^(-0.3)
    # Plot three bracketing L_ref assumptions (aggressive / typical / gentle).
    fig, ax = plt.subplots(figsize=(9, 4))
    for L_ref, label, style in [
        (0.05, r"$L_{ref}=0.05$ m (aggressive)", "--"),
        (0.20, r"$L_{ref}=0.20$ m (typical)",    "-"),
        (1.00, r"$L_{ref}=1.0$ m (gentle)",      ":"),
    ]:
        Tu_x = Tu0 * (1 + x / L_ref) ** (-0.3)
        ax.plot(x, Tu_x, linestyle=style, linewidth=1.8, label=label)
    ax.axhline(Tu0, color="grey", linewidth=0.7, alpha=0.5)
    x_t = findings.get("x_t")
    if x_t:
        ax.axvline(x_t, color="black", linestyle="--", linewidth=1,
                   label=f"$x_t = {x_t:.3f}$ m")
    ax.set_xlim(0, chord)
    ax.set_ylim(0, Tu0 * 1.1)
    ax.set_xlabel("x  (m)")
    ax.set_ylabel(r"$Tu(x)$  (%)")
    ax.set_title(f"Freestream turbulence decay — Fransson Eq.3.1 power law, Tu(0) = {Tu0}%")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "tu_decay.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return PlotArtifact(
        label="tu_decay",
        path=str(path),
        caption=("Tu decay along plate per Fransson Eq.3.1 for three "
                 "L_ref assumptions. Measure actual decay with bare-flow "
                 "hot-wire before final probe placement."),
    )


def _gamma_of_x(x, x_t: float, x_end: float, lam: float | None):
    """Intermittency γ(x): Dhawan-Narasimha γ = 1 − exp(−0.412·ξ²) when λ is
    known, else a smoothstep between x_t and x_end.  Shared by the Cf blend
    and the intermittency plot so they stay consistent."""
    import numpy as np
    if lam and lam > 0:
        xi = np.clip((x - x_t) / lam, 0, None)
        return 1.0 - np.exp(-0.412 * xi ** 2)
    L = max(x_end - x_t, 1e-9)
    xi = np.clip((x - x_t) / L, 0, 1)
    return xi ** 2 * (3 - 2 * xi)


def _plot_cf_x(
    findings: dict[str, float],
    flow: FlowConditions,
    out_dir: Path,
    plt,
) -> PlotArtifact | None:
    """Skin-friction coefficient Cf(x) — THE canonical transition signature.

    Laminar branch  Cf = 0.664 / √Re_x          (Blasius)
    Turbulent branch Cf = 0.0592 / Re_x^0.2      (1/7-power)
    blended through the zone by the intermittency γ(x):
        Cf(x) = (1-γ)·Cf_lam + γ·Cf_turb
    The dip-then-rise is what an experimentalist/CFD reader looks for first.
    """
    x_t = findings.get("x_t")
    x_end = findings.get("x_end") or (x_t + findings.get("L_tr", 0) if x_t else None)
    U = flow.velocity_ms
    nu = flow.kinematic_viscosity_m2s
    chord = flow.chord_m
    if not (x_t and x_end and x_end > x_t and U and nu and chord):
        return None

    import numpy as np
    x = np.linspace(0.001, chord, 400)
    Re_x = U * x / nu
    cf_lam = 0.664 / np.sqrt(Re_x)
    cf_turb = 0.0592 / (Re_x ** 0.2)
    gamma = _gamma_of_x(x, x_t, x_end, findings.get("lambda_dn"))
    cf = (1.0 - gamma) * cf_lam + gamma * cf_turb

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, cf_lam * 1e3, color="#1c71d8", ls=":", lw=1.2,
            label=r"laminar  $0.664/\sqrt{Re_x}$")
    ax.plot(x, cf_turb * 1e3, color="#a51d2d", ls=":", lw=1.2,
            label=r"turbulent  $0.0592/Re_x^{0.2}$")
    ax.plot(x, cf * 1e3, color="black", lw=2.2, label=r"$C_f(x)$ ($\gamma$-blended)")
    ax.axvspan(0, x_t, color="#cce7ff", alpha=0.25)
    ax.axvspan(x_t, x_end, color="#fff2b3", alpha=0.25)
    ax.axvspan(x_end, chord, color="#ffc9b3", alpha=0.25)
    for x_st, lb in [(x_t, "$x_t$"), (x_end, "$x_e$")]:
        ax.axvline(x_st, color="black", ls="--", lw=1)
        ax.text(x_st, ax.get_ylim()[1] * 0.95, lb, ha="center",
                fontsize=9, backgroundcolor="white")
    ax.set_xlim(0, chord)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("x  (m)")
    ax.set_ylabel(r"$C_f \times 10^{3}$")
    ax.set_title("Skin-friction $C_f(x)$ — laminar dip → transition rise → turbulent")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "cf_x.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return PlotArtifact(
        label="cf_x",
        path=str(path),
        caption=("Skin-friction Cf(x): the canonical transition signature — the "
                 "laminar branch (0.664/√Re_x) rises through the γ-weighted "
                 "transition zone to the turbulent branch (0.0592/Re_x^0.2)."),
    )


def _plot_re_theta_x(
    findings: dict[str, float],
    flow: FlowConditions,
    out_dir: Path,
    plt,
) -> PlotArtifact | None:
    """Momentum-thickness Reynolds number Re_θ(x) = 0.664·√Re_x (Blasius)
    against the onset criterion Re_θt — shows WHERE/WHY onset triggers
    (transition begins where the growing Re_θ crosses the correlation
    threshold)."""
    U = flow.velocity_ms
    nu = flow.kinematic_viscosity_m2s
    chord = flow.chord_m
    if not (U and nu and chord):
        return None

    import numpy as np
    x = np.linspace(0.001, chord, 400)
    Re_x = U * x / nu
    Re_theta = 0.664 * np.sqrt(Re_x)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, Re_theta, color="#1c71d8", lw=2,
            label=r"$Re_\theta(x)=0.664\sqrt{Re_x}$ (Blasius)")
    drew_onset = False
    for canon, lbl, col in [
        ("re_theta_t_mayle", "Mayle onset", "#26a269"),
        ("re_theta_t_ags",   "AGS onset",   "#e5a50a"),
    ]:
        v = findings.get(canon)
        if v and v > 0:
            ax.axhline(v, color=col, ls="--", lw=1.3,
                       label=f"{lbl}  $Re_{{\\theta t}}={v:.0f}$")
            drew_onset = True
    x_t = findings.get("x_t")
    if x_t:
        ax.axvline(x_t, color="black", ls="--", lw=1, label=f"$x_t={x_t:.3f}$ m")
    ax.set_xlim(0, chord)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("x  (m)")
    ax.set_ylabel(r"$Re_\theta$")
    ax.set_title(r"Momentum-thickness Reynolds $Re_\theta(x)$ vs onset criterion")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "re_theta_x.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return PlotArtifact(
        label="re_theta_x",
        path=str(path),
        caption=("Re_θ grows as 0.664·√Re_x; transition onset is where it crosses "
                 "the correlation threshold Re_θt"
                 + (" (dashed)." if drew_onset else ".")),
    )
