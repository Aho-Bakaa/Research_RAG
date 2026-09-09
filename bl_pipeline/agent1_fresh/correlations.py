"""correlations.py — the algebraic-transition formula library for A1.

This module is the SINGLE SOURCE OF TRUTH for the closed-form correlations
that A1's REASONER can plug numbers into.  It exposes:

  - Grid geometry helpers:
        mesh_size_from_solidity(sigma, d_bar)
        virtual_origin_from_grid(x_grid, mesh_size, k=3)
        estimate_lambda_x_from_grid(sigma, d_bar, x_grid, x, alpha=0.1, n=0.4)

  - Freestream-turbulence decay:
        fransson_decay_tu(x, C, x_0, b)            — Eq.3.1, forward
        fit_fransson_decay(tu_x_pairs)             — fit (C, x_0, b) to data

  - Boundary-layer growth:
        blasius_re_theta(x, U, nu)                 — Re_θ at x

  - Onset correlations (Re_θ,t vs Tu):
        ags_re_theta_t(Tu_pct)                      — AGS 1980 Eq.3
        mayle_re_theta_t(Tu_pct)                    — Mayle 1991 Eq.9
        langtry_menter_re_theta_t(Tu_pct)           — LM 2009 ZPG-simplified
            DEFERRED TO AGENT 2 — γ-Re_θ-SST is a CFD closure model,
            NOT a stand-alone algebraic A1 correlation.  This function
            stays in the module for back-compat and off-pipeline use
            only; A1's envelope / planner / equation-judge / x_t plot
            paths no longer call it.

  - Length-scale-aware transition (Re_x,t vs Tu, Λ_x):
        fs20_eq35_re_tr(Tu_frac)                    — Fransson-Shahinfar 2020 Tu-only
        fs20_eq36_re_tr(Tu_frac, U, Lambda_x, nu)   — Fransson-Shahinfar 2020 Λ-aware
        gonzalez_re_tr(FSTI, L_in_over_delta_in, gamma=0.5)  — Gonzalez 2025 Eq.4-7

  - Transition zone length:
        ags_l_tr_from_x_t(x_t, U, nu)              — AGS Eq.17/18

  - Orchestrators:
        x_sweep_forward(flow, correlation='AGS', ...)
                       — A1's primary forward predictor (X-sweep on Eq.3.1)
        phase2_back_fit(flow, x_t_measured, varying='Lambda_x')
                       — A1 Phase 2 inverse problem (back-fit a parameter)

All formulas are quoted with their paper / equation source in the docstring
so the formula_verifier (task #102) and REASONER prompt can cross-reference.
Tu units are documented per function — Fransson uses FRACTION (0.028 for
2.8%), Mayle/AGS use PERCENT (2.8).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


# ══════════════════════════════════════════════════════════════════════
# Section 1 — Grid geometry helpers
# ══════════════════════════════════════════════════════════════════════

def mesh_size_from_solidity(sigma: float, d_bar_mm: float) -> float:
    """Solve the biplanar-grid solidity equation for the mesh size M.

    Solidity (projected blockage fraction) for a square mesh of round bars
    of diameter d and spacing M, with bars in two perpendicular directions:
        sigma = 2·(d/M) − (d/M)²
    Inverting: let r = d/M, then r² − 2r + sigma = 0
              r = 1 − √(1 − sigma)     (physical root, r < 1)
              M = d / r

    Parameters
    ----------
    sigma : grid solidity in [0, 1]
    d_bar_mm : bar diameter in mm

    Returns
    -------
    M_mm : mesh size in mm

    Notes
    -----
    Returns NaN if sigma >= 1.0 (unphysical).
    """
    if not (0.0 < sigma < 1.0):
        return float("nan")
    r = 1.0 - math.sqrt(1.0 - sigma)
    return d_bar_mm / r


def virtual_origin_from_grid(
    x_grid_m: float, mesh_size_m: float, k: float = 3.0,
) -> float:
    """Estimate Fransson Eq.3.1 virtual origin x_0 from grid position.

    For passive cylindrical-bar grids, the virtual origin sits k·M upstream
    of the physical grid plane (Comte-Bellot & Corrsin 1966 fits;
    k typically 2-5).  Default k=3 is the central estimate.

    Parameters
    ----------
    x_grid_m : physical grid position relative to LE (m, typically negative)
    mesh_size_m : grid mesh size M in metres
    k : virtual-origin offset coefficient (default 3.0, range 2-5)

    Returns
    -------
    x_0_m : virtual origin in m, in the SAME coordinate system as x_grid
            (x = 0 at LE, x_0 < x_grid < 0)
    """
    return x_grid_m - k * mesh_size_m


def estimate_lambda_x_from_grid(
    sigma: float,
    d_bar_mm: float,
    x_grid_m: float,
    x_m: float = 0.0,
    alpha: float = 0.1,
    n: float = 0.4,
    k_virtual_origin: float = 3.0,
) -> float:
    """⚠ ENGINEERING-BLEND ESTIMATE — provenance is NOT a single citation.

    Returns Λ_x via:
        Λ_x(x) = α · M · ((x − x_0) / M)^n
    with α=0.1, n=0.4, k=3 as defaults.  These are an engineering blend
    of values drawn from multiple sources.  No single paper in our
    corpus reports exactly (α=0.1, n=0.4, k=3).

    THIS HELPER IS DEFERRED TO THE REASONER.  When A1's REASONER needs
    Λ_x from a grid spec, it should:
      1. Call lookup_equation('kurian_fransson_2009', 'Eq.(8)') to fetch
         the verbatim formula (KF 2009 Eq.8 is the cleanest single-source
         primary reference in our corpus).
      2. Write a compute() block tagged `# kurian_fransson_2009::Eq.(8)`
         and evaluate Λ_x there.
      3. The formula_verifier will judge the Python RHS against the
         verbatim LaTeX from the deterministic glossary.

    Kept here as an order-of-magnitude pre-estimate that A1 surfaces in
    its FLOW_CONTEXT block so the OPTIMIZER can decide whether FS20-
    based correlations are reachable.  The number is honest with ±30%
    uncertainty; the SOURCE attribution is not honest until the
    reasoner re-derives it via the lookup_equation path.

    Honest coefficient ranges from the literature:
        α ≈ 0.05–0.2  (varies with grid Re_M; no universal value)
        n ≈ 0.3–0.5   (Comte-Bellot & Corrsin 1966 Table 4)
        k (mesh-widths upstream of grid for x_0):
            Comte-Bellot 1966: 2–5
            Kurian-Fransson 2009 Eq.(8) + Fig.8: k ≈ 0 (x_0 ≈ x_grid)

    Verifiable primary references in the corpus:
        - Comte-Bellot & Corrsin (1966) Table 4 — m exponent, square
          grids σ=0.34: m ∈ {0.34, 0.35, 0.40}.
        - Roach (1987) Eq.(18), p.85 — Λ_x/d = I·(x/d)^(1/2) with
          I = 0.20 UNIVERSAL across SMR/SMS/PR/PS grid types (Table 2).
          σ range 0.27–0.72, 10² < R_d < 10⁴.  Roach scales Λ_x with the
          ROD DIAMETER d, NOT the mesh width M.
        - Kurian & Fransson (2009) Eq.(8), p.17 —
          Λ_x/M = A_Λ (x−x_0)^(1/2) M^(−1/2) with A_Λ = 0.1 for the
          LT₁₋₅ grids at M/d ≈ 4–6.  KF scale Λ_x with MESH WIDTH M.
          Note: KF Eq.(8) and Roach Eq.(18) disagree on which length
          scale matters (M vs d).  The reasoner reports BOTH for the
          user's grid.

    NOTE: Real Λ_x for any specific facility can only be obtained by
    hot-wire autocorrelation at the LE.  Treat any grid-derivation
    output as a pre-experiment estimate, not a measured value.

    Parameters
    ----------
    sigma : grid solidity
    d_bar_mm : bar diameter (mm)
    x_grid_m : physical grid position relative to LE (m, negative if upstream)
    x_m : position where Λ_x is wanted (m; default 0 = at LE)
    alpha, n : Comte-Bellot scaling coefficients
    k_virtual_origin : virtual-origin offset (mesh widths upstream of grid)

    Returns
    -------
    Lambda_x_mm : estimated integral length scale (mm) at x_m

    Returns NaN if (x_m − x_0) ≤ 0 (no downstream distance from virtual origin).
    """
    M_mm = mesh_size_from_solidity(sigma, d_bar_mm)
    if math.isnan(M_mm):
        return float("nan")
    M_m = M_mm * 1e-3
    x_0_m = virtual_origin_from_grid(x_grid_m, M_m, k=k_virtual_origin)
    delta_x = x_m - x_0_m
    if delta_x <= 0:
        return float("nan")
    Lambda_x_m = alpha * M_m * (delta_x / M_m) ** n
    return Lambda_x_m * 1e3   # back to mm


# ══════════════════════════════════════════════════════════════════════
# Section 1.5 — Λ_x via hot-wire autocorrelation (A6's primary output 1)
# ══════════════════════════════════════════════════════════════════════

def autocorrelation_lambda_x(
    u_t: "np.ndarray",
    U_inf: float,
    sample_rate_hz: float,
    *,
    max_lag_pts: int | None = None,
) -> dict:
    """Compute integral length scale Λ_x via hot-wire autocorrelation.

    Uses Taylor frozen-turbulence hypothesis to convert temporal
    autocorrelation into a streamwise length scale:

        Tu(τ)·Tu(τ+δτ)/⟨Tu²⟩       (normalised autocorrelation R_uu)
        T_int = ∫₀^τ_0 R_uu(τ) dτ   (integral up to first zero crossing)
        Λ_x   = U_∞ · T_int          (Taylor hypothesis)

    This is the standard procedure used by Fransson, Matsubara, Shahinfar
    and all modern grid-FST experiments.  It is the GOLD STANDARD method
    for measuring the integral length scale of a wind tunnel rig.

    Parameters
    ----------
    u_t : np.ndarray
        Calibrated velocity time series at the LE station (m/s).
        Mean is subtracted internally before autocorrelation.
    U_inf : float
        Freestream velocity used in the Taylor hypothesis conversion.
        Typically the time-averaged mean of u_t.
    sample_rate_hz : float
        DAQ sample rate.  Used to convert lag samples → physical time.
    max_lag_pts : int, optional
        Cap on the lag range to search for the first zero crossing.
        Default: 1/10 of the time series length (memory-friendly for
        long traces).

    Returns
    -------
    {
        "Lambda_x_m":   float,       # the integral length scale (metres)
        "T_integral_s": float,       # integral of R_uu (seconds)
        "tau_zero_s":   float | None,# location of first zero crossing
        "n_samples":    int,         # samples used in the autocorrelation
        "U_inf":        float,       # echoed back for provenance
        "rms_u_prime":  float,       # √⟨u'²⟩ (Tu = this / U_inf)
    }
    """
    u_t = np.asarray(u_t, dtype=float).flatten()
    n = len(u_t)
    if n < 100:
        return {
            "Lambda_x_m": float("nan"), "T_integral_s": float("nan"),
            "tau_zero_s": None, "n_samples": int(n), "U_inf": float(U_inf),
            "rms_u_prime": float("nan"),
            "_error": "time series too short for autocorrelation (need ≥ 100 samples)",
        }

    u_prime = u_t - np.mean(u_t)
    rms = float(np.sqrt(np.mean(u_prime ** 2)))
    if rms <= 0 or not math.isfinite(rms):
        return {
            "Lambda_x_m": float("nan"), "T_integral_s": float("nan"),
            "tau_zero_s": None, "n_samples": int(n), "U_inf": float(U_inf),
            "rms_u_prime": rms,
            "_error": "zero or non-finite RMS — flow is steady",
        }

    if max_lag_pts is None:
        max_lag_pts = max(100, n // 10)
    max_lag_pts = min(max_lag_pts, n - 1)

    # Biased autocorrelation R_uu(τ) — efficient via FFT convolution.
    # np.correlate is O(n²) for long signals; use scipy.signal.correlate
    # with method='fft' to make this O(n log n).
    from scipy.signal import correlate
    full = correlate(u_prime, u_prime, mode="full", method="fft")
    mid  = len(full) // 2
    R    = full[mid : mid + max_lag_pts]
    R    = R / R[0]   # normalise so R_uu(0) = 1

    # First zero crossing.
    sign_changes = np.where(np.diff(np.sign(R)))[0]
    if len(sign_changes) == 0:
        # No zero crossing found within max_lag_pts; integrate to the cap
        tau_zero_pts = max_lag_pts
        tau_zero_s = None
    else:
        tau_zero_pts = int(sign_changes[0]) + 1
        tau_zero_s = float(tau_zero_pts / sample_rate_hz)

    # Integral T_int = trapezoid(R, dτ) from 0 to first zero
    R_trunc = R[: tau_zero_pts + 1]
    T_int_s = float(np.trapz(R_trunc, dx=1.0 / sample_rate_hz))

    Lambda_x_m = float(U_inf * T_int_s)

    return {
        "Lambda_x_m":   Lambda_x_m,
        "T_integral_s": T_int_s,
        "tau_zero_s":   tau_zero_s,
        "n_samples":    int(n),
        "U_inf":        float(U_inf),
        "rms_u_prime":  rms,
    }


# ══════════════════════════════════════════════════════════════════════
# Section 1.6 — Λ_x multi-method (4 estimators + spread + diagnostics)
# ══════════════════════════════════════════════════════════════════════
#
# Computes Λ_x via FOUR independent methods, returns the spread + warning
# flags + diagnostic signals.  Downstream A6 selector LLM reasons over
# this dict to pick the most-trustworthy method by judgment + RAG-cited
# evidence (smart-A6 architecture — see agent6_turbulent_spots/method_selector.py).
#
# Methods:
#   M1 — Autocorrelation + first zero-crossing truncation
#         Source: Roach 1987 (Int. J. Heat Fluid Flow 8:82) Eq.14
#         Source: Hinze 1975 (Turbulence, 2nd ed) §1.4
#         Canonical method.  Fails when R(τ) doesn't cross zero (LF drift).
#
#   M2 — Autocorrelation + 1/e exponential truncation
#         Source: Trush, Pospíšil & Kozmar 2020 (WIT Press AFM20)
#         T_int = τ where R(τ) = 1/e ≈ 0.368.  Robust to LF drift.
#         Assumes exponential decay — overestimates for non-exponential R(τ).
#
#   M3 — Welch-segmented autocorrelation + first zero-crossing
#         Source: Bendat & Piersol 2010 (Random Data, 4th ed) §8.5
#         Variance reduction at large τ.  Good for noisy data.
#
#   M4 — Spectral E(f→0) limit (integration-free)
#         Source: Roach 1987 Eq.15
#         Λ_x = [E(f)·U / (4·u'²)]_{f→0}.  Independent of integration choice;
#         best cross-check.  Fails if LF noise floor swamps the f→0 region.

def lambda_x_multi_method(
    u_t: "np.ndarray",
    U_inf: float,
    sample_rate_hz: float,
    *,
    detrend: bool = True,
    hp_cutoff_hz: float | None = None,
    welch_segment_len_s: float = 2.0,
    welch_overlap: float = 0.5,
    spectral_lf_fit_fraction: float = 0.05,
) -> dict:
    """Compute Λ_x via 4 independent methods + spread + diagnostics.

    Pre-processing:
      • detrend (default True): subtract a linear trend from u(t) before
        computing fluctuations.  Removes slow tunnel drift.
      • hp_cutoff_hz (optional): if set, apply a Butterworth high-pass
        filter at this cutoff.  Removes acoustic/mechanical LF
        oscillations that can contaminate the autocorrelation tail.

    Returns
    -------
    {
      "values_m": {
        "M1_zero_crossing":  float,
        "M2_one_over_e":     float,
        "M3_welch_zero":     float,
        "M4_spectral":       float,
        "consensus_m":       float,  # median of the four
        "spread_pct":        float,  # (max−min)/median × 100
      },
      "diagnostics": {
        "r_zero_crossing_found":       bool,
        "tau_zero_s":                  float | None,
        "n_integral_scales_record":    float,
        "detrend_applied":             bool,
        "hp_cutoff_hz":                float | None,
        "u_rms_m_s":                   float,
        "Tu_pct":                      float,
        "spectral_lf_plateau_R2":      float,
        "welch_n_segments":            int,
      },
      "warnings":  list[str],        # populated when methods disagree
      "citations": dict[str, str],   # method → primary source
      "_error":    str | None,
    }
    """
    u_t = np.asarray(u_t, dtype=float).flatten()
    n = len(u_t)
    if n < 1000:
        return {"_error": "time series too short (need ≥1000 samples)",
                "values_m": {}, "diagnostics": {}, "warnings": [],
                "citations": {}}

    # ── Pre-processing ─────────────────────────────────────────────
    u_work = u_t.copy()
    if detrend:
        # Linear detrend removes slow drift.  More aggressive than just
        # mean-subtract; recommended by Bruun 1995 §11.3.
        from scipy.signal import detrend as _detrend
        u_work = _detrend(u_work, type="linear")
    if hp_cutoff_hz is not None and hp_cutoff_hz > 0:
        # 2nd-order Butterworth high-pass; filtfilt for zero-phase.
        from scipy.signal import butter, filtfilt
        nyq = sample_rate_hz / 2.0
        b_hp, a_hp = butter(2, hp_cutoff_hz / nyq, btype="high")
        u_work = filtfilt(b_hp, a_hp, u_work)

    u_prime = u_work - np.mean(u_work)
    rms = float(np.sqrt(np.mean(u_prime ** 2)))
    if rms <= 0 or not math.isfinite(rms):
        return {"_error": "zero/non-finite RMS — flow is steady",
                "values_m": {}, "diagnostics": {}, "warnings": [],
                "citations": {}}

    Tu_pct = float(rms / U_inf * 100.0)
    dt = 1.0 / sample_rate_hz

    # ── M1 — Autocorrelation + first zero crossing (canonical) ────
    from scipy.signal import correlate
    max_lag = min(n - 1, n // 4)
    full = correlate(u_prime, u_prime, mode="full", method="fft")
    mid  = len(full) // 2
    R    = full[mid : mid + max_lag]
    R    = R / R[0]
    sign_changes = np.where(np.diff(np.sign(R)))[0]
    if len(sign_changes) == 0:
        zero_found = False
        zero_idx = max_lag - 1
        tau_zero_s = None
    else:
        zero_found = True
        zero_idx = int(sign_changes[0]) + 1
        tau_zero_s = float(zero_idx * dt)
    # np.trapz removed in numpy 2.0 — use trapezoid manually so we don't
    # break on older numpy either.
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    T_int_M1 = float(_trapz(R[: zero_idx + 1], dx=dt))
    Lambda_M1 = float(U_inf * T_int_M1)

    # ── M2 — Autocorrelation + 1/e exponential ────────────────────
    threshold = 1.0 / math.e
    below = np.where(R <= threshold)[0]
    if len(below) == 0:
        # R never reaches 1/e — pathological, use M1 value
        Lambda_M2 = Lambda_M1
        tau_1e_s = None
    else:
        idx_1e = int(below[0])
        tau_1e_s = float(idx_1e * dt)
        # For exponential R(τ) = exp(−τ/T_int), the 1/e point IS T_int
        Lambda_M2 = float(U_inf * tau_1e_s)

    # ── M3 — Welch-segmented autocorrelation + zero crossing ──────
    seg_N = int(welch_segment_len_s * sample_rate_hz)
    if seg_N < 100 or seg_N >= n:
        # Segment too short or longer than record — fall back to M1
        Lambda_M3 = Lambda_M1
        n_segments = 1
    else:
        step = max(1, int(seg_N * (1.0 - welch_overlap)))
        n_segments = (n - seg_N) // step + 1
        R_avg = None
        for m in range(n_segments):
            seg = u_prime[m * step : m * step + seg_N]
            seg_var = float(np.mean(seg ** 2))
            if seg_var <= 0:
                continue
            c = correlate(seg, seg, mode="full", method="fft")
            seg_mid = len(c) // 2
            R_m = c[seg_mid : seg_mid + seg_N // 2] / c[seg_mid]
            if R_avg is None:
                R_avg = R_m
            else:
                R_avg = R_avg + R_m
        R_avg = R_avg / n_segments
        sc3 = np.where(np.diff(np.sign(R_avg)))[0]
        zero_M3 = int(sc3[0]) + 1 if len(sc3) > 0 else len(R_avg) - 1
        T_int_M3 = float(_trapz(R_avg[: zero_M3 + 1], dx=dt))
        Lambda_M3 = float(U_inf * T_int_M3)

    # ── M4 — Spectral E(f→0) limit (Roach 1987 Eq.15) ─────────────
    from scipy.signal import welch
    welch_nperseg = min(n, int(welch_segment_len_s * sample_rate_hz))
    if welch_nperseg < 64:
        welch_nperseg = min(n, 64)
    f, E = welch(u_prime, fs=sample_rate_hz, nperseg=welch_nperseg,
                 noverlap=int(welch_nperseg * welch_overlap),
                 scaling="density")
    # Λ_x = E(f→0) · U / (4 · u'²) — Roach 1987 Eq.15
    # Fit a constant (or low-order polynomial) over the lowest fraction
    # of frequencies, then read off f→0 value.
    n_lf = max(3, int(len(f) * spectral_lf_fit_fraction))
    f_lf = f[1 : n_lf + 1]   # skip f=0 (DC bin can be biased)
    E_lf = E[1 : n_lf + 1]
    if len(E_lf) > 0 and np.all(E_lf > 0):
        # Linear fit in log space → extrapolate to f→0
        # (Roach 1987 implicitly assumes E(f→0) is finite + flat)
        try:
            log_f = np.log(f_lf)
            log_E = np.log(E_lf)
            slope, intercept = np.polyfit(log_f, log_E, 1)
            E_at_f_min = E_lf[0]   # use the lowest measured f (conservative)
            # R² of fit (quality flag)
            resid = log_E - (slope * log_f + intercept)
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((log_E - log_E.mean()) ** 2))
            spectral_R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        except Exception:
            E_at_f_min = E_lf[0]
            spectral_R2 = 0.0
        Lambda_M4 = float(E_at_f_min * U_inf / (4.0 * rms ** 2))
    else:
        Lambda_M4 = float("nan")
        spectral_R2 = 0.0

    # ── Consensus + spread ────────────────────────────────────────
    candidates = [v for v in (Lambda_M1, Lambda_M2, Lambda_M3, Lambda_M4)
                  if math.isfinite(v) and v > 0]
    consensus = float(np.median(candidates)) if candidates else float("nan")
    if candidates and consensus > 0:
        spread_pct = float((max(candidates) - min(candidates))
                           / consensus * 100.0)
    else:
        spread_pct = float("nan")

    # ── Warnings — pattern-match the disagreement signature ───────
    warnings: list[str] = []
    if not zero_found:
        warnings.append(
            "M1 first zero crossing NOT found within max_lag range. "
            "Likely cause: LF drift or insufficient mean subtraction. "
            "Trust M2 (1/e) or M4 (spectral)."
        )
    if math.isfinite(spread_pct) and spread_pct > 30:
        # Diagnose which method is the outlier
        vals = {"M1": Lambda_M1, "M2": Lambda_M2,
                "M3": Lambda_M3, "M4": Lambda_M4}
        med = consensus
        outlier_factor = {k: (v / med) if med > 0 else 1.0 for k, v in vals.items()
                          if math.isfinite(v) and v > 0}
        worst = max(outlier_factor, key=lambda k: abs(outlier_factor[k] - 1.0))
        worst_ratio = outlier_factor[worst]
        warnings.append(
            f"Inter-method spread {spread_pct:.0f}% (>30% threshold). "
            f"{worst} is {worst_ratio:.1f}× the median — likely outlier."
        )
        # Pattern signatures
        if Lambda_M1 > 0 and Lambda_M2 > 0 and Lambda_M1 / Lambda_M2 > 2:
            warnings.append(
                "Signature: M1 ≫ M2 → LF drift inflating zero-crossing integral. "
                "Recommend re-run with hp_cutoff_hz=1.0 (Trush et al. 2020)."
            )
        if Lambda_M4 > 0 and consensus > 0 and Lambda_M4 / consensus > 2:
            warnings.append(
                "Signature: M4 ≫ median → LF noise in spectrum. "
                "Spectral f→0 extrapolation may be biased."
            )
    n_int_scales = float((n * dt) / (consensus / U_inf)) if (
        consensus > 0 and math.isfinite(consensus)
    ) else float("nan")
    if math.isfinite(n_int_scales) and n_int_scales < 100:
        warnings.append(
            f"Record contains only {n_int_scales:.0f} integral scales. "
            f"Hurst-Vassilicos 2007 criterion: need ≥100 (preferably ≥10^4) "
            f"for converged autocorrelation."
        )

    return {
        "values_m": {
            "M1_zero_crossing": Lambda_M1,
            "M2_one_over_e":    Lambda_M2,
            "M3_welch_zero":    Lambda_M3,
            "M4_spectral":      Lambda_M4,
            "consensus_m":      consensus,
            "spread_pct":       spread_pct,
        },
        "diagnostics": {
            "r_zero_crossing_found":     zero_found,
            "tau_zero_s":                tau_zero_s,
            "tau_one_over_e_s":          tau_1e_s,
            "n_integral_scales_record":  n_int_scales,
            "detrend_applied":           detrend,
            "hp_cutoff_hz":              hp_cutoff_hz,
            "u_rms_m_s":                 rms,
            "Tu_pct":                    Tu_pct,
            "spectral_lf_plateau_R2":    spectral_R2,
            "welch_n_segments":          n_segments,
            "n_samples":                 int(n),
            "record_duration_s":         float(n * dt),
        },
        "warnings": warnings,
        "citations": {
            "M1":  "Roach 1987 (Int. J. Heat Fluid Flow 8:82) Eq.14 + "
                   "Hinze 1975 (Turbulence, 2nd ed) §1.4",
            "M2":  "Trush, Pospíšil & Kozmar 2020 (WIT Press AFM20)",
            "M3":  "Bendat & Piersol 2010 (Random Data, 4th ed) §8.5",
            "M4":  "Roach 1987 (Int. J. Heat Fluid Flow 8:82) Eq.15",
            "Taylor_hypothesis":  "Taylor 1938 (Proc. R. Soc. A 164:476)",
            "definition_R_tau":   "Hinze 1975 §1.4",
            "detrend_practice":   "Bruun 1995 (Hot-Wire Anemometry) §11.3",
        },
        "_error": None,
    }


# ══════════════════════════════════════════════════════════════════════
# Section 2 — Freestream-turbulence decay (Fransson, Matsubara & Alfredsson 2005 §3.1)
# ══════════════════════════════════════════════════════════════════════

def fransson_decay_tu(x_m: float, C: float, x_0_m: float, b: float) -> float:
    """Fransson Eq.3.1 — Tu(x) along streamwise direction.

    Tu(x) = C · (x − x_0)^(−b)        Fransson, Matsubara & Alfredsson 2005, Eq.3.1

    CRITICAL: this is the RMS form (LHS is Tu = u_rms / U_∞, NOT u_rms²).
    Exponent is −b, NOT −b/2.  Mistaking the RMS form for energy form
    (u_rms²) is a recurring bug — guarded against by I2 in PLANNER_SYSTEM.

    Parameters
    ----------
    x_m : streamwise position (m, in LE-anchored coordinate)
    C : decay constant (units depend on units of x; C = Tu(0) · (−x_0)^b)
    x_0_m : virtual origin (m, x_0 < 0 for typical upstream grid)
    b : decay exponent (Fransson 2005 default ≈ 0.6 for cylindrical bar grid)

    Returns
    -------
    Tu(x) — Tu value at x (same units as C / Tu boundary condition)

    Returns NaN if (x − x_0) ≤ 0 (point upstream of virtual origin).
    """
    delta = x_m - x_0_m
    if delta <= 0:
        return float("nan")
    return C * delta ** (-b)


def fit_fransson_decay(
    tu_x_pairs: Sequence[tuple[float, float]],
    b_init: float = 0.6,
    x_0_init: float | None = None,
) -> dict:
    """Fit Fransson Eq.3.1 parameters (C, x_0, b) to measured Tu(x_i) data.

    Performs nonlinear least-squares on log(Tu) = log(C) − b·log(x − x_0).
    Returns the fitted parameters plus residuals and a flag for fit quality.

    Parameters
    ----------
    tu_x_pairs : list of (x_m, Tu) measurements; Tu in same units as you want
                 for C (typically percent).  Need >= 3 stations for unique fit.
    b_init : initial guess for decay exponent (0.6 is the canonical default)
    x_0_init : initial guess for virtual origin (m, NEGATIVE for upstream
               grid).  If None, defaults to 2·min(x_i)−1.

    Returns
    -------
    {"C": float, "x_0": float, "b": float, "residuals": ndarray,
     "rms_residual_pct": float, "n_points": int, "converged": bool}
    """
    from scipy.optimize import least_squares
    xs = np.array([p[0] for p in tu_x_pairs], dtype=float)
    tus = np.array([p[1] for p in tu_x_pairs], dtype=float)
    if len(xs) < 3:
        return {
            "C": float("nan"), "x_0": float("nan"), "b": float("nan"),
            "residuals": np.array([]), "rms_residual_pct": float("nan"),
            "n_points": len(xs), "converged": False,
            "_note": "Need >= 3 stations for unique (C, x_0, b) fit.",
        }
    if x_0_init is None:
        x_0_init = 2.0 * float(np.min(xs)) - 1.0
    def residual(params):
        log_C, x_0, b = params
        delta = xs - x_0
        if np.any(delta <= 0):
            return np.full_like(xs, 1e6)
        log_pred = log_C - b * np.log(delta)
        return np.log(tus) - log_pred
    try:
        x_0_init_guarded = min(x_0_init, float(np.min(xs)) - 1e-3)
        res = least_squares(
            residual,
            x0=[float(np.log(np.mean(tus))), x_0_init_guarded, b_init],
        )
        log_C, x_0, b = res.x
        C = math.exp(log_C)
        pred = np.array([fransson_decay_tu(xi, C, x_0, b) for xi in xs])
        rms = float(np.sqrt(np.mean((pred - tus) ** 2)))
        return {
            "C": C, "x_0": x_0, "b": b,
            "residuals": pred - tus,
            "rms_residual_pct": rms,
            "n_points": int(len(xs)),
            "converged": bool(res.success),
        }
    except Exception as e:
        return {
            "C": float("nan"), "x_0": float("nan"), "b": float("nan"),
            "residuals": np.array([]), "rms_residual_pct": float("nan"),
            "n_points": int(len(xs)), "converged": False,
            "_error": str(e),
        }


# ══════════════════════════════════════════════════════════════════════
# Section 3 — Boundary-layer growth (Blasius)
# ══════════════════════════════════════════════════════════════════════

def blasius_re_theta(x_m: float, U: float, nu: float) -> float:
    """Blasius laminar momentum-thickness Reynolds number.

    Re_θ(x) = 0.664 · √(Re_x),   where Re_x = U·x/ν

    Parameters
    ----------
    x_m : position from LE (m, > 0)
    U : freestream velocity (m/s)
    nu : kinematic viscosity (m²/s)

    Returns
    -------
    Re_θ at x.
    """
    if x_m <= 0:
        return 0.0
    Re_x = U * x_m / nu
    return 0.664 * math.sqrt(Re_x)


# ══════════════════════════════════════════════════════════════════════
# Section 4 — Onset correlations: Re_θ,t = f(Tu)
# ══════════════════════════════════════════════════════════════════════

def ags_re_theta_t(Tu_pct: float) -> float:
    """Abu-Ghannam & Shaw 1980 Eq.3 — ZPG transition onset.

    Re_θ,t = 163 + exp(6.91 − Tu)       Tu in PERCENT, valid Tu ∈ [0.3, 5]%

    NOTE on Tu convention: per Dick & Kubacki 2017 p.9, the Tu in this
    correlation is the AVERAGE between the leading edge and the onset
    location, NOT Tu at LE.  Two defensible ways to fix the Tu input:
      (a) X-sweep: at each candidate x, use the LOCAL Tu(x); the smallest
          x satisfying Re_θ(x) ≥ Re_θ,t(Tu(x)) is x_t.  Self-consistent
          but assumes Blasius BL from LE.
      (b) Virtual-origin: when measured δ(x) data are available, fit
          δ = K·√(x − x_0) and use the Tu evaluated AT x_0 (single-shot,
          no iteration).  Anchors the BL to measurements and is preferred
          at Tu > ~1 % where Klebanoff thickening lifts δ above Blasius.
    """
    return 163.0 + math.exp(6.91 - Tu_pct)


def mayle_re_theta_t(Tu_pct: float) -> float:
    """Mayle 1991 Eq.9 — bypass transition onset.

    Re_θ,t = 400 · Tu^(−5/8)            Tu in PERCENT, valid Tu ≥ 3% reliably

    NOTE on Tu < ~3% validity:
    Dick & Kubacki 2017 §3.1 caution that this correlation's empirical
    basis (low-Tu data set) is sparse for Tu < ~3 %, so the fit may be
    extrapolating.  This does NOT mean Mayle systematically over- or
    under-predicts at low Tu — empirically, Mayle Eq.(9) often lands
    within 5% of measured onset for Tu = 2–3% cases where AGS
    over-predicts by 20–30% (the AGS exponential is more Tu-sensitive
    at low Tu than Mayle's power law).  PLANNER should report BOTH
    Mayle Eq.(9) and AGS Eq.(3) as primary co-equal predictions in
    this range and let the experimental anchor (if available) arbitrate.
    DO NOT bias the planner toward AGS-as-primary at low Tu without
    evidence — the historical "AGS preferred at low Tu" heuristic was
    based on a single hand-calculation that did NOT iterate Tu(x_t)
    self-consistently and turned out to be misleading on the senior's
    357 mm anchor (Mayle hit 361 mm = 1% error, AGS hit 449 mm = 26%
    error after self-consistent iteration on the same query).
    """
    return 400.0 * Tu_pct ** (-5.0 / 8.0)


def langtry_menter_re_theta_t(Tu_pct: float) -> float:
    """Langtry-Menter 2009 ZPG-simplified onset correlation.

    For Tu in PERCENT, ZPG, λ_θ = 0:
        Tu ≤ 1.3 : Re_θ,t = 1173.51 − 589.428·Tu + 0.2196·Tu^(−2)
        Tu >  1.3: Re_θ,t = 331.50·(Tu − 0.5658)^(−0.671)

    Numerical limits in the paper: Tu ≥ 0.027 %, Re_θ,t ≥ 20.

    Strictly speaking LM is a transport-equation model; this is the local
    closed-form fit it uses internally and is treated here as a cross-check
    correlation, NOT as a primary algebraic predictor.
    """
    if Tu_pct < 0.027:
        return float("nan")
    if Tu_pct <= 1.3:
        return max(20.0, 1173.51 - 589.428 * Tu_pct + 0.2196 * Tu_pct ** (-2))
    return max(20.0, 331.50 * (Tu_pct - 0.5658) ** (-0.671))


# ══════════════════════════════════════════════════════════════════════
# Section 5 — Length-scale-aware transition: Re_x,t = f(Tu, Λ_x)
# ══════════════════════════════════════════════════════════════════════

# Fransson-Shahinfar 2020 fit coefficients (from §3 of the paper).
# Eq.3.5 — Tu-only:  (Re_tr)_cf^Tu = B_1·Tu^(−2) + B_2  ;  Tu in FRACTION.
_FS20_B1 = 148.0
_FS20_B2 = 31_956.0

# Eq.3.6 — Re_FST-based:  (Re_tr)_cf = C_1·Re_FST^(−2) + C_2  ;  Tu in FRACTION.
_FS20_C1 = 7.6961e9
_FS20_C2 = 62_538.0


def fs20_eq35_re_tr(Tu_frac: float) -> float:
    """Fransson-Shahinfar 2020 Eq.3.5 — Tu-only transition Reynolds number.

    (Re_tr)_cf^Tu = B_1 · Tu^(−2) + B_2,
        (B_1, B_2) = (148, 31 956),  Tu in FRACTION (0.028, NOT 2.8).

    ⚠ CRITICAL — DEFINITION DIFFERS FROM AGS/MAYLE:
    Per FS20 §2.4 (page A23-10), the Re_tr in this paper is defined as
        Re_tr = U_∞ · x_tr / ν   with   x_tr ≡ x_{γ=0.5}
    i.e. MID-TRANSITION (γ = 0.5), NOT transition ONSET.
    AGS Eq.(3) and Mayle Eq.(9) report Re_θ,t at ONSET (γ ≈ 0).
    To compare numerically, subtract ≈ Δx_tr/2 from the FS20 x_tr,
    where Δx_tr = x_{γ=0.9} − x_{γ=0.1} (≈ Fransson 2005 Eq. 5.1 width).

    The subscript 'cf' stands for "empirical curve fit" (FS20 p.16), NOT
    skin-friction.

    Calibration range (FS20 Table 2): Tu ∈ [1.81, 6.19]%, Λ_x ∈ [16.05,
    25.61] mm.  Outside this band the fit extrapolates.

    Empirically observed (page 16) to differ from Eq.3.6 by a factor ~2 in
    the asymptotic Re_FST → ∞ limit — both fits are valid in their own
    Re_FST range; Eq.3.5 implicitly absorbs typical decay for grid-FST
    setups similar to KTH's.
    """
    if Tu_frac <= 0:
        return float("nan")
    return _FS20_B1 * Tu_frac ** (-2) + _FS20_B2


def fs20_eq36_re_tr(
    Tu_frac: float, U: float, Lambda_x_m: float, nu: float,
) -> float:
    """Fransson-Shahinfar 2020 Eq.3.6 — Re_FST-based transition Re_x,t.

    (Re_tr)_cf = C_1 · Re_FST^(−2) + C_2,
        (C_1, C_2) = (7.6961e9, 62 538),
        Re_FST = Tu · Re_Λ,   Re_Λ = U · Λ_x / ν,   Tu in FRACTION.

    Parameters
    ----------
    Tu_frac : Tu as FRACTION (e.g. 0.028 for 2.8%)
    U : freestream velocity (m/s)
    Lambda_x_m : integral length scale at LE (m)
    nu : kinematic viscosity (m²/s)

    Returns
    -------
    Re_x,t — transitional streamwise Reynolds number; x_t = Re_x,t · ν / U.

    ⚠ CRITICAL — DEFINITION DIFFERS FROM AGS/MAYLE:
    Per FS20 §2.4 (page A23-10), Re_tr in this paper is defined as
        Re_tr = U_∞ · x_tr / ν   with   x_tr ≡ x_{γ=0.5}
    i.e. MID-TRANSITION (γ = 0.5), NOT transition ONSET.
    AGS Eq.(3) and Mayle Eq.(9) report Re_θ,t at ONSET (γ ≈ 0).
    The variable name `re_tr` returned here is therefore NOT directly
    comparable to AGS / Mayle / LM Re_x,t.  To compare with onset,
    subtract ≈ Δx_tr/2 where Δx_tr = x_{γ=0.9} − x_{γ=0.1}
    (≈ Fransson 2005 Eq. 5.1 transition-zone 10–90% width).

    The subscript 'cf' stands for "empirical curve fit" (FS20 p.16), NOT
    skin-friction.

    Calibration range (FS20 Table 2): Tu ∈ [1.81, 6.19]%, Λ_x ∈ [16.05,
    25.61] mm.  Outside this band the fit extrapolates; in particular,
    Λ_x < 16 mm pushes Re_FST below the calibration cluster (~200–500)
    and the 1/Re_FST^2 term blows up, leading to gross over-prediction.

    Note: this baseline Eq.3.6 form is NOT yet length-scale-corrected by
    the scale-matching model (Eq.4.7+).  The Eq.4.7 correction needs κ,
    which depends on the regime — applied separately when Λ_x/δ_tr is far
    from optimal (≈ 15).
    """
    if Tu_frac <= 0 or Lambda_x_m <= 0:
        return float("nan")
    Re_Lambda = U * Lambda_x_m / nu
    Re_FST = Tu_frac * Re_Lambda
    return _FS20_C1 * Re_FST ** (-2) + _FS20_C2


def gonzalez_re_tr(
    FSTI_pct: float,
    L_in_over_delta_in: float,
    gamma: float = 0.5,
) -> float:
    """Gonzalez, Agrawal & Wu 2025 Eq.4-7 — DNS-fitted Re_x,t correlation.

    Re_x,t = P1(L/δ) · exp(P2(γ)) · g(FSTI_in)
        P1(L/δ)    = −2.41e-4·(L/δ)^3 + 2.77e-3·(L/δ)^2 + 5.19·(L/δ) + 271.6
        P2(γ)      = 1.68·γ^3 − 2.84·γ^2 + 1.92·γ + 2.72
        g(FSTI)    = 5.23 − exp(5.92 − 1.05·FSTI)        FSTI in PERCENT

    Validity: ZPG flat plate, bypass narrow-sense, FSTI ∈ [0.75, 6]%,
    γ ∈ [0, 1].  At γ=0 it reduces to AGS Eq.3.

    Parameters
    ----------
    FSTI_pct : inlet freestream turbulence intensity in PERCENT
    L_in_over_delta_in : inlet integral length scale / inlet BL thickness
    gamma : intermittency threshold (default 0.5 = mid-transition;
                                     0.0 recovers AGS onset)

    Returns
    -------
    Re_x,t at γ-threshold for this (FSTI, L/δ) pair.
    """
    L_d = L_in_over_delta_in
    P1 = -2.41e-4 * L_d ** 3 + 2.77e-3 * L_d ** 2 + 5.19 * L_d + 271.6
    P2 = 1.68 * gamma ** 3 - 2.84 * gamma ** 2 + 1.92 * gamma + 2.72
    g = 5.23 - math.exp(5.92 - 1.05 * FSTI_pct)
    return P1 * math.exp(P2) * g


# ══════════════════════════════════════════════════════════════════════
# Section 6 — Transition zone length: L_tr = f(x_t)
# ══════════════════════════════════════════════════════════════════════

def ags_l_tr_from_x_t(x_t_m: float, U: float, nu: float) -> float:
    """Abu-Ghannam & Shaw 1980 Eq.17/18 — full transition-zone length.

    R_L = 16.8 · R_XS^0.8,    L_tr = R_L · ν / U
    where R_XS = U·x_t/ν.

    AGS L_tr is the FULL zone (γ ≈ 0 to γ ≈ 0.99) — different from
    Fransson Eq.5.5 (which is 10-90% intermittency width, smaller).
    Use [I4] in PLANNER to compare like-with-like.
    """
    if x_t_m <= 0:
        return 0.0
    R_XS = U * x_t_m / nu
    R_L = 16.8 * R_XS ** 0.8
    return R_L * nu / U


def l_tr_feasibility(
    L_tr_pred_m: float,
    x_t_m: float,
    Tu_LE_pct: float,
) -> dict:
    """Literature-grounded sanity-check on a predicted L_tr.

    Returns a verdict + a recommended L_tr_eff for downstream consumers
    (Agent 3 station planner, Agent 4 y-traverse profile).

    Bypass-transition literature L_tr/x_t bands (Tu_LE ≳ 1%):

      Correlation                       L_tr / x_t band     Notes
      ───────────────────────────────   ─────────────────   ──────────────
      Mayle 1991 Fig.36                 0.20 – 0.80         5%-95% γ zone
      Solomon-Walker 1995 (Tu ~ 2-4%)   0.30 – 0.60         tight bypass
      Abu-Ghannam-Shaw 1980 Eq.17/18    ~ 1.0 – 1.7         FULL zone γ≈0-99%

    The function uses Mayle's wide band as the pass criterion and
    Solomon-Walker's midpoint (0.30·x_t) as the safety floor.  Natural
    transition (Tu_LE < 0.5%) is OUT-OF-SCOPE for this check — the
    bands above do not apply; returns verdict='OUT_OF_SCOPE' and the
    raw prediction.

    Returns
    -------
    dict with keys
        verdict   : 'FEASIBLE' | 'BELOW_MAYLE_BAND' | 'ABOVE_AGS' |
                    'OUT_OF_SCOPE' | 'INVALID_INPUT'
        ratio     : L_tr_pred / x_t (or None if invalid)
        L_tr_eff_m: the value downstream consumers should USE.  When
                    BELOW_MAYLE_BAND, this is max(L_tr_pred, 0.30·x_t).
                    Otherwise it equals L_tr_pred.
        safety_factor : L_tr_eff / L_tr_pred  (1.0 = no adjustment)
        rationale : one-sentence human-readable explanation
        citation  : the literature anchor for the floor / pass
    """
    # ── Input validation ──────────────────────────────────────────
    if L_tr_pred_m is None or x_t_m is None or x_t_m <= 0:
        return {
            "verdict": "INVALID_INPUT",
            "ratio": None,
            "L_tr_eff_m": L_tr_pred_m,
            "safety_factor": 1.0,
            "rationale": "x_t must be > 0 and L_tr must be supplied",
            "citation": "",
        }
    if L_tr_pred_m < 0:
        return {
            "verdict": "INVALID_INPUT",
            "ratio": None,
            "L_tr_eff_m": 0.0,
            "safety_factor": 0.0,
            "rationale": "L_tr cannot be negative",
            "citation": "",
        }

    # ── Regime gate: natural transition is out of band scope ──────
    if Tu_LE_pct is not None and Tu_LE_pct < 0.5:
        return {
            "verdict": "OUT_OF_SCOPE",
            "ratio": L_tr_pred_m / x_t_m,
            "L_tr_eff_m": L_tr_pred_m,
            "safety_factor": 1.0,
            "rationale": (
                f"Tu_LE={Tu_LE_pct:.2f}% is below the bypass-transition "
                "regime (~0.5%); Mayle/Solomon-Walker L_tr/x_t bands do "
                "not apply.  Using L_tr_pred as-is."
            ),
            "citation": "Mayle 1991 §3 (regime threshold)",
        }

    ratio = L_tr_pred_m / x_t_m
    MAYLE_LOWER  = 0.20    # Mayle 1991 Fig.36 lower bound (bypass, wide)
    MAYLE_UPPER  = 0.80    # Mayle 1991 Fig.36 upper bound (bypass, wide)
    SW_LOWER     = 0.30    # Solomon-Walker 1995 lower bound (Tu ≈ 2-4%, tight)
    SW_MIDPOINT  = 0.30    # Solomon-Walker 1995 lower band edge → use as floor
    AGS_UPPER    = 1.70    # AGS Eq.17/18 full zone is the absolute ceiling

    # ── Below Solomon-Walker tight band → apply safety floor ──────
    #
    # We trigger on SW_LOWER (0.30), not on the wider MAYLE_LOWER (0.20),
    # for two reasons:
    #   (1) Solomon-Walker 1995 is the tightest empirical bypass band
    #       in the corpus for Tu = 2-4% — the canonical lab-tunnel range.
    #       A ratio < 0.30 in this regime indicates the prediction is
    #       too short to bracket the 5%-95% γ zone reliably.
    #   (2) The 2026-06-03 pilot exposed A1 emitting ratio = 0.25
    #       (technically inside Mayle's wide band but well below
    #       Solomon-Walker's tight band).  Coverage truncation by A3
    #       cost a complete station list that missed the turbulent
    #       plateau.  This tighter trigger is the regression fix.
    if ratio < SW_LOWER:
        L_tr_eff = SW_MIDPOINT * x_t_m
        return {
            "verdict": "BELOW_MAYLE_BAND",
            "ratio": ratio,
            "L_tr_eff_m": L_tr_eff,
            "safety_factor": L_tr_eff / L_tr_pred_m if L_tr_pred_m > 0 else float("inf"),
            "rationale": (
                f"L_tr_pred={L_tr_pred_m*1000:.1f}mm gives ratio "
                f"L_tr/x_t={ratio:.2f}, below the Solomon-Walker 1995 bypass "
                f"lower bound (0.30) for Tu={Tu_LE_pct:.1f}%.  Using "
                f"L_tr_eff={L_tr_eff*1000:.1f}mm (=0.30·x_t) as safety "
                f"floor for station planning."
            ),
            "citation": "Solomon-Walker 1995 (tight band); Mayle 1991 Fig.36",
        }

    # ── Above AGS full-zone ceiling → flag but don't change ───────
    if ratio > AGS_UPPER:
        return {
            "verdict": "ABOVE_AGS",
            "ratio": ratio,
            "L_tr_eff_m": L_tr_pred_m,
            "safety_factor": 1.0,
            "rationale": (
                f"L_tr_pred={L_tr_pred_m*1000:.1f}mm gives ratio "
                f"L_tr/x_t={ratio:.2f}, above AGS Eq.17/18 full-zone "
                f"ceiling (1.7).  Unusual but not adjusted; consider "
                "natural-transition or laminar-separation regime."
            ),
            "citation": "Abu-Ghannam-Shaw 1980 Eq.17/18",
        }

    # ── Inside Mayle band → feasible, use as-is ────────────────────
    return {
        "verdict": "FEASIBLE",
        "ratio": ratio,
        "L_tr_eff_m": L_tr_pred_m,
        "safety_factor": 1.0,
        "rationale": (
            f"L_tr_pred={L_tr_pred_m*1000:.1f}mm gives ratio "
            f"L_tr/x_t={ratio:.2f}, inside the Mayle 1991 bypass band "
            f"[0.20, 0.80].  Using A1's prediction directly."
        ),
        "citation": "Mayle 1991 Fig.36 (bypass band)",
    }


# ══════════════════════════════════════════════════════════════════════
# Section 7 — Orchestrators: X-sweep forward + Phase 2 inverse fit
# ══════════════════════════════════════════════════════════════════════

@dataclass
class XSweepResult:
    """Result of one forward X-sweep on Fransson Eq.3.1 + chosen onset correlation."""
    x_t_m: float                # transition ONSET location (m) — γ≈0 for ALL correlations
    Re_theta_t: float           # Re_θ at x_t
    Tu_at_x_t_pct: float        # local Tu at x_t (percent)
    correlation: str            # which onset correlation was used
    n_sweep_points: int         # how many x candidates were checked
    converged: bool             # whether a crossing was found within chord
    # FS20 Eq.(3.6) and Gonzalez natively report the γ=0.5 (mid-transition)
    # location, NOT onset.  x_t_m above is the ONSET-corrected value so it is
    # apples-to-apples with AGS/Mayle; the raw γ=0.5 value is kept here for
    # transparency, and gamma_note records exactly how x_t_m was obtained.
    x_mid_transition_m: float | None = None   # native γ=0.5 location (FS20/Gonzalez)
    gamma_note: str = ""                        # provenance of the onset correction


def x_sweep_forward(
    *,
    Tu_LE_pct: float,
    U: float,
    nu: float,
    chord_m: float,
    x_0_m: float,
    b: float = 0.6,
    correlation: str = "AGS",
    Lambda_x_m: float | None = None,
    n_steps: int = 800,
) -> XSweepResult:
    """Forward X-sweep on Fransson Eq.3.1 — A1's primary forward predictor.

    For each candidate x in [chord/n_steps, chord], evaluates Tu(x) via
    Fransson Eq.3.1 (decay), Re_θ(x) via Blasius, and Re_θ,t(Tu(x)) via the
    chosen correlation.  Returns the smallest x where Re_θ(x) ≥ Re_θ,t(Tu(x)).

    Solves the "which Tu to plug in" question by using LOCAL Tu(x) at every
    step — never inlet Tu_0, never hand-picked midway-Tu.

    Parameters
    ----------
    Tu_LE_pct : Tu at LE in PERCENT (boundary condition for decay)
    U : freestream velocity (m/s)
    nu : kinematic viscosity (m²/s)
    chord_m : plate length (m)
    x_0_m : virtual origin (m, negative — upstream of LE)
    b : Fransson decay exponent (default 0.6)
    correlation : "AGS" | "Mayle" | "LM" | "FS20_eq36" | "Gonzalez"
                  ("FS20_eq36" requires Lambda_x_m to be provided)
    Lambda_x_m : integral length scale at LE (m) — only for FS20_eq36/Gonzalez
    n_steps : number of x candidates to sweep (default 800)
    """
    # Derive C from boundary condition: Tu(0) = Tu_LE.
    if x_0_m >= 0:
        raise ValueError("x_0_m must be negative (upstream of LE)")
    C = Tu_LE_pct * (-x_0_m) ** b

    # ── FS20 / Gonzalez are Re_x-based and CALIBRATED FOR INLET Tu. ────
    # Both papers fit their correlations using LE-measured Tu, not local
    # Tu(x_t).  So we evaluate them DIRECTLY at inlet conditions and skip
    # the X-sweep entirely.
    #
    # ⚠ ONSET CORRECTION (task: FS20 onset-vs-mid-transition, prompts.py I9):
    # FS20 Eq.(3.6) and Gonzalez Eq.4-7 (at γ=0.5) return the MID-TRANSITION
    # location x_{γ=0.5}, NOT onset.  AGS/Mayle fill x_t_m with ONSET (γ≈0).
    # Placing the γ=0.5 value in x_t_m biases FS20 a full transition half-length
    # too far downstream and makes any "correlation spread" claim apples-to-
    # oranges.  We therefore convert to a TRUE onset before assigning x_t_m:
    #   • FS20: subtract the Dhawan-Narasimha back-step 1.297·Λ_DN, where
    #       Λ_DN = R_λ·ν/U,  R_λ = 5.0·Re_x,½^0.8,  and
    #       1.297 = √(ln2/0.412) is the ξ at which γ = 1−exp(−0.412·ξ²) = 0.5.
    #   • Gonzalez: evaluate its own γ-parametrised fit at γ=0 (its onset form,
    #       which reduces to AGS Eq.3), avoiding any back-step approximation.
    # The native γ=0.5 value is preserved in XSweepResult.x_mid_transition_m.
    if correlation in ("FS20_eq36", "Gonzalez"):
        if Lambda_x_m is None:
            raise ValueError(
                f"{correlation} requires Lambda_x_m (the LE integral length scale)"
            )
        x_mid = None
        note = ""
        if correlation == "FS20_eq36":
            Re_x_mid = fs20_eq36_re_tr(Tu_LE_pct / 100.0, U, Lambda_x_m, nu)
            if not (Re_x_mid and Re_x_mid > 0):
                return XSweepResult(
                    x_t_m=float("nan"), Re_theta_t=float("nan"),
                    Tu_at_x_t_pct=float("nan"), correlation=correlation,
                    n_sweep_points=0, converged=False,
                )
            x_mid = Re_x_mid * nu / U                       # x_{γ=0.5} (mid-transition)
            R_lambda = 5.0 * Re_x_mid ** 0.8                # Dhawan-Narasimha length scale
            Lambda_DN = R_lambda * nu / U
            backstep = 1.297 * Lambda_DN                    # 1.297 = √(ln2/0.412)
            x_t = x_mid - backstep                          # → true onset (γ≈0)
            note = (f"FS20 Eq.3.6 native x_{{γ=0.5}}={x_mid:.4f} m; onset = "
                    f"x_{{γ=0.5}} − 1.297·Λ_DN ({backstep:.4f} m, Dhawan-Narasimha)")
        else:  # Gonzalez — use its own γ=0 onset form (no back-step needed).
            Re_x_onset = gonzalez_re_tr(Tu_LE_pct, L_in_over_delta_in=15.0, gamma=0.0)
            Re_x_mid   = gonzalez_re_tr(Tu_LE_pct, L_in_over_delta_in=15.0, gamma=0.5)
            if not (Re_x_onset and Re_x_onset > 0):
                return XSweepResult(
                    x_t_m=float("nan"), Re_theta_t=float("nan"),
                    Tu_at_x_t_pct=float("nan"), correlation=correlation,
                    n_sweep_points=0, converged=False,
                )
            x_t = Re_x_onset * nu / U
            x_mid = (Re_x_mid * nu / U) if (Re_x_mid and Re_x_mid > 0) else None
            note = "Gonzalez Eq.4-7 evaluated at γ=0 (its own onset form; reduces to AGS Eq.3)"
        # Onset must be a physical, in-chord value (a back-step larger than
        # x_{γ=0.5} pushes onset ≤ 0 → FS20 out of its calibrated regime).
        if not np.isfinite(x_t) or x_t <= 0 or x_t > chord_m:
            return XSweepResult(
                x_t_m=float("nan"), Re_theta_t=float("nan"),
                Tu_at_x_t_pct=float("nan"), correlation=correlation,
                n_sweep_points=0, converged=False,
                x_mid_transition_m=(float(x_mid) if x_mid is not None else None),
                gamma_note=note + " — onset non-physical (≤0 or >chord)",
            )
        Re_th_at_x_t = blasius_re_theta(x_t, U, nu)
        Tu_at_x_t   = fransson_decay_tu(x_t, C, x_0_m, b)  # reported for context only
        return XSweepResult(
            x_t_m=float(x_t), Re_theta_t=float(Re_th_at_x_t),
            Tu_at_x_t_pct=float(Tu_at_x_t), correlation=correlation,
            n_sweep_points=0, converged=True,   # 0 sweep points → direct evaluation
            x_mid_transition_m=(float(x_mid) if x_mid is not None else None),
            gamma_note=note,
        )

    # ── Mayle 1991: NO ITERATION — use Tu @ LE directly ──────────────
    # Per user 2026-06-15: Mayle re-evaluates Re_θ_t with Tu_LE (no Tu(x_t)
    # iteration), then solves Re_θ(x) = Re_θ_t via Blasius for x_t.  This
    # is a non-iterative single-shot use of the correlation.
    if correlation == "Mayle":
        Re_th_t = mayle_re_theta_t(Tu_LE_pct)
        # Blasius: Re_θ = 0.664·sqrt(Re_x)  →  Re_x = (Re_θ/0.664)²
        Re_x_t = (Re_th_t / 0.664) ** 2
        x_t = Re_x_t * nu / U
        if not np.isfinite(x_t) or x_t <= 0 or x_t > chord_m:
            return XSweepResult(
                x_t_m=float("nan"), Re_theta_t=float("nan"),
                Tu_at_x_t_pct=float("nan"), correlation="Mayle",
                n_sweep_points=0, converged=False,
            )
        return XSweepResult(
            x_t_m=float(x_t), Re_theta_t=float(Re_th_t),
            Tu_at_x_t_pct=float(Tu_LE_pct),  # we used Tu_LE, not Tu(x_t)
            correlation="Mayle",
            n_sweep_points=0, converged=True,
        )

    # ── AGS 1980: iterate as usual, THEN re-evaluate at Tu(x_t/2) ────
    # Per user 2026-06-15: AGS sweeps through x with local Tu(x) until
    # Re_θ(x) ≥ Re_θ,t(Tu(x)) — same as before — to find a converged x_t.
    # Then the FINAL AGS equation is re-evaluated using Tu(x_t/2) — the Tu
    # at half the converged onset — and the final x_t is recomputed from
    # that Re_θ_t via Blasius.  The reported Tu_at_x_t_pct is Tu(x_t/2).
    if correlation == "AGS":
        x_t_iter = float("nan")
        x_grid = np.linspace(chord_m / n_steps, chord_m, n_steps)
        for x in x_grid:
            Tu_x = fransson_decay_tu(x, C, x_0_m, b)
            Re_th = blasius_re_theta(x, U, nu)
            Re_th_t = ags_re_theta_t(Tu_x)
            if Re_th >= Re_th_t:
                x_t_iter = float(x)
                break
        if not np.isfinite(x_t_iter):
            return XSweepResult(
                x_t_m=float("nan"), Re_theta_t=float("nan"),
                Tu_at_x_t_pct=float("nan"), correlation="AGS",
                n_sweep_points=n_steps, converged=False,
            )
        # Re-evaluate AGS at Tu(x_t/2)
        x_half = 0.5 * x_t_iter
        Tu_half = fransson_decay_tu(x_half, C, x_0_m, b)
        Re_th_t_final = ags_re_theta_t(Tu_half)
        Re_x_t_final = (Re_th_t_final / 0.664) ** 2
        x_t_final = Re_x_t_final * nu / U
        if not np.isfinite(x_t_final) or x_t_final <= 0 or x_t_final > chord_m:
            return XSweepResult(
                x_t_m=float("nan"), Re_theta_t=float("nan"),
                Tu_at_x_t_pct=float("nan"), correlation="AGS",
                n_sweep_points=n_steps, converged=False,
            )
        return XSweepResult(
            x_t_m=float(x_t_final), Re_theta_t=float(Re_th_t_final),
            Tu_at_x_t_pct=float(Tu_half),       # Tu @ x_t/2 (per user spec)
            correlation="AGS",
            n_sweep_points=n_steps, converged=True,
        )

    if correlation == "LM":
        raise ValueError(
            "correlation 'LM' (Langtry-Menter 2009) is a CFD closure "
            "model deferred to Agent 2; the A1 x_t sweep no longer "
            "treats it as an algebraic correlation"
        )
    raise ValueError(f"unknown correlation '{correlation}'")


def phase2_back_fit(
    *,
    x_t_measured_m: float,
    Tu_LE_pct: float,
    U: float,
    nu: float,
    chord_m: float,
    x_0_m: float,
    b: float = 0.6,
    correlation: str = "AGS",
    varying: str = "Lambda_x",
    search_range: tuple = (1.0, 30.0),
    n_steps: int = 200,
) -> dict:
    """A1 Phase 2 retrospective — back-fit a parameter to match A5 measurement.

    Given A5's measured x_t, sweep ONE parameter to find the value that
    makes the forward predictor reproduce x_t_measured.  Currently supports:
      - varying="Lambda_x"  : sweep Λ_x in `search_range` (mm).  Uses FS20.
      - varying="b"         : sweep Fransson decay exponent.
      - varying="Tu_LE"     : sweep inlet Tu in `search_range` (percent).

    Returns the fitted value, residual, and a diagnostic note.  Used to
    diagnose WHICH assumption was wrong when A1 Phase 1 disagreed with A5.
    """
    candidates = np.linspace(search_range[0], search_range[1], n_steps)
    best_val, best_residual = None, float("inf")
    for cand in candidates:
        if varying == "Lambda_x":
            r = x_sweep_forward(
                Tu_LE_pct=Tu_LE_pct, U=U, nu=nu, chord_m=chord_m,
                x_0_m=x_0_m, b=b,
                correlation="FS20_eq36",
                Lambda_x_m=float(cand) * 1e-3,
            )
        elif varying == "b":
            r = x_sweep_forward(
                Tu_LE_pct=Tu_LE_pct, U=U, nu=nu, chord_m=chord_m,
                x_0_m=x_0_m, b=float(cand), correlation=correlation,
            )
        elif varying == "Tu_LE":
            r = x_sweep_forward(
                Tu_LE_pct=float(cand), U=U, nu=nu, chord_m=chord_m,
                x_0_m=x_0_m, b=b, correlation=correlation,
            )
        else:
            raise ValueError(f"unknown varying='{varying}'")
        if not r.converged:
            continue
        residual = abs(r.x_t_m - x_t_measured_m)
        if residual < best_residual:
            best_residual = residual
            best_val = cand
    return {
        "varying": varying,
        "fitted_value": float(best_val) if best_val is not None else float("nan"),
        "residual_m": float(best_residual) if best_val is not None else float("nan"),
        "x_t_target_m": x_t_measured_m,
        "search_range": search_range,
        "n_candidates_tried": n_steps,
        "converged": best_val is not None,
    }
