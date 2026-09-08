"""RAG collection definitions — semantic naming + role-based split.

Architecture (2026-05-14 refactor):

  ALGEBRAIC PREDICTORS (Agent 1 — runnable in Python):
    algebraic_onset            — Standalone onset predictors (Mayle Eq.9,
                                 AGS Eq.12). Input: flow conditions.
                                 Output: Re_θ,t / x_t directly.

    algebraic_intermittency    — Intermittency / transition-zone formulas
                                 that CONSUME x_t from an onset producer
                                 (DN γ(x), Narasimha γ(x), Mayle Eq.8 σ̂_n).
                                 Cannot run standalone; pair with an
                                 onset_producer in composition.

    stability_methods          — eN method, AFT, Mack N_cr correlation.
                                 Algebraic once N-factor is computed
                                 (which itself needs a stability solver).

    roughness_criteria         — Re_k criteria for roughness-induced
                                 transition. Niche but useful.

  AGENT 2 TERRITORY:
    cfd_models                 — γ-Reθ-SST, kkLω, Suzen-Huang transport,
                                 single-eq γ, V-SA. Require CFD solvers.
                                 Agent 1's query expander does NOT include
                                 this in target_cols.

  VALIDATION:
    benchmark_data             — T3A/T3B + ERCOFTAC + DNS data + V&V studies.

  SECONDARY (context, not extracted-from):
    theory                     — Linear stability + Floquet (Mack, Herbert).
    stability_physics          — Mechanisms (Reed/Saric/Arnal, Kachanov).
    bypass_physics             — Bypass mechanisms (Zaki, Brandt, Westin).
    reviews                    — Major reviews (Mayle 1991, Durbin 2017).
    separation_physics         — Separation-induced (Marxen, Diwan).
    foundations                — Foundational concepts (Emmons, Tani).

  DEPRECATED:
    primary_models             — Retired 2026-05-14. Old chunks still exist
                                 in ChromaDB but the registry no longer
                                 references it as a query target.

Naming reasoning: the legacy "primary_*" / "secondary_*" prefix was a TIER
distinction (answer-producing vs explanation). After the algebraic-vs-CFD
audit, the role of each collection is more specific than its old tier
prefix conveyed. Semantic names make Agent 1's algebraic-only scope
obvious to any reader of the code (or the thesis methodology chapter).
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class CollectionConfig:
    name: str
    rag_tier: str  # "primary" or "secondary" (semantic role)
    description: str
    chunk_size: int
    chunk_overlap: int
    metadata_fields: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# ALGEBRAIC PREDICTORS — Agent 1's runnable library
# ═══════════════════════════════════════════════════════════════════

ALGEBRAIC_ONSET = CollectionConfig(
    name="algebraic_onset",
    rag_tier="primary",
    description=(
        "Standalone algebraic onset predictors — closed-form formulas "
        "for Re_θ,t or x_t given flow conditions (U_inf, Tu, λ_θ, ν). "
        "Canonical entries: Mayle 1991 Eq.9, Abu-Ghannam & Shaw 1980 "
        "Eq.12. Each chunk in this collection contains a formula that "
        "can run standalone in Python — no external solver, no need "
        "for an upstream x_t input."
    ),
    chunk_size=512,
    chunk_overlap=64,
    metadata_fields=["correlation_name", "quantity", "validity_range", "regime", "paper_id", "page_number"],
)

ALGEBRAIC_INTERMITTENCY = CollectionConfig(
    name="algebraic_intermittency",
    rag_tier="primary",
    description=(
        "Intermittency / transition-zone length formulas that CONSUME "
        "an upstream x_t (transition onset) and produce γ(x) or L_tr. "
        "Cannot run standalone — must be composed with an algebraic_onset "
        "predictor. Canonical entries: Dhawan-Narasimha 1958 γ(x), "
        "Narasimha 1985 refined γ(x), and Mayle 1991 Eq.8 spot-formation "
        "rate σ̂_n. The composition_solver pairs these with onset "
        "producers (e.g., Mayle Eq.9 + DN γ(x) → x_t + γ(x) → L_tr)."
    ),
    chunk_size=512,
    chunk_overlap=64,
    metadata_fields=["correlation_name", "quantity", "needs_onset_input", "regime", "paper_id", "page_number"],
)

STABILITY_METHODS = CollectionConfig(
    name="stability_methods",
    rag_tier="primary",
    description=(
        "Stability-based transition prediction — eN method, amplification-"
        "factor transport (AFT), Mack's N_cr correlation, Floquet theory. "
        "Algebraic in their final form (N_cr = -8.43 - 2.4·ln(Tu)) but "
        "may require a stability solver upstream to compute N-factor envelopes. "
        "Canonical entries: Van Ingen 1956 + 2008, Mack 1984, Herbert 1988, "
        "Coder-Maughmer 2014."
    ),
    chunk_size=512,
    chunk_overlap=64,
    metadata_fields=["method", "quantity", "paper_id", "page_number"],
)

ROUGHNESS_CRITERIA = CollectionConfig(
    name="roughness_criteria",
    rag_tier="primary",
    description=(
        "Roughness-induced transition criteria — Braslow's Re_k criterion "
        "and related empirical algebraic forms. Applicable when surface "
        "roughness drives transition before natural/bypass modes."
    ),
    chunk_size=512,
    chunk_overlap=64,
    metadata_fields=["roughness_type", "quantity", "regime", "paper_id", "page_number"],
)


# ═══════════════════════════════════════════════════════════════════
# CFD MODELS — Agent 2's territory (PDE-based)
# ═══════════════════════════════════════════════════════════════════

CFD_MODELS = CollectionConfig(
    name="cfd_models",
    rag_tier="primary",
    description=(
        "CFD-resident transition closures: γ-Reθ-SST (Langtry-Menter 2006/"
        "2009), kkLω (Walters-Cokljat 2008 / Walters-Leylek 2004), Suzen-"
        "Huang γ transport, single-eq γ (Menter 2015, Durbin 2012, Ge-Arolla-"
        "Durbin 2014), 3-eq V-SA, plus implementations and V&V studies. "
        "These models require a CFD solver to evaluate; they are NOT "
        "algebraic standalone. AGENT 2's TERRITORY — Agent 1's query "
        "expander should NOT include this in target_cols."
    ),
    chunk_size=512,
    chunk_overlap=64,
    metadata_fields=["model_name", "equation_type", "validity_range", "regime", "paper_id", "page_number"],
)


# ═══════════════════════════════════════════════════════════════════
# BENCHMARKS — validation data
# ═══════════════════════════════════════════════════════════════════

BENCHMARK_DATA = CollectionConfig(
    name="benchmark_data",
    rag_tier="primary",
    description=(
        "Benchmark datasets and V&V studies: T3A/T3B (ERCOFTAC), Schubauer-"
        "Skramstad classical Tollmien-Schlichting experiment, DNS bypass "
        "(Jacobs-Durbin, Andersson-Brandt), plus V&V studies of CFD models "
        "(Furst, Gonzalez, Liu, Wang, Saru, Ghimire) cross-tagged from "
        "cfd_models."
    ),
    chunk_size=768,
    chunk_overlap=96,
    metadata_fields=["dataset_name", "geometry", "re_range", "tu_range", "paper_id", "page_number"],
)


# ═══════════════════════════════════════════════════════════════════
# SECONDARY — physics narrative, reviews, foundational concepts
# ═══════════════════════════════════════════════════════════════════

THEORY = CollectionConfig(
    name="theory",
    rag_tier="secondary",
    description="Core boundary-layer theory + linear stability theory (Mack), Floquet secondary-instability theory (Herbert).",
    chunk_size=1024,
    chunk_overlap=128,
    metadata_fields=["topic", "paper_id", "page_number"],
)

STABILITY_PHYSICS = CollectionConfig(
    name="stability_physics",
    rag_tier="secondary",
    description="Stability/transition physics — Reed/Saric/Arnal reviews, Kachanov experiments, Yoshikawa-Wesfreid KH instability.",
    chunk_size=1024,
    chunk_overlap=128,
    metadata_fields=["mechanism", "regime", "paper_id", "page_number"],
)

BYPASS_PHYSICS = CollectionConfig(
    name="bypass_physics",
    rag_tier="secondary",
    description="Bypass transition physics — Zaki-Durbin mode interaction, Brandt streak breakdown, Westin/Matsubara Klebanoff modes, Jacobs-Durbin DNS.",
    chunk_size=1024,
    chunk_overlap=128,
    metadata_fields=["mechanism", "paper_id", "page_number"],
)

REVIEWS = CollectionConfig(
    name="reviews",
    rag_tier="secondary",
    description="Major review papers — Mayle 1991 gas-turbine, Durbin 2017 bypass, Dick-Kubacki 2017 γ-Reθ, Narasimha 1985 intermittency, Sreenivasan 1989 re-transition.",
    chunk_size=1024,
    chunk_overlap=128,
    metadata_fields=["topic", "paper_id", "page_number"],
)

SEPARATION_PHYSICS = CollectionConfig(
    name="separation_physics",
    rag_tier="secondary",
    description="Separation-induced transition — Marxen-Henningson, Diwan-Ramesh, McAuliffe-Yaras.",
    chunk_size=1024,
    chunk_overlap=128,
    metadata_fields=["mechanism", "paper_id", "page_number"],
)

FOUNDATIONS = CollectionConfig(
    name="foundations",
    rag_tier="secondary",
    description="Foundational transition concepts — Emmons 1951 spots, Tani 1969 roughness, Reshotko 2001 transient growth.",
    chunk_size=768,
    chunk_overlap=96,
    metadata_fields=["concept", "paper_id", "page_number"],
)


# ═══════════════════════════════════════════════════════════════════
# DEPRECATED — left for backwards compatibility
# ═══════════════════════════════════════════════════════════════════

PRIMARY_MODELS = CollectionConfig(
    name="primary_models",
    rag_tier="deprecated",
    description=(
        "DEPRECATED 2026-05-14. The original 'primary_models' collection "
        "lumped algebraic correlations and CFD-resident PDE models "
        "together, which caused γ-Reθ-SST to surface in Agent 1's "
        "algebraic shortlist. After the audit, content was split into "
        "algebraic_onset, algebraic_intermittency, stability_methods, "
        "and cfd_models based on each paper's dominant algorithmic "
        "character. Old chunks under this name remain in ChromaDB but "
        "this collection is NOT in PRIMARY_COLLECTIONS, so the query "
        "expander does not target it. Safe to delete from ChromaDB "
        "after a clean re-ingest."
    ),
    chunk_size=512,
    chunk_overlap=64,
    metadata_fields=["model_name", "equation_type", "validity_range", "regime", "paper_id", "page_number"],
)

# Legacy aliases (for old code that imports these symbols by name)
# All semantically replaced by the renamed collections above.
PRIMARY_CORRELATIONS = ALGEBRAIC_ONSET           # rough back-compat (loses intermittency role)
PRIMARY_STABILITY    = STABILITY_METHODS
PRIMARY_ROUGHNESS    = ROUGHNESS_CRITERIA
PRIMARY_BENCHMARKS   = BENCHMARK_DATA
PRIMARY_CFD_CLOSURES = CFD_MODELS
SECONDARY_THEORY            = THEORY
SECONDARY_STABILITY_PHYSICS = STABILITY_PHYSICS
SECONDARY_BYPASS            = BYPASS_PHYSICS
SECONDARY_REVIEWS           = REVIEWS
SECONDARY_SEPARATION        = SEPARATION_PHYSICS
SECONDARY_FOUNDATIONS       = FOUNDATIONS


# ═══════════════════════════════════════════════════════════════════
# Registries
# ═══════════════════════════════════════════════════════════════════

# Active primary collections — these are what the query expander offers
# as target_cols. PRIMARY_MODELS is intentionally absent (deprecated).
PRIMARY_COLLECTIONS = [
    ALGEBRAIC_ONSET,
    ALGEBRAIC_INTERMITTENCY,
    STABILITY_METHODS,
    ROUGHNESS_CRITERIA,
    BENCHMARK_DATA,
    CFD_MODELS,
]

SECONDARY_COLLECTIONS = [
    THEORY,
    STABILITY_PHYSICS,
    BYPASS_PHYSICS,
    REVIEWS,
    SEPARATION_PHYSICS,
    FOUNDATIONS,
]

ALL_COLLECTIONS = PRIMARY_COLLECTIONS + SECONDARY_COLLECTIONS

# COLLECTION_MAP includes the deprecated primary_models so any legacy
# code path that looks up "primary_models" by name doesn't crash. New
# code should reference the semantic names directly.
COLLECTION_MAP: dict[str, CollectionConfig] = {
    c.name: c for c in (ALL_COLLECTIONS + [PRIMARY_MODELS])
}


# ═══════════════════════════════════════════════════════════════════
# PAPER → COLLECTION mapping
# ═══════════════════════════════════════════════════════════════════
#
# Each paper is classified by its DOMINANT algorithmic role and the
# CONTENT each chunk would actually contribute. Cross-tagging is used
# when one paper genuinely serves two roles (e.g., Mayle 1991 contains
# both algebraic onset Eq.9 AND algebraic intermittency Eq.8 AND is a
# major review — three legitimate tags).
#
# Last reconciled with disk: 2026-05-14 (algebraic-vs-CFD audit
# + semantic-naming refactor).

PAPER_COLLECTION_MAP: dict[str, list[str]] = {

    # ═══════════════════════════════════════════════════════════════
    # ALGEBRAIC ONSET PREDICTORS — standalone, runnable in Python
    # ═══════════════════════════════════════════════════════════════
    "mayle_1991":            ["algebraic_onset", "algebraic_intermittency", "reviews"],
        # Eq.9 onset (algebraic_onset) + Eq.8 spot rate σ̂_n
        # (algebraic_intermittency, since spot rate → L_tr after pairing
        # with onset) + foundational gas-turbine review.
    "abu_ghannam_shaw_1980": ["algebraic_onset"],
        # Eq.12 Tu + λ_θ onset. Pure algebraic standalone.

    # ═══════════════════════════════════════════════════════════════
    # ALGEBRAIC INTERMITTENCY — needs x_t input, gives γ(x) / L_tr
    # ═══════════════════════════════════════════════════════════════
    "dhawan_narasimha_1958": ["algebraic_intermittency"],
        # γ(x) = 1 - exp[-A(x-x_t)²/L²]. Onset-consuming.
    "narasimha_1985":        ["algebraic_intermittency", "reviews"],
        # Refined γ(x). Onset-consuming. Also major review.

    # ═══════════════════════════════════════════════════════════════
    # STABILITY-BASED METHODS — eN, AFT, Mack N_cr, Floquet
    # ═══════════════════════════════════════════════════════════════
    "van_ingen_1956":        ["stability_methods"],
    "van_ingen_2008":        ["stability_methods"],
    "coder_maughmer_2014":   ["stability_methods"],
        # AFT — algebraic N-factor amplification transport.
    "mack_1984":             ["stability_methods", "theory"],
        # Stability bible + N_cr = -8.43 - 2.4·ln(Tu).
    "herbert_1988":          ["stability_methods", "theory"],
        # Floquet secondary-instability theory.

    # ═══════════════════════════════════════════════════════════════
    # CFD-RESIDENT TRANSITION MODELS — Agent 2 territory
    # ═══════════════════════════════════════════════════════════════
    # γ-Reθ-SST family (Langtry-Menter)
    "langtry_menter_2006":   ["cfd_models"],
    "langtry_menter_2009":   ["cfd_models"],
    "menter_1994":           ["cfd_models"],   # SST k-ω baseline (foundation for γ-Reθ-SST)
    "menter_2015":           ["cfd_models"],
    # kkLω family
    "walters_cokljat_2008":  ["cfd_models"],
    "walters_leylek_2004":   ["cfd_models"],
    # Other PDE-based models
    "suzen_huang_2000":      ["cfd_models"],
    "durbin_2012":           ["cfd_models"],
    "ge_arolla_durbin_2014": ["cfd_models"],
    "zhang_chen_zhao_liu_yan_2022": ["cfd_models"],
    "xia_chen_2016":         ["cfd_models"],
    # Modified γ-Reθ implementations + V&V (dual-tagged with benchmark_data)
    "liu_lu_wang_wang_yan_2022":    ["cfd_models", "benchmark_data"],
    "wang_zhang_li_meng_2015":      ["cfd_models", "benchmark_data"],
    "friedlander_georgiadis_2023":  ["cfd_models"],
    "gonzalez_agrawal_wu_2025":     ["cfd_models", "benchmark_data"],
    "furst_2012":                   ["cfd_models", "benchmark_data"],
    "furst_2013":                   ["cfd_models", "benchmark_data"],
    # Fürst, Straka, Příhoda & Šimurda (2013) EPJ Web Conf 45, 01032.
    # Multi-model comparison: algebraic Straka-Příhoda (in-house), kkLOmega
    # (their OpenFOAM impl), γ-Reθ (ANSYS Fluent). Strongest verbatim source
    # for SSTLM/γ-Reθ T3B over-trip at Tu≈6% — cited in A2 Phase 2
    # diagnostic_citations.SSTLM_HIGH_TU_OVERTRIP.
    "furst_straka_prihoda_simurda_2013": ["cfd_models", "benchmark_data"],
    "saru_ersan_pulat_2025":        ["cfd_models", "benchmark_data"],
    "ghimire_ni_wang_2025":         ["cfd_models", "benchmark_data"],
    # Agent 2 closure-tuning support — convention-source primaries +
    # mesh-refinement primary + a secondary that walks the Spalart-Rumsey
    # analytical-inversion math (Spalart-Rumsey itself is paywalled and
    # is cited via Malan).
    "suluksna_juntasaro_2008":      ["cfd_models", "benchmark_data"],
    "yin_pavesi_yuan_2023":         ["cfd_models", "benchmark_data"],
    "spalart_rumsey_2007":          ["cfd_models"],
    "malan_suluksna_juntasaro_2009": ["cfd_models", "benchmark_data"],

    # ═══════════════════════════════════════════════════════════════
    # ROUGHNESS-INDUCED
    # ═══════════════════════════════════════════════════════════════
    "braslow_1960":               ["roughness_criteria"],
    "ergin_white_2006":           ["roughness_criteria"],
    "gbadebo_hynes_cumpsty_2004": ["roughness_criteria"],

    # ═══════════════════════════════════════════════════════════════
    # PHYSICS / DNS / EXPERIMENT — secondary context + benchmarks
    # ═══════════════════════════════════════════════════════════════
    # Stability physics
    "reed_saric_arnal_1996":   ["stability_physics"],
    "saric_reed_white_2003":   ["stability_physics"],
    "kachanov_1994":           ["stability_physics"],
    "yoshikawa_wesfreid_2011": ["stability_physics"],

    # Bypass transition physics (DNS-rich, several cross-tag as benchmarks)
    "zaki_durbin_2005":                           ["bypass_physics"],
    "brandt_2004":                                ["bypass_physics", "stability_physics"],
    "andersson_brandt_bottaro_henningson_2001":   ["bypass_physics", "stability_physics", "benchmark_data"],
    "fransson_matsubara_alfredsson_2005":         ["bypass_physics", "benchmark_data"],
    "westin_et_al_1994":                          ["bypass_physics", "benchmark_data"],
    "matsubara_alfredsson_2001":                  ["bypass_physics"],
    "jacobs_durbin_2001":                         ["bypass_physics", "benchmark_data"],   # DNS T3A
    "durbin_wu_2007":                             ["bypass_physics"],

    # Reviews (Mayle and Narasimha are cross-tagged from algebraic side)
    "durbin_2017":      ["reviews"],
    "dick_kubacki_2017":["reviews"],
    "sreenivasan_1989": ["reviews", "foundations"],

    # Separation-induced
    "marxen_henningson_2011": ["separation_physics"],
    "diwan_ramesh_2009":      ["separation_physics"],
    "mcauliffe_yaras_2010":   ["separation_physics"],

    # Foundations
    "emmons_1951":   ["foundations"],
    "tani_1969":     ["foundations"],
    "reshotko_2001": ["foundations"],

    # ═══════════════════════════════════════════════════════════════
    # EXPERIMENTAL DATA — Validation / Ground Truth
    # ═══════════════════════════════════════════════════════════════
    "schubauer_skramstad_1947":      ["benchmark_data"],
    "schubauer_skramstad_1947_data": ["benchmark_data"],
    # Note: Coupland 1990 ERCOFTAC T3A/T3B raw data is at
    # bl_pipeline/agent2_cfd/validation/reference_data/ercoftac/
    # (not in RAG as a PDF; consumed directly by metrics.py).

    # ═══════════════════════════════════════════════════════════════
    # MEASUREMENT METHODS + TURBULENT BL FOUNDATIONS — added 2026-05-18
    # via Marker (open-source PDF→md, $0).  These cover hot-wire
    # anemometry practice, log-law / wall-flow physics, and turbulent-
    # BL Reynolds-stress data — all foundational references that
    # Agent 1/3/4 cite when justifying probe-spacing, sampling-rate,
    # and BL-profile reasoning.
    # ═══════════════════════════════════════════════════════════════
    "comte_bellot_1976":                       ["foundations", "reviews"],
        # Annu. Rev. Fluid Mech. 8:209 — hot-wire anemometry review.
        # Cited for traverse practice + standard measurement protocols.
    "kurian_fransson_2009":                    ["foundations", "reviews", "benchmark_data"],
        # Fluid Dyn. Res. 41:021403 — comprehensive review of grid-
        # generated turbulence with verbatim length-scale formula
        # (Eq.8: Λ_x/M = A_Λ (x-x_0)^(1/2) M^(-1/2)) and seven validated
        # grids (LT_{1-5} + A + E).  Primary source for A1's grid-
        # derived Λ_x reasoning when σ + d_bar + x_grid supplied but
        # Λ_x not directly measured.  Replaces the previous hand-rolled
        # estimate_lambda_x_from_grid() helper.
    "roach_1987":                              ["foundations", "reviews", "benchmark_data"],
        # Int. J. Heat Fluid Flow 8(2):82-92 — "The generation of
        # nearly isotropic turbulence by means of grids".  Verbatim
        # Λ_x formula: Eq.(18) Λ_x/d = I·(x/d)^(1/2) with I=0.20 for
        # all four bar grid types (SMR, SMS, PR, PS) — see Table 2.
        # Also gives Tu(x) decay law (Eq.2) and pressure-loss
        # correlations (Eqs.1, 23).  Roach's σ range covers 0.27-0.72,
        # matching typical wind-tunnel grids.  PRIMARY companion to
        # Kurian-Fransson 2009 for A1's grid-derived Λ_x reasoning;
        # the two papers disagree on whether d (Roach) or M (KF)
        # is the relevant scaling length, so the REASONER reports
        # both estimates and the spread.
    "osterlund_johansson_nagib_hites_2000":    ["foundations", "benchmark_data"],
        # Phys. Fluids 12:1 — log-law overlap region experimental
        # validation (κ=0.38, B=4.1, Re_θ>6000 lower bound).
        # Benchmark BL data + log-law citation source.
    "degraaff_eaton_2000":                     ["foundations", "benchmark_data"],
        # J. Fluid Mech. 422:319 — Reynolds-stress scaling of flat-
        # plate TBL via LDA, Re_θ=1430–31000.  Documented log-
        # stretched measurement grid.
    "hutchins_nickels_marusic_chong_2009":     ["foundations"],
        # J. Fluid Mech. 635:103 — hot-wire spatial-resolution effects.
        # Cited for y_min lower bound in probe-placement justifications.
    "stainback_nagabushana_1997":              ["foundations", "reviews"],
        # NASA report — comprehensive HWA review (Bruun substitute,
        # public-domain).
    "pope_2000":                               ["foundations", "theory"],
        # Pope Turbulent Flows textbook — curated chapters 3 + 6 + 7.
        # Ch.7 §7.1.4 is THE log-law citation source (κ formula).
        # Ch.3 covers turbulent statistics (skewness, intermittency).
        # Ch.6 covers Kolmogorov scales + energy spectra.
    "schlichting_gersten_2017":                ["foundations", "theory", "stability_methods"],
        # Schlichting & Gersten 9th ed — curated chapters 2 + 6 + 7 +
        # 15 + 16 + 17 + 18.  Ch.6 has Blasius δ=5x/√Re_x (laminar);
        # Ch.18 §18.2.5 has Schlichting 1/7-power δ=0.37x/Re_x^0.2
        # (turbulent).  Ch.15 covers stability + transition.
    "trefethen_2000":                          ["theory"],
        # Spectral Methods in MATLAB — curated chapters 5 + 6 only.
        # Cited for cosine/Chebyshev spacing in Y_SPACING_PRESETS;
        # not core to BL transition but supports rare two-boundary
        # spacing arguments.

    # ═══════════════════════════════════════════════════════════════
    # DEFERRED — pending Claude API key for vlm_ocr step
    # ═══════════════════════════════════════════════════════════════
    # blair_werle_1980, comte_bellot_corrsin_1971 — PDF moved to
    # data/_deferred/; VLM-cascade ingest stalls on these scans.
    #
    # Future additions identified in 2026-05-14 lit-gap audit (require
    # Claude API for OCR/markdown extraction; user has deferred these):
    #   - solomon_walker_1996      → algebraic_intermittency
    #   - walker_gostelow_1990     → algebraic_intermittency
    #   - michel_1951              → algebraic_onset
    #   - cebeci_smith_1974        → algebraic_onset (textbook excerpt)
}
