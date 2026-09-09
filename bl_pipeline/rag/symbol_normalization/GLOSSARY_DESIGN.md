# Glossary design — paper-aware symbol normalization

*Last reviewed: 2026-05.  Read this before editing `canonical_symbols.yaml`.*

## Why this exists

The retrieval failure we hit in May 2026 was a clean **vocabulary mismatch**:
Abu-Ghannam & Shaw 1980 contains the onset formula
`R_{θ,S} = 163 + exp[F(λ_θ)·(1 - τ_t/6.91)]` on page 7, but the user's
query about `Re_θ,t correlation Tu` retrieves Mayle and Dhawan-Narasimha
chunks first — because AGS writes the same physical quantity using
different LaTeX tokens (`R_{θ,S}` vs `Re_θ,t`, `τ_t` vs `Tu`).  The
embedding model represents these as different concepts in vector space,
so the AGS chunk ranks low even though it is the most relevant one.

The fix is to **normalize symbol notation at INGEST time** so every
paper's chunks embed under canonical tokens.  But several symbols
mean **different things in different papers** — bare `τ_0` is wall
shear stress in Mayle 1991 and (separately) wall shear in
Suzen-Huang 2000 papers that adopt that convention, while `τ` in
Mayle means wake-passing period (seconds!).  A naive global rewrite
of `τ_0 → τ_w` would mangle papers that use the symbol differently.

## The architecture

The glossary has TWO alias tables per canonical:

```yaml
turbulence_intensity:
  canonical: Tu
  display: Tu
  unit: '%'
  description: Freestream turbulence intensity
  aliases:                                  # ← GLOBAL: applied to every chunk
  - FSTI
  - Ti
  - turbulence intensity
  paper_specific_aliases:                   # ← PAPER-SCOPED: applied only
    abu_ghannam_shaw_1980:                  #   when ingesting a chunk whose
    - τ_t                                   #   paper_id matches
    - τₜ
    - \tau_t
    narasimha_1985:
    - q
```

The `SymbolNormalizer.normalize_text_with_log(text, paper_id=…)`
method runs in **two passes**:

1. **Paper-specific pass** — when `paper_id` is supplied, the
   normalizer applies the paper-scoped aliases for that paper first.
   Each one is regex-substituted with word-boundary matching.
2. **Global pass** — the standard alias table is then applied.

Paper-specific aliases win where they exist; global aliases handle
everything else.

The ingest path (`bl_pipeline/rag/ingestion.py`) passes `paper_id`
to `normalize_text` for every chunk it builds, so the per-paper
notation gets canonicalized BEFORE the chunk text is embedded into
ChromaDB.  This is what lets cross-paper retrieval find AGS's
onset chunk when the query talks about `Re_θ,t`.

## Known symbol collisions

Edit this table when you add new papers — a fresh paper's nomenclature
should be checked against every line below before adding global aliases.

| Symbol | Meaning A | Meaning B | Resolution |
|---|---|---|---|
| `τ` | Wall shear stress (LM, modern RANS) | Wake-passing period — seconds (Mayle 1991) | Paper-scoped: τ→τ_w only for `langtry_menter_2009`; τ stays as wake-period in `mayle_1991` |
| `τ_0` | Wall shear stress (Mayle 1991) | Stagnation point shear, reference shear (other papers) | Paper-scoped: τ_0→τ_w only for `mayle_1991` |
| `τ_t` | Turbulence intensity in percent (AGS 1980) | Reynolds shear stress (modern RANS literature) | Paper-scoped: τ_t→Tu only for `abu_ghannam_shaw_1980` |
| `ω` | Specific dissipation rate (modern k-ω models) | Wake-passing frequency — Hz (Mayle 1991) | Paper-scoped: ω→ω_wake only for `mayle_1991`; global ω→ω stays as dissipation |
| `K` | Acceleration parameter (Mayle 1991, canonical) | Turbulent kinetic energy (Narasimha 1985) | Paper-scoped: K→k only for `narasimha_1985` |
| `k` | Turbulent kinetic energy (LM, canonical) | Thermal conductivity (Mayle 1991) | Mayle's k stays (no rewrite) — too risky for single-char alias |
| `k_L` | Laminar kinetic energy (Walters-Cokljat) | "Laminar k" — sometimes legacy alias for TKE | Global: k_L→laminar_kinetic_energy; removed from TKE global aliases |
| `n` | N-factor exponent (e^N method) | Spot formation rate (Mayle, DN, Narasimha) | Distinct canonicals: n_factor vs spot_formation_rate; aliases never overlap |
| `λ` | Pressure gradient parameter (when subscripted: λ_θ) | Intermittency-zone length x_(γ=0.75) − x_(γ=0.25) (DN 1958, Narasimha 1985) | Paper-scoped: bare λ→L_γ for `dhawan_narasimha_1958` and `narasimha_1985` |
| `λ_0` | Pressure gradient parameter (Mayle 1991's notation) | Reference wavelength (other contexts) | Paper-scoped: λ_0→λ_θ only for `mayle_1991` |
| `Λ` | Integral length scale (canonical) | Same — Mayle writes it as `L` | Paper-scoped: L→Λ for `mayle_1991` and `narasimha_1985` (turbulence macroscale) |
| `L` | Length of transition (AGS, Mayle subscript LT) | Integral length scale (Mayle, Narasimha) | DEPENDS on context — paper-scoped: `mayle_1991`+`narasimha_1985` send L→Λ; `abu_ghannam_shaw_1980` sends L→Δx_tr |
| `δ_0` | Momentum thickness (Mayle 1991's notation) | BL thickness at reference station (other papers) | Paper-scoped: δ_0→θ only for `mayle_1991` |
| `R_T` | Viscosity ratio μ_t/μ (LM 2009, Walters-Cokljat) | Correlation coefficient (AGS nomenclature) | Paper-scoped: R_T→viscosity_ratio only for `langtry_menter_2009`+`walters_cokljat_2008` |
| `D` | Sphere diameter (AGS) | Leading-edge diameter (Mayle) | Both → roughness_diameter is wrong; no global alias; left alone |
| `N` | N-factor (e^N stability) | Non-dimensional spot rate "crumble" (Narasimha 1985) | Paper-scoped: N→N_crumble only for `narasimha_1985` |
| `β` | Half-angle of spot envelope (Narasimha) | Falkner-Skan parameter (Narasimha §7) | Single-char alias not added — too ambiguous even per-paper |
| `R` | Mack stability Re = √(Re_x) | Chord Reynolds (VI 1956); rotating-disk Re (SRW); Re_δ* (Westin); generic Re (canonical) | Paper-scoped: Mack/RSA bare R → mack_stability_reynolds; SS/Westin bare R → re_displacement_thickness; SRW bare R → mack_stability_reynolds (rotating-disk variant) |
| `F` | Mack dimensionless frequency F = ω*ν*/U*² | AGS pressure-gradient function F(λ_θ); LM F_onset/F_length; Emmons cumulative probability; Brandt 2004 volume-forcing amplitude | Paper-scoped: stability F (Mack/VI/RSA/Herbert/Westin/JD/SS) → mack_dimensionless_frequency; AGS keeps F(λ_θ) → ags_pressure_gradient_function |
| `α` | Streamwise wavenumber α = α_r + iα_i (Mack/RSA/Herbert/ABH/Brandt/JD/MA) | Emmons spot half-angle tan ≈ 0.17; angle of incidence (VI 1956) | Paper-scoped: stability bare α → streamwise_wavenumber; spatial growth rate stays as -α_i (separate canonical) |
| `β` (revisited) | Spanwise wavenumber 2π/λ_z (RSA/Herbert/ABH/Brandt/JD/MA/Kachanov) | Hartree pressure-gradient parameter (VI 1956/2008/SRW 2003); Emmons spot-center velocity ratio ≈ 0.43; DN correlation exponent 0.8; Clauser β_K (canonical) | Paper-scoped: stability papers bare β → spanwise_wavenumber; VI/SRW bare β → falkner_skan_hartree; RSA uses G for Hartree to disambiguate |
| `γ` (revisited) | Floquet exponent (Herbert γ = γ_r+iγ_i; ABH detuning) | Ratio of specific heats (Mack); wall-normal wavenumber (Brandt 2004); oblique-wave angle (JD 2001); interfacial tension (YW 2011); soliton asymmetry (Kachanov); OS-Squire coupling coefficient (Reshotko); specific-heat ratio γ_g (Liu) | Strict paper-scope on EACH non-intermittency use; canonical γ = intermittency stays unchanged |
| `σ` | Mack/VI 1956 spatial growth rate / amplification factor (= old N) | DN/Narasimha/Emmons spot-propagation parameter (~0.1–0.4); Floquet temporal eigenvalue (Herbert); subharmonic growth rate (Kachanov); YW 2011 temporal-mode growth rate; Mack Appendix A Prandtl number | Paper-scoped: Mack σ → spatial_growth_rate; VI 1956 σ → n_factor; DN/Narasimha keep σ → relaxation_length_dn family; Mack Pr-σ paper-scoped to prandtl |
| `λ` (revisited) | Thwaites λ_θ (canonical) | DN/Narasimha intermittency-zone length L_γ; Tani Pohlhausen (δ²/ν) dU/dx; Emmons spot-area ≈ 2; Mack 2nd-viscosity coefficient; Mack azimuthal wavelength on disk; SRW crossflow wavelength λ_CF; Sreenivasan Barenblatt power-law exponent; VI 1956 wave-length λ_x | Paper-scoped: DN/Narasimha λ → intermittency_zone_length; Tani/VI/SS λ → pohlhausen_pg_parameter; SRW λ → streak_spanwise_wavelength (CF variant); Mack λ stays paper-private; Sreenivasan λ paper-private |
| `Λ` (revisited) | Integral length scale of FST (canonical) | DN/Narasimha intermittency-zone length L_γ (Λ); Sreenivasan streak spanwise spacing ≈ 100 wall units (Λ⁺); DR2009 Pohlhausen (δ²/ν)(dU/dx); Mack azimuthal wavelength | Paper-scoped: DN/Narasimha → intermittency_zone_length; Sreenivasan → streak_spanwise_wavelength; DR → pohlhausen_pg_parameter |
| `K` (revisited) | Acceleration parameter (Mayle, canonical) | Narasimha TKE (already paper-scoped); AGS Karman constant (Law-of-Wall calibration); Herbert/Kachanov K-type (regime LABEL — string token); VI 2008 Pohlhausen pressure-gradient parameter (≡ λ_θ); YW 2011 Keulegan-Carpenter number; Ghimire K_γ-SST model name; Fransson curve-fit constant; Sreenivasan numbers/constants | Paper-scoped: AGS K → von_karman_constant; VI 2008 K → pressure_gradient_parameter; YW K paper-private; Herbert/Kachanov K-type is a STRING token (breakdown_type_label) NOT an aliased numeric |
| `k` (revisited) | TKE (canonical) | Mack wavenumber magnitude √(α_r²+β_r²); Sreenivasan spectral wavenumber; roughness height (canonical separate); Walters k_T (paper-scoped) | Mack k → streamwise_wavenumber (paper-scope); Sreenivasan k → streak/wavenumber paper-private |
| `N` (revisited) | N-factor e^N (canonical) | Narasimha crumble rate (paper-scoped); Emmons spot frequency at P; Kachanov N-type regime label (string token); Mack station-index AND mode-number (paper-private); Coder N_crit (= critical N) | Paper-scoped: Mack N → ln(A/A_0) → canonical n_factor; Kachanov N-type → breakdown_type_label string; Emmons N(P) paper-private |
| `δ_0` (revisited) | Mayle momentum thickness θ (paper-scoped) | Inflow BL thickness (ABH); inflow displacement thickness δ_0* (Brandt 2004); inflow 99% thickness (Jacobs-Durbin 2001) | Paper-scoped: Mayle δ_0 → θ; other papers' δ_0 paper-private (not aliased so they don't collapse to Mayle's θ) |
| `R_y` (revisited) | LM viscosity-ratio-related wall-distance Reynolds ρy√k/μ | Coder-Maughmer 2014 Re_y = ρUd/μ (velocity-based, different formula); Dick-Kubacki Re_y = √k·y/ν (matches LM); Durbin 2012 R_ν = d²|Ω|/(2.188ν) | Paper-scoped: LM/Dick/WL keep R_y → lm_wall_distance_reynolds; Coder Re_y is DIFFERENT (paper-private notation; do NOT alias to LM R_y) |
| `Re_θ,S` | AGS start of transition (Re_θ at xS) | Marxen/DR/McAuliffe Re_θ at SEPARATION (a different physical location); Dick Re_θs in turbomachinery cascades | Paper-scoped: AGS R_{θ,S} → re_momentum_thickness_transition (canonical Re_θt); Marxen/DR/McAuliffe/Dick Re_{θ,S} → separation_momentum_thickness_reynolds (NEW canonical Re_θs) |
| `St` | Stanton number (Mayle 1991; Friedlander 2023; Liu 2022) | Shear-layer Strouhal St_f / Sr_θ (McAuliffe 2010; Ergin-White 2006 Sr); Wake-passing reduced frequency St = f_d·b_s/U_0 (Dick 2017) | Three distinct canonicals: Stanton (stanton_number), shear-layer Strouhal (shear_layer_strouhal), wake-passing Strouhal (wake_passing_strouhal); each paper-scoped to disambiguate |
| `x_t` (revisited) | Canonical transition onset location (start) | McAuliffe-Yaras 2010 uses x_t for END of transition (x_st for start); Marxen 2011 x_T for transition location inside bubble; Tani x_s for spark-induced spot origin | Paper-scoped: McAuliffe x_t is DELIBERATELY NOT aliased (would invert canonical meaning); their x_st → transition_onset_location only if explicitly added |
| `T_ω` | Durbin 2012 / GAD 2014 / Durbin 2017 turbulent time-scale ratio T_ω = N_t·|Ω|/ω | None significant outside Durbin family | New canonical (durbin_tomega); paper-scoped to Durbin papers |
| `ñ` | Narasimha 1985 dimensionless spot-formation rate (n·σ·ν²/U³) | Coder-Maughmer 2014 AFT envelope amplification factor (transported, e^N PDE form); Fransson 2005 n̂ = nν²/U_∞³ (no σ) | Strict paper-scope: Narasimha ñ → narasimha_dimensionless_spot_rate; Coder ñ → aft_amplification_factor (NEW canonical); Fransson n̂ paper-private |
| `H_L` | LM 2009 fully laminar shape factor (canonical aliasing) | Coder-Maughmer 2014 AFT local pressure-gradient parameter S·d/U_e (NEW model variable) | Paper-scoped: Coder H_L → aft_local_pg_parameter (NEW canonical); LM/canonical H_L stays mapped to shape_factor H |
| `ε` | Dissipation rate (k-ε models); Sreenivasan ⟨ε⟩ | Floquet detuning parameter ε (Herbert; 0=fundamental, 1=subharmonic, 0<ε<1 detuned); Mack reduction parameter ε ≈ (αδ*R·U'_c/U_0)^(-1/3) in SS 1947; small-scale parameter | Paper-scoped: Herbert ε → floquet_detuning (NEW canonical); k-ε ε stays as standard dissipation |
| `Re_Ω` / `Re_v` | Vorticity Reynolds number d²Ω/ν (LM 2009 canonical Re_v = ρy²Ω/μ) | Walters-Cokljat Re_Ω = d²Ω/ν (same up to ρ); Furst Re_Ω; Dick-Kubacki Re_Ω | All map to lm_vorticity_reynolds (canonical Re_v) via paper-specific aliases; NEW alias canonical re_vorticity_y2omega added for documentation |
| `R_T` (revisited) | LM viscosity ratio μ_t/μ (canonical) | Durbin 2012 N_t = ν_T/ν; GAD 2014 R_t (lowercase t); Durbin 2017 R_T (also as Menter γ-R_T transport variable, distinct from viscosity ratio); WC/WL R_T; Friedlander μ_t/μ | All map to lm_viscosity_ratio under paper-scoped aliases; Durbin 2017's "γ-R_T" transport-variable usage is paper-private and NOT aliased to viscosity ratio (paper documents the separate physics) |
| `c_p` | Pressure coefficient C_p (Mayle/AGS/Gbadebo/Marxen/DR/Westin) | Specific heat at constant pressure (Mack 1984; Friedlander) | Two distinct canonicals: pressure_coefficient (C_p) and specific_heat_constant_pressure (c_p_heat); per-paper context disambiguates by subscript or capitalisation |
| `M` | Mach number (Mack/RSA/Tani/Braslow/Friedlander/Liu) | AGS turbulence-grid mesh spacing (paper-scoped already); VI 2008 Falkner-Skan exponent U=u_1·x^M | Paper-scoped: AGS M → grid_mesh_spacing; VI 2008 M → falkner_skan_m; Mach M → mach_number (NEW canonical) |
| `T_w` / `T_r` | Wall temperature / recovery temperature (Mack/Tani/RSA/Reshotko/Liu/Friedlander) | No collision outside compressible-BL context | New canonicals wall_temperature and recovery_temperature; paper-scoped where they appear |
| `Λ⁺` | Sreenivasan 1989 streak spanwise spacing ≈ 100 wall units | Bypass papers λ_z (spanwise wavelength of streaks) | Same physical concept — both alias to streak_spanwise_wavelength (NEW canonical) but Sreenivasan uses wall-unit scaling Λ⁺ = Λ·u*/ν |
| `λ_2` | Jeong-Hussain vortex-identification criterion (Brandt 2004; Marxen 2011) | Pressure-gradient parameter λ_θ; intermittency-zone length λ_γ | Paper-scoped: Brandt/Marxen λ_2 → lambda2_vortex_criterion (NEW canonical), with subscript 2 mandatory to disambiguate from λ_θ |
| `f_SS`, `β_BP`, `β_NAT`, `β_TS`, `R_BP`, `R_NAT` | Walters family / Furst / Liu / Dick transition closure functions | β collides with Hartree/Clauser but the subscripts (BP, NAT, TS) avoid alias-collision | Each gets its own canonical (walters_shear_sheltering_function, walters_bypass_threshold, etc.); no single-char β alias added |
| `Tu_L` / `λ_θ,L` | Menter 2015 LOCAL turbulence intensity / local pressure-gradient parameter (computed from k, ω, wall distance) | Standard canonical Tu / λ_θ | Already covered by paper_specific_aliases on canonical Tu and λ_θ; Menter 2015's local forms preserve the canonical meaning |

## What's NEW in the 2026-05 expansion

### Wave 1 (initial 7-paper coverage)

Added paper-specific aliases for these papers (read word-by-word from
the markdown_cache, top 5+ papers most likely to surface in queries):

* `abu_ghannam_shaw_1980` — full nomenclature page + Eqs. 6, 9, 11, 12, 13, 17
* `mayle_1991` — complete nomenclature table (their Nomenclature p.215)
* `dhawan_narasimha_1958` — γ + λ + ξ definitions (Eq. 1)
* `narasimha_1985` — full Principal Notation section
* `langtry_menter_2009` — Nomenclature table (front matter)
* `walters_cokljat_2008` — model description (introduces k_L)
* `suzen_huang_2000` — intermittency transport intro
* `saric_reed_white_2003` — 3D BL review (mostly Mack notation, deferred)

### Wave 2 (2026-05 multi-agent expansion — full 13-agent synthesis)

Added paper-specific aliases for all the 13 agent-inventoried papers,
covering the COMPLETE corpus needed for the transition-prediction RAG:

* **Mayle/AGS group** — `mayle_1991`, `abu_ghannam_shaw_1980`
  (already covered, augmented further)
* **DN/Emmons/Tani/Narasimha group** — `dhawan_narasimha_1958`,
  `emmons_1951`, `tani_1969`, `narasimha_1985`
* **Compressible stability (Mack)** — `mack_1984`
* **Stability methods** — `van_ingen_1956`, `van_ingen_2008`,
  `saric_reed_white_2003`, `reed_saric_arnal_1996`
* **Schubauer-Skramstad foundational** — `schubauer_skramstad_1947`
* **Herbert/Kachanov/Sreenivasan secondary instability** —
  `herbert_1988`, `kachanov_1994`, `sreenivasan_1989`
* **γ-Re_θ-SST family** — `menter_1994`, `langtry_menter_2006`,
  `langtry_menter_2009`, `menter_2015`
* **kkLω / intermittency transport** — `walters_cokljat_2008`,
  `walters_leylek_2004`, `suzen_huang_2000`, `durbin_2012`,
  `durbin_2017`, `ge_arolla_durbin_2014`
* **Klebanoff modes / streak DNS** —
  `andersson_brandt_bottaro_henningson_2001`, `brandt_2004`,
  `westin_et_al_1994`, `matsubara_alfredsson_2001`,
  `jacobs_durbin_2001`
* **Separation / bypass dynamics** — `diwan_ramesh_2009`,
  `marxen_henningson_2011`, `yoshikawa_wesfreid_2011` (deferred —
  immiscible two-layer KH, outside BL scope), `zaki_durbin_2005`,
  `durbin_wu_2007`
* **Roughness-induced transition** — `braslow_1960`,
  `ergin_white_2006`, `reshotko_2001`
* **LP-turbine / wake-passing** — `dick_kubacki_2017`,
  `gbadebo_hynes_cumpsty_2004`, `mcauliffe_yaras_2010`,
  `furst_2012`, `furst_2013`, `liu_lu_wang_wang_yan_2022`
* **Recent papers (2014–2025)** — `fransson_matsubara_alfredsson_2005`,
  `coder_maughmer_2014`, `friedlander_georgiadis_2023`,
  `saru_ersan_pulat_2025`, `ghimire_ni_wang_2025` (deferred — mostly
  hybrid-RANS-LES variants outside symbol normalisation scope),
  `gonzalez_agrawal_wu_2025`, `wang_zhang_li_meng_2015`,
  `xia_chen_2016`, `zhang_chen_zhao_liu_yan_2022`

Total: paper-specific alias coverage went from 7 papers in the
initial design to 50 papers in Wave 2.

NEW canonical entries added (quantities that appear in multiple papers
but were missing):

### Wave 1 canonicals (already present in glossary):

* `re_momentum_thickness_end_of_transition` (`Re_θE`)
* `re_x_end_of_transition` (`Re_xE`)
* `re_intermittency_zone_length` (`Re_Lγ`)
* `intermittency_zone_length` (`L_γ`)
* `shape_factor_start_of_transition` (`H_S`)
* `shape_factor_end_of_transition` (`H_E`)
* `shape_factor_normalized_ags` (`H'`)
* `ags_pressure_gradient_function` (`F(λ_θ)`)
* `tunnel_reference_velocity` (`U_R`)
* `streamwise_macroscale` (`L_x`)
* `transverse_macroscale` (`L_y`)
* `grid_mesh_spacing` (`M_grid`)
* `narasimha_intermittency_transform` (`F(γ)`)
* `narasimha_dimensionless_spot_rate` (`ñ`)
* `narasimha_crumble_rate` (`N_crumble`)
* `mayle_acceleration_at_transition` (`K_t`)
* `mayle_wake_passing_period` (`τ_wake`)
* `mayle_wake_passing_frequency` (`ω_wake`)
* `mayle_thermal_intermittency` (`γ_h`)
* `mayle_dimensionless_pg_parameter` (`λ_0`)
* `lm_local_transition_reynolds` (`Re_θt_local`)
* `lm_critical_transition_reynolds` (`Re_θc`)
* `lm_viscosity_ratio` (`R_T`)
* `lm_wall_distance_reynolds` (`R_y`)
* `lm_vorticity_reynolds` (`Re_v`)
* `laminar_kinetic_energy` (`k_L`)

### Wave 2 canonicals (2026-05 multi-agent expansion):

Compressible stability / Mack 1984 family:

* `mack_stability_reynolds` (`R` = √(Re_x)) — Mack/RSA primary
* `mack_critical_reynolds` (`R_cr`) — Mack/RSA/VI/SS
* `mack_dimensionless_frequency` (`F` = ω*ν*/U*²) — primary stability F
* `mack_edge_mach` (`M_e` / `M_1` / `M_∞`)
* `mack_relative_mach` (`M̄`) — 2nd-mode driver
* `falkner_skan_hartree` (`β_h`)
* `falkner_skan_m` (`m_fs`)

Wavenumbers / stability theory:

* `streamwise_wavenumber` (`α_w`)
* `spanwise_wavenumber` (`β_w`)
* `floquet_exponent` (`γ_floq`) — Herbert/ABH
* `phase_velocity_complex` (`c_phase`)

Klebanoff modes / streak instability:

* `streak_amplitude` (`A_streak`)
* `klebanoff_mode` (concept label)
* `floquet_detuning` (`ε_detune`)
* `breakdown_type_label` (K-type, H-type, C-type, N-type)

Walters family auxiliary quantities:

* `total_fluctuation_kinetic_energy` (`k_TOT`)
* `walters_eff_length_scale` (`λ_eff`)
* `walters_turb_length_scale` (`λ_T_turb`)
* `walters_shear_sheltering_function` (`f_SS`)
* `walters_bypass_threshold` (`β_BP`)
* `walters_natural_threshold` (`β_NAT`)
* `walters_TS_threshold` (`β_TS`)
* `walters_bypass_transfer` (`R_BP`)
* `walters_natural_transfer` (`R_NAT`)
* `re_vorticity_y2omega` (`Re_Ω`)
* `walters_intermittency_damping` (`f_INT`)
* `walters_wall_damping` (`f_W`)

Durbin γ-Re_θ family:

* `durbin_tomega` (`T_ω`)
* `durbin_critical_reynolds` (`R_c_thresh`)

Coder AFT model:

* `aft_amplification_factor` (`ñ_aft`)
* `aft_local_pg_parameter` (`H_L_aft`)

Compressibility / heat transfer:

* `mach_number` (`M_mach`)
* `wall_temperature` (`T_w`)
* `recovery_temperature` (`T_r`)
* `stanton_number` (`St_heat`)
* `specific_heats_ratio` (`γ_h`)
* `specific_heat_constant_pressure` (`c_p_heat`)
* `pressure_coefficient` (`C_p_pressure`)
* `heat_transfer_coefficient` (`h_heat`)
* `nusselt_number` (`Nu`)

Separation-bubble geometry:

* `bubble_separation_location` (`x_S`)
* `bubble_reattachment_location` (`x_R`)
* `separation_momentum_thickness_reynolds` (`Re_θs`)

Pressure-gradient & wall units:

* `pohlhausen_pg_parameter` (`Λ_pohl`) — distinct from λ_θ
* `re_displacement_thickness` (`Re_δ*`)
* `re_displacement_thickness_critical` (`Re_δ*_cr`)
* `von_karman_constant` (`κ_vK`)
* `log_law_intercept` (`B_log`)
* `coles_wake_parameter` (`Π_coles`)

Stability / DNS / vortex:

* `orr_sommerfeld_eigenfunction` (`φ_OS`)
* `group_velocity` (`c_g`)
* `streak_spanwise_wavelength` (`λ_z_streak`)
* `wake_passing_strouhal` (`St_wake`)
* `shear_layer_strouhal` (`Sr_θ`)
* `lambda2_vortex_criterion` (`λ_2`)
* `streamwise_vorticity` (`ω_x`)

Concept tokens (string-valued, not numeric):

* `shear_sheltering` — Hunt-Durbin sheltering label
* `lift_up_effect` — Landahl mechanism
* `kelvin_helmholtz_instability` — KH instability label
* `bypass_transition` — Morkovin bypass label

Roughness-induced:

* `roughness_reynolds_local` (`Re_k_local`) — covers Braslow/Ergin
  variants with local viscosity

## Validation

The end-to-end check: AGS Eq. 11 chunk
```
R_{θ S} = 163 + exp{F(λ_θ) - F(λ_θ)/6.91·τ_t}
```
with `paper_id="abu_ghannam_shaw_1980"` now rewrites to
```
Re_θt = 163 + exp{F(λ_θ) - F(λ_θ)/6.91·Tu}
```
which embeds under the same canonical tokens as Mayle and Dhawan-Narasimha
chunks discussing the same physics.  Future queries about `Re_theta_t Tu
correlation` retrieve AGS's onset chunk uniformly with the others.

## Process for adding a new paper

1. Read its nomenclature section (if it has one) word-by-word.
2. For each symbol whose meaning matches an EXISTING canonical:
   * If the paper uses standard notation → no work needed (global alias handles it).
   * If the paper uses non-standard notation → add to that canonical's `paper_specific_aliases` block keyed by `paper_id`.
3. For each symbol whose meaning is NEW:
   * Add a NEW canonical entry at the bottom of the file.
   * Document any collision in this README's table.
4. Update the COLLISIONS TABLE above if a new collision is introduced.
5. Validate with a test snippet from one of the paper's actual chunks.
6. Re-ingest the corpus.

## When NOT to add a paper-specific alias

* If the alias is a single character AND appears in plain prose in
  that paper (risk of over-matching).  Example: bare `K` for TKE in
  Narasimha is on the borderline — we added it, but the normalizer's
  built-in `len(alias) < 2: continue` safety still skips single-char
  aliases.  This is intentional.
* If you're not sure whether the meaning is exclusive in that paper.
  Better to leave the symbol unnormalized and accept a small retrieval
  cost than to make a wrong substitution everywhere.
