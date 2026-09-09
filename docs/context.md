# MTech Defense — Project Context

**Student:** Priyanshi Dubey  
**Supervisor:** Dr. Sourabh S. Diwan  
**Department:** Aerospace Engineering, Indian Institute of Science, Bangalore  
**Date:** June 2026  
**Keywords:** Boundary-layer transition, bypass transition, agentic AI, retrieval-augmented generation, OpenFOAM, hot-wire anemometry, intermittency, transition-sensitive CFD, workflow automation, boundary-layer characterization

---

## 1. Thesis at a Glance

An **agentic AI framework for automating boundary-layer transition research workflows**, demonstrated on a free-stream-turbulence-induced (Tu = 2.7%) bypass transition case over a smooth flat plate at zero pressure gradient (ZPG) in the LT1 wind tunnel at IISc.

The framework is a **6-agent pipeline + central orchestrator** (LangGraph-based) that coordinates:

| Agent | Role | Output |
|-------|------|--------|
| **A1 — Theoretical** | Retrieval-augmented correlation lookup (AGS, Mayle, FS20) | Transition onset envelope: 0.315 ≤ xt ≤ 0.501 m, xend ≈ 0.879 m |
| **A2 — CFD** | Transition-sensitive RANS in OpenFOAM (γ–fReθt and k–kL–ω closures) | xt = 0.418 m (LCTM), xt = 0.774 m (kkLOmega) |
| **A3 — Pilot** | 14-station hot-wire pilot survey + King's law calibration + FMA-2005 intermittency | xt = 611.3 mm, xe = 1324 mm, Ltr = 712.7 mm |
| **A4 — Traverse** | Deterministic full traverse planning (6 × 33 = 198 points) | Traverse plan at 850 RPM |
| **A5 — Analysis** | Hot-wire data reduction: profiles, boundary-layer parameters, conditional ⟨(∂u/∂t)²⟩ diagnostic | xt,H=1.7 = 862.9 mm; conditional diagnostic separates only at low-γ station |
| **A6 — Synthesis** | Cross-validation + report generation | Consolidated technical report (Appendix C) |

**Orchestrator responsibilities:** query decomposition (tailors sub-queries per agent), parallel execution of A1/A2, envelope union → A3 pilot planning, retrospective feedback pathways (A3 → A1 Phase 2, A3 → A2 Phase 2), and final cross-validation synthesis by A6.

**Stack:** LangGraph (orchestration), ChromaDB (vector DB for literature), NVIDIA Llama-Embed-Nemotron-8B (embeddings), BAAI bge-reranker-v2-m3 (reranker), BGE-M3 (dense), BM25 (lexical), PyMuPDF (PDF parsing with fallback cascade), python-docx, scipy (signal processing), NI-DAQ (hardware), OpenFOAM (CFD solver), Claude models (reasoning core).

---

## 2. Demonstration Case (the "main query")

- **Geometry:** Smooth flat plate, ZPG, chord = 1.6 m, width = 0.05 m, sharp leading edge
- **Flow:** U∞ = 6.2 m/s, TuLE = 2.7%, ν = 1.516×10⁻⁵ m²/s (air, 25°C)
- **Turbulence grid:** Square-bar biplane grid, solidity σ = 0.30, bar size d = 5 mm, offset xgrid = 1.0 m upstream of LE
- **Facility:** LT1 open-circuit wind tunnel, IISc Low-Speed Aerodynamics Laboratory
- **Instrumentation:** Dantec 55P11 single-wire hot-wire probe (5 µm W, L_eff = 1.2 mm), Dantec StreamLine Pro CTA, NI 16-bit DAQ at 25 kHz, 10 kHz hardware LPF

### 2.1 Transition Onset — Three-Pillar Cross-Validation

| Pillar | Method / Correlation | xt (m) | Notes |
|--------|---------------------|--------|-------|
| **Theory (A1)** | Abu-Ghannam & Shaw [1] (primary) | 0.342 | Reθ,t = 248.2; only correlation fully in own calibration band |
| | Mayle [35] | 0.315 | Out-of-range: Tu(xt) = 2.29% < 3% lower bound |
| | Fransson-Shahinfar [16] w/ Kurian-Fransson Λx | 0.501 | Indicative only; Roach Λx rejected (below FS20 band) |
| | Dhawan-Narasimha cross-check | 0.342 | Within 5% of AGS closure |
| **CFD (A2)** | kOmegaSSTLM (γ–fReθt) | 0.418 | Converged in 789 iterations; γ peak = 0.947 |
| | kkLOmega (k–kL–ω) | 0.774 | Converged in 1656 iterations; verdict: TRANSITION_AMBIGUOUS (no γ field) |
| **Experiment (A3/A5)** | FMA-2005 + Narasimha sigmoid | 0.611 | R² = 0.996 on sigmoid fit |

**Key discrepancy:** Theory (0.342 m) and CFD-low (0.418 m) **under-predict** experiment (0.611 m). Root cause identified in Phase 2: **integral length scale Λx** — Phase 1 assumed ~12.8 mm (Roach far-field power law), but direct measurement = **39.4 mm** (Roach M1 zero-crossing estimator, 8.5% uncertainty), which sits well outside the FS20 [16–26 mm] calibration band. This elevated Λx drives the bypass transition onset downstream.

### 2.2 Boundary-Layer Characterization (Agent 5)

- **Shape factor H(x):** Falls from 2.14 (upstream, near-laminar Blasius 2.59) → 1.45 (downstream, near-turbulent ~1.4)
- **Transition midpoint by H=1.7 crossing:** xH=1.7 = 862.9 mm (agrees with FMA mid-transition xγ=0.5 = 866.4 mm to within 0.4%)
- **δ99(x):** Measured lifts off Blasius curve near upstream station; at x=1400 mm, δ99 = 26.2 mm (2.7× Blasius, 71% of 1/7-power-law)
- **Reθ(x):** Rises from 377 (x=410 mm) → 1026 (x=1400 mm)
- **Conditional ⟨(∂u/∂t)² | I=1⟩ diagnostic:** Wallward peak migration from y/δ99 = 0.75 (upstream) → 0.08 (downstream), amplitude grows ×23. Separates from unconditional curve only at x=610 mm (γmin = 0.029); collapses elsewhere (γ≈1). Contribution = FMA-indicator-gated acceleration-variance on single-wire traverse.

---

## 3. Key Physics & Methods

### Correlations (Appendix A.1)
- **Abu-Ghannam & Shaw [1]:** Reθ,t = 163 + exp[6.91·(1 − Tu/6.91)], Tu in % (bypass, Tu ≈ 1–5%)
- **Mayle [35]:** Reθ,t = 400·Tu^(−5/8), Tu in % (bypass, turbomachinery, Tu ≈ 1–10%)
- **Fransson-Shahinfar [16]:** Re_x,γ=0.5 = C1·Tu^(−p) [Tu as fraction] or C2·Re_FST^(−q) where Re_FST = Tu·Λx·U∞/ν (mid-transition location)
- **Narasimha [39] intermittency:** γ(x) = 1 − exp[−α(x−xt)²] or concentrated-breakdown form (c=2 sigmoid for γ=0.5)
- **Blasius inversion:** xt = (Reθ,t / 0.664)² · ν/U∞ for ZPG laminar BL (θ = 0.664√(νx/U∞))

### Hot-Wire Methods
- **King's law:** E² = A + B·Uⁿ (linearized: E² vs Uⁿ); calibration from 5-point sweep, R² = 0.9999, A=1.0271, B=0.9600, n=0.3739
- **FMA-2005 intermittency detection [15]:** High-pass filter (f_cut = U∞/(5·δ99)) → detector D(t) = |u_h(t)| → criterion C(t) = moving avg → threshold sweep → exponential fit γ_obs(u_s) ≈ c·exp(α·u_s) → intercept c = γ_true
- **Transition definitions:** xt at γ=0.01, xγ=0.5 at γ=0.50, xe at γ=0.99, Ltr = xe − xt

### CFD Models
- **Langtry-Menter γ–fReθt (kOmegaSSTLM):** Two extra transport equations (intermittency γ, Reθt seed = 215). Grid: 400×50×1, 20k cells, Δy1 = 2.26×10⁻⁵ m, y+ ≈ 0.26, simpleFoam steady. Inlet Tu from Kurian-Fransson Λx = 17.50 mm. Converged in 789 iterations.
- **Walters-Cokljat k–kL–ω (kkLOmega):** Three-equation model (kL, kT, ω). Transition via kL→kT transfer. Converged in 1656 iterations. Onset = 0.774 m (360 mm downstream of LCTM).

---

## 4. RAG / Retrieval Architecture (Appendix B)

Agent 1's RAG subsystem for literature-grounded transition-prediction:

- **Corpus:** Curated boundary-layer transition literature (PDFs), parsed via page-parsing pipeline with fallback cascade (parser-agnostic). Cached after parse.
- **Chunking:** Equation-aware — preserves local relationship between equations, variable definitions, and surrounding text. Metadata: paper_id, page, section, subsection, equation presence, table presence, figure references.
- **Indexing:** Hybrid — dense (Llama-Embed-Nemotron-8B) + BM25 (lexical). HyDE supported (diagnostic analysis only).
- **Retrieval pipeline (Algorithm B.1):** Query → extract flow params → generate focused queries (onset, length, turbulence, equations, applicability) → dense + BM25 → RRF/fusion → dedup → cross-encoder rerank (BAAI bge-reranker-v2-m3) → sufficiency check → adaptive retry if gaps.
- **Equation inventory:** Structured store (paper_id, eq_id, page, symbolic form) for formula verification.
- **Verification layers:** Formula-verification-before-computation (rejects hard-coded results, compares against source equation in inventory) → source-checking of reported values → coverage verification + critique → reconciliation.
- **Runtime config (Table B.1):** Hybrid dense+lexical retrieval, BAAI reranker, adaptive retrieval enabled, ReAct-style tool-assisted reasoning, formula verification enabled, source verification enabled, coverage verification enabled, critique+revision enabled.

### RAG Architecture Notes Relevant to Current Work
- ChromaDB used (lightweight, local) — review report §"Enterprise Vector Database Integration" recommends Qdrant for multi-tenant production scaling.
- Metadata-based representation (no explicit parent-child hierarchy) — noted as potential future enhancement.
- Documents chunked with **section-level metadata**, **equation-awareness**, and **figure/table annotations** — directly relevant to the chunking discussions.

---

## 5. Formal Review & Scalability Assessment — Key Recommendations

From the review report (full text extracted, 14.3 KB). Main themes:

### Observability & CI/CD
- **Langfuse** for glass-box trajectory evaluation: capture multi-step reasoning, tool orchestrations, RAG operations as hierarchical "spans" in a single trace.
- **LLM-as-a-Judge scoring** (asynchronous): score final reports against technical rubrics (scientific accuracy, hallucination, context completeness) without blocking workflow. Supports multi-model provider comparison.
- **Benchmark datasets:** Aggregate historical wind-tunnel measurements, calibration constants, flow configs into centralized benchmark for offline experiments.

### State Management & Resilience
- **Asynchronous state management:** Persist workflow state to DB; yield compute while CFD/experiments run; resume on webhook/message-broker callbacks.
- **Evaluator-Optimizer pattern:** Separate optimizer agent (generates code) from independent evaluator agent (distinct model or strict rubric) — replaces fragile self-critique.
- **Bounded execution guardrails:** Max 3 tool-call attempts per node; escalate to historical baseline or human-in-the-loop on threshold breach.

### Infrastructure Scaling
- **Model cascade:** Lightweight models for routine tasks (query parsing, state routing); frontier models reserved for deep scientific reasoning + final synthesis.
- **Enterprise vector DB:** Migrate ChromaDB → Qdrant for metadata filtering, horizontal scalability, multi-tenant throughput.
- **Orchestration:** Use OpenAI Agents SDK with robust guardrails + global state wrappers; agents interact via validated input/output filters + shared state contexts.

### Continuous Learning (Phase 2)
- **Episodic memory:** Store diagnostic insights (root causes of prediction errors) as vectorized memory; query on future similar cases to inject as few-shot context.
- **Dynamic RAG corpus weighting:** Penalize retrieval weight of papers that consistently yield high validation errors against physical ground truth.

---

## 6. Working Notes & Ongoing RAG Design

**Architecture doc:** `mtech-defense/RAG_ARCHITECTURE.md` (parser-agnostic; deterministic structure graph built from parsing — nodes are chunks, edges are parsed citations/refs; hybrid dense+BM25 → cross-encoder rerank → MMR diversity λ≈0.7–0.8 → parent expansion; exact-hash + near-dup dedup at index time; multimodal track for figures with VLM descriptions).

**Files in working directory:**
- `context.md` — this file
- `presentation_extracted.txt` — full extracted text of MTech_Defense_Presentation.pdf (154 pages, ~292K chars)
- `report_extracted.txt` — full extracted text of Formal Review and Scalability Assessment Report.docx (~14.3 KB)
- `RAG_ARCHITECTURE.md` — RAG design notes

---

## 7. Source Files

- **Source 1:** `D:\Downloads\MTech_Defense_Presentation.pdf` — 154 pages, 291,619 chars (extracted)
- **Source 2:** `D:\Downloads\Formal Review and Scalability Assessment Report.docx` — 14,351 chars (extracted)

---

*This context file is the working memory for the MTech defense RAG work. It should be updated as the project evolves.*
