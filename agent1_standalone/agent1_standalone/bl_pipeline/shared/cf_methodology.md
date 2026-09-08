<!--
cf_methodology.md — task #163

Reusable methodology block for Cf(x) extraction from hot-wire y-sweep
data. Embedded verbatim in A5's manuscript whenever A5 publishes
station-by-station Cf estimates. The text is intentionally written in
manuscript voice so it drops directly into a §Methods subsection.

Citation discipline (per the project audit doc):
  • Schlichting BLT, Clauser (1954), Kármán (1921), Blasius (1908), and
    Narasimha (1957) are foundational textbook / classical references
    and do not require a RAG chunk_id.
  • Narasimha's intermittency framework is also available verbatim in
    the project RAG (paper_ids `narasimha_1985` and
    `dhawan_narasimha_1958`).
  • Every citation in the References section was verified via WebSearch
    on 2026-06-04 against at least two independent sources for
    journal, volume, year, page range. Verification notes:
      - Blasius 1908 — verified via SciRP citation database +
        Heidelberg University Library + Google Books catalogue entry.
      - Clauser 1954 — verified via SciRP, Semantic Scholar, and the
        AIAA ARC DOI 10.2514/8.2938.
      - Kármán 1921 — verified via Wiley Online Library DOI
        10.1002/zamm.19210010401 + Zenodo archive + NASA ADS.
      - Narasimha 1957 — verified via U.S. NAS biographical memoir
        for R. Narasimha + Sadhana Vol. 14 (1989) reproduction.
      - Schlichting 1979 — verified via Wiley ZAMM 1980 book review
        + WorldCat ISBN-10 0070553343 / ISBN-13 9780070553347.
-->

### Skin-friction estimation methodology

For a transitional boundary layer surveyed station-by-station with a
single-wire hot-wire probe, the skin-friction coefficient
$C_f = 2\tau_w / (\rho U_\infty^2)$ cannot be obtained from a single
universal procedure: the laminar, transition and turbulent regions each
satisfy different constraints, and the probe itself cannot resolve
$\mathrm{d}U/\mathrm{d}y$ at $y \to 0$ because near-wall heat conduction
to the plate corrupts hot-wire output below $y \approx 0.5\text{–}1$ mm.
We therefore report three independent estimates per station, each
applicable in a specific régime, with a clear label of which estimate is
defensible at that station:

1.  **Blasius reference (laminar régime).**
    Stations clearly upstream of transition onset
    ($\gamma < 0.05$ from the Narasimha intermittency fit performed by
    Agent 6) are quoted against the Blasius flat-plate solution,
    $C_f = 0.664 / \sqrt{Re_x}$ (Schlichting 1979, §17). The check is
    twofold: the measured $\delta_{99}(x)$ must agree with
    $\delta_{99,\,\text{Blasius}} = 5.0\,x / \sqrt{Re_x}$, and the
    integral momentum thickness $\theta(x)$ extracted from the
    $U(y)$ profile must satisfy $Re_\theta = 0.664 \sqrt{Re_x}$. When
    both checks pass within the calibration uncertainty,
    $C_f^{(\text{Blasius})}$ is treated as the station value.

2.  **Clauser-chart fit (turbulent régime).**
    Stations clearly downstream of transition completion
    ($\gamma > 0.95$) are reduced via the Clauser-chart method
    (Clauser 1954): the log-law region of the velocity profile,
    $u^+ = (1/\kappa)\,\ln(y^+) + B$ with $\kappa = 0.41$ and
    $B = 5.0$, is fitted to extract the friction velocity $u_\tau$, from
    which $C_f^{(\text{Clauser})} = 2\,(u_\tau / U_\infty)^2$. The fit
    is accepted only when the recovered profile spans at least five
    measurement points in the range $30 < y^+ < 200$ (the validity
    window for the canonical log-law constants); if it does not, the
    estimate is flagged as under-resolved and reported alongside the
    momentum-integral estimate below.

3.  **Momentum-integral check (transition régime and consistency
    cross-check).**
    For stations within the transition region ($0.05 \leq \gamma \leq
    0.95$), neither the Blasius nor the Clauser path applies, and
    $C_f$ is instead obtained from the Kármán momentum-integral
    equation for a zero-pressure-gradient boundary layer
    (Kármán 1921): $\mathrm{d}\theta / \mathrm{d}x = C_f / 2$. With
    $\theta(x)$ measured at three adjacent x-stations, the central
    difference $C_f^{(\text{momentum})} = 2\,
    [\theta(x_{i+1}) - \theta(x_{i-1})] /
    (x_{i+1} - x_{i-1})$ provides a per-station estimate that is
    independent of any wall-law assumption. The same expression is
    evaluated at the laminar and turbulent stations as a consistency
    cross-check against estimates 1 and 2; agreement within
    $\pm 15\%$ confirms the choice of régime, while a larger
    discrepancy is flagged in the manuscript table as a station whose
    régime assignment is uncertain.

The intermittency $\gamma(x)$ that delimits the three régimes is the
single deterministic output of Agent 6's turbulent-spot detection
(Fransson, Matsubara & Alfredsson 2005) coupled with the Narasimha
intermittency distribution
$\gamma(x) = 1 - \exp[-\hat n\,\sigma\,(x - x_t)^2 / U_\infty]$
(Narasimha 1957; see also Dhawan & Narasimha 1958, available in the
project RAG as `dhawan_narasimha_1958`). The transition onset $x_t$ and
the spot-formation-rate parameter $\hat n \sigma$ are extracted by
Agent 6 from the same hot-wire time series used for the present
$C_f$ analysis, ensuring that the régime-assignment and the
skin-friction reduction are based on a self-consistent dataset.

#### Computation of $Re_x$ from hot-wire data

$Re_x = U_\infty\,x / \nu$ does not come from the voltage signal
directly. The three factors are sourced as follows:

* $U_\infty$ is the mean velocity at the topmost $y$-position of each
  station's y-sweep, after King's-law calibration
  $V^2 = A + B\,U^n$ converts the recorded voltage to velocity.
* $x$ is read from the traverse-stage motor encoder position, recorded
  in the data filename by the rig acquisition script
  (`Hotwire_x<xxx>_y<yyy>_z<zzz>_rpm<nnnn>.txt`); this is the rig's
  source of truth for streamwise location.
* $\nu$ is taken from the air-property tables at the measured tunnel
  temperature ($\nu = 1.516 \times 10^{-5}\,\text{m}^2/\text{s}$ for
  standard air at 20 °C); when the tunnel temperature is not recorded
  the value is held at the standard-air default and a 3% uncertainty
  is propagated to $Re_x$.

### References

* **Blasius, H. (1908).** Grenzschichten in Flüssigkeiten mit kleiner
  Reibung. *Z. Math. Phys.* **56**, 1–37. *(Original derivation of the
  flat-plate $C_f$ scaling. Cited here via its reproduction in
  Schlichting 1979.)*
* **Clauser, F. H. (1954).** Turbulent boundary layers in adverse
  pressure gradients. *J. Aeronaut. Sci.* **21**(2), 91–108. *(The
  original Clauser-chart method for $C_f$ from a log-law fit.)*
* **Dhawan, S. & Narasimha, R. (1958).** Some properties of boundary
  layer flow during transition from laminar to turbulent motion.
  *J. Fluid Mech.* **3**(4), 418–436. *(In project RAG as
  `dhawan_narasimha_1958`.)*
* **Fransson, J. H. M., Matsubara, M. & Alfredsson, P. H. (2005).**
  Transition induced by free-stream turbulence. *J. Fluid Mech.*
  **527**, 1–25. *(In project RAG as `fransson_matsubara_alfredsson_2005`;
  used by Agent 6 for turbulent-spot detection.)*
* **Kármán, T. von (1921).** Über laminare und turbulente Reibung.
  *Z. Angew. Math. Mech.* **1**(4), 233–252. *(Original momentum-
  integral derivation.)*
* **Narasimha, R. (1957).** On the distribution of intermittency in the
  transition region of a boundary layer. *J. Aeronaut. Sci.* **24**,
  711–712. *(Original intermittency-distribution formula; see also
  Narasimha 1985 in the project RAG as `narasimha_1985`.)*
* **Schlichting, H. (1979).** *Boundary-Layer Theory*, 7th ed.
  McGraw-Hill, New York. ISBN 0-07-055334-3. *(Standard textbook
  reference for the Blasius and Schlichting $C_f$ formulas. §17.)*
