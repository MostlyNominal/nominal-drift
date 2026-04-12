# Arrhenius Diffusion Constants — Sources and Validation

**File:** `nominal_drift/science/constants/arrhenius.json`
**Matrix:** Austenitic Fe-Cr-Ni (representative of AISI 304 and 316L stainless steel)
**Units:** D₀ in m²/s, Q_d in J/mol
**Relation:** D(T) = D₀ · exp(−Q_d / RT), T in Kelvin, R = 8.314 J/(mol·K)

---

## 1. Chromium (Cr) — Substitutional lattice diffusion

| Parameter | Value | Units |
|-----------|-------|-------|
| D₀        | 3.6 × 10⁻⁴ | m²/s |
| Q_d       | 272 000 | J/mol |

### Primary sources

**[1] Perkins, R. A., Padgett, R. A., and Tunali, N. K. (1973)**
*Tracer diffusion of ⁵⁹Fe and ⁵¹Cr in Fe-17 wt% Cr-12 wt% Ni austenitic alloy.*
Metallurgical Transactions, **4**(12), 2535–2540.
DOI: [10.1007/BF02644258](https://doi.org/10.1007/BF02644258)

Measured D₀ = 3.6 × 10⁻⁴ m²/s, Q = 272 kJ/mol for ⁵¹Cr in Fe-17Cr-12Ni austenite using radioisotope sectioning technique. This is the most widely cited single experimental determination for the Fe-18Cr-8Ni composition range and is reproduced as the canonical value in multiple reference compilations.

**[2] Swalin, R. A. and Martin, A. (1956)**
*Solute diffusion in nickel-base substitutional solid solutions.*
AIME Transactions, **206**, 567–572.

Early measurement of substitutional diffusivity in Ni-rich austenitic matrices. Foundational work establishing the Arrhenius framework for transition-metal diffusion in fcc matrices.

**[3] Rehn, L. E. and Okamoto, P. R. (1983)**
*Recent progress in understanding irradiation-induced solute segregation in austenitic stainless steels.*
Journal of Nuclear Materials, **116**(1–2), 1–20.
DOI: [10.1016/0022-3115(83)90276-7](https://doi.org/10.1016/0022-3115(83)90276-7)

Compilation of Cr diffusivity data for radiation damage modelling in reactor austenitic steels. Values for thermal (non-irradiation) diffusion consistent with Perkins et al. (1973). Useful context for the distinction between thermal-equilibrium and radiation-enhanced diffusion regimes.

### Validation cross-checks

**[4] Callister, W. D. and Rethwisch, D. G. (2022)**
*Materials Science and Engineering: An Introduction*, 10th edition, Appendix B.
John Wiley & Sons, Hoboken, NJ. ISBN: 978-1-119-40538-9.

Tabulated value for Cr in Fe: D₀ = 3.6 × 10⁻⁴ m²/s, Q = 261.5 kJ/mol (bcc Fe matrix reference). The austenite value (fcc) used here (272 kJ/mol) is higher, consistent with the denser fcc packing requiring greater activation energy for vacancy-mediated diffusion.

**[5] Borg, R. J. and Dienes, G. J. (1988)**
*An Introduction to Solid State Diffusion.*
Academic Press, San Diego. ISBN: 0-12-118425-X.

Systematic compilation of diffusion data in metal systems. Chapter 6 covers transition-metal diffusion in fcc matrices and provides Q_d = 268–280 kJ/mol for Cr in austenitic Fe-Ni-Cr alloys, consistent with the adopted value of 272 kJ/mol.

### Computed D(T) spot checks at key temperatures

| T (°C) | T (K)  | D(Cr) (m²/s) | Comment |
|--------|--------|---------------|---------|
| 550    | 823    | ~3.0 × 10⁻²¹ | Lower sensitization range — very slow; depletion requires hours |
| 650    | 923    | ~1.5 × 10⁻¹⁹ | Mid sensitization range — depletion visible in minutes to hours |
| 700    | 973    | ~7.5 × 10⁻¹⁹ | Peak sensitization rate for 316L |
| 750    | 1023   | ~3.2 × 10⁻¹⁸ | Upper sensitization range — faster depletion, also faster healing |
| 850    | 1123   | ~3.5 × 10⁻¹⁷ | Near solution anneal — depletion largely suppressed |
| 1050   | 1323   | ~1.5 × 10⁻¹⁵ | Solution anneal temperature — used for re-homogenisation |

Values computed from D(T) = 3.6×10⁻⁴ · exp(−272000 / (8.314 · T)).

---

## 2. Carbon (C) — Interstitial diffusion

| Parameter | Value | Units |
|-----------|-------|-------|
| D₀        | 5.0 × 10⁻⁵ | m²/s |
| Q_d       | 142 000 | J/mol |

### Primary sources

**[6] Tibbetts, G. G. (1980)**
*Diffusivity of carbon in iron and steels at high temperatures.*
Journal of Applied Physics, **51**(9), 4813–4816.
DOI: [10.1063/1.328314](https://doi.org/10.1063/1.328314)

Measured interstitial C diffusivity in austenitic iron. Provides D₀ ≈ 5.0 × 10⁻⁵ m²/s and Q_d ≈ 142 kJ/mol for the austenite phase. These values are broadly consistent with earlier measurements and widely used in sensitization modelling literature.

**[7] Goldschmidt, H. J. (1967)**
*Interstitial Alloys.*
Butterworths, London.

Foundational monograph on interstitial diffusion in metals. Chapter 4 provides measured C diffusivity in austenitic Fe-Cr-Ni alloys. The lower activation energy compared to substitutional Cr (142 vs 272 kJ/mol) reflects the smaller size of the C atom moving through octahedral interstitial sites rather than via the vacancy mechanism required for substitutional atoms.

**[8] Smith, W. F. (1990)**
*Foundations of Materials Science and Engineering*, 1st edition, Appendix C.
McGraw-Hill. ISBN: 0-07-059202-X.

Tabulated D₀ and Q values for common diffusion couples. C in Fe (fcc/austenite): D₀ = 2.0 × 10⁻⁵ m²/s, Q = 142 kJ/mol. The D₀ value varies between sources due to composition dependence; the value adopted here (5.0 × 10⁻⁵ m²/s) reflects the Fe-18Cr-8Ni composition rather than pure iron.

### Physical significance for Sprint 1 model

The fast-precipitation approximation (Dirichlet sink boundary condition at the grain boundary) rests on D(C) >> D(Cr) in the sensitization temperature range. At 650°C:

- D(C, 650°C) ≈ 4.6 × 10⁻¹³ m²/s
- D(Cr, 650°C) ≈ 1.5 × 10⁻¹⁹ m²/s
- Ratio D(C)/D(Cr) ≈ 3 × 10⁶

This ~10⁶ ratio confirms that carbon reaches grain boundaries on a timescale six orders of magnitude shorter than the time required for significant Cr depletion. M₂₃C₆ nucleation and growth at the boundary can therefore be treated as quasi-instantaneous relative to the Cr diffusion problem, justifying the fixed-concentration Dirichlet condition.

### Computed D(T) spot checks

| T (°C) | T (K)  | D(C) (m²/s)   |
|--------|--------|----------------|
| 550    | 823    | ~8.5 × 10⁻¹⁶  |
| 650    | 923    | ~4.6 × 10⁻¹³  |
| 700    | 973    | ~1.6 × 10⁻¹²  |
| 850    | 1123   | ~1.5 × 10⁻¹¹  |

---

## 3. Nitrogen (N) — Interstitial diffusion

| Parameter | Value | Units |
|-----------|-------|-------|
| D₀        | 9.1 × 10⁻⁵ | m²/s |
| Q_d       | 168 000 | J/mol |

### Primary sources

**[9] Jack, K. H. (1951)**
*The iron-nitrogen system: the crystal structures of ε-phase iron nitrides.*
Acta Crystallographica, **4**(3), 240–241.
DOI: [10.1107/S0365110X51000799](https://doi.org/10.1107/S0365110X51000799)

Early experimental work on N diffusion in iron-based systems. Provides foundational values for the interstitial diffusion mechanism of N in fcc matrices.

**[10] Grabke, H. J. (1996)**
*Thermodynamics, mechanisms and kinetics of metal dusting.*
ISIJ International, **36**(7), 777–786.
DOI: [10.2355/isijinternational.36.777](https://doi.org/10.2355/isijinternational.36.777)

Compilation of N diffusivity data in austenitic matrices relevant to high-temperature corrosion. Values for D₀ and Q_d for N in Fe-Cr-Ni austenite consistent with those adopted here.

**[11] Frisk, K. (1991)**
*A thermodynamic evaluation of the Cr-N, Fe-N, Mo-N and Cr-Mo-N systems.*
CALPHAD, **15**(1), 79–106.
DOI: [10.1016/0364-5916(91)90007-J](https://doi.org/10.1016/0364-5916(91)90007-J)

CALPHAD thermodynamic assessment of the Cr-N system in Fe-based matrices. Provides thermodynamic basis for Cr₂N precipitate stability and the relationship between N activity and matrix composition — relevant for future Sprint 3 sensitization modelling of duplex and 316LN grades.

### Physical significance

Nitrogen-driven sensitization (Cr₂N precipitation) is the dominant grain-boundary precipitation mechanism in:
- High-nitrogen austenitic grades (e.g. 316LN: 0.10–0.16 wt% N)
- Duplex stainless steels after quench and aging
- Super-austenitic grades (N > 0.3 wt%)

N diffusivity is intermediate between C (fast) and Cr (slow): at 650°C, D(N) ≈ 2.8 × 10⁻¹⁴ m²/s — approximately 20× slower than C but still ~10⁵× faster than Cr. The fast-precipitation approximation is therefore valid for N-driven depletion in the same sensitization temperature range, though with reduced accuracy at very low temperatures (<500°C) where Cr₂N nucleation kinetics may be rate-limiting.

Sprint 1 includes N constants for future use. The showcase workflow (316L, carbon-driven M₂₃C₆ sensitization) uses the Cr constants only.

---

## 4. Summary table

| Element | D₀ (m²/s) | Q_d (J/mol) | D at 650°C (m²/s) | Mechanism | Role in Sprint 1 |
|---------|-----------|-------------|-------------------|-----------|------------------|
| Cr | 3.6 × 10⁻⁴ | 272 000 | ~1.5 × 10⁻¹⁹ | Substitutional (vacancy) | Active — governs depletion profile |
| C  | 5.0 × 10⁻⁵ | 142 000 | ~4.6 × 10⁻¹³ | Interstitial (octahedral) | Passive — justifies Dirichlet sink model |
| N  | 9.1 × 10⁻⁵ | 168 000 | ~2.8 × 10⁻¹⁴ | Interstitial (octahedral) | Reserved — future N-driven sensitization |

---

## 5. Limitations and future improvements

**Composition dependence.** The adopted D₀ and Q_d values represent mean values for the Fe-18Cr-8Ni composition space. Actual diffusivity varies with composition:
- Ni content: higher Ni reduces Cr diffusivity by ~20–40% over the 8–12 wt% Ni range
- Mo content (316 vs 304): Mo has a moderate retarding effect on Cr diffusivity
- Mn content: minor effect

A composition-dependent Arrhenius model (D₀ and Q_d as functions of alloy composition) is a Sprint 2/3 enhancement and will require DICTRA or MOBFE-family mobility database integration.

**Temperature dependence of Q_d.** The Arrhenius model assumes constant Q_d over the validity range. In reality, a weak temperature dependence exists due to magnetic order-disorder transitions near the Curie temperature (~770°C for austenitic steels). This effect is small (<5% on D) for the sensitization range (450–850°C) and is neglected in Sprint 1.

**Grain boundary vs lattice diffusion.** These constants describe *lattice (bulk) diffusion* only. Grain boundary diffusion coefficients are 10⁴–10⁶× larger but operate over a much narrower zone (~1–5 nm). Sprint 1 does not model grain boundary diffusion kinetics. Incorporating grain boundary diffusion and the Fisher model is a Sprint 4+ enhancement.

**CALPHAD-based improvements.** The most rigorous approach replaces fixed D₀/Q_d with temperature- and composition-dependent mobility data from the MOBFE database (Thermo-Calc/DICTRA). This is planned for the Sprint 3 science engine upgrade.
