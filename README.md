# Nominal Drift

**Where failed assumptions become working systems.**

Nominal Drift is a **local-first scientific AI workstation** for materials science, corrosion, diffusion physics, and crystal intelligence.

It is being built as a hybrid system where:

- **LLM reasoning** supports scientific dialogue, orchestration, and reporting
- **deterministic numerical engines** produce inspectable physical outputs
- **mechanism-aware visualisation** makes diffusion behaviour visible
- **experiment persistence and retrieval** preserve validation history
- **crystal dataset pipelines** support structure search, parsing, and evaluation workflows

This is **not a generic chatbot**.

The model reasons.  
**The science lives in the tools.**

---

## What it does

Nominal Drift is designed to answer scientific questions with **traceable computational pathways**.

Its current focus is to keep numerical outputs tied to explicit models and named data sources rather than hidden heuristics or vague “AI intuition”.

Core workflow areas include:

- diffusion solvers
- sensitisation screening
- heat-treatment studies
- thermodynamic layer integration
- crystal dataset ingestion and search
- animation and engineering visualisation
- experiment storage and retrieval
- reporting and scientific narration

The objective is simple:

> Ask scientific questions in natural language  
> and receive physically grounded, inspectable outputs.

From chromium depletion in stainless steels to crystal structure browsing and future thermodynamic extensions, Nominal Drift is being built as a **scientific copilot that stays local**.

---

## Current capabilities

### Materials and corrosion workflows

- 1D Fickian diffusion engine (**Crank–Nicolson**)
- chromium / carbon / nitrogen diffusion workflows
- austenitic and ferritic material preset paths
- sensitisation and depletion studies
- heat-treatment mechanism studies
- engineering-profile diffusion animation from real solver output
- schematic microstructure-style visualisation derived from computed depletion fields
- runtime provenance display for constants and active material settings
- experiment memory and comparison database
- CLI and GUI orchestration

### Thermodynamic layer *(in active integration)*

Nominal Drift is being extended with a distinct **Layer 2 thermodynamic path** using `pycalphad`.

This layer is intended to sit alongside diffusion, not replace it:

- **Layer 1** = baseline diffusion physics
- **Layer 2** = thermodynamic equilibrium / phase plausibility
- **Layer 3** = crystal structure and dataset workflows

Current design goals for Layer 2:

- explicit TDB-backed thermodynamic queries
- composition + temperature phase plausibility
- separate provenance from diffusion outputs
- support for both:
  - **316L / austenitic Fe-Cr-Ni-C-N**
  - **430 stainless steel / ferritic Fe-Cr-C-N**

Important:
- diffusion outputs remain authoritative only for diffusion
- thermodynamic outputs must remain clearly separated
- precipitation kinetics, segregation energetics, and CALPHAD extensions must never be implied unless actually implemented

### Crystal intelligence lane

The crystal lane is operational as a **separate scientific lane**.

Supported datasets include:

- `MP-20`
- `MPTS-52`
- `Perov-5`
- `Carbon-24`

Capabilities include:

- dataset import and on-disk status reporting
- normalized `structures.jsonl` pipelines
- `pymatgen`-driven structure parsing
- crystal search and filtering utilities
- CIF-style structure inspection workflows
- generative crystal evaluation bridge
- future `DiffCSP`-style structure workflows

Important:  
these datasets are **not currently used to drive diffusion coefficients or sensitisation calculations**.  
They belong to the structure-aware crystal lane unless explicitly connected by a validated scientific model.

---

## Scientific architecture

Nominal Drift is being structured as a three-layer scientific system.

### Layer 1 — Baseline physics

This is the authoritative numerical layer for diffusion outputs.

It currently includes:

- Arrhenius diffusion constants
- Fick’s Second Law
- Crank–Nicolson finite-difference solving
- explicit material presets
- strict element/system gating
- no silent fallbacks

This layer is responsible for outputs such as:

- concentration profiles
- depletion depth
- minimum chromium concentration
- time-evolving diffusion fields

### Layer 2 — Thermodynamic / kinetics layer

This layer is being introduced to answer questions that diffusion alone cannot answer, including:

- which phases are thermodynamically plausible at a given temperature
- whether carbide/nitride formation is thermodynamically accessible
- how alloy chemistry interacts with phase stability

This layer is where `pycalphad` belongs.

It is intended to provide:

- TDB-backed phase plausibility
- explicit composition–temperature equilibrium queries
- provenance on active database, phases, and assumptions

This layer is **not** yet equivalent to full precipitation kinetics, grain-boundary segregation energetics, or DFT-backed migration modelling.

### Layer 3 — Structure-aware layer

This layer contains:

- `MPTS-52`
- `MP-20`
- `Perov-5`
- `Carbon-24`
- `pymatgen` structure tools
- normalized crystal records
- retrieval / search / evaluation pipelines

Its current honest roles are:

- structure search
- reference retrieval
- crystal data inspection
- generative model evaluation
- future structure-informed workflows

It is **not currently authoritative for diffusion outputs**.

---

## Grounded in

Nominal Drift is built on a combination of:

- literature-grounded diffusion constants
- classical transport theory
- corrosion science
- experimentally interpretable engineering workflows
- thermodynamic modelling pathways
- crystal dataset ecosystems

### Chromium diffusion *(substitutional)*

Primary reference currently used:

**Perkins, Padgett & Tunali (1973)**  
*Tracer diffusion of Fe and Cr in Fe-17 wt% Cr-12 wt% Ni austenitic alloy*  
*Metallurgical Transactions*, 4(12), 2535–2540  
DOI: 10.1007/BF02644258

Canonical values currently used:

- **D₀ = 3.6 × 10⁻⁴ m²/s**
- **Q = 272 kJ/mol**

This remains the primary reference for Cr transport in the 18-8 / 316L composition space.

### Carbon diffusion *(interstitial)*

Grounded in:

- Tibbetts (1980)
- Goldschmidt (1967)
- Smith (1990)

### Nitrogen diffusion

Grounded in:

- Jack (1951)
- Grabke (1996)
- Frisk (1991)

### Numerical foundation

The transport layer is built on:

- **Fick’s Second Law**
- **Arrhenius diffusion formalism**
- **Crank–Nicolson finite-difference solving**

Validated against:

- Crank, *The Mathematics of Diffusion* (1975)
- Borg & Dienes (1988)

---

## What the loaded crystal datasets do — and do not do

The normalized crystal datasets currently support:

- structure parsing
- crystal search
- dataset status inspection
- evaluation workflows
- future structure-aware scientific extensions

They do **not currently**:

- provide diffusion coefficients
- override `D₀` or `Q`
- drive the diffusion solver
- supply segregation energies
- supply precipitation kinetics
- replace CALPHAD or DFT

In the current architecture:

- **diffusion calculations** use literature-backed constants and material presets
- **crystal datasets** support the crystal intelligence lane
- **future bridges** between these lanes must remain explicit and scientifically justified

---

## Visualisation philosophy

Nominal Drift distinguishes carefully between:

- **real numerical outputs**
- **derived engineering visualisations**
- **future, unimplemented scientific claims**

Current animation modes are designed to avoid misleading presentation:

- the engineering profile is driven directly by solver output
- the schematic view is derived from the computed depletion field
- static or decorative visuals should never be presented as if they were true atomistic motion

Nominal Drift does **not** currently claim:

- literal atomic trajectories
- true segregation energetics
- true precipitation kinetics
- DFT-backed clustering behaviour
- CALPHAD-backed phase evolution unless the thermodynamic layer is explicitly active

---

## Dataset and experiment workflows

### Dataset import

The dataset import system reports **real on-disk status**.

Supported dataset lanes include:

- `Perov-5`
- `MP-20`
- `MPTS-52`
- `Carbon-24`

The import page is intended to show what is actually present locally, rather than pretending that all datasets are always available.

### Experiment database

Nominal Drift includes an experiment database for saving and reviewing results.

This supports:

- saving diffusion runs
- preserving notes and labels
- comparing previous runs
- keeping local engineering history inspectable

The experiment database is for **result persistence**, not for driving diffusion coefficients or thermodynamic predictions.

---

## Architecture

```text
Llama / Ollama
       ↓
Scientific Router
       ↓
Core Toolchain
       ├── Layer 1: diffusion_engine
       ├── Layer 2: thermodynamic_layer
       ├── mechanism_animator
       ├── experiment_store
       ├── crystal_datasets
       ├── pymatgen_structure_lane
       ├── reporting
       └── retrieval
