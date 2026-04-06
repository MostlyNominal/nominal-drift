"""
nominal_drift.crystal
==================
Track 2 — Crystal Structure Generation / Scientific AI.

This sub-package is reserved for crystal-structure-related workflows:

  - Crystal structure generation (equivariant diffusion models, flow-based)
  - DFT-informed data pipelines
  - Benchmarking on crystal datasets (e.g. MPTS-52 via matbench-genmetrics)
  - Evaluation integration with Joint Equivariant Diffusion (JED) frameworks
  - Structure validation and novelty / coverage / uniqueness metrics

Nothing is implemented here in Sprint 1.  The sub-package exists to
make the dual-track architecture explicit from day one and to prevent
Track 1 modules from being named or scoped as if they were the only
scientific capability of the system.
"""
