"""
nominal_drift.science
==================
Deterministic scientific engine — Tier B of the NominalDrift three-tier
architecture.

All numerical computations live here.  The LLM layer (Tier A) may NOT
perform calculations; it calls this layer via the orchestrator (Tier C)
and narrates the results.

Current modules (Sprint 1)
--------------------------
diffusion_engine      : 1-D Fick solver (Crank–Nicolson) — Day 3+
constants/            : Curated Arrhenius and precipitation constant data

Planned modules (later sprints)
--------------------------------
sensitization_model   : Cr-depletion EPR-TOS risk scoring (C- and N-driven)
homogenisation_estimator : Temperature / time window suggestion
grain_growth_model    : Arrhenius-based grain size evolution
ht_interpreter        : Heat-treatment history parser
doe_module            : Design-of-experiments matrix generation
adapters/matbench_adapter : matbench-genmetrics plugin wrapper (Track 2)
"""
