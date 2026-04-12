"""
nominal_drift — Local Scientific AI Workbench for Materials Engineering
=======================================================================

NominalDrift is a fully local, modular scientific AI system that couples a
deterministic physics engine with an LLM reasoning layer.  The name reflects
the two core ideas: *nominal* compositions as starting inputs, and *drift* as
the physical process of elemental diffusion away from those nominal values —
most critically, chromium depletion toward grain boundaries during sensitization.

Two long-term scientific tracks are planned:

Track 1 — Metallurgy / Process Intelligence
    Alloy composition input, heat-treatment route modelling, diffusion
    and depletion modelling (Cr / C / N and future multi-species), grain
    growth, sensitization risk, homogenisation estimation, experiment
    logging and retrieval, TTT/CCT-style risk maps, process-window
    explorer.

Track 2 — Crystal Structure Generation / Scientific AI
    Crystal structure generation workflows, integration with equivariant
    diffusion models, DFT-informed data pipelines, benchmarking on
    datasets such as MPTS-52. A bridge to ``matbench-genmetrics`` exists
    in ``nominal_drift.datasets.matbench_bridge`` for evaluation, but
    the dependency is *optional and not installed by default*; the
    bridge raises ``ImportError`` if called when the package is missing,
    and the GUI surfaces the real install status via
    ``matbench_genmetrics_status()``.

Sprint 1 implements the foundation of Track 1 (diffusion engine,
visualisation, experiment store) while the package structure remains
intentionally extensible for both tracks.
"""

__version__ = "0.1.0"
__author__ = "NominalDrift Contributors"
