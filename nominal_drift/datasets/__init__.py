"""
nominal_drift.datasets
======================
Lane B — crystal / benchmark / DFT-oriented dataset layer.

This package is intentionally isolated from Lane A (metallurgy/process).
Lane A modules (science, knowledge, llm, core, cli) do not import from
here, and this package does not import from Lane A modules.

Current public API (schema only — adapters, normaliser, loader, and
index are Sprint 2B+):

    LatticeParams      — unit-cell dimensions and angles
    AtomicSite         — single atomic site with fractional coordinates
    CrystalRecord      — canonical internal record for one crystal structure
    DatasetManifest    — metadata written alongside each normalised dataset

Extending
---------
To add a new dataset (e.g. MPTS-52):
  1. Write ``nominal_drift/datasets/adapters/mpts52.py``
     implementing ``BaseAdapter`` (Sprint 2B).
  2. Register the name in ``nominal_drift/datasets/registry.py``.
  3. Add a 5-record fixture under
     ``tests/unit/test_datasets/fixtures/``.

No other files in this package need to change.
"""

from nominal_drift.datasets.schema import (
    AtomicSite,
    CrystalRecord,
    DatasetManifest,
    LatticeParams,
)

__all__: list[str] = [
    "AtomicSite",
    "CrystalRecord",
    "DatasetManifest",
    "LatticeParams",
]
