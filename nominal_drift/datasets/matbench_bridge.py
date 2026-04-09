"""
nominal_drift.datasets.matbench_bridge
=======================================
Bridge between Nominal Drift's CrystalRecord schema and the
matbench-genmetrics evaluation framework.

matbench-genmetrics defines four generation metrics for crystal structure
models (Validity, Coverage, Novelty, Uniqueness) and uses
``MPTimeSplit`` as the reference dataset / temporal-split ground truth.
This module provides:

  1. ``evaluate_generated(generated_records, fold, mpt)``
     Run all four matbench-genmetrics against generated CrystalRecord
     structures, returning a ``GenMetricsResult``.

  2. ``records_to_structures(records)``
     Convert a list of CrystalRecord → list[pymatgen.Structure].
     (Thin wrapper around pymatgen_bridge; useful on its own.)

  3. ``matbench_metrics_for_fold(generated_records, fold, mpt)``
     Return raw ``matbench_genmetrics.GenMatcher`` results for one fold.

Metric definitions (from matbench-genmetrics)
----------------------------------------------
  Validity    — fraction of generated structures with physically valid
                stoichiometry (checked by pymatgen composition).
  Coverage    — fraction of val structures "matched" by at least one
                generated structure (ElementProperty + SiteStatsFingerprint
                fingerprint matching).
  Novelty     — fraction of generated structures NOT matched by any
                training structure.
  Uniqueness  — fraction of generated structures not duplicated within the
                generated set itself.

Usage
-----
>>> from nominal_drift.datasets.mp_time_split_bridge import load_mpts52
>>> from nominal_drift.datasets.matbench_bridge import evaluate_generated
>>>
>>> mpt = load_mpts52(dummy=True)
>>> fold = 0
>>> generated = [...]  # list[CrystalRecord] produced by your model
>>> result = evaluate_generated(generated, fold=fold, mpt=mpt)
>>> print(result)
GenMetricsResult(validity=0.95, coverage=0.30, novelty=0.87, uniqueness=0.99)

Graceful degradation
--------------------
If ``matbench-genmetrics`` is not installed the module still imports
successfully; evaluation functions raise a helpful ``ImportError`` only
when actually called.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nominal_drift.datasets.pymatgen_bridge import crystal_record_to_structure
from nominal_drift.datasets.schema import CrystalRecord

if TYPE_CHECKING:
    pass  # keep type checking imports here to avoid circular imports

_MATBENCH_GENMETRICS_AVAILABLE: bool | None = None  # lazy-checked


def _check_matbench() -> None:
    global _MATBENCH_GENMETRICS_AVAILABLE
    if _MATBENCH_GENMETRICS_AVAILABLE is None:
        try:
            import matbench_genmetrics  # noqa: F401
            _MATBENCH_GENMETRICS_AVAILABLE = True
        except ImportError:
            _MATBENCH_GENMETRICS_AVAILABLE = False
    if not _MATBENCH_GENMETRICS_AVAILABLE:
        raise ImportError(
            "matbench-genmetrics is required for evaluation.  "
            "Install it with: pip install matbench-genmetrics"
        )


# ---------------------------------------------------------------------------
# GenMetricsResult
# ---------------------------------------------------------------------------

@dataclass
class GenMetricsResult:
    """Result of a matbench-genmetrics evaluation run.

    All metrics are fractions in [0, 1].

    Attributes
    ----------
    fold : int
        The MPTimeSplit CV fold used.
    n_generated : int
        Number of generated structures submitted.
    n_train : int
        Number of training structures for this fold.
    n_val : int
        Number of validation structures for this fold.
    validity : float
        Fraction of generated structures with valid stoichiometry.
    coverage : float
        Fraction of val structures matched by at least one generated
        structure (recall).
    novelty : float
        Fraction of generated structures not already in the training set.
    uniqueness : float
        Fraction of generated structures not duplicated within the
        generated set.
    """
    fold: int
    n_generated: int
    n_train: int
    n_val: int
    validity: float
    coverage: float
    novelty: float
    uniqueness: float

    def __str__(self) -> str:
        return (
            f"GenMetricsResult(fold={self.fold}, "
            f"n_generated={self.n_generated:,}, "
            f"validity={self.validity:.3f}, "
            f"coverage={self.coverage:.3f}, "
            f"novelty={self.novelty:.3f}, "
            f"uniqueness={self.uniqueness:.3f})"
        )

    def to_dict(self) -> dict:
        return {
            "fold": self.fold,
            "n_generated": self.n_generated,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "validity": self.validity,
            "coverage": self.coverage,
            "novelty": self.novelty,
            "uniqueness": self.uniqueness,
        }


# ---------------------------------------------------------------------------
# Utility: list[CrystalRecord] → list[pymatgen.Structure]
# ---------------------------------------------------------------------------

def records_to_structures(records: list[CrystalRecord]) -> list:
    """Convert a list of CrystalRecord objects to pymatgen Structures.

    Structures that fail reconstruction are silently skipped.

    Parameters
    ----------
    records : list[CrystalRecord]

    Returns
    -------
    list[pymatgen.core.Structure]
    """
    structures = []
    for rec in records:
        try:
            s = crystal_record_to_structure(rec)
            structures.append(s)
        except Exception:
            pass
    return structures


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_generated(
    generated_records: list[CrystalRecord],
    fold: int,
    mpt,
    verbose: bool = False,
) -> GenMetricsResult:
    """Evaluate generated structures against the MPTimeSplit benchmark.

    Uses matbench-genmetrics to compute Validity, Coverage, Novelty, and
    Uniqueness for the given CV fold.

    Parameters
    ----------
    generated_records : list[CrystalRecord]
        Crystal structures produced by a generative model.
    fold : int
        MPTimeSplit fold index (0–4).
    mpt : MPTimeSplit
        A loaded MPTimeSplit instance.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    GenMetricsResult

    Raises
    ------
    ImportError
        If matbench-genmetrics is not installed.
    """
    _check_matbench()
    from matbench_genmetrics.core import GenMatcher

    from nominal_drift.datasets.mp_time_split_bridge import FOLDS

    if fold not in FOLDS:
        raise ValueError(f"fold={fold!r} must be one of {FOLDS}")

    # Get train/val structures for this fold
    train_inputs, val_inputs, _train_out, _val_out = mpt.get_train_and_val_data(fold)
    train_structures = list(train_inputs.values)
    val_structures = list(val_inputs.values)

    # Convert generated CrystalRecords → pymatgen Structures
    generated_structures = records_to_structures(generated_records)
    if not generated_structures:
        # Nothing to evaluate
        return GenMetricsResult(
            fold=fold,
            n_generated=0,
            n_train=len(train_structures),
            n_val=len(val_structures),
            validity=0.0,
            coverage=0.0,
            novelty=0.0,
            uniqueness=0.0,
        )

    if verbose:
        print(
            f"  [matbench] fold={fold}: "
            f"{len(generated_structures)} generated | "
            f"{len(train_structures)} train | "
            f"{len(val_structures)} val"
        )

    # Run matbench-genmetrics evaluation
    gen_matcher = GenMatcher(
        train_structures=train_structures,
        gen_structures=generated_structures,
        test_structures=val_structures,
    )
    gen_matcher.calc_metrics()

    return GenMetricsResult(
        fold=fold,
        n_generated=len(generated_structures),
        n_train=len(train_structures),
        n_val=len(val_structures),
        validity=float(gen_matcher.validity),
        coverage=float(gen_matcher.coverage),
        novelty=float(gen_matcher.novelty),
        uniqueness=float(gen_matcher.uniqueness),
    )


# ---------------------------------------------------------------------------
# Multi-fold evaluation
# ---------------------------------------------------------------------------

def evaluate_all_folds(
    generated_records_per_fold: dict[int, list[CrystalRecord]],
    mpt,
    verbose: bool = False,
) -> dict[int, GenMetricsResult]:
    """Evaluate generated structures across all CV folds.

    Parameters
    ----------
    generated_records_per_fold : dict[int, list[CrystalRecord]]
        Mapping of fold → list of generated CrystalRecord.
    mpt : MPTimeSplit
        Loaded MPTimeSplit instance.
    verbose : bool

    Returns
    -------
    dict[int, GenMetricsResult]
    """
    from nominal_drift.datasets.mp_time_split_bridge import FOLDS

    results: dict[int, GenMetricsResult] = {}
    for fold in FOLDS:
        if fold not in generated_records_per_fold:
            continue
        results[fold] = evaluate_generated(
            generated_records_per_fold[fold],
            fold=fold,
            mpt=mpt,
            verbose=verbose,
        )
    return results


# ---------------------------------------------------------------------------
# Validity check (no matbench-genmetrics required)
# ---------------------------------------------------------------------------

def check_validity(records: list[CrystalRecord]) -> tuple[int, int]:
    """Check structural validity of records without requiring matbench-genmetrics.

    A structure is considered valid if:
      - it has at least 1 atom
      - all element symbols are recognised by pymatgen

    Parameters
    ----------
    records : list[CrystalRecord]

    Returns
    -------
    tuple[int, int]
        ``(n_valid, n_total)``
    """
    try:
        from pymatgen.core import Element
        def _is_valid(rec: CrystalRecord) -> bool:
            if rec.n_atoms < 1:
                return False
            for sym in rec.elements:
                try:
                    Element(sym)
                except ValueError:
                    return False
            return True
    except ImportError:
        # Without pymatgen, just check n_atoms > 0
        def _is_valid(rec: CrystalRecord) -> bool:
            return rec.n_atoms > 0

    n_valid = sum(1 for r in records if _is_valid(r))
    return n_valid, len(records)
