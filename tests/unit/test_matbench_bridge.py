"""
tests/unit/test_matbench_bridge.py
====================================
Unit tests for nominal_drift.datasets.matbench_bridge.

matbench-genmetrics itself may not be installed in the test environment,
so tests that require it are guarded with ``pytest.importorskip``.

Tests that only need pymatgen (records_to_structures, check_validity) run
unconditionally.

Run with:
    pytest tests/unit/test_matbench_bridge.py -v
"""
from __future__ import annotations

import pytest

from nominal_drift.datasets.schema import AtomicSite, CrystalRecord, LatticeParams


# ---------------------------------------------------------------------------
# Helpers: minimal CrystalRecords
# ---------------------------------------------------------------------------

def _make_record(
    elements: tuple[str, ...] = ("Ba", "Ti"),
    n_atoms: int = 2,
    split: str | None = "train",
    source_index: int = 0,
) -> CrystalRecord:
    """Build a minimal valid CrystalRecord for testing."""
    sites = tuple(
        AtomicSite(
            species=elem,
            frac_coords=(i * 0.5, i * 0.5, i * 0.5),
        )
        for i, elem in enumerate(elements)
    )
    return CrystalRecord(
        record_id=f"test-record-{source_index}",
        source_dataset="mpts-52",
        source_index=source_index,
        split=split,
        elements=tuple(sorted(set(e for e in elements))),
        n_atoms=n_atoms,
        lattice=LatticeParams(a=4.0, b=4.0, c=4.0, alpha=90.0, beta=90.0, gamma=90.0),
        sites=sites,
        properties={"energy_above_hull": -1.0, "mp_id": f"mp-{1000 + source_index}"},
        raw_path=None,
    )


def _make_records(n: int = 5, split: str | None = "train") -> list[CrystalRecord]:
    return [_make_record(source_index=i, split=split) for i in range(n)]


# ---------------------------------------------------------------------------
# GenMetricsResult
# ---------------------------------------------------------------------------

class TestGenMetricsResult:

    def _result(self, **kwargs):
        from nominal_drift.datasets.matbench_bridge import GenMetricsResult
        defaults = dict(
            fold=0, n_generated=10, n_train=100, n_val=50,
            validity=0.9, coverage=0.4, novelty=0.8, uniqueness=0.95,
        )
        defaults.update(kwargs)
        return GenMetricsResult(**defaults)

    def test_str_contains_fold(self):
        r = self._result(fold=2)
        assert "fold=2" in str(r)

    def test_str_contains_validity(self):
        r = self._result(validity=0.75)
        assert "0.750" in str(r)

    def test_to_dict_has_all_keys(self):
        r = self._result()
        d = r.to_dict()
        for key in ("fold", "n_generated", "n_train", "n_val",
                    "validity", "coverage", "novelty", "uniqueness"):
            assert key in d

    def test_to_dict_values_correct(self):
        r = self._result(fold=3, validity=0.85, coverage=0.3)
        d = r.to_dict()
        assert d["fold"] == 3
        assert d["validity"] == pytest.approx(0.85)
        assert d["coverage"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# records_to_structures
# ---------------------------------------------------------------------------

class TestRecordsToStructures:

    def test_converts_valid_records(self):
        from nominal_drift.datasets.matbench_bridge import records_to_structures
        records = _make_records(3)
        structures = records_to_structures(records)
        assert len(structures) == 3

    def test_returns_pymatgen_structures(self):
        from nominal_drift.datasets.matbench_bridge import records_to_structures
        from pymatgen.core import Structure
        records = _make_records(2)
        structures = records_to_structures(records)
        for s in structures:
            assert isinstance(s, Structure)

    def test_empty_input_returns_empty_list(self):
        from nominal_drift.datasets.matbench_bridge import records_to_structures
        assert records_to_structures([]) == []

    def test_structure_has_correct_n_sites(self):
        from nominal_drift.datasets.matbench_bridge import records_to_structures
        records = [_make_record(elements=("Ba", "Ti"), n_atoms=2)]
        structures = records_to_structures(records)
        assert len(structures[0]) == 2

    def test_structure_lattice_preserved(self):
        from nominal_drift.datasets.matbench_bridge import records_to_structures
        records = [_make_record()]
        structures = records_to_structures(records)
        assert abs(structures[0].lattice.a - 4.0) < 1e-6


# ---------------------------------------------------------------------------
# check_validity
# ---------------------------------------------------------------------------

class TestCheckValidity:

    def test_all_valid_records(self):
        from nominal_drift.datasets.matbench_bridge import check_validity
        records = _make_records(5)
        n_valid, n_total = check_validity(records)
        assert n_total == 5
        assert n_valid == 5

    def test_empty_input(self):
        from nominal_drift.datasets.matbench_bridge import check_validity
        n_valid, n_total = check_validity([])
        assert n_valid == 0
        assert n_total == 0

    def test_returns_tuple_of_ints(self):
        from nominal_drift.datasets.matbench_bridge import check_validity
        result = check_validity(_make_records(3))
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, int) for v in result)

    def test_fraction_correct(self):
        from nominal_drift.datasets.matbench_bridge import check_validity
        records = _make_records(4)
        n_valid, n_total = check_validity(records)
        assert n_valid / n_total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# evaluate_generated — graceful ImportError when matbench not installed
# ---------------------------------------------------------------------------

class TestEvaluateGeneratedImportGuard:

    def test_raises_import_error_gracefully(self, monkeypatch):
        """If matbench_genmetrics is absent, a clear ImportError is raised."""
        import nominal_drift.datasets.matbench_bridge as mb
        # Force _MATBENCH_GENMETRICS_AVAILABLE to False
        monkeypatch.setattr(mb, "_MATBENCH_GENMETRICS_AVAILABLE", False)

        with pytest.raises(ImportError, match="matbench-genmetrics"):
            mb.evaluate_generated([], fold=0, mpt=None)


# ---------------------------------------------------------------------------
# evaluate_all_folds — no matbench required (empty inputs per fold)
# ---------------------------------------------------------------------------

class TestEvaluateAllFoldsGuard:

    def test_raises_import_error_gracefully(self, monkeypatch):
        import nominal_drift.datasets.matbench_bridge as mb
        monkeypatch.setattr(mb, "_MATBENCH_GENMETRICS_AVAILABLE", False)

        with pytest.raises(ImportError, match="matbench-genmetrics"):
            mb.evaluate_all_folds({0: []}, mpt=None)


# ---------------------------------------------------------------------------
# evaluate_generated — full test if matbench-genmetrics is installed
# ---------------------------------------------------------------------------

import importlib as _importlib

@pytest.mark.skipif(
    _importlib.util.find_spec("matbench_genmetrics") is None,
    reason="matbench_genmetrics not installed",
)
class TestEvaluateGeneratedIntegration:
    """Full integration test — only runs when matbench-genmetrics is installed."""

    def _mock_mpt(self):
        """Tiny mock MPTimeSplit for integration testing."""
        import pandas as pd
        from pymatgen.core import Structure, Lattice

        def _s(a):
            return Structure(Lattice.cubic(a), ["Ba", "Ti"],
                             [[0, 0, 0], [0.5, 0.5, 0.5]])

        structures = [_s(3.9 + i * 0.05) for i in range(10)]
        ids = [f"mp-{i}" for i in range(10)]

        class _MockMPT:
            target = "energy_above_hull"
            folds = [0, 1, 2, 3, 4]
            inputs = pd.Series(structures, index=ids)
            outputs = pd.Series([-1.0] * 10, index=ids)
            trainval_splits = [[[0, 1, 2, 3, 4], [5, 6]]] * 5
            test_split = [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9]]

            def get_train_and_val_data(self, fold):
                t, v = self.trainval_splits[fold]
                return self.inputs.iloc[t], self.inputs.iloc[v], \
                       self.outputs.iloc[t], self.outputs.iloc[v]

        return _MockMPT()

    def test_returns_gen_metrics_result(self):
        from nominal_drift.datasets.matbench_bridge import evaluate_generated, GenMetricsResult
        mpt = self._mock_mpt()
        generated = _make_records(3)
        result = evaluate_generated(generated, fold=0, mpt=mpt)
        assert isinstance(result, GenMetricsResult)

    def test_n_generated_set(self):
        from nominal_drift.datasets.matbench_bridge import evaluate_generated
        mpt = self._mock_mpt()
        generated = _make_records(5)
        result = evaluate_generated(generated, fold=0, mpt=mpt)
        assert result.n_generated == 5

    def test_empty_generated_returns_zero_metrics(self):
        from nominal_drift.datasets.matbench_bridge import evaluate_generated
        mpt = self._mock_mpt()
        result = evaluate_generated([], fold=0, mpt=mpt)
        assert result.validity == 0.0
        assert result.n_generated == 0

    def test_all_metrics_in_zero_to_one(self):
        from nominal_drift.datasets.matbench_bridge import evaluate_generated
        mpt = self._mock_mpt()
        result = evaluate_generated(_make_records(4), fold=0, mpt=mpt)
        for metric in (result.validity, result.coverage, result.novelty, result.uniqueness):
            assert 0.0 <= metric <= 1.0
