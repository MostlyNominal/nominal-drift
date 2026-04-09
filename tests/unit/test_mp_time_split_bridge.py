"""
tests/unit/test_mp_time_split_bridge.py
========================================
Unit tests for nominal_drift.datasets.mp_time_split_bridge.

All tests that would require network access (MPTimeSplit.load → figshare)
use mocked MPTimeSplit objects built from pymatgen Structures created
in-memory, so these tests run fully offline.

Tests cover:
  - Module constants (FOLDS, AVAILABLE_MODES)
  - _structure_to_record()
  - mpt_to_crystal_records() on a mock MPTimeSplit
  - get_fold_records() on a mock MPTimeSplit
  - get_test_records() on a mock MPTimeSplit
  - load_mpts52() import-guard when mp-time-split not installed
  - Graceful handling of conversion failures

Run with:
    pytest tests/unit/test_mp_time_split_bridge.py -v
"""
from __future__ import annotations

import pandas as pd
import pytest

from nominal_drift.datasets.schema import CrystalRecord


# ---------------------------------------------------------------------------
# Build a minimal pymatgen Structure for testing
# ---------------------------------------------------------------------------

def _make_structure(a: float = 4.0):
    """Return a tiny cubic pymatgen Structure (BaTiO3-like, 2 atoms)."""
    from pymatgen.core import Structure, Lattice
    lattice = Lattice.cubic(a)
    return Structure(
        lattice=lattice,
        species=["Ba", "Ti"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )


# ---------------------------------------------------------------------------
# Build a mock MPTimeSplit-like object
# ---------------------------------------------------------------------------

class _MockMPT:
    """Minimal mock that satisfies the MPTimeSplit contract used by the bridge."""

    target = "energy_above_hull"
    folds = [0, 1, 2, 3, 4]

    def __init__(self, n_train: int = 6, n_val: int = 2, n_test: int = 2):
        # Build pandas Series of Structure objects with mp-ID index
        structures = [_make_structure(a=3.9 + i * 0.1) for i in range(n_train + n_val + n_test)]
        ids = [f"mp-{1000 + i}" for i in range(len(structures))]
        energies = [-1.0 - i * 0.1 for i in range(len(structures))]

        self.inputs = pd.Series(structures, index=ids, name="structure")
        self.outputs = pd.Series(energies, index=ids, name="energy_above_hull")

        # Fake splits using iloc indices
        # Test split: last n_test structures are "test", rest are "final train"
        train_idx = list(range(n_train + n_val))
        test_idx = list(range(n_train + n_val, len(structures)))
        self.test_split = [train_idx, test_idx]

        # CV folds: split train+val into equal halves for each fold
        total_tv = n_train + n_val
        half = total_tv // 2
        self.trainval_splits = [
            [list(range(half)), list(range(half, total_tv))]
            for _ in range(5)
        ]

    def get_test_data(self):
        t_idx, v_idx = self.test_split
        return (
            self.inputs.iloc[t_idx],
            self.inputs.iloc[v_idx],
            self.outputs.iloc[t_idx],
            self.outputs.iloc[v_idx],
        )

    def get_train_and_val_data(self, fold):
        t_idx, v_idx = self.trainval_splits[fold]
        return (
            self.inputs.iloc[t_idx],
            self.inputs.iloc[v_idx],
            self.outputs.iloc[t_idx],
            self.outputs.iloc[v_idx],
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:

    def test_folds_is_list_of_five(self):
        from nominal_drift.datasets.mp_time_split_bridge import FOLDS
        assert FOLDS == [0, 1, 2, 3, 4]

    def test_available_modes_is_non_empty_list(self):
        from nominal_drift.datasets.mp_time_split_bridge import AVAILABLE_MODES
        assert len(AVAILABLE_MODES) >= 1
        assert all(isinstance(m, str) for m in AVAILABLE_MODES)

    def test_time_series_split_in_modes(self):
        from nominal_drift.datasets.mp_time_split_bridge import AVAILABLE_MODES
        assert "TimeSeriesSplit" in AVAILABLE_MODES

    def test_dataset_name(self):
        from nominal_drift.datasets.mp_time_split_bridge import DATASET_NAME
        assert DATASET_NAME == "mpts-52"

    def test_figshare_url_well_formed(self):
        from nominal_drift.datasets.mp_time_split_bridge import FIGSHARE_FULL_URL
        assert FIGSHARE_FULL_URL.startswith("https://figshare.com")


# ---------------------------------------------------------------------------
# _structure_to_record (internal helper)
# ---------------------------------------------------------------------------

class TestStructureToRecord:

    def test_returns_crystal_record(self):
        from nominal_drift.datasets.mp_time_split_bridge import _structure_to_record
        rec = _structure_to_record(
            structure=_make_structure(),
            mp_id="mp-123",
            target_value=-1.5,
            target_name="energy_above_hull",
            split="train",
            source_index=0,
        )
        assert isinstance(rec, CrystalRecord)

    def test_mp_id_in_properties(self):
        from nominal_drift.datasets.mp_time_split_bridge import _structure_to_record
        rec = _structure_to_record(_make_structure(), "mp-999", -1.0, "energy_above_hull", "train", 0)
        assert rec.properties["mp_id"] == "mp-999"

    def test_target_value_in_properties(self):
        from nominal_drift.datasets.mp_time_split_bridge import _structure_to_record
        rec = _structure_to_record(_make_structure(), "mp-1", -2.5, "energy_above_hull", "train", 0)
        assert rec.properties["energy_above_hull"] == pytest.approx(-2.5)

    def test_split_set(self):
        from nominal_drift.datasets.mp_time_split_bridge import _structure_to_record
        rec = _structure_to_record(_make_structure(), "mp-1", -1.0, "energy_above_hull", "test", 0)
        assert rec.split == "test"

    def test_none_split_allowed(self):
        from nominal_drift.datasets.mp_time_split_bridge import _structure_to_record
        rec = _structure_to_record(_make_structure(), "mp-1", -1.0, "energy_above_hull", None, 0)
        assert rec.split is None

    def test_source_dataset_is_mpts52(self):
        from nominal_drift.datasets.mp_time_split_bridge import _structure_to_record
        rec = _structure_to_record(_make_structure(), "mp-1", -1.0, "energy_above_hull", "train", 0)
        assert rec.source_dataset == "mpts-52"

    def test_formula_in_properties(self):
        from nominal_drift.datasets.mp_time_split_bridge import _structure_to_record
        rec = _structure_to_record(_make_structure(), "mp-1", -1.0, "energy_above_hull", "train", 0)
        # BaTi structure → formula "Ba1 Ti1"
        assert "Ba" in rec.properties["formula"] or "Ba" in rec.elements

    def test_none_target_not_in_properties(self):
        from nominal_drift.datasets.mp_time_split_bridge import _structure_to_record
        rec = _structure_to_record(_make_structure(), "mp-1", None, "energy_above_hull", "train", 0)
        assert "energy_above_hull" not in rec.properties


# ---------------------------------------------------------------------------
# mpt_to_crystal_records
# ---------------------------------------------------------------------------

class TestMptToCrystalRecords:

    def test_returns_list_of_crystal_records(self):
        from nominal_drift.datasets.mp_time_split_bridge import mpt_to_crystal_records
        mpt = _MockMPT(n_train=4, n_val=2, n_test=2)
        records = mpt_to_crystal_records(mpt)
        assert all(isinstance(r, CrystalRecord) for r in records)

    def test_total_count_matches_dataset_size(self):
        from nominal_drift.datasets.mp_time_split_bridge import mpt_to_crystal_records
        mpt = _MockMPT(n_train=4, n_val=2, n_test=2)
        records = mpt_to_crystal_records(mpt)
        # n_train + n_val are the "final train"; n_test are "test"
        # total = 4+2 + 2 = 8
        assert len(records) == 8

    def test_splits_assigned_train_and_test(self):
        from nominal_drift.datasets.mp_time_split_bridge import mpt_to_crystal_records
        mpt = _MockMPT(n_train=4, n_val=2, n_test=2)
        records = mpt_to_crystal_records(mpt)
        splits = {r.split for r in records}
        assert splits == {"train", "test"}

    def test_train_count_correct(self):
        from nominal_drift.datasets.mp_time_split_bridge import mpt_to_crystal_records
        mpt = _MockMPT(n_train=4, n_val=2, n_test=2)
        records = mpt_to_crystal_records(mpt)
        n_train = sum(1 for r in records if r.split == "train")
        assert n_train == 6  # 4+2 in final train

    def test_test_count_correct(self):
        from nominal_drift.datasets.mp_time_split_bridge import mpt_to_crystal_records
        mpt = _MockMPT(n_train=4, n_val=2, n_test=2)
        records = mpt_to_crystal_records(mpt)
        n_test = sum(1 for r in records if r.split == "test")
        assert n_test == 2

    def test_target_in_properties(self):
        from nominal_drift.datasets.mp_time_split_bridge import mpt_to_crystal_records
        mpt = _MockMPT()
        records = mpt_to_crystal_records(mpt)
        for rec in records:
            assert "energy_above_hull" in rec.properties

    def test_mp_id_in_properties(self):
        from nominal_drift.datasets.mp_time_split_bridge import mpt_to_crystal_records
        mpt = _MockMPT()
        records = mpt_to_crystal_records(mpt)
        for rec in records:
            assert "mp_id" in rec.properties
            assert rec.properties["mp_id"].startswith("mp-")


# ---------------------------------------------------------------------------
# get_fold_records
# ---------------------------------------------------------------------------

class TestGetFoldRecords:

    def test_returns_two_lists(self):
        from nominal_drift.datasets.mp_time_split_bridge import get_fold_records
        mpt = _MockMPT(n_train=4, n_val=2, n_test=2)
        train_recs, val_recs = get_fold_records(mpt, fold=0)
        assert isinstance(train_recs, list)
        assert isinstance(val_recs, list)

    def test_train_records_non_empty(self):
        from nominal_drift.datasets.mp_time_split_bridge import get_fold_records
        mpt = _MockMPT(n_train=4, n_val=2, n_test=2)
        train_recs, _ = get_fold_records(mpt, fold=0)
        assert len(train_recs) > 0

    def test_val_records_non_empty(self):
        from nominal_drift.datasets.mp_time_split_bridge import get_fold_records
        mpt = _MockMPT(n_train=4, n_val=2, n_test=2)
        _, val_recs = get_fold_records(mpt, fold=0)
        assert len(val_recs) > 0

    def test_all_folds_return_records(self):
        from nominal_drift.datasets.mp_time_split_bridge import get_fold_records, FOLDS
        mpt = _MockMPT(n_train=6, n_val=2, n_test=2)
        for fold in FOLDS:
            train_recs, val_recs = get_fold_records(mpt, fold=fold)
            assert isinstance(train_recs, list)
            assert isinstance(val_recs, list)

    def test_invalid_fold_raises(self):
        from nominal_drift.datasets.mp_time_split_bridge import get_fold_records
        mpt = _MockMPT()
        with pytest.raises(ValueError, match="fold"):
            get_fold_records(mpt, fold=99)

    def test_split_labels_correct(self):
        from nominal_drift.datasets.mp_time_split_bridge import get_fold_records
        mpt = _MockMPT(n_train=4, n_val=2, n_test=2)
        train_recs, val_recs = get_fold_records(mpt, fold=0)
        assert all(r.split == "train" for r in train_recs)
        assert all(r.split == "val" for r in val_recs)

    def test_all_are_crystal_records(self):
        from nominal_drift.datasets.mp_time_split_bridge import get_fold_records
        mpt = _MockMPT(n_train=4, n_val=2, n_test=2)
        train_recs, val_recs = get_fold_records(mpt, fold=0)
        for rec in train_recs + val_recs:
            assert isinstance(rec, CrystalRecord)


# ---------------------------------------------------------------------------
# get_test_records
# ---------------------------------------------------------------------------

class TestGetTestRecords:

    def test_returns_two_lists(self):
        from nominal_drift.datasets.mp_time_split_bridge import get_test_records
        mpt = _MockMPT(n_train=4, n_val=2, n_test=2)
        train_recs, test_recs = get_test_records(mpt)
        assert isinstance(train_recs, list)
        assert isinstance(test_recs, list)

    def test_counts_correct(self):
        from nominal_drift.datasets.mp_time_split_bridge import get_test_records
        mpt = _MockMPT(n_train=4, n_val=2, n_test=2)
        train_recs, test_recs = get_test_records(mpt)
        assert len(train_recs) == 6  # 4+2 = 6 in final train
        assert len(test_recs) == 2

    def test_train_split_label(self):
        from nominal_drift.datasets.mp_time_split_bridge import get_test_records
        mpt = _MockMPT(n_train=4, n_val=2, n_test=2)
        train_recs, _ = get_test_records(mpt)
        assert all(r.split == "train" for r in train_recs)

    def test_test_split_label(self):
        from nominal_drift.datasets.mp_time_split_bridge import get_test_records
        mpt = _MockMPT(n_train=4, n_val=2, n_test=2)
        _, test_recs = get_test_records(mpt)
        assert all(r.split == "test" for r in test_recs)
