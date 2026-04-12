"""
tests/unit/test_pymatgen_bridge.py
====================================
Unit tests for nominal_drift.datasets.pymatgen_bridge.

Tests use a minimal but syntactically complete CIF string that pymatgen can
parse without network access.  We also test the CrystalRecord round-trip
and the canonical column mappings on real-world CSV rows (synthetic dicts
matching the actual perov-5 CSV format).

Run with:
    pytest tests/unit/test_pymatgen_bridge.py -v
"""
from __future__ import annotations

import pytest

from nominal_drift.datasets.schema import CrystalRecord, LatticeParams, AtomicSite


# ---------------------------------------------------------------------------
# Minimal valid CIF for TlCoN2O (taken from perov-5 train.csv row 0)
# ---------------------------------------------------------------------------

_CIF_TLCO = (
    "# generated using pymatgen\n"
    "data_TlCoN2O\n"
    "_symmetry_space_group_name_H-M   'P 1'\n"
    "_cell_length_a   4.24596403\n"
    "_cell_length_b   4.24596403\n"
    "_cell_length_c   4.24596403\n"
    "_cell_angle_alpha   90.00000000\n"
    "_cell_angle_beta   90.00000000\n"
    "_cell_angle_gamma   90.00000000\n"
    "_symmetry_Int_Tables_number   1\n"
    "_chemical_formula_structural   TlCoN2O\n"
    "_chemical_formula_sum   'Tl1 Co1 N2 O1'\n"
    "_cell_volume   76.54713370\n"
    "_cell_formula_units_Z   1\n"
    "loop_\n"
    " _symmetry_equiv_pos_site_id\n"
    " _symmetry_equiv_pos_as_xyz\n"
    "  1  'x, y, z'\n"
    "loop_\n"
    " _atom_site_type_symbol\n"
    " _atom_site_label\n"
    " _atom_site_symmetry_multiplicity\n"
    " _atom_site_fract_x\n"
    " _atom_site_fract_y\n"
    " _atom_site_fract_z\n"
    " _atom_site_occupancy\n"
    "  Tl  Tl0  1  0.50015703  0.50000000  0.50000000  1\n"
    "  Co  Co1  1  0.00265771  0.00000000  0.00000000  1\n"
    "  N  N2  1  0.50108143  0.00000000  0.50000000  1\n"
    "  N  N3  1  0.50108143  0.50000000  0.00000000  1\n"
    "  O  O4  1  0.00111568  0.50000000  0.50000000  1\n"
)

_CIF_INVALID = "not a cif at all"
_CIF_EMPTY = ""


# ---------------------------------------------------------------------------
# Tests: cif_string_to_structure
# ---------------------------------------------------------------------------

class TestCifStringToStructure:

    def test_returns_structure_for_valid_cif(self):
        from nominal_drift.datasets.pymatgen_bridge import cif_string_to_structure
        s = cif_string_to_structure(_CIF_TLCO)
        assert s is not None

    def test_structure_has_correct_n_sites(self):
        from nominal_drift.datasets.pymatgen_bridge import cif_string_to_structure
        s = cif_string_to_structure(_CIF_TLCO)
        assert len(s) == 5  # Tl + Co + 2N + O

    def test_structure_lattice_a(self):
        from nominal_drift.datasets.pymatgen_bridge import cif_string_to_structure
        s = cif_string_to_structure(_CIF_TLCO)
        assert abs(s.lattice.a - 4.24596403) < 1e-4

    def test_structure_has_correct_elements(self):
        from nominal_drift.datasets.pymatgen_bridge import cif_string_to_structure
        s = cif_string_to_structure(_CIF_TLCO)
        symbols = {str(e) for e in s.composition.elements}
        assert symbols == {"Tl", "Co", "N", "O"}

    def test_returns_none_for_invalid_cif(self):
        from nominal_drift.datasets.pymatgen_bridge import cif_string_to_structure
        s = cif_string_to_structure(_CIF_INVALID)
        assert s is None

    def test_returns_none_for_empty_string(self):
        from nominal_drift.datasets.pymatgen_bridge import cif_string_to_structure
        s = cif_string_to_structure(_CIF_EMPTY)
        assert s is None


# ---------------------------------------------------------------------------
# Tests: structure_to_crystal_record
# ---------------------------------------------------------------------------

class TestStructureToCrystalRecord:

    def _get_structure(self):
        from nominal_drift.datasets.pymatgen_bridge import cif_string_to_structure
        return cif_string_to_structure(_CIF_TLCO)

    def test_returns_crystal_record(self):
        from nominal_drift.datasets.pymatgen_bridge import structure_to_crystal_record
        rec = structure_to_crystal_record(
            self._get_structure(), "perov-5", 0, "train"
        )
        assert isinstance(rec, CrystalRecord)

    def test_source_dataset_set(self):
        from nominal_drift.datasets.pymatgen_bridge import structure_to_crystal_record
        rec = structure_to_crystal_record(self._get_structure(), "perov-5", 0, "train")
        assert rec.source_dataset == "perov-5"

    def test_split_set(self):
        from nominal_drift.datasets.pymatgen_bridge import structure_to_crystal_record
        rec = structure_to_crystal_record(self._get_structure(), "perov-5", 42, "val")
        assert rec.split == "val"

    def test_source_index_set(self):
        from nominal_drift.datasets.pymatgen_bridge import structure_to_crystal_record
        rec = structure_to_crystal_record(self._get_structure(), "perov-5", 99, "train")
        assert rec.source_index == 99

    def test_n_atoms(self):
        from nominal_drift.datasets.pymatgen_bridge import structure_to_crystal_record
        rec = structure_to_crystal_record(self._get_structure(), "perov-5", 0, "train")
        assert rec.n_atoms == 5

    def test_elements_sorted(self):
        from nominal_drift.datasets.pymatgen_bridge import structure_to_crystal_record
        rec = structure_to_crystal_record(self._get_structure(), "perov-5", 0, "train")
        assert rec.elements == tuple(sorted(rec.elements))

    def test_elements_match_sites(self):
        from nominal_drift.datasets.pymatgen_bridge import structure_to_crystal_record
        rec = structure_to_crystal_record(self._get_structure(), "perov-5", 0, "train")
        assert set(rec.elements) == {s.species for s in rec.sites}

    def test_lattice_a_preserved(self):
        from nominal_drift.datasets.pymatgen_bridge import structure_to_crystal_record
        rec = structure_to_crystal_record(self._get_structure(), "perov-5", 0, "train")
        assert abs(rec.lattice.a - 4.24596403) < 1e-4

    def test_properties_injected(self):
        from nominal_drift.datasets.pymatgen_bridge import structure_to_crystal_record
        rec = structure_to_crystal_record(
            self._get_structure(), "perov-5", 0, "train",
            properties={"formation_energy_per_atom": -1.5, "formula": "TlCoN2O"},
        )
        assert rec.properties["formation_energy_per_atom"] == -1.5
        assert rec.properties["formula"] == "TlCoN2O"

    def test_record_id_generated(self):
        from nominal_drift.datasets.pymatgen_bridge import structure_to_crystal_record
        rec = structure_to_crystal_record(self._get_structure(), "perov-5", 0, "train")
        assert rec.record_id  # non-empty UUID string

    def test_record_id_override(self):
        from nominal_drift.datasets.pymatgen_bridge import structure_to_crystal_record
        rec = structure_to_crystal_record(
            self._get_structure(), "perov-5", 0, "train", record_id="fixed-id"
        )
        assert rec.record_id == "fixed-id"

    def test_split_none_allowed(self):
        from nominal_drift.datasets.pymatgen_bridge import structure_to_crystal_record
        rec = structure_to_crystal_record(self._get_structure(), "perov-5", 0, None)
        assert rec.split is None


# ---------------------------------------------------------------------------
# Tests: crystal_record_to_structure (round-trip)
# ---------------------------------------------------------------------------

class TestCrystalRecordToStructure:

    def _make_record(self):
        from nominal_drift.datasets.pymatgen_bridge import (
            cif_string_to_structure,
            structure_to_crystal_record,
        )
        s = cif_string_to_structure(_CIF_TLCO)
        return structure_to_crystal_record(s, "perov-5", 0, "train")

    def test_round_trip_n_sites(self):
        from nominal_drift.datasets.pymatgen_bridge import crystal_record_to_structure
        rec = self._make_record()
        s = crystal_record_to_structure(rec)
        assert len(s) == rec.n_atoms

    def test_round_trip_lattice_a(self):
        from nominal_drift.datasets.pymatgen_bridge import crystal_record_to_structure
        rec = self._make_record()
        s = crystal_record_to_structure(rec)
        assert abs(s.lattice.a - rec.lattice.a) < 1e-6

    def test_round_trip_elements(self):
        from nominal_drift.datasets.pymatgen_bridge import crystal_record_to_structure
        rec = self._make_record()
        s = crystal_record_to_structure(rec)
        symbols = {str(e) for e in s.composition.elements}
        assert symbols == set(rec.elements)

    def test_returns_pymatgen_structure(self):
        from nominal_drift.datasets.pymatgen_bridge import crystal_record_to_structure
        from pymatgen.core import Structure
        rec = self._make_record()
        s = crystal_record_to_structure(rec)
        assert isinstance(s, Structure)


# ---------------------------------------------------------------------------
# Tests: cif_string_to_crystal_record (convenience)
# ---------------------------------------------------------------------------

class TestCifStringToCrystalRecord:

    def test_valid_cif_returns_record(self):
        from nominal_drift.datasets.pymatgen_bridge import cif_string_to_crystal_record
        rec = cif_string_to_crystal_record(
            _CIF_TLCO, "perov-5", 0, split="train",
            properties={"heat_all": 2.72},
        )
        assert isinstance(rec, CrystalRecord)
        assert rec.properties["heat_all"] == 2.72

    def test_invalid_cif_returns_none(self):
        from nominal_drift.datasets.pymatgen_bridge import cif_string_to_crystal_record
        rec = cif_string_to_crystal_record(_CIF_INVALID, "perov-5", 0)
        assert rec is None

    def test_empty_cif_returns_none(self):
        from nominal_drift.datasets.pymatgen_bridge import cif_string_to_crystal_record
        rec = cif_string_to_crystal_record("", "perov-5", 0)
        assert rec is None
