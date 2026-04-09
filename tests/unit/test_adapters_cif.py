"""
tests/unit/test_adapters_cif.py
=================================
Unit tests for the CIF-aware dataset adapters (v0.2.0).

Tests use synthetic CSV-row dicts matching the CDVAE / DiffCSP on-disk
format (material_id, cif, formula, heat_all, …) rather than touching
real downloaded files, so they run fully offline.

A real but minimal CIF string (from perov-5 row 0) is embedded here so
that pymatgen parsing is exercised without network access.

Run with:
    pytest tests/unit/test_adapters_cif.py -v
"""
from __future__ import annotations

import pytest

from nominal_drift.datasets.adapters import (
    ADAPTER_REGISTRY,
    Carbon24Adapter,
    MP20Adapter,
    MPTS52Adapter,
    Perov5Adapter,
    get_adapter,
    normalise_records,
)
from nominal_drift.datasets.schema import CrystalRecord


# ---------------------------------------------------------------------------
# Shared CIF fixture (TlCoN2O, 5-atom perovskite, from perov-5 train row 0)
# ---------------------------------------------------------------------------

_CIF_VALID = (
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

_ROW_PEROV5 = {
    "": "0",
    "material_id": "6334",
    "cif": _CIF_VALID,
    "formula": "CoTlON2",
    "heat_all": "2.72",
    "heat_ref": "2.6248636002164694",
    "dir_gap": "0.0",
    "ind_gap": "0.0",
    "split": "train",
}

_ROW_MP20 = {
    "": "0",
    "material_id": "mp-1234",
    "cif": _CIF_VALID,
    "formula": "TlCoN2O",
    "heat_all": "-1.5",
    "heat_ref": "-1.4",
    "dir_gap": "0.5",
    "ind_gap": "0.4",
    "e_above_hull": "0.02",
    "split": "val",
}

_ROW_CARBON24 = {
    "": "0",
    "material_id": "c24_001",
    "cif": _CIF_VALID,
    "formula": "TlCoN2O",
    "heat_all": "-8.1",
    "energy_per_atom": "-8.1",
    "split": "test",
}

_ROW_MPTS52 = {
    "": "0",
    "material_id": "mpts-9999",
    "cif": _CIF_VALID,
    "formula": "TlCoN2O",
    "heat_all": "-2.0",
    "heat_ref": "-1.9",
    "dir_gap": "0.0",
    "ind_gap": "0.0",
    "split": "train",
}

_ROW_MISSING_CIF = {
    "material_id": "bad",
    "cif": "",
    "formula": "X",
    "split": "train",
}

_ROW_INVALID_CIF = {
    "material_id": "bad2",
    "cif": "not a cif",
    "formula": "X",
    "split": "train",
}


# ---------------------------------------------------------------------------
# ADAPTER_REGISTRY
# ---------------------------------------------------------------------------

class TestAdapterRegistry:

    def test_has_four_entries(self):
        assert len(ADAPTER_REGISTRY) == 4

    def test_all_known_datasets_present(self):
        for name in ("perov-5", "mp-20", "carbon-24", "mpts-52"):
            assert name in ADAPTER_REGISTRY

    def test_get_adapter_returns_instance(self):
        adapter = get_adapter("perov-5")
        assert isinstance(adapter, Perov5Adapter)

    def test_get_adapter_raises_for_unknown(self):
        with pytest.raises(KeyError):
            get_adapter("nonexistent")

    def test_all_adapters_have_dataset_name(self):
        for name, cls in ADAPTER_REGISTRY.items():
            assert cls().dataset_name == name

    def test_all_adapters_have_source_url(self):
        for cls in ADAPTER_REGISTRY.values():
            assert cls().source_url.startswith("http")


# ---------------------------------------------------------------------------
# Perov5Adapter
# ---------------------------------------------------------------------------

class TestPerov5Adapter:

    def _convert(self, row=None):
        return Perov5Adapter().convert(row or _ROW_PEROV5, source_index=0)

    def test_returns_crystal_record(self):
        assert isinstance(self._convert(), CrystalRecord)

    def test_source_dataset(self):
        assert self._convert().source_dataset == "perov-5"

    def test_split_train(self):
        assert self._convert().split == "train"

    def test_n_atoms(self):
        assert self._convert().n_atoms == 5

    def test_elements_sorted(self):
        rec = self._convert()
        assert list(rec.elements) == sorted(rec.elements)

    def test_material_id_in_properties(self):
        rec = self._convert()
        assert rec.properties["material_id"] == "6334"

    def test_formula_in_properties(self):
        rec = self._convert()
        assert rec.properties["formula"] == "CoTlON2"

    def test_heat_all_captured(self):
        rec = self._convert()
        assert rec.properties["heat_all"] == pytest.approx(2.72)

    def test_formation_energy_canonical_alias(self):
        rec = self._convert()
        assert rec.properties["formation_energy_per_atom"] == pytest.approx(2.72)

    def test_band_gap_alias(self):
        rec = self._convert()
        # band_gap = max(dir_gap, ind_gap) = max(0, 0) = 0
        assert rec.properties["band_gap"] == pytest.approx(0.0)

    def test_lattice_a(self):
        rec = self._convert()
        assert abs(rec.lattice.a - 4.24596403) < 1e-4

    def test_missing_cif_raises_value_error(self):
        with pytest.raises(ValueError, match="cif"):
            Perov5Adapter().convert(_ROW_MISSING_CIF, source_index=0)

    def test_invalid_cif_raises_value_error(self):
        with pytest.raises(ValueError, match="CIF parsing failed"):
            Perov5Adapter().convert(_ROW_INVALID_CIF, source_index=0)

    def test_split_absent_from_row_gives_none(self):
        row = {**_ROW_PEROV5}
        row.pop("split")
        rec = Perov5Adapter().convert(row, source_index=0)
        assert rec.split is None


# ---------------------------------------------------------------------------
# MP20Adapter
# ---------------------------------------------------------------------------

class TestMP20Adapter:

    def test_returns_crystal_record(self):
        assert isinstance(MP20Adapter().convert(_ROW_MP20, 0), CrystalRecord)

    def test_split_val(self):
        assert MP20Adapter().convert(_ROW_MP20, 0).split == "val"

    def test_formation_energy(self):
        rec = MP20Adapter().convert(_ROW_MP20, 0)
        assert rec.properties["formation_energy_per_atom"] == pytest.approx(-1.5)

    def test_e_above_hull(self):
        rec = MP20Adapter().convert(_ROW_MP20, 0)
        assert rec.properties["e_above_hull"] == pytest.approx(0.02)

    def test_band_gap_is_max_of_gaps(self):
        rec = MP20Adapter().convert(_ROW_MP20, 0)
        # max(0.5, 0.4) = 0.5
        assert rec.properties["band_gap"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Carbon24Adapter
# ---------------------------------------------------------------------------

class TestCarbon24Adapter:

    def test_returns_crystal_record(self):
        assert isinstance(Carbon24Adapter().convert(_ROW_CARBON24, 0), CrystalRecord)

    def test_split_test(self):
        assert Carbon24Adapter().convert(_ROW_CARBON24, 0).split == "test"

    def test_energy_per_atom(self):
        rec = Carbon24Adapter().convert(_ROW_CARBON24, 0)
        assert rec.properties["energy_per_atom"] == pytest.approx(-8.1)


# ---------------------------------------------------------------------------
# MPTS52Adapter
# ---------------------------------------------------------------------------

class TestMPTS52Adapter:

    def test_returns_crystal_record(self):
        assert isinstance(MPTS52Adapter().convert(_ROW_MPTS52, 0), CrystalRecord)

    def test_source_dataset(self):
        assert MPTS52Adapter().convert(_ROW_MPTS52, 0).source_dataset == "mpts-52"

    def test_formation_energy(self):
        rec = MPTS52Adapter().convert(_ROW_MPTS52, 0)
        assert rec.properties["formation_energy_per_atom"] == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# normalise_records + manifest
# ---------------------------------------------------------------------------

class TestNormaliseRecords:

    def _make_records(self, n=3):
        adapter = Perov5Adapter()
        records = []
        for i in range(n):
            row = {**_ROW_PEROV5, "split": "train" if i < 2 else "val"}
            records.append(adapter.convert(row, source_index=i))
        return records

    def test_writes_jsonl(self, tmp_path):
        records = self._make_records(3)
        normalise_records(records, str(tmp_path), "perov-5")
        jsonl = tmp_path / "structures.jsonl"
        assert jsonl.exists()
        lines = [l for l in jsonl.read_text().splitlines() if l.strip()]
        assert len(lines) == 3

    def test_writes_manifest(self, tmp_path):
        records = self._make_records(2)
        normalise_records(records, str(tmp_path), "perov-5")
        manifest_file = tmp_path / "manifest.json"
        assert manifest_file.exists()

    def test_manifest_n_structures(self, tmp_path):
        import json
        records = self._make_records(3)
        normalise_records(records, str(tmp_path), "perov-5")
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["n_structures"] == 3

    def test_manifest_splits(self, tmp_path):
        import json
        records = self._make_records(3)
        normalise_records(records, str(tmp_path), "perov-5")
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["splits"]["train"] == 2
        assert manifest["splits"]["val"] == 1

    def test_manifest_elements_sorted(self, tmp_path):
        import json
        records = self._make_records(2)
        normalise_records(records, str(tmp_path), "perov-5")
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        elems = manifest["elements_present"]
        assert elems == sorted(elems)

    def test_jsonl_records_round_trip(self, tmp_path):
        import json
        records = self._make_records(2)
        normalise_records(records, str(tmp_path), "perov-5")
        jsonl = tmp_path / "structures.jsonl"
        loaded = [CrystalRecord.model_validate_json(l) for l in jsonl.read_text().splitlines() if l.strip()]
        assert len(loaded) == 2
        assert all(isinstance(r, CrystalRecord) for r in loaded)
