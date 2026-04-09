"""
tests/unit/test_adapters.py
===========================
Unit tests for nominal_drift.datasets.adapters (CIF-aware v0.2.0).

Fixtures use minimal but syntactically complete CIF strings that pymatgen
can parse without any network access.  Each CIF encodes a distinct
chemistry so element/atom-count assertions remain meaningful.

Run with:
    pytest tests/unit/test_adapters.py -v
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from nominal_drift.datasets.adapters import (
    ADAPTER_REGISTRY,
    ADAPTER_VERSION,
    BaseAdapter,
    Carbon24Adapter,
    MP20Adapter,
    MPTS52Adapter,
    Perov5Adapter,
    get_adapter,
    normalise_records,
)
from nominal_drift.datasets.schema import (
    AtomicSite,
    CrystalRecord,
    LatticeParams,
)


# ---------------------------------------------------------------------------
# Shared CIF strings (all from real perov-5 data, offline-safe)
# ---------------------------------------------------------------------------

# TlCoN2O — 5 atoms, cubic cell, a=4.246 Å
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

# BaBiN3 — 5 atoms, cubic, a=4.542 Å
_CIF_BABIN3 = (
    "# generated using pymatgen\n"
    "data_BaBiN3\n"
    "_symmetry_space_group_name_H-M   'P 1'\n"
    "_cell_length_a   4.54173211\n"
    "_cell_length_b   4.54173211\n"
    "_cell_length_c   4.54173211\n"
    "_cell_angle_alpha   90.00000000\n"
    "_cell_angle_beta   90.00000000\n"
    "_cell_angle_gamma   90.00000000\n"
    "_symmetry_Int_Tables_number   1\n"
    "_chemical_formula_structural   BaBiN3\n"
    "_chemical_formula_sum   'Ba1 Bi1 N3'\n"
    "_cell_volume   93.68380947\n"
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
    "  Bi  Bi0  1  0.00135646  0.00000000  0.00000000  1\n"
    "  Ba  Ba1  1  0.50129553  0.50000000  0.50000000  1\n"
    "  N  N2  1  0.50114834  0.00000000  0.50000000  1\n"
    "  N  N3  1  0.50114834  0.50000000  0.00000000  1\n"
    "  N  N4  1  0.00110768  0.50000000  0.50000000  1\n"
)


# ---------------------------------------------------------------------------
# Fixtures — CIF-aware row dicts (matching CDVAE CSV format)
# ---------------------------------------------------------------------------

@pytest.fixture()
def perov5_raw_dict() -> dict:
    """Minimal valid Perov5 CSV row dict (CIF format)."""
    return {
        "": "0",
        "material_id": "perov_001",
        "cif": _CIF_TLCO,
        "formula": "TlCoN2O",
        "heat_all": "-2.5",
        "heat_ref": "-2.4",
        "dir_gap": "0.5",
        "ind_gap": "0.4",
        "split": "train",
    }


@pytest.fixture()
def mp20_raw_dict() -> dict:
    """Minimal valid MP20 CSV row dict (CIF format)."""
    return {
        "": "0",
        "material_id": "mp_001",
        "cif": _CIF_BABIN3,
        "formula": "BaBiN3",
        "heat_all": "-1.5",
        "heat_ref": "-1.4",
        "dir_gap": "5.2",
        "ind_gap": "4.9",
        "e_above_hull": "0.1",
        "split": "val",
    }


@pytest.fixture()
def carbon24_raw_dict() -> dict:
    """Minimal valid Carbon24 CSV row dict (CIF format, single-element)."""
    # Use TlCoN2O but as a stand-in (adapters don't enforce formula/CIF match)
    return {
        "": "0",
        "material_id": "carbon_001",
        "cif": _CIF_TLCO,
        "formula": "TlCoN2O",
        "heat_all": "-10.0",
        "energy_per_atom": "-10.0",
        "split": "test",
    }


@pytest.fixture()
def mpts52_raw_dict() -> dict:
    """Minimal valid MPTS52 CSV row dict (CIF format)."""
    return {
        "": "0",
        "material_id": "mpts_001",
        "cif": _CIF_TLCO,
        "formula": "TlCoN2O",
        "heat_all": "-2.0",
        "heat_ref": "-1.9",
        "dir_gap": "0.0",
        "ind_gap": "0.0",
        "spacegroup": "Pm-3m",
        "volume": "76.5",
        "split": "train",
    }


# =============================================================================
# Perov5 Adapter Tests
# =============================================================================

def test_perov5_adapter_attributes() -> None:
    adapter = Perov5Adapter()
    assert adapter.dataset_name == "perov-5"
    assert "txie-93" in adapter.source_url or "cdvae" in adapter.source_url


def test_perov5_convert_basic(perov5_raw_dict: dict) -> None:
    adapter = Perov5Adapter()
    rec = adapter.convert(perov5_raw_dict, source_index=0)

    assert rec.source_dataset == "perov-5"
    assert rec.source_index == 0
    assert rec.n_atoms == 5  # TlCoN2O = 5 atoms
    assert rec.split == "train"
    assert "TlCoN2O" in rec.properties.values()


def test_perov5_lattice(perov5_raw_dict: dict) -> None:
    adapter = Perov5Adapter()
    rec = adapter.convert(perov5_raw_dict, source_index=0)

    # Lattice params come from the CIF: a=b=c=4.246 Å
    assert abs(rec.lattice.a - 4.24596403) < 1e-4
    assert abs(rec.lattice.b - 4.24596403) < 1e-4
    assert abs(rec.lattice.c - 4.24596403) < 1e-4


def test_perov5_energy_property(perov5_raw_dict: dict) -> None:
    adapter = Perov5Adapter()
    rec = adapter.convert(perov5_raw_dict, source_index=0)

    assert "formation_energy_per_atom" in rec.properties
    assert rec.properties["formation_energy_per_atom"] == pytest.approx(-2.5)


def test_perov5_no_energy(perov5_raw_dict: dict) -> None:
    perov5_raw_dict["heat_all"] = ""
    perov5_raw_dict["heat_ref"] = ""
    adapter = Perov5Adapter()
    rec = adapter.convert(perov5_raw_dict, source_index=1)

    # formation_energy_per_atom should not be present if heat_all is empty
    assert "formation_energy_per_atom" not in rec.properties


# =============================================================================
# MP20 Adapter Tests
# =============================================================================

def test_mp20_adapter_attributes() -> None:
    adapter = MP20Adapter()
    assert adapter.dataset_name == "mp-20"


def test_mp20_convert_basic(mp20_raw_dict: dict) -> None:
    adapter = MP20Adapter()
    rec = adapter.convert(mp20_raw_dict, source_index=0)

    assert rec.source_dataset == "mp-20"
    assert rec.n_atoms == 5  # BaBiN3 = 5 atoms
    assert set(rec.elements) == {"Ba", "Bi", "N"}
    assert rec.split == "val"


def test_mp20_all_properties(mp20_raw_dict: dict) -> None:
    adapter = MP20Adapter()
    rec = adapter.convert(mp20_raw_dict, source_index=0)

    assert "formation_energy_per_atom" in rec.properties
    assert "e_above_hull" in rec.properties
    assert "band_gap" in rec.properties
    # band_gap = max(dir_gap=5.2, ind_gap=4.9) = 5.2
    assert rec.properties["band_gap"] == pytest.approx(5.2)


def test_mp20_missing_optional_properties(perov5_raw_dict: dict) -> None:
    """Row with no e_above_hull → that key absent from properties."""
    row = {**perov5_raw_dict, "material_id": "mp_test"}
    row.pop("e_above_hull", None)
    adapter = MP20Adapter()
    rec = adapter.convert(row, source_index=0)

    assert rec.n_atoms == 5
    assert "e_above_hull" not in rec.properties


# =============================================================================
# Carbon24 Adapter Tests
# =============================================================================

def test_carbon24_adapter_attributes() -> None:
    adapter = Carbon24Adapter()
    assert adapter.dataset_name == "carbon-24"


def test_carbon24_convert_basic(carbon24_raw_dict: dict) -> None:
    adapter = Carbon24Adapter()
    rec = adapter.convert(carbon24_raw_dict, source_index=0)

    assert rec.source_dataset == "carbon-24"
    assert rec.n_atoms == 5
    assert rec.split == "test"


def test_carbon24_energy_property(carbon24_raw_dict: dict) -> None:
    adapter = Carbon24Adapter()
    rec = adapter.convert(carbon24_raw_dict, source_index=0)

    assert "energy_per_atom" in rec.properties
    assert rec.properties["energy_per_atom"] == pytest.approx(-10.0)


# =============================================================================
# MPTS52 Adapter Tests
# =============================================================================

def test_mpts52_adapter_attributes() -> None:
    adapter = MPTS52Adapter()
    assert adapter.dataset_name == "mpts-52"


def test_mpts52_convert_basic(mpts52_raw_dict: dict) -> None:
    adapter = MPTS52Adapter()
    rec = adapter.convert(mpts52_raw_dict, source_index=0)

    assert rec.source_dataset == "mpts-52"
    assert rec.n_atoms == 5


def test_mpts52_all_properties(mpts52_raw_dict: dict) -> None:
    adapter = MPTS52Adapter()
    rec = adapter.convert(mpts52_raw_dict, source_index=0)

    assert "spacegroup" in rec.properties
    assert "volume" in rec.properties
    assert rec.properties["spacegroup"] == "Pm-3m"
    assert rec.properties["volume"] == pytest.approx(76.5)


# =============================================================================
# Manifest Generation Tests
# =============================================================================

def test_manifest_single_record(perov5_raw_dict: dict) -> None:
    adapter = Perov5Adapter()
    rec = adapter.convert(perov5_raw_dict, source_index=0)
    manifest = adapter.make_manifest([rec])

    assert manifest.dataset_name == "perov-5"
    assert manifest.n_structures == 1
    assert manifest.nominal_drift_schema_version == "1.0"
    assert manifest.importer_version == ADAPTER_VERSION


def test_manifest_multiple_records(perov5_raw_dict: dict) -> None:
    adapter = Perov5Adapter()
    recs = [
        adapter.convert(perov5_raw_dict, source_index=0),
        adapter.convert(perov5_raw_dict, source_index=1),
    ]
    manifest = adapter.make_manifest(recs)
    assert manifest.n_structures == 2


def test_manifest_splits(perov5_raw_dict: dict) -> None:
    adapter = Perov5Adapter()
    r1 = {**perov5_raw_dict, "split": "train"}
    r2 = {**perov5_raw_dict, "split": "val"}
    r3 = {**perov5_raw_dict, "split": "test"}
    recs = [
        adapter.convert(r1, 0),
        adapter.convert(r2, 1),
        adapter.convert(r3, 2),
    ]
    manifest = adapter.make_manifest(recs)
    assert manifest.splits == {"train": 1, "val": 1, "test": 1}


def test_manifest_elements(perov5_raw_dict: dict) -> None:
    adapter = Perov5Adapter()
    rec = adapter.convert(perov5_raw_dict, source_index=0)
    manifest = adapter.make_manifest([rec])

    # TlCoN2O → Tl, Co, N, O
    assert "N" in manifest.elements_present
    assert "O" in manifest.elements_present
    assert manifest.elements_present == sorted(manifest.elements_present)


def test_manifest_property_keys(mp20_raw_dict: dict) -> None:
    adapter = MP20Adapter()
    rec = adapter.convert(mp20_raw_dict, source_index=0)
    manifest = adapter.make_manifest([rec])

    assert "formation_energy_per_atom" in manifest.property_keys
    assert "band_gap" in manifest.property_keys
    assert manifest.property_keys == sorted(manifest.property_keys)


def test_manifest_timestamp_iso() -> None:
    adapter = Perov5Adapter()
    manifest = adapter.make_manifest([])
    assert "T" in manifest.imported_at
    assert "Z" in manifest.imported_at


# =============================================================================
# Adapter Registry Tests
# =============================================================================

def test_adapter_registry_keys() -> None:
    assert "perov-5" in ADAPTER_REGISTRY
    assert "mp-20" in ADAPTER_REGISTRY
    assert "carbon-24" in ADAPTER_REGISTRY
    assert "mpts-52" in ADAPTER_REGISTRY


def test_adapter_registry_values() -> None:
    assert ADAPTER_REGISTRY["perov-5"] == Perov5Adapter
    assert ADAPTER_REGISTRY["mp-20"] == MP20Adapter
    assert ADAPTER_REGISTRY["carbon-24"] == Carbon24Adapter
    assert ADAPTER_REGISTRY["mpts-52"] == MPTS52Adapter


def test_get_adapter_perov5() -> None:
    assert isinstance(get_adapter("perov-5"), Perov5Adapter)


def test_get_adapter_mp20() -> None:
    assert isinstance(get_adapter("mp-20"), MP20Adapter)


def test_get_adapter_unknown() -> None:
    with pytest.raises(KeyError):
        get_adapter("unknown-dataset")


# =============================================================================
# Batch Normalization Tests
# =============================================================================

def test_normalise_records_creates_jsonl(perov5_raw_dict: dict) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = Perov5Adapter()
        rec = adapter.convert(perov5_raw_dict, source_index=0)
        normalise_records([rec], tmpdir, "perov-5")
        assert (Path(tmpdir) / "structures.jsonl").exists()


def test_normalise_records_creates_manifest(perov5_raw_dict: dict) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = Perov5Adapter()
        rec = adapter.convert(perov5_raw_dict, source_index=0)
        normalise_records([rec], tmpdir, "perov-5")
        assert (Path(tmpdir) / "manifest.json").exists()


def test_normalise_records_jsonl_content(perov5_raw_dict: dict) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = Perov5Adapter()
        rec = adapter.convert(perov5_raw_dict, source_index=0)
        normalise_records([rec], tmpdir, "perov-5")
        lines = (Path(tmpdir) / "structures.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1


def test_normalise_records_manifest_return(perov5_raw_dict: dict) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = Perov5Adapter()
        rec = adapter.convert(perov5_raw_dict, source_index=0)
        manifest = normalise_records([rec], tmpdir, "perov-5")
        assert manifest.n_structures == 1
        assert manifest.dataset_name == "perov-5"


def test_normalise_records_multiple_records(perov5_raw_dict: dict, mp20_raw_dict: dict) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = Perov5Adapter()
        rec1 = adapter.convert(perov5_raw_dict, 0)
        rec2 = adapter.convert(perov5_raw_dict, 1)
        manifest = normalise_records([rec1, rec2], tmpdir, "perov-5")
        assert manifest.n_structures == 2
        lines = (Path(tmpdir) / "structures.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2


def test_normalise_records_creates_output_dir(perov5_raw_dict: dict) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = f"{tmpdir}/new_dir/nested"
        adapter = Perov5Adapter()
        rec = adapter.convert(perov5_raw_dict, source_index=0)
        normalise_records([rec], output_dir, "perov-5")
        assert Path(output_dir).exists()


def test_normalise_records_different_adapters(mp20_raw_dict: dict) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = MP20Adapter()
        rec = adapter.convert(mp20_raw_dict, 0)
        manifest = normalise_records([rec], tmpdir, "mp-20")
        assert manifest.dataset_name == "mp-20"


# =============================================================================
# Edge Cases and Validation Tests
# =============================================================================

def test_adapter_validate_n_atoms(perov5_raw_dict: dict) -> None:
    adapter = Perov5Adapter()
    rec = adapter.convert(perov5_raw_dict, 0)
    assert rec.n_atoms == len(rec.sites)


def test_adapter_validate_elements_sorted(perov5_raw_dict: dict) -> None:
    adapter = Perov5Adapter()
    rec = adapter.convert(perov5_raw_dict, 0)
    assert rec.elements == tuple(sorted(rec.elements))


def test_adapter_generate_unique_record_ids(perov5_raw_dict: dict) -> None:
    adapter = Perov5Adapter()
    rec1 = adapter.convert(perov5_raw_dict, 0)
    rec2 = adapter.convert(perov5_raw_dict, 1)
    assert rec1.record_id != rec2.record_id


def test_adapter_preserve_fractional_coords(perov5_raw_dict: dict) -> None:
    """Fractional coordinates are preserved from the CIF."""
    adapter = Perov5Adapter()
    rec = adapter.convert(perov5_raw_dict, 0)
    # All frac_coords must be 3-tuples of finite floats
    for site in rec.sites:
        assert len(site.frac_coords) == 3
        assert all(isinstance(v, float) for v in site.frac_coords)


# =============================================================================
# Integration Tests
# =============================================================================

def test_full_workflow_perov5(perov5_raw_dict: dict) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = get_adapter("perov-5")
        rec = adapter.convert(perov5_raw_dict, 0)
        manifest = normalise_records([rec], tmpdir, "perov-5")

        assert manifest.n_structures == 1
        assert manifest.dataset_name == "perov-5"
        assert Path(tmpdir, "structures.jsonl").exists()
        assert Path(tmpdir, "manifest.json").exists()


def test_cross_adapter_element_compatibility(perov5_raw_dict: dict, mp20_raw_dict: dict) -> None:
    """Both adapters return CrystalRecords with sorted, non-empty elements."""
    rec1 = Perov5Adapter().convert(perov5_raw_dict, 0)
    rec2 = MP20Adapter().convert(mp20_raw_dict, 0)

    for rec in (rec1, rec2):
        assert len(rec.elements) > 0
        assert rec.elements == tuple(sorted(rec.elements))
        assert set(rec.elements) == {s.species for s in rec.sites}
