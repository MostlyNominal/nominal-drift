"""
tests/unit/test_adapters.py
===========================

Unit tests for nominal_drift.datasets.adapters.

Tests cover:
  - BaseAdapter abstraction
  - Dataset-specific adapters (Perov5, MP20, Carbon24, MPTS52)
  - Manifest generation
  - Adapter registry and factory
  - Batch normalization

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


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture()
def perov5_raw_dict() -> dict:
    """Minimal valid Perov5 raw dict."""
    return {
        "material_id": "perov_001",
        "formula": "BaTiO3",
        "a": 4.0,
        "b": 4.0,
        "c": 4.1,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
        "sites": [
            {"species": "Ba", "xyz": [0.5, 0.5, 0.5]},
            {"species": "Ti", "xyz": [0.0, 0.0, 0.0]},
            {"species": "O", "xyz": [0.0, 0.5, 0.5]},
            {"species": "O", "xyz": [0.5, 0.0, 0.5]},
            {"species": "O", "xyz": [0.5, 0.5, 0.0]},
        ],
        "formation_energy_per_atom": -2.5,
        "split": "train",
    }


@pytest.fixture()
def mp20_raw_dict() -> dict:
    """Minimal valid MP20 raw dict."""
    return {
        "material_id": "mp_001",
        "formula": "SiO2",
        "a": 5.0,
        "b": 5.0,
        "c": 6.0,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 120.0,
        "sites": [
            {"species": "Si", "xyz": [0.0, 0.0, 0.0]},
            {"species": "Si", "xyz": [0.5, 0.5, 0.5]},
            {"species": "O", "xyz": [0.3, 0.0, 0.0]},
            {"species": "O", "xyz": [0.0, 0.3, 0.0]},
            {"species": "O", "xyz": [0.0, 0.0, 0.3]},
            {"species": "O", "xyz": [0.7, 0.0, 0.0]},
        ],
        "formation_energy_per_atom": -1.5,
        "e_above_hull": 0.1,
        "band_gap": 5.2,
        "split": "val",
    }


@pytest.fixture()
def carbon24_raw_dict() -> dict:
    """Minimal valid Carbon24 raw dict."""
    return {
        "structure_id": "carbon_001",
        "a": 2.5,
        "b": 2.5,
        "c": 4.0,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 120.0,
        "sites": [
            {"species": "C", "xyz": [0.0, 0.0, 0.0]},
            {"species": "C", "xyz": [0.5, 0.5, 0.0]},
            {"species": "C", "xyz": [0.333, 0.667, 0.5]},
        ],
        "energy_per_atom": -10.0,
        "split": "test",
    }


@pytest.fixture()
def mpts52_raw_dict() -> dict:
    """Minimal valid MPTS52 raw dict."""
    return {
        "material_id": "mpts_001",
        "formula": "Fe2O3",
        "a": 5.0,
        "b": 5.0,
        "c": 14.0,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 120.0,
        "sites": [
            {"species": "Fe", "xyz": [0.0, 0.0, 0.0]},
            {"species": "Fe", "xyz": [0.333, 0.667, 0.333]},
            {"species": "Fe", "xyz": [0.667, 0.333, 0.667]},
            {"species": "O", "xyz": [0.3, 0.0, 0.0]},
            {"species": "O", "xyz": [0.0, 0.3, 0.0]},
            {"species": "O", "xyz": [0.0, 0.0, 0.3]},
        ],
        "spacegroup": "R-3m",
        "volume": 100.0,
        "split": "train",
    }


# =============================================================================
# Perov5 Adapter Tests
# =============================================================================

def test_perov5_adapter_attributes() -> None:
    """Test Perov5 adapter class attributes."""
    adapter = Perov5Adapter()
    assert adapter.dataset_name == "perov-5"
    assert "txie-93" in adapter.source_url


def test_perov5_convert_basic(perov5_raw_dict: dict) -> None:
    """Test basic Perov5 conversion."""
    adapter = Perov5Adapter()
    rec = adapter.convert(perov5_raw_dict, source_index=0)

    assert rec.source_dataset == "perov-5"
    assert rec.source_index == 0
    assert rec.n_atoms == 5
    assert rec.elements == ("Ba", "O", "Ti")
    assert rec.split == "train"
    assert "BaTiO3" in rec.properties.values()


def test_perov5_lattice(perov5_raw_dict: dict) -> None:
    """Test Perov5 lattice parameters."""
    adapter = Perov5Adapter()
    rec = adapter.convert(perov5_raw_dict, source_index=0)

    assert rec.lattice.a == 4.0
    assert rec.lattice.b == 4.0
    assert rec.lattice.c == 4.1


def test_perov5_energy_property(perov5_raw_dict: dict) -> None:
    """Test energy property in Perov5."""
    adapter = Perov5Adapter()
    rec = adapter.convert(perov5_raw_dict, source_index=0)

    assert "formation_energy_per_atom" in rec.properties
    assert rec.properties["formation_energy_per_atom"] == -2.5


def test_perov5_no_energy(perov5_raw_dict: dict) -> None:
    """Test Perov5 with no energy data."""
    perov5_raw_dict["formation_energy_per_atom"] = None
    adapter = Perov5Adapter()
    rec = adapter.convert(perov5_raw_dict, source_index=1)

    assert "formation_energy_per_atom" not in rec.properties or \
           rec.properties["formation_energy_per_atom"] is None


# =============================================================================
# MP20 Adapter Tests
# =============================================================================

def test_mp20_adapter_attributes() -> None:
    """Test MP20 adapter class attributes."""
    adapter = MP20Adapter()
    assert adapter.dataset_name == "mp-20"


def test_mp20_convert_basic(mp20_raw_dict: dict) -> None:
    """Test basic MP20 conversion."""
    adapter = MP20Adapter()
    rec = adapter.convert(mp20_raw_dict, source_index=0)

    assert rec.source_dataset == "mp-20"
    assert rec.n_atoms == 6
    assert rec.elements == ("O", "Si")
    assert rec.split == "val"


def test_mp20_all_properties(mp20_raw_dict: dict) -> None:
    """Test MP20 with all optional properties."""
    adapter = MP20Adapter()
    rec = adapter.convert(mp20_raw_dict, source_index=0)

    assert "formation_energy_per_atom" in rec.properties
    assert "e_above_hull" in rec.properties
    assert "band_gap" in rec.properties
    assert rec.properties["band_gap"] == 5.2


def test_mp20_missing_optional_properties() -> None:
    """Test MP20 with missing optional properties."""
    raw = {
        "material_id": "mp_test",
        "formula": "Test",
        "a": 5.0,
        "b": 5.0,
        "c": 6.0,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
        "sites": [{"species": "X", "xyz": [0.0, 0.0, 0.0]}],
    }
    adapter = MP20Adapter()
    rec = adapter.convert(raw, source_index=0)

    assert rec.n_atoms == 1
    # Properties should be present but optional ones may be absent


# =============================================================================
# Carbon24 Adapter Tests
# =============================================================================

def test_carbon24_adapter_attributes() -> None:
    """Test Carbon24 adapter class attributes."""
    adapter = Carbon24Adapter()
    assert adapter.dataset_name == "carbon-24"


def test_carbon24_convert_basic(carbon24_raw_dict: dict) -> None:
    """Test basic Carbon24 conversion."""
    adapter = Carbon24Adapter()
    rec = adapter.convert(carbon24_raw_dict, source_index=0)

    assert rec.source_dataset == "carbon-24"
    assert rec.n_atoms == 3
    assert rec.elements == ("C",)
    assert rec.split == "test"


def test_carbon24_energy_property(carbon24_raw_dict: dict) -> None:
    """Test energy property in Carbon24."""
    adapter = Carbon24Adapter()
    rec = adapter.convert(carbon24_raw_dict, source_index=0)

    assert "energy_per_atom" in rec.properties
    assert rec.properties["energy_per_atom"] == -10.0


# =============================================================================
# MPTS52 Adapter Tests
# =============================================================================

def test_mpts52_adapter_attributes() -> None:
    """Test MPTS52 adapter class attributes."""
    adapter = MPTS52Adapter()
    assert adapter.dataset_name == "mpts-52"


def test_mpts52_convert_basic(mpts52_raw_dict: dict) -> None:
    """Test basic MPTS52 conversion."""
    adapter = MPTS52Adapter()
    rec = adapter.convert(mpts52_raw_dict, source_index=0)

    assert rec.source_dataset == "mpts-52"
    assert rec.n_atoms == 6
    assert rec.elements == ("Fe", "O")


def test_mpts52_all_properties(mpts52_raw_dict: dict) -> None:
    """Test MPTS52 with all properties."""
    adapter = MPTS52Adapter()
    rec = adapter.convert(mpts52_raw_dict, source_index=0)

    assert "spacegroup" in rec.properties
    assert "volume" in rec.properties
    assert rec.properties["spacegroup"] == "R-3m"
    assert rec.properties["volume"] == 100.0


# =============================================================================
# Manifest Generation Tests
# =============================================================================

def test_manifest_single_record(perov5_raw_dict: dict) -> None:
    """Test manifest generation with single record."""
    adapter = Perov5Adapter()
    rec = adapter.convert(perov5_raw_dict, source_index=0)
    manifest = adapter.make_manifest([rec])

    assert manifest.dataset_name == "perov-5"
    assert manifest.n_structures == 1
    assert manifest.nominal_drift_schema_version == "1.0"
    assert manifest.importer_version == ADAPTER_VERSION


def test_manifest_multiple_records(perov5_raw_dict: dict) -> None:
    """Test manifest with multiple records."""
    adapter = Perov5Adapter()
    recs = [
        adapter.convert(perov5_raw_dict, source_index=0),
        adapter.convert(perov5_raw_dict, source_index=1),
    ]
    manifest = adapter.make_manifest(recs)

    assert manifest.n_structures == 2


def test_manifest_splits(perov5_raw_dict: dict) -> None:
    """Test manifest split tracking."""
    adapter = Perov5Adapter()
    raw1 = perov5_raw_dict.copy()
    raw1["split"] = "train"
    raw2 = perov5_raw_dict.copy()
    raw2["split"] = "val"
    raw3 = perov5_raw_dict.copy()
    raw3["split"] = "test"

    recs = [
        adapter.convert(raw1, 0),
        adapter.convert(raw2, 1),
        adapter.convert(raw3, 2),
    ]
    manifest = adapter.make_manifest(recs)

    assert manifest.splits == {"train": 1, "val": 1, "test": 1}


def test_manifest_elements(perov5_raw_dict: dict) -> None:
    """Test manifest element collection."""
    adapter = Perov5Adapter()
    rec = adapter.convert(perov5_raw_dict, source_index=0)
    manifest = adapter.make_manifest([rec])

    assert "Ba" in manifest.elements_present
    assert "Ti" in manifest.elements_present
    assert "O" in manifest.elements_present
    assert manifest.elements_present == sorted(manifest.elements_present)


def test_manifest_property_keys(mp20_raw_dict: dict) -> None:
    """Test manifest property key collection."""
    adapter = MP20Adapter()
    rec = adapter.convert(mp20_raw_dict, source_index=0)
    manifest = adapter.make_manifest([rec])

    assert "formation_energy_per_atom" in manifest.property_keys
    assert "band_gap" in manifest.property_keys
    assert manifest.property_keys == sorted(manifest.property_keys)


def test_manifest_timestamp_iso() -> None:
    """Test manifest timestamp is ISO format."""
    adapter = Perov5Adapter()
    manifest = adapter.make_manifest([])

    assert "T" in manifest.imported_at
    assert "Z" in manifest.imported_at


# =============================================================================
# Adapter Registry Tests
# =============================================================================

def test_adapter_registry_keys() -> None:
    """Test all adapters are in registry."""
    assert "perov-5" in ADAPTER_REGISTRY
    assert "mp-20" in ADAPTER_REGISTRY
    assert "carbon-24" in ADAPTER_REGISTRY
    assert "mpts-52" in ADAPTER_REGISTRY


def test_adapter_registry_values() -> None:
    """Test registry values are adapter classes."""
    assert ADAPTER_REGISTRY["perov-5"] == Perov5Adapter
    assert ADAPTER_REGISTRY["mp-20"] == MP20Adapter
    assert ADAPTER_REGISTRY["carbon-24"] == Carbon24Adapter
    assert ADAPTER_REGISTRY["mpts-52"] == MPTS52Adapter


def test_get_adapter_perov5() -> None:
    """Test getting Perov5 adapter."""
    adapter = get_adapter("perov-5")
    assert isinstance(adapter, Perov5Adapter)


def test_get_adapter_mp20() -> None:
    """Test getting MP20 adapter."""
    adapter = get_adapter("mp-20")
    assert isinstance(adapter, MP20Adapter)


def test_get_adapter_unknown() -> None:
    """Test getting unknown adapter raises KeyError."""
    with pytest.raises(KeyError):
        get_adapter("unknown-dataset")


# =============================================================================
# Batch Normalization Tests
# =============================================================================

def test_normalise_records_creates_jsonl(perov5_raw_dict: dict) -> None:
    """Test normalise_records creates JSONL file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = Perov5Adapter()
        rec = adapter.convert(perov5_raw_dict, source_index=0)
        manifest = normalise_records([rec], tmpdir, "perov-5")

        jsonl_file = Path(tmpdir) / "structures.jsonl"
        assert jsonl_file.exists()


def test_normalise_records_creates_manifest(perov5_raw_dict: dict) -> None:
    """Test normalise_records creates manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = Perov5Adapter()
        rec = adapter.convert(perov5_raw_dict, source_index=0)
        manifest = normalise_records([rec], tmpdir, "perov-5")

        manifest_file = Path(tmpdir) / "manifest.json"
        assert manifest_file.exists()


def test_normalise_records_jsonl_content(perov5_raw_dict: dict) -> None:
    """Test JSONL file contains correct data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = Perov5Adapter()
        rec = adapter.convert(perov5_raw_dict, source_index=0)
        normalise_records([rec], tmpdir, "perov-5")

        jsonl_file = Path(tmpdir) / "structures.jsonl"
        lines = jsonl_file.read_text().strip().split("\n")
        assert len(lines) == 1


def test_normalise_records_manifest_return(perov5_raw_dict: dict) -> None:
    """Test normalise_records returns manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = Perov5Adapter()
        rec = adapter.convert(perov5_raw_dict, source_index=0)
        manifest = normalise_records([rec], tmpdir, "perov-5")

        assert manifest.n_structures == 1
        assert manifest.dataset_name == "perov-5"


def test_normalise_records_multiple_records(
    perov5_raw_dict: dict, mp20_raw_dict: dict
) -> None:
    """Test normalise_records with multiple records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        perov_adapter = Perov5Adapter()
        rec1 = perov_adapter.convert(perov5_raw_dict, 0)
        rec2 = perov_adapter.convert(perov5_raw_dict, 1)
        manifest = normalise_records([rec1, rec2], tmpdir, "perov-5")

        assert manifest.n_structures == 2
        jsonl_file = Path(tmpdir) / "structures.jsonl"
        lines = jsonl_file.read_text().strip().split("\n")
        assert len(lines) == 2


def test_normalise_records_creates_output_dir(perov5_raw_dict: dict) -> None:
    """Test normalise_records creates output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = f"{tmpdir}/new_dir/nested"
        adapter = Perov5Adapter()
        rec = adapter.convert(perov5_raw_dict, source_index=0)
        normalise_records([rec], output_dir, "perov-5")

        assert Path(output_dir).exists()


def test_normalise_records_different_adapters() -> None:
    """Test normalise_records with different datasets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = MP20Adapter()
        raw = {
            "material_id": "test",
            "formula": "Test",
            "a": 5.0,
            "b": 5.0,
            "c": 6.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
            "sites": [{"species": "X", "xyz": [0.0, 0.0, 0.0]}],
        }
        rec = adapter.convert(raw, 0)
        manifest = normalise_records([rec], tmpdir, "mp-20")

        assert manifest.dataset_name == "mp-20"


# =============================================================================
# Edge Cases and Validation Tests
# =============================================================================

def test_adapter_validate_n_atoms(perov5_raw_dict: dict) -> None:
    """Test that n_atoms matches len(sites)."""
    adapter = Perov5Adapter()
    rec = adapter.convert(perov5_raw_dict, 0)

    assert rec.n_atoms == len(rec.sites)


def test_adapter_validate_elements_sorted(perov5_raw_dict: dict) -> None:
    """Test that elements are sorted."""
    adapter = Perov5Adapter()
    rec = adapter.convert(perov5_raw_dict, 0)

    assert rec.elements == tuple(sorted(rec.elements))


def test_adapter_generate_unique_record_ids(perov5_raw_dict: dict) -> None:
    """Test that record IDs are unique."""
    adapter = Perov5Adapter()
    rec1 = adapter.convert(perov5_raw_dict, 0)
    rec2 = adapter.convert(perov5_raw_dict, 1)

    assert rec1.record_id != rec2.record_id


def test_adapter_preserve_fractional_coords(perov5_raw_dict: dict) -> None:
    """Test that fractional coordinates are preserved."""
    adapter = Perov5Adapter()
    rec = adapter.convert(perov5_raw_dict, 0)

    assert rec.sites[0].frac_coords == (0.5, 0.5, 0.5)


# =============================================================================
# Integration Tests
# =============================================================================

def test_full_workflow_perov5(perov5_raw_dict: dict) -> None:
    """Test full workflow for Perov5."""
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = get_adapter("perov-5")
        rec = adapter.convert(perov5_raw_dict, 0)
        manifest = normalise_records([rec], tmpdir, "perov-5")

        # Verify output
        assert manifest.n_structures == 1
        assert manifest.dataset_name == "perov-5"
        assert Path(tmpdir, "structures.jsonl").exists()
        assert Path(tmpdir, "manifest.json").exists()


def test_cross_adapter_element_compatibility() -> None:
    """Test that different adapters handle elements consistently."""
    perov_adapter = Perov5Adapter()
    mp_adapter = MP20Adapter()

    perov_raw = {
        "material_id": "p1",
        "formula": "BaTiO3",
        "a": 4.0,
        "b": 4.0,
        "c": 4.1,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
        "sites": [
            {"species": "Ba", "xyz": [0.5, 0.5, 0.5]},
            {"species": "Ti", "xyz": [0.0, 0.0, 0.0]},
            {"species": "O", "xyz": [0.0, 0.5, 0.5]},
        ],
    }

    mp_raw = {
        "material_id": "m1",
        "formula": "BaTiO3",
        "a": 4.0,
        "b": 4.0,
        "c": 4.1,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
        "sites": [
            {"species": "Ba", "xyz": [0.5, 0.5, 0.5]},
            {"species": "Ti", "xyz": [0.0, 0.0, 0.0]},
            {"species": "O", "xyz": [0.0, 0.5, 0.5]},
        ],
    }

    rec1 = perov_adapter.convert(perov_raw, 0)
    rec2 = mp_adapter.convert(mp_raw, 0)

    # Elements should be the same
    assert rec1.elements == rec2.elements
