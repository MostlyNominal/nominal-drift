"""
tests/unit/test_datasets/test_schema.py
========================================
Unit tests for ``nominal_drift.datasets.schema``.

Covers:
  - Valid construction of all four models
  - All specified field validators (lengths, angles, species, indices, splits)
  - Cross-field model validators in CrystalRecord
  - Frozen-model immutability
  - JSON round-trip (model_dump_json / model_validate_json)
  - properties dict accepting float, str, and None values
  - DatasetManifest sorted/deduplicated list validators
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nominal_drift.datasets.schema import (
    AtomicSite,
    CrystalRecord,
    DatasetManifest,
    LatticeParams,
)


# ---------------------------------------------------------------------------
# Shared factory helpers
# ---------------------------------------------------------------------------

def _lattice(**overrides) -> dict:
    base = dict(a=4.0, b=4.0, c=4.0, alpha=90.0, beta=90.0, gamma=90.0)
    base.update(overrides)
    return base


def _site(species: str = "Ba", coords: tuple = (0.0, 0.0, 0.0)) -> AtomicSite:
    return AtomicSite(species=species, frac_coords=coords)


def _record(**overrides) -> dict:
    """Return a dict of valid CrystalRecord kwargs, with optional overrides."""
    base = dict(
        record_id="rec-001",
        source_dataset="perov-5",
        source_index=0,
        split="train",
        elements=("Ba", "O", "Ti"),
        n_atoms=5,
        lattice=LatticeParams(**_lattice()),
        sites=(
            _site("Ba", (0.0, 0.0, 0.0)),
            _site("Ti", (0.5, 0.5, 0.5)),
            _site("O",  (0.5, 0.5, 0.0)),
            _site("O",  (0.5, 0.0, 0.5)),
            _site("O",  (0.0, 0.5, 0.5)),
        ),
        properties={"formation_energy_per_atom": -2.34},
        raw_path=None,
    )
    base.update(overrides)
    return base


def _manifest(**overrides) -> dict:
    base = dict(
        dataset_name="perov-5",
        nominal_drift_schema_version="1.0",
        source_url="https://example.com/perov5.tar.gz",
        n_structures=18928,
        splits={"train": 17928, "val": 500, "test": 500},
        elements_present=["Ba", "Ca", "La", "O", "Ti"],
        property_keys=["formation_energy_per_atom"],
        imported_at="2026-04-07T10:00:00Z",
        importer_version="0.1.0",
        raw_path="data/datasets/raw/perov-5/",
        checksum_raw="sha256:abc123",
    )
    base.update(overrides)
    return base


# ===========================================================================
# TestLatticeParams
# ===========================================================================

class TestLatticeParams:

    def test_valid_cubic(self):
        lp = LatticeParams(**_lattice())
        assert lp.a == 4.0
        assert lp.alpha == 90.0

    def test_valid_triclinic(self):
        lp = LatticeParams(a=3.1, b=4.2, c=5.8, alpha=75.0, beta=82.0, gamma=110.0)
        assert lp.gamma == 110.0

    def test_length_a_zero_raises(self):
        with pytest.raises(ValidationError, match="a"):
            LatticeParams(**_lattice(a=0.0))

    def test_length_b_negative_raises(self):
        with pytest.raises(ValidationError, match="b"):
            LatticeParams(**_lattice(b=-1.0))

    def test_length_c_zero_raises(self):
        with pytest.raises(ValidationError, match="c"):
            LatticeParams(**_lattice(c=0.0))

    def test_angle_alpha_zero_raises(self):
        with pytest.raises(ValidationError, match="alpha"):
            LatticeParams(**_lattice(alpha=0.0))

    def test_angle_gamma_180_raises(self):
        with pytest.raises(ValidationError, match="gamma"):
            LatticeParams(**_lattice(gamma=180.0))

    def test_angle_beta_negative_raises(self):
        with pytest.raises(ValidationError, match="beta"):
            LatticeParams(**_lattice(beta=-10.0))

    def test_frozen_immutability(self):
        lp = LatticeParams(**_lattice())
        with pytest.raises(ValidationError):
            lp.a = 99.0  # type: ignore[misc]


# ===========================================================================
# TestAtomicSite
# ===========================================================================

class TestAtomicSite:

    def test_valid_site(self):
        s = _site("Cr", (0.25, 0.25, 0.25))
        assert s.species == "Cr"
        assert s.frac_coords == (0.25, 0.25, 0.25)

    def test_coords_outside_unit_cell_are_allowed(self):
        # Fractional coords outside [0,1] must NOT be rejected.
        s = AtomicSite(species="N", frac_coords=(-0.1, 1.3, 0.5))
        assert s.frac_coords[0] == pytest.approx(-0.1)

    def test_empty_species_raises(self):
        with pytest.raises(ValidationError, match="species"):
            AtomicSite(species="", frac_coords=(0.0, 0.0, 0.0))

    def test_inf_coord_raises(self):
        with pytest.raises(ValidationError, match="finite"):
            AtomicSite(species="Fe", frac_coords=(float("inf"), 0.0, 0.0))

    def test_nan_coord_raises(self):
        with pytest.raises(ValidationError, match="finite"):
            AtomicSite(species="Fe", frac_coords=(float("nan"), 0.0, 0.0))

    def test_frozen_immutability(self):
        s = _site()
        with pytest.raises(ValidationError):
            s.species = "Ni"  # type: ignore[misc]


# ===========================================================================
# TestCrystalRecord
# ===========================================================================

class TestCrystalRecord:

    def test_valid_batio3(self):
        rec = CrystalRecord(**_record())
        assert rec.source_dataset == "perov-5"
        assert rec.n_atoms == 5
        assert rec.elements == ("Ba", "O", "Ti")

    def test_split_none_valid(self):
        rec = CrystalRecord(**_record(split=None))
        assert rec.split is None

    def test_split_val_valid(self):
        rec = CrystalRecord(**_record(split="val"))
        assert rec.split == "val"

    def test_split_test_valid(self):
        rec = CrystalRecord(**_record(split="test"))
        assert rec.split == "test"

    def test_invalid_split_raises(self):
        with pytest.raises(ValidationError, match="split"):
            CrystalRecord(**_record(split="holdout"))

    def test_empty_record_id_raises(self):
        with pytest.raises(ValidationError, match="record_id"):
            CrystalRecord(**_record(record_id=""))

    def test_empty_source_dataset_raises(self):
        with pytest.raises(ValidationError, match="source_dataset"):
            CrystalRecord(**_record(source_dataset=""))

    def test_negative_source_index_raises(self):
        with pytest.raises(ValidationError, match="source_index"):
            CrystalRecord(**_record(source_index=-1))

    def test_source_index_zero_valid(self):
        rec = CrystalRecord(**_record(source_index=0))
        assert rec.source_index == 0

    def test_n_atoms_zero_raises(self):
        with pytest.raises(ValidationError, match="n_atoms"):
            CrystalRecord(**_record(n_atoms=0))

    def test_n_atoms_mismatch_raises(self):
        # n_atoms says 3 but sites has 5 entries
        with pytest.raises(ValidationError, match="n_atoms"):
            CrystalRecord(**_record(n_atoms=3))

    def test_unsorted_elements_raises(self):
        with pytest.raises(ValidationError, match="sorted"):
            CrystalRecord(**_record(elements=("Ti", "Ba", "O")))

    def test_duplicate_elements_raises(self):
        with pytest.raises(ValidationError, match="duplicates"):
            CrystalRecord(**_record(elements=("Ba", "Ba", "O", "Ti")))

    def test_elements_species_mismatch_raises(self):
        # elements says ("Ba", "Fe") but sites contain Ba, Ti, O
        with pytest.raises(ValidationError, match="species"):
            CrystalRecord(**_record(elements=("Ba", "Fe")))

    def test_frozen_immutability(self):
        rec = CrystalRecord(**_record())
        with pytest.raises(ValidationError):
            rec.record_id = "new-id"  # type: ignore[misc]

    def test_json_roundtrip(self):
        rec = CrystalRecord(**_record())
        json_str = rec.model_dump_json()
        rec2 = CrystalRecord.model_validate_json(json_str)
        assert rec == rec2
        assert rec2.elements == ("Ba", "O", "Ti")
        assert rec2.sites[1].species == "Ti"

    def test_properties_float(self):
        rec = CrystalRecord(**_record(properties={"band_gap": 1.75}))
        assert rec.properties["band_gap"] == pytest.approx(1.75)

    def test_properties_str(self):
        rec = CrystalRecord(**_record(properties={"space_group": "Pm-3m"}))
        assert rec.properties["space_group"] == "Pm-3m"

    def test_properties_none(self):
        rec = CrystalRecord(**_record(properties={"hull_distance": None}))
        assert rec.properties["hull_distance"] is None

    def test_properties_mixed_types(self):
        props = {
            "formation_energy_per_atom": -2.34,
            "space_group": "Pm-3m",
            "hull_distance": None,
        }
        rec = CrystalRecord(**_record(properties=props))
        assert len(rec.properties) == 3

    def test_raw_path_none_valid(self):
        rec = CrystalRecord(**_record(raw_path=None))
        assert rec.raw_path is None

    def test_raw_path_string_valid(self):
        rec = CrystalRecord(**_record(raw_path="data/datasets/raw/perov-5/001.cif"))
        assert "001.cif" in rec.raw_path


# ===========================================================================
# TestDatasetManifest
# ===========================================================================

class TestDatasetManifest:

    def test_valid_manifest(self):
        m = DatasetManifest(**_manifest())
        assert m.dataset_name == "perov-5"
        assert m.n_structures == 18928
        assert m.splits["train"] == 17928

    def test_empty_dataset_name_raises(self):
        with pytest.raises(ValidationError, match="dataset_name"):
            DatasetManifest(**_manifest(dataset_name=""))

    def test_negative_n_structures_raises(self):
        with pytest.raises(ValidationError, match="n_structures"):
            DatasetManifest(**_manifest(n_structures=-1))

    def test_zero_n_structures_valid(self):
        m = DatasetManifest(**_manifest(n_structures=0))
        assert m.n_structures == 0

    def test_unsorted_elements_present_raises(self):
        with pytest.raises(ValidationError, match="elements_present"):
            DatasetManifest(**_manifest(elements_present=["Ti", "Ba", "O"]))

    def test_duplicate_elements_present_raises(self):
        with pytest.raises(ValidationError, match="elements_present"):
            DatasetManifest(**_manifest(elements_present=["Ba", "Ba", "Ti"]))

    def test_unsorted_property_keys_raises(self):
        with pytest.raises(ValidationError, match="property_keys"):
            DatasetManifest(**_manifest(property_keys=["z_key", "a_key"]))

    def test_duplicate_property_keys_raises(self):
        with pytest.raises(ValidationError, match="property_keys"):
            DatasetManifest(**_manifest(property_keys=["band_gap", "band_gap"]))

    def test_source_url_none_valid(self):
        m = DatasetManifest(**_manifest(source_url=None))
        assert m.source_url is None

    def test_checksum_none_valid(self):
        m = DatasetManifest(**_manifest(checksum_raw=None))
        assert m.checksum_raw is None

    def test_empty_splits_valid(self):
        m = DatasetManifest(**_manifest(splits={}))
        assert m.splits == {}

    def test_frozen_immutability(self):
        m = DatasetManifest(**_manifest())
        with pytest.raises(ValidationError):
            m.dataset_name = "carbon-24"  # type: ignore[misc]

    def test_json_roundtrip(self):
        m = DatasetManifest(**_manifest())
        m2 = DatasetManifest.model_validate_json(m.model_dump_json())
        assert m == m2
        assert m2.elements_present == ["Ba", "Ca", "La", "O", "Ti"]
