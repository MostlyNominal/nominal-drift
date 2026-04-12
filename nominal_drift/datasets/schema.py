"""
nominal_drift.datasets.schema
==============================
Canonical internal schema for Lane B dataset records.

All models are Pydantic v2 **frozen** — they are immutable after
construction and fully JSON-serialisable via ``model_dump_json()`` /
``model_validate_json()``.

This module is the single source of truth for what a "crystal record"
looks like inside Nominal Drift, regardless of which raw dataset it
came from (Perov-5, Carbon-24, MP-20, MPTS-52, user CIF folders, …).
Adapters (Sprint 2B) convert raw source formats → ``CrystalRecord``.
The normaliser (Sprint 2B) streams records to JSONL on disk.

Public API
----------
``LatticeParams``
    Unit-cell lengths (Å) and angles (°).

``AtomicSite``
    One atomic site: element symbol + fractional coordinates.

``CrystalRecord``
    Complete canonical record for one crystal structure.

``DatasetManifest``
    Metadata written alongside each normalised dataset on disk.

Design notes
------------
* ``CrystalRecord.properties`` is an open-ended ``dict[str, float | str | None]``
  so dataset-specific quantities (formation energy, band gap, hull
  distance, space group string, …) can be stored without schema changes.
* Fractional coordinates are NOT clamped to [0, 1].  Some datasets
  store coordinates outside the unit cell; clamping is the adapter's
  responsibility if desired.
* ``elements`` is the sorted, deduplicated tuple of species present in
  ``sites``.  It is stored explicitly so downstream code can filter by
  element set without unpacking all sites.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, field_validator, model_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_SPLITS: frozenset[str] = frozenset({"train", "val", "test"})


# ---------------------------------------------------------------------------
# LatticeParams
# ---------------------------------------------------------------------------

class LatticeParams(BaseModel, frozen=True):
    """Unit-cell parameters for a periodic crystal structure.

    Attributes
    ----------
    a, b, c : float
        Lattice lengths in Ångströms.  Must be strictly positive.
    alpha, beta, gamma : float
        Inter-axial angles in degrees.  Must satisfy 0 < angle < 180.
    """

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float

    @field_validator("a", "b", "c")
    @classmethod
    def _lengths_positive(cls, v: float, info) -> float:
        if v <= 0.0:
            raise ValueError(
                f"Lattice length '{info.field_name}' must be > 0, got {v!r}"
            )
        return v

    @field_validator("alpha", "beta", "gamma")
    @classmethod
    def _angles_in_range(cls, v: float, info) -> float:
        if not (0.0 < v < 180.0):
            raise ValueError(
                f"Lattice angle '{info.field_name}' must be > 0 and < 180, "
                f"got {v!r}"
            )
        return v


# ---------------------------------------------------------------------------
# AtomicSite
# ---------------------------------------------------------------------------

class AtomicSite(BaseModel, frozen=True):
    """One atomic site within a crystal structure.

    Attributes
    ----------
    species : str
        Element symbol (e.g. ``"Cr"``, ``"Ba"``).  Must be non-empty.
    frac_coords : tuple[float, float, float]
        Fractional coordinates (x, y, z) relative to the lattice vectors.
        All three values must be finite (no ``inf`` or ``nan``).
        Coordinates are **not** clamped to [0, 1] — values outside the
        unit cell are legal at this schema level.
    """

    species: str
    frac_coords: tuple[float, float, float]

    @field_validator("species")
    @classmethod
    def _species_non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("'species' must be a non-empty string")
        return v

    @field_validator("frac_coords")
    @classmethod
    def _coords_finite(
        cls, v: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        for i, coord in enumerate(v):
            if not math.isfinite(coord):
                raise ValueError(
                    f"frac_coords[{i}] must be finite, got {coord!r}"
                )
        return v


# ---------------------------------------------------------------------------
# CrystalRecord
# ---------------------------------------------------------------------------

class CrystalRecord(BaseModel, frozen=True):
    """Canonical internal record for one crystal structure.

    This is the single internal currency for all Lane B workflows.
    Adapters convert raw dataset formats into ``CrystalRecord`` objects;
    the normaliser streams them to JSONL; the loader reconstructs them.

    Attributes
    ----------
    record_id : str
        Globally unique identifier (UUID4 recommended).
    source_dataset : str
        Name of the originating dataset, e.g. ``"perov-5"``.
    source_index : int
        Zero-based row / entry index in the original dataset (≥ 0).
    split : str | None
        Dataset split: one of ``"train"``, ``"val"``, ``"test"``, or
        ``None`` if the dataset has no predefined splits.
    elements : tuple[str, ...]
        Sorted, deduplicated tuple of element symbols present in the
        structure.  Must exactly match ``{site.species for site in sites}``.
    n_atoms : int
        Total number of atomic sites (> 0).  Must equal ``len(sites)``.
    lattice : LatticeParams
        Unit-cell parameters.
    sites : tuple[AtomicSite, ...]
        All atomic sites in the structure.
    properties : dict[str, float | str | None]
        Open-ended key-value store for dataset-specific quantities such
        as formation energy, band gap, space group, hull distance, etc.
        New properties are added as new keys — no schema migration required.
    raw_path : str | None
        Relative path to the source file within ``data/datasets/raw/``,
        or ``None`` if the raw file is not tracked per-structure.
    """

    record_id:      str
    source_dataset: str
    source_index:   int
    split:          str | None
    elements:       tuple[str, ...]
    n_atoms:        int
    lattice:        LatticeParams
    sites:          tuple[AtomicSite, ...]
    properties:     dict[str, float | str | None]
    raw_path:       str | None

    # ------------------------------------------------------------------
    # Field-level validators
    # ------------------------------------------------------------------

    @field_validator("record_id")
    @classmethod
    def _record_id_non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("'record_id' must be a non-empty string")
        return v

    @field_validator("source_dataset")
    @classmethod
    def _source_dataset_non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("'source_dataset' must be a non-empty string")
        return v

    @field_validator("source_index")
    @classmethod
    def _source_index_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(
                f"'source_index' must be >= 0, got {v!r}"
            )
        return v

    @field_validator("n_atoms")
    @classmethod
    def _n_atoms_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"'n_atoms' must be > 0, got {v!r}")
        return v

    @field_validator("split")
    @classmethod
    def _split_valid(cls, v: str | None) -> str | None:
        if v is not None and v not in _VALID_SPLITS:
            raise ValueError(
                f"'split' must be 'train', 'val', 'test', or None; "
                f"got {v!r}"
            )
        return v

    @field_validator("elements")
    @classmethod
    def _elements_non_empty(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("'elements' must be a non-empty tuple")
        return v

    # ------------------------------------------------------------------
    # Cross-field model validator (runs after all field validators pass)
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _cross_field_checks(self) -> "CrystalRecord":
        # 1. n_atoms must equal the number of sites
        if self.n_atoms != len(self.sites):
            raise ValueError(
                f"'n_atoms' ({self.n_atoms}) must equal len(sites) "
                f"({len(self.sites)})"
            )

        # 2. elements must be sorted and deduplicated
        elements_list = list(self.elements)
        if elements_list != sorted(set(elements_list)):
            raise ValueError(
                f"'elements' must be sorted and contain no duplicates; "
                f"got {self.elements!r}"
            )

        # 3. elements must exactly match the unique species in sites
        expected = tuple(sorted({s.species for s in self.sites}))
        if self.elements != expected:
            raise ValueError(
                f"'elements' {self.elements!r} does not match the unique "
                f"species in 'sites' {expected!r}"
            )

        return self


# ---------------------------------------------------------------------------
# DatasetManifest
# ---------------------------------------------------------------------------

class DatasetManifest(BaseModel, frozen=True):
    """Metadata written alongside each normalised dataset on disk.

    One ``manifest.json`` file is created per dataset inside
    ``data/datasets/normalized/<dataset-name>/``.  It is the
    authoritative record for reproducibility and is also used by the
    dataset registry to discover which datasets are locally available.

    Attributes
    ----------
    dataset_name : str
        Short canonical name, e.g. ``"perov-5"``.
    nominal_drift_schema_version : str
        Version of the ``CrystalRecord`` schema used during import
        (e.g. ``"1.0"``).
    source_url : str | None
        URL where the raw dataset was downloaded from, if known.
    n_structures : int
        Total number of structures in the normalised JSONL file (≥ 0).
    splits : dict[str, int]
        Mapping of split name → count, e.g.
        ``{"train": 17928, "val": 500, "test": 500}``.
    elements_present : list[str]
        Sorted, deduplicated list of all element symbols found across the
        dataset.
    property_keys : list[str]
        Sorted, deduplicated list of all property keys present across
        ``CrystalRecord.properties`` in this dataset.
    imported_at : str
        ISO-8601 UTC timestamp of when the import was run.
    importer_version : str
        Version string of the adapter used (e.g. ``"0.1.0"``).
    raw_path : str | None
        Relative path to the raw source directory or file, if tracked.
    checksum_raw : str | None
        SHA-256 checksum of the raw source (``"sha256:<hex>"``), or
        ``None`` if not computed.
    """

    dataset_name:                  str
    nominal_drift_schema_version:  str
    source_url:                    str | None
    n_structures:                  int
    splits:                        dict[str, int]
    elements_present:              list[str]
    property_keys:                 list[str]
    imported_at:                   str
    importer_version:              str
    raw_path:                      str | None
    checksum_raw:                  str | None

    # ------------------------------------------------------------------
    # Field-level validators
    # ------------------------------------------------------------------

    @field_validator("dataset_name")
    @classmethod
    def _dataset_name_non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("'dataset_name' must be a non-empty string")
        return v

    @field_validator("n_structures")
    @classmethod
    def _n_structures_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(
                f"'n_structures' must be >= 0, got {v!r}"
            )
        return v

    @field_validator("elements_present")
    @classmethod
    def _elements_present_sorted_unique(cls, v: list[str]) -> list[str]:
        if v != sorted(set(v)):
            raise ValueError(
                f"'elements_present' must be sorted and contain no "
                f"duplicates; got {v!r}"
            )
        return v

    @field_validator("property_keys")
    @classmethod
    def _property_keys_sorted_unique(cls, v: list[str]) -> list[str]:
        if v != sorted(set(v)):
            raise ValueError(
                f"'property_keys' must be sorted and contain no "
                f"duplicates; got {v!r}"
            )
        return v
