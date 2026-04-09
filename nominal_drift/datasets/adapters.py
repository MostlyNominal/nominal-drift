"""
nominal_drift.datasets.adapters
================================
Adapters that convert raw dataset records into the canonical CrystalRecord
schema.

All CDVAE / DiffCSP datasets (perov-5, mp-20, carbon-24, mpts-52) share
the same on-disk CSV format:

    column 0     : row index (unnamed)
    material_id  : dataset-internal identifier
    cif          : full pymatgen-generated CIF string
    formula      : reduced chemical formula (e.g. "BaTiO3")
    heat_all     : formation enthalpy per atom [eV/atom] (perov-5)
    heat_ref     : reference-state-corrected formation energy (perov-5)
    dir_gap      : direct band gap [eV]
    ind_gap      : indirect band gap [eV]
    energy_per_atom : energy above convex hull [eV/atom] (carbon-24)

Each adapter inherits from ``CifBaseAdapter`` and overrides only the
``_extract_properties`` method to capture dataset-specific columns.

The heavy lifting — CIF parsing, pymatgen Structure extraction, and
conversion to CrystalRecord — is delegated to
``nominal_drift.datasets.pymatgen_bridge``.

Public API
----------
``BaseAdapter``
    Abstract base (maintained for backward compatibility).

``CifBaseAdapter``
    CIF-aware adapter for CDVAE/DiffCSP CSV rows.

``Perov5Adapter``, ``MP20Adapter``, ``Carbon24Adapter``, ``MPTS52Adapter``
    Concrete adapters for the four registered datasets.

``ADAPTER_REGISTRY``
    Dict mapping dataset name → adapter class.

``get_adapter(name)``
    Return an instantiated adapter.

``normalise_records(records, output_dir, dataset_name)``
    Stream records to ``structures.jsonl`` + ``manifest.json``.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from nominal_drift.datasets.schema import (
    AtomicSite,
    CrystalRecord,
    DatasetManifest,
    LatticeParams,
)

ADAPTER_VERSION = "0.2.0"


# ---------------------------------------------------------------------------
# Base Adapter (abstract — preserved for backward compatibility)
# ---------------------------------------------------------------------------

class BaseAdapter(ABC):
    """Abstract base adapter for all dataset formats.

    Subclasses must define:
        - dataset_name: str (class attribute)
        - source_url: str (class attribute)
        - convert(raw, source_index): CrystalRecord (instance method)
    """

    dataset_name: str = ""
    source_url: str = ""

    @abstractmethod
    def convert(self, raw: dict, source_index: int) -> CrystalRecord:
        """Convert raw dict to CrystalRecord.

        Parameters
        ----------
        raw : dict
            Raw record from the source dataset (CSV row as DictReader gives it).
        source_index : int
            Zero-based row index in the source file.

        Returns
        -------
        CrystalRecord

        Raises
        ------
        ValueError
            If the CIF cannot be parsed or the record is otherwise invalid.
        """
        raise NotImplementedError

    def make_manifest(
        self,
        records: list[CrystalRecord],
        raw_path: str | None = None,
    ) -> DatasetManifest:
        """Create a DatasetManifest from a list of CrystalRecords."""
        splits: dict[str, int] = {}
        for rec in records:
            if rec.split:
                splits[rec.split] = splits.get(rec.split, 0) + 1

        all_elements: set[str] = set()
        for rec in records:
            all_elements.update(rec.elements)
        elements_present = sorted(all_elements)

        all_property_keys: set[str] = set()
        for rec in records:
            all_property_keys.update(rec.properties.keys())
        property_keys = sorted(all_property_keys)

        imported_at = datetime.utcnow().isoformat() + "Z"

        return DatasetManifest(
            dataset_name=self.dataset_name,
            nominal_drift_schema_version="1.0",
            source_url=self.source_url,
            n_structures=len(records),
            splits=splits,
            elements_present=elements_present,
            property_keys=property_keys,
            imported_at=imported_at,
            importer_version=ADAPTER_VERSION,
            raw_path=raw_path,
            checksum_raw=None,
        )


# ---------------------------------------------------------------------------
# CIF-aware base adapter (CDVAE / DiffCSP CSV format)
# ---------------------------------------------------------------------------

class CifBaseAdapter(BaseAdapter):
    """Base adapter for CDVAE/DiffCSP datasets whose CSV rows contain a
    full CIF string in the ``cif`` column.

    Concrete subclasses only need to override ``_extract_properties`` to
    pull dataset-specific numeric columns (energy, band gap, etc.).
    """

    def convert(self, raw: dict, source_index: int) -> CrystalRecord:
        """Parse the ``cif`` column and return a CrystalRecord.

        Parameters
        ----------
        raw : dict
            CSV row (DictReader dict).
        source_index : int
            Zero-based row index.

        Returns
        -------
        CrystalRecord

        Raises
        ------
        ValueError
            If the CIF string is missing, empty, or cannot be parsed.
        """
        from nominal_drift.datasets.pymatgen_bridge import cif_string_to_crystal_record

        cif_str = (raw.get("cif") or "").strip()
        if not cif_str:
            raise ValueError(
                f"Row {source_index}: 'cif' column is empty or missing"
            )

        material_id = (
            raw.get("material_id") or raw.get("") or f"{self.dataset_name}_{source_index}"
        )
        formula = raw.get("formula", "")
        split = raw.get("split") or None

        properties: dict[str, float | str | None] = {
            "material_id": str(material_id),
            "formula": formula,
        }
        properties.update(self._extract_properties(raw))

        record = cif_string_to_crystal_record(
            cif_str=cif_str,
            source_dataset=self.dataset_name,
            source_index=source_index,
            split=split,
            properties=properties,
        )
        if record is None:
            raise ValueError(
                f"Row {source_index} (material_id={material_id}): "
                f"CIF parsing failed"
            )
        return record

    def _extract_properties(self, raw: dict) -> dict[str, float | str | None]:
        """Return dataset-specific properties extracted from the CSV row.

        Subclasses override this.  The default implementation returns an
        empty dict.
        """
        return {}


# ---------------------------------------------------------------------------
# Perov-5 Adapter
# ---------------------------------------------------------------------------

class Perov5Adapter(CifBaseAdapter):
    """Adapter for Perov-5 (18 928 perovskite ABX₃ structures).

    CSV columns used beyond ``cif`` / ``formula`` / ``material_id``:
        heat_all   — formation enthalpy [eV/atom]
        heat_ref   — reference-corrected formation energy [eV/atom]
        dir_gap    — direct band gap [eV]
        ind_gap    — indirect band gap [eV]
    """

    dataset_name = "perov-5"
    source_url = "https://github.com/txie-93/cdvae"

    def _extract_properties(self, raw: dict) -> dict[str, float | str | None]:
        props: dict[str, float | str | None] = {}
        for key in ("heat_all", "heat_ref", "dir_gap", "ind_gap"):
            val = raw.get(key)
            if val not in (None, "", "nan", "NaN"):
                try:
                    props[key] = float(val)
                except (TypeError, ValueError):
                    pass
        # Expose formation_energy_per_atom as the canonical key
        if "heat_all" in props:
            props["formation_energy_per_atom"] = props["heat_all"]
        if "dir_gap" in props or "ind_gap" in props:
            # Use the larger of the two as "band_gap" (conservative)
            gaps = [v for v in (props.get("dir_gap"), props.get("ind_gap")) if v is not None]
            if gaps:
                props["band_gap"] = max(gaps)
        return props


# ---------------------------------------------------------------------------
# MP-20 Adapter
# ---------------------------------------------------------------------------

class MP20Adapter(CifBaseAdapter):
    """Adapter for MP-20 (~45 000 Materials Project structures).

    CSV columns used beyond ``cif`` / ``formula`` / ``material_id``:
        heat_all              — formation enthalpy [eV/atom]
        heat_ref              — reference-corrected formation energy [eV/atom]
        dir_gap / ind_gap     — band gaps [eV]
        e_above_hull          — distance to convex hull [eV/atom] (if present)
        spacegroup            — international space-group symbol (if present)
        volume                — unit-cell volume [Å³] (if present)
    """

    dataset_name = "mp-20"
    source_url = "https://github.com/txie-93/cdvae"

    def _extract_properties(self, raw: dict) -> dict[str, float | str | None]:
        props: dict[str, float | str | None] = {}
        for float_key in (
            "heat_all", "heat_ref", "dir_gap", "ind_gap",
            "e_above_hull", "volume", "formation_energy_per_atom",
            "band_gap",
        ):
            val = raw.get(float_key)
            if val not in (None, "", "nan", "NaN"):
                try:
                    props[float_key] = float(val)
                except (TypeError, ValueError):
                    pass
        for str_key in ("spacegroup",):
            val = raw.get(str_key)
            if val not in (None, ""):
                props[str_key] = str(val)
        # Canonical aliases
        if "heat_all" in props and "formation_energy_per_atom" not in props:
            props["formation_energy_per_atom"] = props["heat_all"]
        if "dir_gap" in props or "ind_gap" in props:
            gaps = [v for v in (props.get("dir_gap"), props.get("ind_gap")) if v is not None]
            if gaps and "band_gap" not in props:
                props["band_gap"] = max(gaps)
        return props


# ---------------------------------------------------------------------------
# Carbon-24 Adapter
# ---------------------------------------------------------------------------

class Carbon24Adapter(CifBaseAdapter):
    """Adapter for Carbon-24 (10 153 carbon allotropes).

    CSV columns used beyond ``cif`` / ``formula`` / ``material_id``:
        heat_all          — formation/cohesive energy [eV/atom]
        energy_per_atom   — energy above hull [eV/atom] (alias)
    """

    dataset_name = "carbon-24"
    source_url = "https://github.com/txie-93/cdvae"

    def _extract_properties(self, raw: dict) -> dict[str, float | str | None]:
        props: dict[str, float | str | None] = {}
        for key in ("heat_all", "heat_ref", "energy_per_atom", "dir_gap", "ind_gap"):
            val = raw.get(key)
            if val not in (None, "", "nan", "NaN"):
                try:
                    props[key] = float(val)
                except (TypeError, ValueError):
                    pass
        if "heat_all" in props and "energy_per_atom" not in props:
            props["energy_per_atom"] = props["heat_all"]
        return props


# ---------------------------------------------------------------------------
# MPTS-52 Adapter
# ---------------------------------------------------------------------------

class MPTS52Adapter(CifBaseAdapter):
    """Adapter for MPTS-52 (~40 000 Materials Project structures, larger set).

    Same CSV format as MP-20 but sourced from the ml-evs/mpts repository.
    Extra columns if present: ``spacegroup``, ``volume``.
    """

    dataset_name = "mpts-52"
    source_url = "https://github.com/ml-evs/mpts"

    def _extract_properties(self, raw: dict) -> dict[str, float | str | None]:
        props: dict[str, float | str | None] = {}
        for float_key in (
            "heat_all", "heat_ref", "dir_gap", "ind_gap",
            "e_above_hull", "volume", "formation_energy_per_atom",
            "band_gap",
        ):
            val = raw.get(float_key)
            if val not in (None, "", "nan", "NaN"):
                try:
                    props[float_key] = float(val)
                except (TypeError, ValueError):
                    pass
        for str_key in ("spacegroup",):
            val = raw.get(str_key)
            if val not in (None, ""):
                props[str_key] = str(val)
        if "heat_all" in props and "formation_energy_per_atom" not in props:
            props["formation_energy_per_atom"] = props["heat_all"]
        if "dir_gap" in props or "ind_gap" in props:
            gaps = [v for v in (props.get("dir_gap"), props.get("ind_gap")) if v is not None]
            if gaps and "band_gap" not in props:
                props["band_gap"] = max(gaps)
        return props


# ---------------------------------------------------------------------------
# Adapter Registry
# ---------------------------------------------------------------------------

ADAPTER_REGISTRY: dict[str, type[BaseAdapter]] = {
    "perov-5": Perov5Adapter,
    "mp-20": MP20Adapter,
    "carbon-24": Carbon24Adapter,
    "mpts-52": MPTS52Adapter,
}


def get_adapter(dataset_name: str) -> BaseAdapter:
    """Return an instantiated adapter for *dataset_name*.

    Raises
    ------
    KeyError
        If dataset_name is not in the registry.
    """
    return ADAPTER_REGISTRY[dataset_name]()


# ---------------------------------------------------------------------------
# Batch Processing — stream records to JSONL + manifest
# ---------------------------------------------------------------------------

def normalise_records(
    records: list[CrystalRecord],
    output_dir: str,
    dataset_name: str,
) -> DatasetManifest:
    """Stream *records* to ``{output_dir}/structures.jsonl`` + ``manifest.json``.

    Creates *output_dir* if it does not exist.

    Parameters
    ----------
    records : list[CrystalRecord]
    output_dir : str
    dataset_name : str
        Used to look up the adapter for manifest creation.

    Returns
    -------
    DatasetManifest
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    structures_file = output_path / "structures.jsonl"
    with open(structures_file, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")

    adapter = get_adapter(dataset_name)
    manifest = adapter.make_manifest(records, raw_path=None)
    manifest_file = output_path / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))

    return manifest
