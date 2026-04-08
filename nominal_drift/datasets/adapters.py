"""
nominal_drift.datasets.adapters
================================

Adapters that convert raw dataset records into the canonical CrystalRecord schema.

Each adapter handles one dataset format. All adapters inherit from BaseAdapter
and implement the convert() method to transform raw dicts into CrystalRecord objects.
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

ADAPTER_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Base Adapter
# ---------------------------------------------------------------------------

class BaseAdapter(ABC):
    """Abstract base adapter for all dataset formats.

    Subclasses must define:
        - dataset_name: str (class attribute)
        - source_url: str (class attribute)
        - convert(): CrystalRecord (instance method)
    """

    dataset_name: str = ""
    source_url: str = ""

    @abstractmethod
    def convert(self, raw: dict, source_index: int) -> CrystalRecord:
        """Convert raw dict to CrystalRecord.

        Parameters
        ----------
        raw : dict
            Raw record from the source dataset.
        source_index : int
            Zero-based row index in the source dataset.

        Returns
        -------
        CrystalRecord
            Canonical crystal record.

        Raises
        ------
        KeyError
            If required keys are missing from raw dict.
        ValueError
            If validation fails.
        """
        raise NotImplementedError

    def make_manifest(
        self,
        records: list[CrystalRecord],
        raw_path: str | None = None,
    ) -> DatasetManifest:
        """Create a DatasetManifest from a list of CrystalRecords.

        Computes: n_structures, splits, elements_present, property_keys.

        Parameters
        ----------
        records : list[CrystalRecord]
            List of crystal records.
        raw_path : str | None, optional
            Relative path to raw data source (default: None).

        Returns
        -------
        DatasetManifest
            Frozen manifest.
        """
        # Count splits
        splits: dict[str, int] = {}
        for rec in records:
            if rec.split:
                splits[rec.split] = splits.get(rec.split, 0) + 1

        # Collect all unique elements
        all_elements = set()
        for rec in records:
            all_elements.update(rec.elements)
        elements_present = sorted(list(all_elements))

        # Collect all unique property keys
        all_property_keys = set()
        for rec in records:
            all_property_keys.update(rec.properties.keys())
        property_keys = sorted(list(all_property_keys))

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
# Perov-5 Adapter
# ---------------------------------------------------------------------------

class Perov5Adapter(BaseAdapter):
    """Adapter for Perov-5: 18928 perovskites ABX3.

    Expected raw dict keys:
        - material_id: str
        - formula: str
        - a, b, c, alpha, beta, gamma: float
        - sites: list[{"species": str, "xyz": [x, y, z]}]
        - formation_energy_per_atom: float | None
        - split: "train"|"val"|"test"|None
    """

    dataset_name = "perov-5"
    source_url = "https://github.com/txie-93/cdvae"

    def convert(self, raw: dict, source_index: int) -> CrystalRecord:
        """Convert perovskite record to CrystalRecord."""
        record_id = str(uuid4())
        material_id = raw.get("material_id", f"perov5_{source_index}")
        formula = raw.get("formula", "ABX3")
        a = float(raw["a"])
        b = float(raw["b"])
        c = float(raw["c"])
        alpha = float(raw["alpha"])
        beta = float(raw["beta"])
        gamma = float(raw["gamma"])

        # Build lattice
        lattice = LatticeParams(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)

        # Build sites
        sites_list = []
        for site_dict in raw.get("sites", []):
            species = site_dict["species"]
            xyz = site_dict["xyz"]
            site = AtomicSite(species=species, frac_coords=tuple(xyz))
            sites_list.append(site)

        sites = tuple(sites_list)
        n_atoms = len(sites)
        elements = tuple(sorted(set(s.species for s in sites)))

        # Build properties
        properties: dict[str, float | str | None] = {
            "formula": formula,
            "material_id": material_id,
        }
        if "formation_energy_per_atom" in raw and raw["formation_energy_per_atom"] is not None:
            properties["formation_energy_per_atom"] = float(raw["formation_energy_per_atom"])

        split = raw.get("split")

        return CrystalRecord(
            record_id=record_id,
            source_dataset=self.dataset_name,
            source_index=source_index,
            split=split,
            elements=elements,
            n_atoms=n_atoms,
            lattice=lattice,
            sites=sites,
            properties=properties,
            raw_path=None,
        )


# ---------------------------------------------------------------------------
# MP-20 Adapter
# ---------------------------------------------------------------------------

class MP20Adapter(BaseAdapter):
    """Adapter for MP-20: ~45k Materials Project structures.

    Expected raw dict keys:
        - material_id: str
        - formula: str
        - a, b, c, alpha, beta, gamma: float
        - sites: list[{"species": str, "xyz": [x, y, z]}]
        - formation_energy_per_atom: float | None
        - e_above_hull: float | None
        - band_gap: float | None
        - split: "train"|"val"|"test"|None
    """

    dataset_name = "mp-20"
    source_url = "https://github.com/txie-93/cdvae"

    def convert(self, raw: dict, source_index: int) -> CrystalRecord:
        """Convert Materials Project record to CrystalRecord."""
        record_id = str(uuid4())
        material_id = raw.get("material_id", f"mp20_{source_index}")
        formula = raw.get("formula", "")
        a = float(raw["a"])
        b = float(raw["b"])
        c = float(raw["c"])
        alpha = float(raw["alpha"])
        beta = float(raw["beta"])
        gamma = float(raw["gamma"])

        lattice = LatticeParams(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)

        sites_list = []
        for site_dict in raw.get("sites", []):
            species = site_dict["species"]
            xyz = site_dict["xyz"]
            site = AtomicSite(species=species, frac_coords=tuple(xyz))
            sites_list.append(site)

        sites = tuple(sites_list)
        n_atoms = len(sites)
        elements = tuple(sorted(set(s.species for s in sites)))

        properties: dict[str, float | str | None] = {
            "formula": formula,
            "material_id": material_id,
        }
        if "formation_energy_per_atom" in raw and raw["formation_energy_per_atom"] is not None:
            properties["formation_energy_per_atom"] = float(raw["formation_energy_per_atom"])
        if "e_above_hull" in raw and raw["e_above_hull"] is not None:
            properties["e_above_hull"] = float(raw["e_above_hull"])
        if "band_gap" in raw and raw["band_gap"] is not None:
            properties["band_gap"] = float(raw["band_gap"])

        split = raw.get("split")

        return CrystalRecord(
            record_id=record_id,
            source_dataset=self.dataset_name,
            source_index=source_index,
            split=split,
            elements=elements,
            n_atoms=n_atoms,
            lattice=lattice,
            sites=sites,
            properties=properties,
            raw_path=None,
        )


# ---------------------------------------------------------------------------
# Carbon-24 Adapter
# ---------------------------------------------------------------------------

class Carbon24Adapter(BaseAdapter):
    """Adapter for Carbon-24: allotropes of carbon.

    Expected raw dict keys:
        - structure_id: str
        - a, b, c, alpha, beta, gamma: float
        - sites: list[{"species": str, "xyz": [x, y, z]}]
        - energy_per_atom: float | None
        - split: "train"|"val"|"test"|None
    """

    dataset_name = "carbon-24"
    source_url = "https://github.com/txie-93/cdvae"

    def convert(self, raw: dict, source_index: int) -> CrystalRecord:
        """Convert carbon allotrope record to CrystalRecord."""
        record_id = str(uuid4())
        structure_id = raw.get("structure_id", f"carbon24_{source_index}")
        a = float(raw["a"])
        b = float(raw["b"])
        c = float(raw["c"])
        alpha = float(raw["alpha"])
        beta = float(raw["beta"])
        gamma = float(raw["gamma"])

        lattice = LatticeParams(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)

        sites_list = []
        for site_dict in raw.get("sites", []):
            species = site_dict["species"]
            xyz = site_dict["xyz"]
            site = AtomicSite(species=species, frac_coords=tuple(xyz))
            sites_list.append(site)

        sites = tuple(sites_list)
        n_atoms = len(sites)
        elements = tuple(sorted(set(s.species for s in sites)))

        properties: dict[str, float | str | None] = {"structure_id": structure_id}
        if "energy_per_atom" in raw and raw["energy_per_atom"] is not None:
            properties["energy_per_atom"] = float(raw["energy_per_atom"])

        split = raw.get("split")

        return CrystalRecord(
            record_id=record_id,
            source_dataset=self.dataset_name,
            source_index=source_index,
            split=split,
            elements=elements,
            n_atoms=n_atoms,
            lattice=lattice,
            sites=sites,
            properties=properties,
            raw_path=None,
        )


# ---------------------------------------------------------------------------
# MPTS-52 Adapter
# ---------------------------------------------------------------------------

class MPTS52Adapter(BaseAdapter):
    """Adapter for MPTS-52: 40k Materials Project structures (larger set).

    Expected raw dict keys similar to MP20 plus:
        - spacegroup: str | None
        - volume: float | None
    """

    dataset_name = "mpts-52"
    source_url = "https://github.com/ml-evs/mpts"

    def convert(self, raw: dict, source_index: int) -> CrystalRecord:
        """Convert MPTS record to CrystalRecord."""
        record_id = str(uuid4())
        material_id = raw.get("material_id", f"mpts52_{source_index}")
        formula = raw.get("formula", "")
        a = float(raw["a"])
        b = float(raw["b"])
        c = float(raw["c"])
        alpha = float(raw["alpha"])
        beta = float(raw["beta"])
        gamma = float(raw["gamma"])

        lattice = LatticeParams(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)

        sites_list = []
        for site_dict in raw.get("sites", []):
            species = site_dict["species"]
            xyz = site_dict["xyz"]
            site = AtomicSite(species=species, frac_coords=tuple(xyz))
            sites_list.append(site)

        sites = tuple(sites_list)
        n_atoms = len(sites)
        elements = tuple(sorted(set(s.species for s in sites)))

        properties: dict[str, float | str | None] = {
            "formula": formula,
            "material_id": material_id,
        }
        if "formation_energy_per_atom" in raw and raw["formation_energy_per_atom"] is not None:
            properties["formation_energy_per_atom"] = float(raw["formation_energy_per_atom"])
        if "e_above_hull" in raw and raw["e_above_hull"] is not None:
            properties["e_above_hull"] = float(raw["e_above_hull"])
        if "band_gap" in raw and raw["band_gap"] is not None:
            properties["band_gap"] = float(raw["band_gap"])
        if "spacegroup" in raw and raw["spacegroup"] is not None:
            properties["spacegroup"] = str(raw["spacegroup"])
        if "volume" in raw and raw["volume"] is not None:
            properties["volume"] = float(raw["volume"])

        split = raw.get("split")

        return CrystalRecord(
            record_id=record_id,
            source_dataset=self.dataset_name,
            source_index=source_index,
            split=split,
            elements=elements,
            n_atoms=n_atoms,
            lattice=lattice,
            sites=sites,
            properties=properties,
            raw_path=None,
        )


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
    """Get an instantiated adapter for the given dataset name.

    Parameters
    ----------
    dataset_name : str
        Dataset name (e.g., "perov-5", "mp-20").

    Returns
    -------
    BaseAdapter
        Instantiated adapter.

    Raises
    ------
    KeyError
        If dataset_name is not in the registry.
    """
    adapter_class = ADAPTER_REGISTRY[dataset_name]
    return adapter_class()


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------

def normalise_records(
    records: list[CrystalRecord],
    output_dir: str,
    dataset_name: str,
) -> DatasetManifest:
    """Stream records to {output_dir}/structures.jsonl and write manifest.

    Creates output_dir if it doesn't exist.

    Parameters
    ----------
    records : list[CrystalRecord]
        List of crystal records to normalize.
    output_dir : str
        Output directory path.
    dataset_name : str
        Dataset name (used for adapter).

    Returns
    -------
    DatasetManifest
        The written manifest.

    Raises
    ------
    KeyError
        If dataset_name is not in the registry.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Write JSONL
    structures_file = output_path / "structures.jsonl"
    with open(structures_file, "w", encoding="utf-8") as f:
        for rec in records:
            json_line = rec.model_dump_json()
            f.write(json_line + "\n")

    # Create and write manifest
    adapter = get_adapter(dataset_name)
    manifest = adapter.make_manifest(records, raw_path=None)
    manifest_file = output_path / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))

    return manifest
