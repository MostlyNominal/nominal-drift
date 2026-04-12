"""
nominal_drift.datasets.pymatgen_bridge
=======================================
Bidirectional bridge between pymatgen Structure objects and the internal
CrystalRecord schema.

This module is the single place that touches pymatgen.  All other Lane B
code works with CrystalRecord and imports only from this module when
pymatgen structures are needed.

Public API
----------
``cif_string_to_structure(cif_str)``
    Parse a CIF string → pymatgen Structure (or None on failure).

``structure_to_crystal_record(structure, source_dataset, source_index, split, properties)``
    Convert a pymatgen Structure → CrystalRecord.

``crystal_record_to_structure(record)``
    Convert a CrystalRecord → pymatgen Structure.
"""
from __future__ import annotations

import io
import warnings
from typing import Any

from nominal_drift.datasets.schema import AtomicSite, CrystalRecord, LatticeParams


# ---------------------------------------------------------------------------
# Lazy imports — pymatgen is optional; callers get a clear ImportError if absent
# ---------------------------------------------------------------------------

def _pmg():
    """Return pymatgen.core module, raising a helpful error if not installed."""
    try:
        import pymatgen.core as _core
        return _core
    except ImportError as exc:
        raise ImportError(
            "pymatgen is required for CIF parsing.  "
            "Install it with: pip install pymatgen"
        ) from exc


def _cif_parser():
    try:
        from pymatgen.io.cif import CifParser
        return CifParser
    except ImportError as exc:
        raise ImportError("pymatgen is required for CIF parsing.") from exc


# ---------------------------------------------------------------------------
# CIF string → pymatgen Structure
# ---------------------------------------------------------------------------

def cif_string_to_structure(cif_str: str, primitive: bool = False):
    """Parse a CIF string and return the first pymatgen Structure found.

    Parameters
    ----------
    cif_str : str
        Raw CIF text (as stored in CDVAE / DiffCSP CSV files).
    primitive : bool
        Whether to return the primitive cell (default False — keep
        the conventional cell so fractional coordinates are as stored).

    Returns
    -------
    pymatgen.core.Structure | None
        Parsed structure, or ``None`` if parsing fails for any reason.
    """
    CifParser = _cif_parser()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser = CifParser.from_str(cif_str)
            structures = parser.parse_structures(primitive=primitive)
        if not structures:
            return None
        return structures[0]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# pymatgen Structure → CrystalRecord
# ---------------------------------------------------------------------------

def structure_to_crystal_record(
    structure,
    source_dataset: str,
    source_index: int,
    split: str | None = None,
    properties: dict[str, Any] | None = None,
    record_id: str | None = None,
    raw_path: str | None = None,
) -> CrystalRecord:
    """Convert a pymatgen Structure to a CrystalRecord.

    Parameters
    ----------
    structure : pymatgen.core.Structure
        The input structure.
    source_dataset : str
        Dataset identifier, e.g. ``"perov-5"``.
    source_index : int
        Zero-based row / entry index in the source file.
    split : str | None
        ``"train"`` / ``"val"`` / ``"test"`` or ``None``.
    properties : dict | None
        Optional dataset-specific properties (formation energy, band gap …).
    record_id : str | None
        UUID to use.  A fresh UUID4 is generated if not supplied.
    raw_path : str | None
        Relative path to the raw source file, if tracked.

    Returns
    -------
    CrystalRecord
        The canonical record.
    """
    from uuid import uuid4

    lattice = structure.lattice
    lp = LatticeParams(
        a=float(lattice.a),
        b=float(lattice.b),
        c=float(lattice.c),
        alpha=float(lattice.alpha),
        beta=float(lattice.beta),
        gamma=float(lattice.gamma),
    )

    sites = tuple(
        AtomicSite(
            species=site.species_string,
            frac_coords=(
                float(site.frac_coords[0]),
                float(site.frac_coords[1]),
                float(site.frac_coords[2]),
            ),
        )
        for site in structure.sites
    )

    elements = tuple(sorted({s.species for s in sites}))
    n_atoms = len(sites)

    return CrystalRecord(
        record_id=record_id or str(uuid4()),
        source_dataset=source_dataset,
        source_index=source_index,
        split=split,
        elements=elements,
        n_atoms=n_atoms,
        lattice=lp,
        sites=sites,
        properties=dict(properties or {}),
        raw_path=raw_path,
    )


# ---------------------------------------------------------------------------
# CrystalRecord → pymatgen Structure
# ---------------------------------------------------------------------------

def crystal_record_to_structure(record: CrystalRecord):
    """Reconstruct a pymatgen Structure from a CrystalRecord.

    Parameters
    ----------
    record : CrystalRecord
        The canonical record.

    Returns
    -------
    pymatgen.core.Structure
        Reconstructed structure.  Fractional coordinates are used as-is
        (no clamping to [0, 1]).
    """
    core = _pmg()
    Structure = core.Structure
    Lattice = core.Lattice

    lattice = Lattice.from_parameters(
        a=record.lattice.a,
        b=record.lattice.b,
        c=record.lattice.c,
        alpha=record.lattice.alpha,
        beta=record.lattice.beta,
        gamma=record.lattice.gamma,
    )

    species = [site.species for site in record.sites]
    frac_coords = [list(site.frac_coords) for site in record.sites]

    return Structure(
        lattice=lattice,
        species=species,
        coords=frac_coords,
        coords_are_cartesian=False,
    )


# ---------------------------------------------------------------------------
# Convenience: parse CIF string → CrystalRecord in one call
# ---------------------------------------------------------------------------

def cif_string_to_crystal_record(
    cif_str: str,
    source_dataset: str,
    source_index: int,
    split: str | None = None,
    properties: dict[str, Any] | None = None,
    record_id: str | None = None,
    raw_path: str | None = None,
) -> CrystalRecord | None:
    """Parse a CIF string and convert directly to a CrystalRecord.

    Returns ``None`` if the CIF cannot be parsed (the caller should
    count and log these as errors rather than raising an exception, to
    keep batch ingestion resilient).

    Parameters
    ----------
    cif_str : str
        Raw CIF text.
    source_dataset : str
        Dataset identifier.
    source_index : int
        Zero-based row index.
    split : str | None
        Dataset split.
    properties : dict | None
        Dataset-specific properties.
    record_id : str | None
        UUID override.
    raw_path : str | None
        Relative raw file path.

    Returns
    -------
    CrystalRecord | None
    """
    structure = cif_string_to_structure(cif_str)
    if structure is None:
        return None
    try:
        return structure_to_crystal_record(
            structure=structure,
            source_dataset=source_dataset,
            source_index=source_index,
            split=split,
            properties=properties,
            record_id=record_id,
            raw_path=raw_path,
        )
    except Exception:
        return None
