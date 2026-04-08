"""
nominal_drift.datasets.dft_bridge
=================================

Prepare CrystalRecord → DFT-ready export formats.

Forward-looking bridge to produce compatible dicts/strings for pymatgen
and other DFT tools without requiring them as dependencies yet.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from nominal_drift.datasets.schema import CrystalRecord

ExportFormat = Literal["cif_stub", "poscar_stub", "json_dict", "summary_text"]


# ---------------------------------------------------------------------------
# DFTExportResult
# ---------------------------------------------------------------------------

class DFTExportResult(BaseModel, frozen=True):
    """Result of exporting a CrystalRecord to DFT format.

    Attributes
    ----------
    record_id : str
        Record identifier.
    source_dataset : str
        Source dataset name.
    export_format : str
        Format used: "cif_stub", "poscar_stub", "json_dict", "summary_text".
    content : str
        String representation of the structure.
    pymatgen_compatible : bool
        True if content is compatible with pymatgen.Structure.from_dict().
    notes : list[str]
        Notes about the export.
    warnings : list[str]
        Warnings (e.g., "fractional coords outside [0,1]").
    """

    record_id: str
    source_dataset: str
    export_format: str
    content: str
    pymatgen_compatible: bool
    notes: list[str] = []
    warnings: list[str] = []


# ---------------------------------------------------------------------------
# Lattice Vector Computation
# ---------------------------------------------------------------------------

def _compute_lattice_vectors(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> list[list[float]]:
    """Compute Cartesian lattice vectors from cell parameters.

    Parameters
    ----------
    a, b, c : float
        Lattice lengths in Ångströms.
    alpha, beta, gamma : float
        Inter-axial angles in degrees.

    Returns
    -------
    list[list[float]]
        3x3 matrix of lattice vectors as rows.
    """
    # Convert to radians
    alpha_r = math.radians(alpha)
    beta_r = math.radians(beta)
    gamma_r = math.radians(gamma)

    # First lattice vector
    a_vec = [a, 0.0, 0.0]

    # Second lattice vector
    b_x = b * math.cos(gamma_r)
    b_y = b * math.sin(gamma_r)
    b_vec = [b_x, b_y, 0.0]

    # Third lattice vector
    c_x = c * math.cos(beta_r)
    c_y = c * (math.cos(alpha_r) - math.cos(beta_r) * math.cos(gamma_r)) / math.sin(gamma_r)
    c_z_sq = c**2 - c_x**2 - c_y**2
    c_z = math.sqrt(max(0.0, c_z_sq))  # Guard against numerical errors
    c_vec = [c_x, c_y, c_z]

    return [a_vec, b_vec, c_vec]


# ---------------------------------------------------------------------------
# Export Functions
# ---------------------------------------------------------------------------

def pymatgen_dict(record: CrystalRecord) -> dict:
    """Return a pymatgen-compatible Structure dict.

    Parameters
    ----------
    record : CrystalRecord
        Crystal record to export.

    Returns
    -------
    dict
        Dictionary compatible with pymatgen.Structure.from_dict().
    """
    lattice_matrix = _compute_lattice_vectors(
        record.lattice.a,
        record.lattice.b,
        record.lattice.c,
        record.lattice.alpha,
        record.lattice.beta,
        record.lattice.gamma,
    )

    sites = []
    for site in record.sites:
        sites.append({
            "species": [{"element": site.species, "occu": 1.0}],
            "abc": list(site.frac_coords),
            "xyz": [0.0, 0.0, 0.0],  # Will be computed by pymatgen
            "label": site.species,
            "properties": {},
        })

    return {
        "@module": "pymatgen.core.structure",
        "@class": "Structure",
        "lattice": {
            "matrix": lattice_matrix,
            "a": record.lattice.a,
            "b": record.lattice.b,
            "c": record.lattice.c,
            "alpha": record.lattice.alpha,
            "beta": record.lattice.beta,
            "gamma": record.lattice.gamma,
            "volume": 0.0,  # pymatgen will compute
        },
        "sites": sites,
        "charge": 0.0,
    }


def _export_json_dict(record: CrystalRecord) -> str:
    """Export as pymatgen-compatible JSON dict string."""
    import json
    data = pymatgen_dict(record)
    return json.dumps(data, indent=2)


def _export_cif_stub(record: CrystalRecord) -> str:
    """Export as minimal CIF text."""
    lines = [
        "data_structure",
        f"_cell_length_a    {record.lattice.a:.6f}",
        f"_cell_length_b    {record.lattice.b:.6f}",
        f"_cell_length_c    {record.lattice.c:.6f}",
        f"_cell_angle_alpha {record.lattice.alpha:.6f}",
        f"_cell_angle_beta  {record.lattice.beta:.6f}",
        f"_cell_angle_gamma {record.lattice.gamma:.6f}",
        "",
        "loop_",
        "  _atom_site_label",
        "  _atom_site_type_symbol",
        "  _atom_site_fract_x",
        "  _atom_site_fract_y",
        "  _atom_site_fract_z",
    ]

    for i, site in enumerate(record.sites):
        x, y, z = site.frac_coords
        lines.append(f"  {site.species}{i+1}  {site.species}  {x:.6f}  {y:.6f}  {z:.6f}")

    return "\n".join(lines)


def _export_poscar_stub(record: CrystalRecord) -> str:
    """Export as VASP POSCAR format."""
    lines = [record.record_id, "1.0"]

    # Lattice vectors
    lattice_matrix = _compute_lattice_vectors(
        record.lattice.a,
        record.lattice.b,
        record.lattice.c,
        record.lattice.alpha,
        record.lattice.beta,
        record.lattice.gamma,
    )
    for vec in lattice_matrix:
        lines.append(f"  {vec[0]:.10f}  {vec[1]:.10f}  {vec[2]:.10f}")

    # Element symbols (unique, sorted)
    elements = sorted(set(s.species for s in record.sites))
    lines.append("  " + "  ".join(elements))

    # Element counts
    counts = {elem: sum(1 for s in record.sites if s.species == elem) for elem in elements}
    lines.append("  " + "  ".join(str(counts[elem]) for elem in elements))

    # Direct (fractional) coordinates
    lines.append("Direct")

    for site in record.sites:
        x, y, z = site.frac_coords
        lines.append(f"  {x:.10f}  {y:.10f}  {z:.10f}")

    return "\n".join(lines)


def _export_summary_text(record: CrystalRecord) -> str:
    """Export as human-readable text summary."""
    lines = [
        f"Record ID: {record.record_id}",
        f"Source: {record.source_dataset} (index {record.source_index})",
        f"Split: {record.split or 'None'}",
        "",
        "Unit Cell:",
        f"  a={record.lattice.a:.6f} Å,  b={record.lattice.b:.6f} Å,  c={record.lattice.c:.6f} Å",
        f"  α={record.lattice.alpha:.2f}°,  β={record.lattice.beta:.2f}°,  γ={record.lattice.gamma:.2f}°",
        "",
        f"Composition: {record.n_atoms} atoms of {', '.join(record.elements)}",
        "",
        "Atomic Sites:",
    ]

    for i, site in enumerate(record.sites):
        x, y, z = site.frac_coords
        lines.append(f"  {i+1}. {site.species:2s} @ ({x:.6f}, {y:.6f}, {z:.6f})")

    if record.properties:
        lines.append("")
        lines.append("Properties:")
        for key, val in sorted(record.properties.items()):
            lines.append(f"  {key}: {val}")

    return "\n".join(lines)


def export_structure(
    record: CrystalRecord,
    fmt: str = "json_dict",
) -> DFTExportResult:
    """Convert CrystalRecord to requested DFT export format.

    Parameters
    ----------
    record : CrystalRecord
        Crystal record to export.
    fmt : str, optional
        Format: "cif_stub", "poscar_stub", "json_dict", "summary_text"
        (default: "json_dict").

    Returns
    -------
    DFTExportResult
        Frozen export result.

    Raises
    ------
    ValueError
        If fmt is not recognized.
    """
    warnings: list[str] = []

    # Check for fractional coords outside [0, 1]
    for site in record.sites:
        for coord in site.frac_coords:
            if not (0.0 <= coord <= 1.0):
                warnings.append(
                    f"Fractional coordinate {coord:.4f} is outside [0, 1]"
                )
                break

    if fmt == "json_dict":
        content = _export_json_dict(record)
        pymatgen_compat = True
        notes = ["JSON dict compatible with pymatgen.Structure.from_dict()"]
    elif fmt == "cif_stub":
        content = _export_cif_stub(record)
        pymatgen_compat = False
        notes = ["Minimal CIF format without symmetry information"]
    elif fmt == "poscar_stub":
        content = _export_poscar_stub(record)
        pymatgen_compat = False
        notes = ["VASP POSCAR format (direct fractional coordinates)"]
    elif fmt == "summary_text":
        content = _export_summary_text(record)
        pymatgen_compat = False
        notes = ["Human-readable text summary"]
    else:
        raise ValueError(f"Unknown format: {fmt}")

    return DFTExportResult(
        record_id=record.record_id,
        source_dataset=record.source_dataset,
        export_format=fmt,
        content=content,
        pymatgen_compatible=pymatgen_compat,
        notes=notes,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Batch Export
# ---------------------------------------------------------------------------

def batch_export(
    records: list[CrystalRecord],
    fmt: str,
    output_dir: str,
) -> list[str]:
    """Export all records to output_dir/{record_id}.{ext}.

    Parameters
    ----------
    records : list[CrystalRecord]
        Crystal records to export.
    fmt : str
        Format: "json_dict", "cif_stub", "poscar_stub", "summary_text".
    output_dir : str
        Output directory path.

    Returns
    -------
    list[str]
        List of written file paths (absolute).

    Raises
    ------
    ValueError
        If fmt is not recognized.
    """
    # Map format to file extension
    ext_map = {
        "json_dict": "json",
        "cif_stub": "cif",
        "poscar_stub": "poscar",
        "summary_text": "txt",
    }
    if fmt not in ext_map:
        raise ValueError(f"Unknown format: {fmt}")

    ext = ext_map[fmt]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_paths = []
    for rec in records:
        result = export_structure(rec, fmt=fmt)
        out_file = output_path / f"{rec.record_id}.{ext}"
        out_file.write_text(result.content, encoding="utf-8")
        written_paths.append(str(out_file.absolute()))

    return written_paths
