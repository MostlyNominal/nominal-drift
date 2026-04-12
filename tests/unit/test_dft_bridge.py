"""
tests/unit/test_dft_bridge.py
============================

Unit tests for nominal_drift.datasets.dft_bridge.

Tests cover:
  - pymatgen_dict output format
  - DFTExportResult model
  - export_structure in all formats
  - batch_export functionality
  - Lattice vector computation

Run with:
    pytest tests/unit/test_dft_bridge.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from nominal_drift.datasets.dft_bridge import (
    DFTExportResult,
    batch_export,
    export_structure,
    pymatgen_dict,
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
def simple_cubic_record() -> CrystalRecord:
    """Simple cubic structure for testing."""
    lattice = LatticeParams(a=5.0, b=5.0, c=5.0, alpha=90.0, beta=90.0, gamma=90.0)
    sites = (
        AtomicSite(species="Ba", frac_coords=(0.5, 0.5, 0.5)),
        AtomicSite(species="Ti", frac_coords=(0.0, 0.0, 0.0)),
        AtomicSite(species="O", frac_coords=(0.5, 0.0, 0.0)),
        AtomicSite(species="O", frac_coords=(0.0, 0.5, 0.0)),
        AtomicSite(species="O", frac_coords=(0.0, 0.0, 0.5)),
    )
    return CrystalRecord(
        record_id="perov_001",
        source_dataset="test",
        source_index=0,
        split="train",
        elements=("Ba", "O", "Ti"),
        n_atoms=5,
        lattice=lattice,
        sites=sites,
        properties={"formula": "BaTiO3"},
        raw_path=None,
    )


@pytest.fixture()
def hexagonal_record() -> CrystalRecord:
    """Hexagonal structure (non-orthogonal)."""
    lattice = LatticeParams(a=2.5, b=2.5, c=4.0, alpha=90.0, beta=90.0, gamma=120.0)
    sites = (
        AtomicSite(species="C", frac_coords=(0.0, 0.0, 0.0)),
        AtomicSite(species="C", frac_coords=(0.5, 0.5, 0.0)),
        AtomicSite(species="C", frac_coords=(0.333, 0.667, 0.5)),
    )
    return CrystalRecord(
        record_id="carbon_001",
        source_dataset="test",
        source_index=0,
        split=None,
        elements=("C",),
        n_atoms=3,
        lattice=lattice,
        sites=sites,
        properties={"energy": -10.0},
        raw_path=None,
    )


# =============================================================================
# DFTExportResult Tests
# =============================================================================

def test_export_result_creation() -> None:
    """Test creating an export result."""
    result = DFTExportResult(
        record_id="test_001",
        source_dataset="test",
        export_format="json_dict",
        content='{"test": "content"}',
        pymatgen_compatible=True,
    )
    assert result.record_id == "test_001"
    assert result.pymatgen_compatible is True


def test_export_result_with_notes_and_warnings() -> None:
    """Test export result with notes and warnings."""
    result = DFTExportResult(
        record_id="test",
        source_dataset="test",
        export_format="cif_stub",
        content="data test\n_cell_length_a 5.0",
        pymatgen_compatible=False,
        notes=["Note 1", "Note 2"],
        warnings=["Warning 1"],
    )
    assert len(result.notes) == 2
    assert len(result.warnings) == 1


def test_export_result_frozen() -> None:
    """Test that DFTExportResult is frozen."""
    result = DFTExportResult(
        record_id="test",
        source_dataset="test",
        export_format="json_dict",
        content="",
        pymatgen_compatible=False,
    )
    with pytest.raises(Exception):
        result.content = "new"  # type: ignore


# =============================================================================
# pymatgen_dict Tests
# =============================================================================

def test_pymatgen_dict_structure(simple_cubic_record: CrystalRecord) -> None:
    """Test pymatgen dict has required structure."""
    data = pymatgen_dict(simple_cubic_record)

    assert "@module" in data
    assert "@class" in data
    assert "lattice" in data
    assert "sites" in data
    assert data["@class"] == "Structure"
    assert data["@module"] == "pymatgen.core.structure"


def test_pymatgen_dict_lattice(simple_cubic_record: CrystalRecord) -> None:
    """Test lattice parameters in pymatgen dict."""
    data = pymatgen_dict(simple_cubic_record)

    lat = data["lattice"]
    assert lat["a"] == 5.0
    assert lat["b"] == 5.0
    assert lat["c"] == 5.0
    assert lat["alpha"] == 90.0
    assert lat["beta"] == 90.0
    assert lat["gamma"] == 90.0
    assert "matrix" in lat


def test_pymatgen_dict_lattice_matrix(simple_cubic_record: CrystalRecord) -> None:
    """Test lattice matrix is 3x3."""
    data = pymatgen_dict(simple_cubic_record)

    matrix = data["lattice"]["matrix"]
    assert len(matrix) == 3
    assert all(len(vec) == 3 for vec in matrix)


def test_pymatgen_dict_sites(simple_cubic_record: CrystalRecord) -> None:
    """Test sites in pymatgen dict."""
    data = pymatgen_dict(simple_cubic_record)

    sites = data["sites"]
    assert len(sites) == 5
    assert all("species" in site for site in sites)
    assert all("abc" in site for site in sites)


def test_pymatgen_dict_charge() -> None:
    """Test charge field in pymatgen dict."""
    lattice = LatticeParams(a=5.0, b=5.0, c=5.0, alpha=90.0, beta=90.0, gamma=90.0)
    rec = CrystalRecord(
        record_id="test",
        source_dataset="test",
        source_index=0,
        split=None,
        elements=("A",),
        n_atoms=1,
        lattice=lattice,
        sites=(AtomicSite(species="A", frac_coords=(0.0, 0.0, 0.0)),),
        properties={},
        raw_path=None,
    )
    data = pymatgen_dict(rec)

    assert "charge" in data
    assert data["charge"] == 0.0


# =============================================================================
# export_structure Tests
# =============================================================================

def test_export_json_dict(simple_cubic_record: CrystalRecord) -> None:
    """Test JSON dict export format."""
    result = export_structure(simple_cubic_record, fmt="json_dict")

    assert result.export_format == "json_dict"
    assert result.pymatgen_compatible is True
    assert "@class" in result.content
    data = json.loads(result.content)
    assert data["@class"] == "Structure"


def test_export_cif_stub(simple_cubic_record: CrystalRecord) -> None:
    """Test CIF stub export format."""
    result = export_structure(simple_cubic_record, fmt="cif_stub")

    assert result.export_format == "cif_stub"
    assert result.pymatgen_compatible is False
    assert "_cell_length_a" in result.content
    assert "_cell_length_b" in result.content
    assert "_cell_length_c" in result.content
    assert "_cell_angle_alpha" in result.content
    assert "_atom_site_label" in result.content


def test_export_cif_coordinates(simple_cubic_record: CrystalRecord) -> None:
    """Test CIF has fractional coordinates."""
    result = export_structure(simple_cubic_record, fmt="cif_stub")

    assert "_atom_site_fract_x" in result.content
    assert "_atom_site_fract_y" in result.content
    assert "_atom_site_fract_z" in result.content


def test_export_poscar_stub(simple_cubic_record: CrystalRecord) -> None:
    """Test POSCAR stub export format."""
    result = export_structure(simple_cubic_record, fmt="poscar_stub")

    assert result.export_format == "poscar_stub"
    assert result.pymatgen_compatible is False
    assert "perov_001" in result.content  # Record ID on line 1
    assert "1.0" in result.content  # Scale factor
    assert "Direct" in result.content


def test_export_poscar_structure(simple_cubic_record: CrystalRecord) -> None:
    """Test POSCAR has correct structure."""
    result = export_structure(simple_cubic_record, fmt="poscar_stub")
    lines = result.content.split("\n")

    # Line 1: ID
    assert "perov_001" in lines[0]
    # Line 2: Scale
    assert "1.0" in lines[1]
    # Lines 3-5: Lattice vectors
    # Line 6: Element symbols
    # Line 7: Element counts
    # Line 8: Direct
    assert "Direct" in result.content


def test_export_summary_text(simple_cubic_record: CrystalRecord) -> None:
    """Test summary text export."""
    result = export_structure(simple_cubic_record, fmt="summary_text")

    assert result.export_format == "summary_text"
    assert result.pymatgen_compatible is False
    assert "perov_001" in result.content
    assert "test" in result.content
    assert "Unit Cell:" in result.content
    assert "Composition:" in result.content


def test_export_summary_has_properties(simple_cubic_record: CrystalRecord) -> None:
    """Test summary text includes properties."""
    result = export_structure(simple_cubic_record, fmt="summary_text")

    assert "BaTiO3" in result.content or "formula" in result.content


def test_export_format_default(simple_cubic_record: CrystalRecord) -> None:
    """Test default export format is json_dict."""
    result = export_structure(simple_cubic_record)

    assert result.export_format == "json_dict"
    assert result.pymatgen_compatible is True


def test_export_invalid_format(simple_cubic_record: CrystalRecord) -> None:
    """Test invalid export format raises error."""
    with pytest.raises(ValueError):
        export_structure(simple_cubic_record, fmt="invalid_format")


def test_export_warnings_outside_unit_cell() -> None:
    """Test warnings for fractional coords outside [0,1]."""
    lattice = LatticeParams(a=5.0, b=5.0, c=5.0, alpha=90.0, beta=90.0, gamma=90.0)
    rec = CrystalRecord(
        record_id="test",
        source_dataset="test",
        source_index=0,
        split=None,
        elements=("A",),
        n_atoms=1,
        lattice=lattice,
        sites=(AtomicSite(species="A", frac_coords=(1.5, 0.0, 0.0)),),
        properties={},
        raw_path=None,
    )
    result = export_structure(rec)

    assert len(result.warnings) > 0
    assert "outside [0, 1]" in result.warnings[0]


def test_export_all_formats(simple_cubic_record: CrystalRecord) -> None:
    """Test all export formats work."""
    formats = ["json_dict", "cif_stub", "poscar_stub", "summary_text"]

    for fmt in formats:
        result = export_structure(simple_cubic_record, fmt=fmt)
        assert result.export_format == fmt
        assert result.content  # Content should not be empty


# =============================================================================
# batch_export Tests
# =============================================================================

def test_batch_export_json_dict(simple_cubic_record: CrystalRecord) -> None:
    """Test batch export in JSON format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        records = [simple_cubic_record]
        paths = batch_export(records, "json_dict", tmpdir)

        assert len(paths) == 1
        assert paths[0].endswith(".json")
        assert Path(paths[0]).exists()


def test_batch_export_cif(simple_cubic_record: CrystalRecord) -> None:
    """Test batch export in CIF format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        records = [simple_cubic_record]
        paths = batch_export(records, "cif_stub", tmpdir)

        assert len(paths) == 1
        assert paths[0].endswith(".cif")


def test_batch_export_poscar(simple_cubic_record: CrystalRecord) -> None:
    """Test batch export in POSCAR format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        records = [simple_cubic_record]
        paths = batch_export(records, "poscar_stub", tmpdir)

        assert len(paths) == 1
        assert paths[0].endswith(".poscar")


def test_batch_export_summary(simple_cubic_record: CrystalRecord) -> None:
    """Test batch export in summary text format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        records = [simple_cubic_record]
        paths = batch_export(records, "summary_text", tmpdir)

        assert len(paths) == 1
        assert paths[0].endswith(".txt")


def test_batch_export_multiple_records(
    simple_cubic_record: CrystalRecord,
    hexagonal_record: CrystalRecord,
) -> None:
    """Test batch export with multiple records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        records = [simple_cubic_record, hexagonal_record]
        paths = batch_export(records, "json_dict", tmpdir)

        assert len(paths) == 2
        assert all(Path(p).exists() for p in paths)


def test_batch_export_creates_directory(
    simple_cubic_record: CrystalRecord,
) -> None:
    """Test batch export creates output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = f"{tmpdir}/nested/output"
        batch_export([simple_cubic_record], "json_dict", output_dir)

        assert Path(output_dir).exists()


def test_batch_export_file_content(simple_cubic_record: CrystalRecord) -> None:
    """Test batch export files have content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = batch_export([simple_cubic_record], "json_dict", tmpdir)

        content = Path(paths[0]).read_text()
        assert content
        data = json.loads(content)
        assert "@class" in data


def test_batch_export_invalid_format(simple_cubic_record: CrystalRecord) -> None:
    """Test batch export with invalid format raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError):
            batch_export([simple_cubic_record], "invalid_fmt", tmpdir)


def test_batch_export_returns_absolute_paths(
    simple_cubic_record: CrystalRecord,
) -> None:
    """Test batch export returns absolute paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = batch_export([simple_cubic_record], "json_dict", tmpdir)

        assert all(Path(p).is_absolute() for p in paths)


def test_batch_export_unique_filenames(
    simple_cubic_record: CrystalRecord,
    hexagonal_record: CrystalRecord,
) -> None:
    """Test batch export creates unique filenames based on record_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = batch_export([simple_cubic_record, hexagonal_record], "json_dict", tmpdir)

        assert paths[0] != paths[1]
        # Filenames should contain record IDs
        assert "perov_001" in paths[0]
        assert "carbon_001" in paths[1]


# =============================================================================
# Lattice Computation Tests
# =============================================================================

def test_export_orthogonal_lattice(simple_cubic_record: CrystalRecord) -> None:
    """Test orthogonal lattice computation."""
    data = pymatgen_dict(simple_cubic_record)
    matrix = data["lattice"]["matrix"]

    # For orthogonal lattice, diagonal elements should match
    assert abs(matrix[0][0] - 5.0) < 0.1
    assert abs(matrix[1][1] - 5.0) < 0.1
    assert abs(matrix[2][2] - 5.0) < 0.1


def test_export_hexagonal_lattice(hexagonal_record: CrystalRecord) -> None:
    """Test hexagonal lattice computation."""
    data = pymatgen_dict(hexagonal_record)
    matrix = data["lattice"]["matrix"]

    # Hexagonal: a = b = 2.5, c = 4.0, gamma = 120°
    assert len(matrix) == 3
    assert all(len(vec) == 3 for vec in matrix)


def test_export_cif_lattice_values(simple_cubic_record: CrystalRecord) -> None:
    """Test CIF lattice parameters."""
    result = export_structure(simple_cubic_record, fmt="cif_stub")

    assert "_cell_length_a    5.0" in result.content
    assert "_cell_length_b    5.0" in result.content


def test_export_poscar_lattice_matrix(simple_cubic_record: CrystalRecord) -> None:
    """Test POSCAR lattice vectors."""
    result = export_structure(simple_cubic_record, fmt="poscar_stub")
    lines = result.content.split("\n")

    # Lines 3-5 should be lattice vectors
    # For orthogonal 5.0 Angstrom cubic:
    # [5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]
    lattice_lines = [l for l in lines[2:5] if l.strip()]
    assert len(lattice_lines) >= 3


# =============================================================================
# Integration Tests
# =============================================================================

def test_export_and_reimport_json(simple_cubic_record: CrystalRecord) -> None:
    """Test JSON export can be re-parsed."""
    result = export_structure(simple_cubic_record, fmt="json_dict")
    data = json.loads(result.content)

    assert "lattice" in data
    assert "sites" in data
    assert len(data["sites"]) == 5


def test_batch_export_all_formats(simple_cubic_record: CrystalRecord) -> None:
    """Test batch export in all formats."""
    formats = ["json_dict", "cif_stub", "poscar_stub", "summary_text"]

    with tempfile.TemporaryDirectory() as tmpdir:
        for fmt in formats:
            paths = batch_export([simple_cubic_record], fmt, tmpdir)
            assert len(paths) == 1
            assert Path(paths[0]).exists()


def test_export_cif_and_poscar_compatibility(
    simple_cubic_record: CrystalRecord,
) -> None:
    """Test CIF and POSCAR exports have related content."""
    cif_result = export_structure(simple_cubic_record, fmt="cif_stub")
    poscar_result = export_structure(simple_cubic_record, fmt="poscar_stub")

    # Both should mention the record ID or have lattice info
    assert "_cell_length" in cif_result.content or "Direct" in poscar_result.content


def test_export_summary_includes_all_sites(simple_cubic_record: CrystalRecord) -> None:
    """Test summary export includes all atomic sites."""
    result = export_structure(simple_cubic_record, fmt="summary_text")

    # Should list all 5 sites
    lines = result.content.count("\n")
    assert lines > 10  # At least header + 5 sites + metadata
