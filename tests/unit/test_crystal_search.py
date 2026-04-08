"""
tests/unit/test_crystal_search.py
=================================

Unit tests for nominal_drift.datasets.crystal_search.

Tests cover:
  - CrystalFilter model
  - search_crystals function with all filter dimensions
  - load_jsonl round-trip
  - element_distribution analysis
  - property_stats computation

Run with:
    pytest tests/unit/test_crystal_search.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from nominal_drift.datasets.crystal_search import (
    CrystalFilter,
    SearchResult,
    element_distribution,
    load_jsonl,
    property_stats,
    search_crystals,
)
from nominal_drift.datasets.schema import (
    AtomicSite,
    CrystalRecord,
    LatticeParams,
)


# =============================================================================
# Fixtures
# =============================================================================

def make_crystal_record(
    record_id: str,
    elements: tuple[str, ...],
    n_atoms: int,
    source_dataset: str = "test-dataset",
    split: str | None = "train",
    properties: dict | None = None,
) -> CrystalRecord:
    """Helper to create a CrystalRecord for testing."""
    if properties is None:
        properties = {}

    lattice = LatticeParams(a=5.0, b=5.0, c=5.0, alpha=90.0, beta=90.0, gamma=90.0)
    sites = tuple(
        AtomicSite(species=elem, frac_coords=(0.0, 0.0, 0.0))
        for _ in range(n_atoms)
        for elem in sorted(set(elements))
    )[:n_atoms]

    return CrystalRecord(
        record_id=record_id,
        source_dataset=source_dataset,
        source_index=0,
        split=split,
        elements=tuple(sorted(set(elements))),
        n_atoms=n_atoms,
        lattice=lattice,
        sites=sites,
        properties=properties,
        raw_path=None,
    )


@pytest.fixture()
def sample_records() -> list[CrystalRecord]:
    """Sample collection of crystal records for testing."""
    return [
        make_crystal_record("rec1", ("Ba", "Ti", "O"), 5, "perov-5", "train", {"energy": -2.5}),
        make_crystal_record("rec2", ("Si", "O"), 3, "mp-20", "train", {"energy": -1.5, "band_gap": 5.2}),
        make_crystal_record("rec3", ("C",), 8, "carbon-24", "val", {"energy": -10.0}),
        make_crystal_record("rec4", ("Fe", "O"), 6, "mp-20", "train", {"energy": -3.0}),
        make_crystal_record("rec5", ("Ni",), 4, "mp-20", "test", {"energy": -0.5}),
    ]


# =============================================================================
# CrystalFilter Tests
# =============================================================================

def test_filter_creation_empty() -> None:
    """Test creating an empty filter."""
    f = CrystalFilter()
    assert f.required_elements is None
    assert f.excluded_elements is None
    assert f.min_atoms is None
    assert f.max_atoms is None


def test_filter_required_elements() -> None:
    """Test filter with required elements."""
    f = CrystalFilter(required_elements=["Si", "O"])
    assert f.required_elements == ["Si", "O"]


def test_filter_excluded_elements() -> None:
    """Test filter with excluded elements."""
    f = CrystalFilter(excluded_elements=["C", "N"])
    assert f.excluded_elements == ["C", "N"]


def test_filter_allowed_elements() -> None:
    """Test filter with allowed elements (exclusive)."""
    f = CrystalFilter(allowed_elements=["Fe", "O", "Ni"])
    assert f.allowed_elements == ["Fe", "O", "Ni"]


def test_filter_atom_counts() -> None:
    """Test filter with atom count constraints."""
    f = CrystalFilter(min_atoms=5, max_atoms=10)
    assert f.min_atoms == 5
    assert f.max_atoms == 10


def test_filter_dataset_and_split() -> None:
    """Test filter with dataset and split."""
    f = CrystalFilter(source_dataset="mp-20", split="val")
    assert f.source_dataset == "mp-20"
    assert f.split == "val"


def test_filter_property_ranges() -> None:
    """Test filter with property ranges."""
    props = {"energy": (-5.0, 0.0), "band_gap": (0.0, 10.0)}
    f = CrystalFilter(property_filters=props)
    assert f.property_filters == props


def test_filter_space_group() -> None:
    """Test filter with space group."""
    f = CrystalFilter(space_group="Fm-3m")
    assert f.space_group == "Fm-3m"


# =============================================================================
# SearchResult Tests
# =============================================================================

def test_search_result_creation(sample_records: list[CrystalRecord]) -> None:
    """Test creating a search result."""
    f = CrystalFilter()
    result = SearchResult(
        records=sample_records,
        n_total_searched=len(sample_records),
        n_matched=len(sample_records),
        filter_applied=f,
    )
    assert result.n_total_searched == 5
    assert result.n_matched == 5
    assert len(result.records) == 5


def test_search_result_frozen() -> None:
    """Test that SearchResult is frozen."""
    f = CrystalFilter()
    result = SearchResult(records=[], n_total_searched=0, n_matched=0, filter_applied=f)
    with pytest.raises(Exception):
        result.records = []  # type: ignore


# =============================================================================
# search_crystals Tests
# =============================================================================

def test_search_no_filter(sample_records: list[CrystalRecord]) -> None:
    """Test search with no filter returns all records."""
    f = CrystalFilter()
    result = search_crystals(sample_records, f)

    assert result.n_total_searched == 5
    assert result.n_matched == 5
    assert len(result.records) == 5


def test_search_required_elements(sample_records: list[CrystalRecord]) -> None:
    """Test search with required elements."""
    f = CrystalFilter(required_elements=["O"])
    result = search_crystals(sample_records, f)

    # rec1 (Ba, Ti, O), rec2 (Si, O), rec4 (Fe, O) have O
    assert result.n_matched == 3


def test_search_required_elements_multiple(sample_records: list[CrystalRecord]) -> None:
    """Test search with multiple required elements."""
    f = CrystalFilter(required_elements=["Ba", "Ti"])
    result = search_crystals(sample_records, f)

    # Only rec1 has both Ba and Ti
    assert result.n_matched == 1
    assert result.records[0].record_id == "rec1"


def test_search_excluded_elements(sample_records: list[CrystalRecord]) -> None:
    """Test search excluding elements."""
    f = CrystalFilter(excluded_elements=["C"])
    result = search_crystals(sample_records, f)

    # Should exclude rec3
    assert result.n_matched == 4


def test_search_allowed_elements(sample_records: list[CrystalRecord]) -> None:
    """Test search with allowed elements (exclusive)."""
    f = CrystalFilter(allowed_elements=["Si", "O"])
    result = search_crystals(sample_records, f)

    # Only rec2 has only Si and O
    assert result.n_matched == 1
    assert result.records[0].record_id == "rec2"


def test_search_min_atoms(sample_records: list[CrystalRecord]) -> None:
    """Test search with minimum atoms constraint."""
    f = CrystalFilter(min_atoms=5)
    result = search_crystals(sample_records, f)

    # rec1 (5 atoms), rec3 (8 atoms), rec4 (6 atoms)
    assert result.n_matched == 3


def test_search_max_atoms(sample_records: list[CrystalRecord]) -> None:
    """Test search with maximum atoms constraint."""
    f = CrystalFilter(max_atoms=4)
    result = search_crystals(sample_records, f)

    # rec2 (3 atoms), rec5 (4 atoms)
    assert result.n_matched == 2


def test_search_atom_range(sample_records: list[CrystalRecord]) -> None:
    """Test search with atom count range."""
    f = CrystalFilter(min_atoms=5, max_atoms=6)
    result = search_crystals(sample_records, f)

    # rec1 (5 atoms), rec4 (6 atoms)
    assert result.n_matched == 2


def test_search_source_dataset(sample_records: list[CrystalRecord]) -> None:
    """Test search by source dataset."""
    f = CrystalFilter(source_dataset="mp-20")
    result = search_crystals(sample_records, f)

    # rec2, rec4, rec5 are from mp-20
    assert result.n_matched == 3


def test_search_split(sample_records: list[CrystalRecord]) -> None:
    """Test search by split."""
    f = CrystalFilter(split="train")
    result = search_crystals(sample_records, f)

    # rec1, rec2, rec4 have split="train"
    assert result.n_matched == 3


def test_search_property_filter_single(sample_records: list[CrystalRecord]) -> None:
    """Test search with single property filter."""
    f = CrystalFilter(property_filters={"energy": (-3.0, -1.0)})
    result = search_crystals(sample_records, f)

    # rec1 (energy=-2.5), rec2 (energy=-1.5), rec4 (energy=-3.0)
    assert result.n_matched == 3


def test_search_property_filter_multiple(sample_records: list[CrystalRecord]) -> None:
    """Test search with multiple property filters."""
    f = CrystalFilter(
        property_filters={
            "energy": (-3.0, 0.0),
            "band_gap": (0.0, 10.0),
        }
    )
    result = search_crystals(sample_records, f)

    # Only rec2 has both properties in range
    assert result.n_matched == 1
    assert result.records[0].record_id == "rec2"


def test_search_combined_filters(sample_records: list[CrystalRecord]) -> None:
    """Test search with combined filters (AND logic)."""
    f = CrystalFilter(
        required_elements=["O"],
        source_dataset="mp-20",
        min_atoms=3,
    )
    result = search_crystals(sample_records, f)

    # rec2 (Si, O), rec4 (Fe, O) from mp-20 with O and >= 3 atoms
    assert result.n_matched == 2


def test_search_max_results_cap(sample_records: list[CrystalRecord]) -> None:
    """Test max_results capping."""
    f = CrystalFilter()
    result = search_crystals(sample_records, f, max_results=3)

    assert len(result.records) == 3
    assert result.n_matched == 5  # True count is still 5


def test_search_max_results_zero_match() -> None:
    """Test search with max_results when no matches."""
    records = [make_crystal_record("r1", ("Si", "O"), 3)]
    f = CrystalFilter(required_elements=["C"])
    result = search_crystals(records, f, max_results=10)

    assert len(result.records) == 0
    assert result.n_matched == 0


def test_search_no_matches(sample_records: list[CrystalRecord]) -> None:
    """Test search with no matches."""
    f = CrystalFilter(required_elements=["Pt"])
    result = search_crystals(sample_records, f)

    assert result.n_matched == 0
    assert len(result.records) == 0


def test_search_space_group_filter() -> None:
    """Test search with space group filter."""
    rec1 = make_crystal_record("r1", ("A",), 1, properties={"spacegroup": "Fm-3m"})
    rec2 = make_crystal_record("r2", ("B",), 1, properties={"spacegroup": "P1"})
    records = [rec1, rec2]

    f = CrystalFilter(space_group="Fm-3m")
    result = search_crystals(records, f)

    assert result.n_matched == 1
    assert result.records[0].record_id == "r1"


# =============================================================================
# load_jsonl Tests
# =============================================================================

def test_load_jsonl_single_record(sample_records: list[CrystalRecord]) -> None:
    """Test loading JSONL with single record."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/test.jsonl"
        with open(path, "w") as f:
            f.write(sample_records[0].model_dump_json() + "\n")

        loaded = load_jsonl(path)
        assert len(loaded) == 1
        assert loaded[0].record_id == "rec1"


def test_load_jsonl_multiple_records(sample_records: list[CrystalRecord]) -> None:
    """Test loading JSONL with multiple records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/test.jsonl"
        with open(path, "w") as f:
            for rec in sample_records:
                f.write(rec.model_dump_json() + "\n")

        loaded = load_jsonl(path)
        assert len(loaded) == 5


def test_load_jsonl_empty_lines(sample_records: list[CrystalRecord]) -> None:
    """Test loading JSONL with empty lines."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/test.jsonl"
        with open(path, "w") as f:
            f.write(sample_records[0].model_dump_json() + "\n")
            f.write("\n")  # Empty line
            f.write(sample_records[1].model_dump_json() + "\n")

        loaded = load_jsonl(path)
        assert len(loaded) == 2


def test_load_jsonl_file_not_found() -> None:
    """Test loading nonexistent file raises error."""
    with pytest.raises(FileNotFoundError):
        load_jsonl("/nonexistent/path.jsonl")


def test_load_jsonl_invalid_json() -> None:
    """Test loading invalid JSON raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/bad.jsonl"
        with open(path, "w") as f:
            f.write("{ invalid json }\n")

        with pytest.raises(json.JSONDecodeError):
            load_jsonl(path)


def test_load_jsonl_round_trip(sample_records: list[CrystalRecord]) -> None:
    """Test JSONL save and load round-trip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/roundtrip.jsonl"
        with open(path, "w") as f:
            for rec in sample_records:
                f.write(rec.model_dump_json() + "\n")

        loaded = load_jsonl(path)
        assert len(loaded) == len(sample_records)
        for orig, restored in zip(sample_records, loaded):
            assert orig.record_id == restored.record_id
            assert orig.n_atoms == restored.n_atoms
            assert orig.elements == restored.elements


# =============================================================================
# element_distribution Tests
# =============================================================================

def test_element_distribution_basic(sample_records: list[CrystalRecord]) -> None:
    """Test element distribution counting."""
    dist = element_distribution(sample_records)

    assert dist["O"] == 3  # rec1, rec2, rec4
    assert dist["Ba"] == 1
    assert dist["Si"] == 1
    assert dist["C"] == 1
    assert dist["Fe"] == 1


def test_element_distribution_empty() -> None:
    """Test element distribution with empty list."""
    dist = element_distribution([])
    assert dist == {}


def test_element_distribution_single_element() -> None:
    """Test distribution with single element type."""
    records = [
        make_crystal_record("r1", ("Ni",), 2),
        make_crystal_record("r2", ("Ni",), 4),
    ]
    dist = element_distribution(records)

    assert dist == {"Ni": 2}


def test_element_distribution_sorted_keys() -> None:
    """Test that element distribution keys can be sorted."""
    dist = element_distribution([
        make_crystal_record("r1", ("Zr", "O", "Al"), 3),
        make_crystal_record("r2", ("Zr", "Al"), 2),
    ])
    assert "Zr" in dist
    assert "O" in dist
    assert "Al" in dist


# =============================================================================
# property_stats Tests
# =============================================================================

def test_property_stats_basic(sample_records: list[CrystalRecord]) -> None:
    """Test basic property statistics."""
    stats = property_stats(sample_records, "energy")

    assert "min" in stats
    assert "max" in stats
    assert "mean" in stats
    assert "count" in stats
    assert stats["count"] == 5


def test_property_stats_values(sample_records: list[CrystalRecord]) -> None:
    """Test property stats values."""
    stats = property_stats(sample_records, "energy")

    assert stats["min"] == -10.0  # rec3
    assert stats["max"] == -0.5   # rec5
    assert stats["count"] == 5


def test_property_stats_mean() -> None:
    """Test mean calculation."""
    records = [
        make_crystal_record("r1", ("A",), 1, properties={"value": 10.0}),
        make_crystal_record("r2", ("B",), 1, properties={"value": 20.0}),
    ]
    stats = property_stats(records, "value")

    assert stats["mean"] == 15.0


def test_property_stats_missing_property() -> None:
    """Test stats with missing property."""
    records = [
        make_crystal_record("r1", ("A",), 1, properties={"energy": 1.0}),
        make_crystal_record("r2", ("B",), 1, properties={}),  # No energy
    ]
    stats = property_stats(records, "energy")

    assert stats["count"] == 1


def test_property_stats_nonexistent_property() -> None:
    """Test stats for property that doesn't exist raises error."""
    records = [make_crystal_record("r1", ("A",), 1, properties={"energy": 1.0})]

    with pytest.raises(ValueError):
        property_stats(records, "nonexistent")


def test_property_stats_non_numeric_property() -> None:
    """Test stats with non-numeric properties are ignored."""
    records = [
        make_crystal_record("r1", ("A",), 1, properties={"desc": "string", "value": 5.0}),
        make_crystal_record("r2", ("B",), 1, properties={"desc": "another", "value": 10.0}),
    ]
    stats = property_stats(records, "value")

    assert stats["count"] == 2
    assert stats["mean"] == 7.5


def test_property_stats_single_value() -> None:
    """Test stats with single value."""
    records = [make_crystal_record("r1", ("A",), 1, properties={"x": 42.0})]
    stats = property_stats(records, "x")

    assert stats["min"] == 42.0
    assert stats["max"] == 42.0
    assert stats["mean"] == 42.0


# =============================================================================
# Integration Tests
# =============================================================================

def test_search_and_load_jsonl(sample_records: list[CrystalRecord]) -> None:
    """Test search integration with JSONL loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save to JSONL
        path = f"{tmpdir}/structures.jsonl"
        with open(path, "w") as f:
            for rec in sample_records:
                f.write(rec.model_dump_json() + "\n")

        # Load and search
        loaded = load_jsonl(path)
        f = CrystalFilter(source_dataset="mp-20")
        result = search_crystals(loaded, f)

        assert result.n_matched == 3


def test_search_and_stats(sample_records: list[CrystalRecord]) -> None:
    """Test combining search with statistics."""
    f = CrystalFilter(required_elements=["O"])
    result = search_crystals(sample_records, f)

    if result.records:
        stats = property_stats(result.records, "energy")
        assert stats["count"] <= 3


def test_element_distribution_after_search(sample_records: list[CrystalRecord]) -> None:
    """Test element distribution on search results."""
    f = CrystalFilter(split="train")
    result = search_crystals(sample_records, f)

    dist = element_distribution(result.records)
    assert len(dist) > 0
