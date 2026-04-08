"""
nominal_drift.datasets.crystal_search
=====================================

Crystal structure searching and filtering.

Provides filtered search over CrystalRecord lists with support for
element composition, atomic counts, property ranges, and dataset/split filtering.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from nominal_drift.datasets.schema import CrystalRecord


# ---------------------------------------------------------------------------
# CrystalFilter
# ---------------------------------------------------------------------------

class CrystalFilter(BaseModel):
    """Filter specification for crystal searches.

    All fields are optional — None means "no filter on this dimension".
    Multiple filters are ANDed together.

    Attributes
    ----------
    required_elements : list[str] | None
        All of these elements must be present.
    excluded_elements : list[str] | None
        None of these elements may be present.
    allowed_elements : list[str] | None
        Only these elements allowed (exclusive whitelist).
    min_atoms : int | None
        Minimum number of atoms.
    max_atoms : int | None
        Maximum number of atoms.
    source_dataset : str | None
        Exact match against source_dataset.
    split : str | None
        Exact match against split ("train"|"val"|"test"|None).
    property_filters : dict[str, tuple[float, float]] | None
        property_name → (min_val, max_val) for numeric properties.
    space_group : str | None
        Exact match against properties["spacegroup"].
    """

    required_elements: list[str] | None = None
    excluded_elements: list[str] | None = None
    allowed_elements: list[str] | None = None
    min_atoms: int | None = None
    max_atoms: int | None = None
    source_dataset: str | None = None
    split: str | None = None
    property_filters: dict[str, tuple[float, float]] | None = None
    space_group: str | None = None


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

class SearchResult(BaseModel, frozen=True):
    """Result of a crystal search.

    Attributes
    ----------
    records : list[CrystalRecord]
        Matching records (capped by max_results if specified).
    n_total_searched : int
        Total number of records searched.
    n_matched : int
        Total number of records matching the filter (before capping).
    filter_applied : CrystalFilter
        The filter that was applied.
    """

    records: list[CrystalRecord]
    n_total_searched: int
    n_matched: int
    filter_applied: CrystalFilter


# ---------------------------------------------------------------------------
# Search and Filter Functions
# ---------------------------------------------------------------------------

def search_crystals(
    records: list[CrystalRecord],
    f: CrystalFilter,
    max_results: int | None = None,
) -> SearchResult:
    """Apply filter to records list and return matching records.

    All filter fields are ANDed together.
    max_results caps the returned list (but n_matched reflects true count).

    Parameters
    ----------
    records : list[CrystalRecord]
        List of crystal records to search.
    f : CrystalFilter
        Filter specification.
    max_results : int | None, optional
        Maximum number of results to return (default: None → no cap).

    Returns
    -------
    SearchResult
        Frozen result containing matching records and metadata.
    """
    matched = []

    for rec in records:
        # required_elements: ALL must be present
        if f.required_elements is not None:
            if not all(elem in rec.elements for elem in f.required_elements):
                continue

        # excluded_elements: NONE may be present
        if f.excluded_elements is not None:
            if any(elem in rec.elements for elem in f.excluded_elements):
                continue

        # allowed_elements: ONLY these allowed
        if f.allowed_elements is not None:
            if not all(elem in f.allowed_elements for elem in rec.elements):
                continue

        # min_atoms
        if f.min_atoms is not None:
            if rec.n_atoms < f.min_atoms:
                continue

        # max_atoms
        if f.max_atoms is not None:
            if rec.n_atoms > f.max_atoms:
                continue

        # source_dataset
        if f.source_dataset is not None:
            if rec.source_dataset != f.source_dataset:
                continue

        # split
        if f.split is not None:
            if rec.split != f.split:
                continue

        # property_filters
        if f.property_filters is not None:
            skip = False
            for prop_name, (min_val, max_val) in f.property_filters.items():
                if prop_name not in rec.properties:
                    skip = True
                    break
                prop_val = rec.properties[prop_name]
                if not isinstance(prop_val, (int, float)):
                    skip = True
                    break
                if not (min_val <= prop_val <= max_val):
                    skip = True
                    break
            if skip:
                continue

        # space_group
        if f.space_group is not None:
            if rec.properties.get("spacegroup") != f.space_group:
                continue

        matched.append(rec)

    n_matched = len(matched)
    if max_results is not None:
        matched = matched[:max_results]

    return SearchResult(
        records=matched,
        n_total_searched=len(records),
        n_matched=n_matched,
        filter_applied=f,
    )


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[CrystalRecord]:
    """Load a structures.jsonl file into a list of CrystalRecord objects.

    Parameters
    ----------
    path : str
        Path to the JSONL file.

    Returns
    -------
    list[CrystalRecord]
        List of crystal records.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    json.JSONDecodeError
        If a line is not valid JSON.
    """
    records = []
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            rec = CrystalRecord.model_validate(data)
            records.append(rec)
    return records


def element_distribution(records: list[CrystalRecord]) -> dict[str, int]:
    """Count how many records contain each element.

    Parameters
    ----------
    records : list[CrystalRecord]
        List of crystal records.

    Returns
    -------
    dict[str, int]
        Element → count mapping.
    """
    counts: dict[str, int] = {}
    for rec in records:
        for elem in rec.elements:
            counts[elem] = counts.get(elem, 0) + 1
    return counts


def property_stats(
    records: list[CrystalRecord],
    property_key: str,
) -> dict[str, float]:
    """Return statistics for a numeric property across records.

    Ignores records where property is None or non-numeric.

    Parameters
    ----------
    records : list[CrystalRecord]
        List of crystal records.
    property_key : str
        Name of the property to analyze.

    Returns
    -------
    dict[str, float]
        Dictionary with keys "min", "max", "mean", "count".

    Raises
    ------
    ValueError
        If no records have numeric values for the property.
    """
    values = []
    for rec in records:
        if property_key not in rec.properties:
            continue
        val = rec.properties[property_key]
        if isinstance(val, (int, float)):
            values.append(float(val))

    if not values:
        raise ValueError(
            f"No numeric values found for property '{property_key}'"
        )

    min_val = min(values)
    max_val = max(values)
    mean_val = sum(values) / len(values)

    return {
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "count": len(values),
    }
