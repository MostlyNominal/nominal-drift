"""
tests/unit/test_dataset_status.py
===================================
Unit tests for nominal_drift.datasets.status.

Tests are fully isolated — they use tmp_path fixtures and never touch
the real data/datasets/ directory.  All counts are verified from actual
on-disk files, not mocked.

Run with:
    pytest tests/unit/test_dataset_status.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nominal_drift.datasets.status import (
    DATASET_REGISTRY,
    DatasetInfo,
    DatasetStatus,
    get_all_statuses,
    get_dataset_status,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def empty_bases(tmp_path):
    """Return (raw_base, norm_base) pointing at empty tmp dirs."""
    raw = tmp_path / "raw"
    norm = tmp_path / "normalized"
    raw.mkdir()
    norm.mkdir()
    return raw, norm


@pytest.fixture()
def perov_raw_complete(tmp_path):
    """Write minimal perov-5 raw CSV files and return raw_base."""
    raw_base = tmp_path / "raw"
    raw_dir = raw_base / "perov-5"
    raw_dir.mkdir(parents=True)
    for fname in ("train.csv", "val.csv", "test.csv"):
        f = raw_dir / fname
        f.write_text("material_id,formula,a,b,c,alpha,beta,gamma,sites\n"
                     "mp-1,BaTiO3,4.01,4.01,4.01,90,90,90,[{...}]\n"
                     "mp-2,SrTiO3,3.90,3.90,3.90,90,90,90,[{...}]\n")
    return raw_base


@pytest.fixture()
def perov_normalized(tmp_path):
    """Write normalised structures.jsonl + manifest for perov-5."""
    raw_base = tmp_path / "raw"
    norm_base = tmp_path / "normalized"
    raw_dir = raw_base / "perov-5"
    norm_dir = norm_base / "perov-5"
    raw_dir.mkdir(parents=True)
    norm_dir.mkdir(parents=True)

    for fname in ("train.csv", "val.csv", "test.csv"):
        (raw_dir / fname).write_text(
            "material_id,formula,a\nmp-1,BaTiO3,4.01\n"
        )

    # Two structure records
    jsonl_path = norm_dir / "structures.jsonl"
    jsonl_path.write_text(
        '{"id":"1","formula":"BaTiO3"}\n'
        '{"id":"2","formula":"SrTiO3"}\n'
    )
    manifest = {
        "dataset_name": "perov-5",
        "n_structures": 2,
        "elements_present": ["Ba", "O", "Sr", "Ti"],
    }
    (norm_dir / "manifest.json").write_text(json.dumps(manifest))
    return raw_base, norm_base


# ---------------------------------------------------------------------------
# DATASET_REGISTRY
# ---------------------------------------------------------------------------

class TestDatasetRegistry:

    def test_has_four_entries(self):
        assert len(DATASET_REGISTRY) == 4

    def test_all_known_datasets_present(self):
        for name in ("perov-5", "mp-20", "carbon-24", "mpts-52"):
            assert name in DATASET_REGISTRY

    def test_all_entries_are_datasetinfo(self):
        for v in DATASET_REGISTRY.values():
            assert isinstance(v, DatasetInfo)

    def test_all_have_source_url(self):
        for info in DATASET_REGISTRY.values():
            assert info.source_url.startswith("http")

    def test_csv_datasets_have_expected_raw_files(self):
        # CSV-based datasets must list their raw files; figshare-based (mpts-52) may have none
        csv_datasets = {k: v for k, v in DATASET_REGISTRY.items() if k != "mpts-52"}
        for info in csv_datasets.values():
            assert len(info.expected_raw_files) >= 1

    def test_mpts52_has_no_raw_csv_files(self):
        # mpts-52 uses figshare via mp-time-split — no CSV raw files expected
        assert DATASET_REGISTRY["mpts-52"].expected_raw_files == ()

    def test_all_have_manual_instructions(self):
        for name, info in DATASET_REGISTRY.items():
            assert info.manual_instructions, f"{name} has empty manual_instructions"

    def test_perov5_expected_files(self):
        info = DATASET_REGISTRY["perov-5"]
        assert set(info.expected_raw_files) == {"train.csv", "val.csv", "test.csv"}

    def test_mpts52_source_url_is_figshare(self):
        info = DATASET_REGISTRY["mpts-52"]
        assert "figshare" in info.source_url


# ---------------------------------------------------------------------------
# get_dataset_status — not downloaded
# ---------------------------------------------------------------------------

class TestStatusNotDownloaded:

    def test_returns_dataset_status(self, empty_bases):
        raw, norm = empty_bases
        s = get_dataset_status("perov-5", raw_base=raw, norm_base=norm)
        assert isinstance(s, DatasetStatus)

    def test_not_raw_complete(self, empty_bases):
        raw, norm = empty_bases
        s = get_dataset_status("perov-5", raw_base=raw, norm_base=norm)
        assert not s.is_raw_complete

    def test_not_normalised(self, empty_bases):
        raw, norm = empty_bases
        s = get_dataset_status("perov-5", raw_base=raw, norm_base=norm)
        assert not s.is_normalised

    def test_status_label_shows_not_downloaded(self, empty_bases):
        raw, norm = empty_bases
        s = get_dataset_status("perov-5", raw_base=raw, norm_base=norm)
        assert "not downloaded" in s.download_status_label.lower() or "❌" in s.download_status_label

    def test_unknown_dataset_raises_key_error(self, empty_bases):
        raw, norm = empty_bases
        with pytest.raises(KeyError):
            get_dataset_status("nonexistent-dataset", raw_base=raw, norm_base=norm)


# ---------------------------------------------------------------------------
# get_dataset_status — raw complete
# ---------------------------------------------------------------------------

class TestStatusRawComplete:

    def test_is_raw_complete(self, perov_raw_complete, tmp_path):
        norm = tmp_path / "normalized"
        norm.mkdir()
        s = get_dataset_status("perov-5", raw_base=perov_raw_complete, norm_base=norm)
        assert s.is_raw_complete

    def test_row_counts_are_nonzero(self, perov_raw_complete, tmp_path):
        norm = tmp_path / "normalized"
        norm.mkdir()
        s = get_dataset_status("perov-5", raw_base=perov_raw_complete, norm_base=norm)
        for fname in ("train.csv", "val.csv", "test.csv"):
            assert s.raw_row_counts[fname] == 2  # two data rows per file

    def test_total_rows(self, perov_raw_complete, tmp_path):
        norm = tmp_path / "normalized"
        norm.mkdir()
        s = get_dataset_status("perov-5", raw_base=perov_raw_complete, norm_base=norm)
        assert s.total_raw_rows == 6

    def test_not_normalised(self, perov_raw_complete, tmp_path):
        norm = tmp_path / "normalized"
        norm.mkdir()
        s = get_dataset_status("perov-5", raw_base=perov_raw_complete, norm_base=norm)
        assert not s.is_normalised

    def test_status_label_shows_raw_present(self, perov_raw_complete, tmp_path):
        norm = tmp_path / "normalized"
        norm.mkdir()
        s = get_dataset_status("perov-5", raw_base=perov_raw_complete, norm_base=norm)
        label = s.download_status_label
        assert "raw present" in label or "📥" in label


# ---------------------------------------------------------------------------
# get_dataset_status — normalised
# ---------------------------------------------------------------------------

class TestStatusNormalised:

    def test_is_normalised(self, perov_normalized):
        raw, norm = perov_normalized
        s = get_dataset_status("perov-5", raw_base=raw, norm_base=norm)
        assert s.is_normalised

    def test_structure_count(self, perov_normalized):
        raw, norm = perov_normalized
        s = get_dataset_status("perov-5", raw_base=raw, norm_base=norm)
        assert s.norm_structure_count == 2

    def test_elements_populated(self, perov_normalized):
        raw, norm = perov_normalized
        s = get_dataset_status("perov-5", raw_base=raw, norm_base=norm)
        assert "Ba" in s.norm_elements
        assert "Ti" in s.norm_elements

    def test_status_label_shows_normalised(self, perov_normalized):
        raw, norm = perov_normalized
        s = get_dataset_status("perov-5", raw_base=raw, norm_base=norm)
        label = s.download_status_label
        assert "normalised" in label or "✅" in label

    def test_manifest_is_parsed(self, perov_normalized):
        raw, norm = perov_normalized
        s = get_dataset_status("perov-5", raw_base=raw, norm_base=norm)
        assert s.norm_manifest is not None
        assert s.norm_manifest["n_structures"] == 2


# ---------------------------------------------------------------------------
# Partial raw (some files missing)
# ---------------------------------------------------------------------------

class TestStatusPartialRaw:

    def test_partial_raw_not_complete(self, tmp_path):
        raw = tmp_path / "raw"
        norm = tmp_path / "normalized"
        raw_dir = raw / "perov-5"
        raw_dir.mkdir(parents=True)
        norm.mkdir()
        # Only train.csv present
        (raw_dir / "train.csv").write_text("material_id,formula\nmp-1,BaTiO3\n")

        s = get_dataset_status("perov-5", raw_base=raw, norm_base=norm)
        assert not s.is_raw_complete
        assert "train.csv" in s.raw_files_present
        assert "val.csv" in s.raw_files_missing
        assert "test.csv" in s.raw_files_missing

    def test_partial_label_mentions_missing(self, tmp_path):
        raw = tmp_path / "raw"
        norm = tmp_path / "normalized"
        raw_dir = raw / "perov-5"
        raw_dir.mkdir(parents=True)
        norm.mkdir()
        (raw_dir / "train.csv").write_text("material_id\nmp-1\n")

        s = get_dataset_status("perov-5", raw_base=raw, norm_base=norm)
        label = s.download_status_label
        assert "partial" in label.lower() or "⚠️" in label


# ---------------------------------------------------------------------------
# get_all_statuses
# ---------------------------------------------------------------------------

class TestGetAllStatuses:

    def test_returns_four_datasets(self, empty_bases):
        raw, norm = empty_bases
        result = get_all_statuses(raw_base=raw, norm_base=norm)
        assert len(result) == 4

    def test_keys_match_registry(self, empty_bases):
        raw, norm = empty_bases
        result = get_all_statuses(raw_base=raw, norm_base=norm)
        assert set(result.keys()) == set(DATASET_REGISTRY.keys())

    def test_csv_datasets_not_raw_complete_when_empty(self, empty_bases):
        # CSV-based datasets must NOT be raw_complete when their raw dir is empty.
        # Figshare-based (mpts-52) is always raw_complete (download = normalise step).
        raw, norm = empty_bases
        statuses = get_all_statuses(raw_base=raw, norm_base=norm)
        csv_names = [n for n, s in statuses.items() if not s.uses_figshare_loader]
        for name in csv_names:
            assert not statuses[name].is_raw_complete, f"{name} should not be raw_complete"

    def test_mpts52_raw_complete_is_always_true(self, empty_bases):
        # mpts-52 uses figshare loader — is_raw_complete is True regardless of disk state
        raw, norm = empty_bases
        s = get_all_statuses(raw_base=raw, norm_base=norm)["mpts-52"]
        assert s.uses_figshare_loader
        assert s.is_raw_complete  # always True for figshare datasets
