"""
tests/unit/test_ingest.py
==========================
Unit tests for nominal_drift.datasets.ingest.

All tests are fully isolated: they create temporary raw directories with
tiny synthetic CSV files (using the same minimal CIF string as the other
ingestion tests) and never touch the real data/datasets/ directory.

Run with:
    pytest tests/unit/test_ingest.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nominal_drift.datasets.ingest import IngestResult, ingest_dataset


# ---------------------------------------------------------------------------
# Shared minimal CIF (5-atom perovskite, valid pymatgen output)
# ---------------------------------------------------------------------------

_CIF = (
    "# generated using pymatgen\n"
    "data_TlCoN2O\n"
    "_symmetry_space_group_name_H-M   'P 1'\n"
    "_cell_length_a   4.24596403\n"
    "_cell_length_b   4.24596403\n"
    "_cell_length_c   4.24596403\n"
    "_cell_angle_alpha   90.00000000\n"
    "_cell_angle_beta   90.00000000\n"
    "_cell_angle_gamma   90.00000000\n"
    "_symmetry_Int_Tables_number   1\n"
    "_chemical_formula_structural   TlCoN2O\n"
    "_chemical_formula_sum   'Tl1 Co1 N2 O1'\n"
    "_cell_volume   76.54713370\n"
    "_cell_formula_units_Z   1\n"
    "loop_\n"
    " _symmetry_equiv_pos_site_id\n"
    " _symmetry_equiv_pos_as_xyz\n"
    "  1  'x, y, z'\n"
    "loop_\n"
    " _atom_site_type_symbol\n"
    " _atom_site_label\n"
    " _atom_site_symmetry_multiplicity\n"
    " _atom_site_fract_x\n"
    " _atom_site_fract_y\n"
    " _atom_site_fract_z\n"
    " _atom_site_occupancy\n"
    "  Tl  Tl0  1  0.50015703  0.50000000  0.50000000  1\n"
    "  Co  Co1  1  0.00265771  0.00000000  0.00000000  1\n"
    "  N  N2  1  0.50108143  0.00000000  0.50000000  1\n"
    "  N  N3  1  0.50108143  0.50000000  0.00000000  1\n"
    "  O  O4  1  0.00111568  0.50000000  0.50000000  1\n"
)

def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write CSV rows with proper quoting (handles multiline CIF strings)."""
    import csv as _csv
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


_GOOD_ROW_1 = {"": "0", "material_id": "1001", "cif": _CIF, "formula": "TlCoN2O",
               "heat_all": "2.72", "heat_ref": "2.62", "dir_gap": "0.0", "ind_gap": "0.0"}
_GOOD_ROW_2 = {"": "1", "material_id": "1002", "cif": _CIF, "formula": "TlCoN2O",
               "heat_all": "2.80", "heat_ref": "2.70", "dir_gap": "0.1", "ind_gap": "0.1"}
_BAD_ROW    = {"": "1", "material_id": "bad", "cif": "not a cif at all", "formula": "X",
               "heat_all": "0.0", "heat_ref": "0.0", "dir_gap": "0.0", "ind_gap": "0.0"}


@pytest.fixture()
def perov_raw(tmp_path) -> tuple[Path, Path]:
    """Write minimal perov-5 CSV files (2 good rows each); return (raw_base, norm_base)."""
    raw_base = tmp_path / "raw"
    norm_base = tmp_path / "normalized"
    raw_dir = raw_base / "perov-5"
    raw_dir.mkdir(parents=True)
    norm_base.mkdir()
    for fname in ("train.csv", "val.csv", "test.csv"):
        _write_csv(raw_dir / fname, [_GOOD_ROW_1, _GOOD_ROW_2])
    return raw_base, norm_base


@pytest.fixture()
def perov_raw_partial(tmp_path) -> tuple[Path, Path]:
    """Only train.csv is present."""
    raw_base = tmp_path / "raw"
    norm_base = tmp_path / "normalized"
    (raw_base / "perov-5").mkdir(parents=True)
    norm_base.mkdir()
    _write_csv(raw_base / "perov-5" / "train.csv", [_GOOD_ROW_1, _GOOD_ROW_2])
    return raw_base, norm_base


@pytest.fixture()
def perov_raw_one_bad_row(tmp_path) -> tuple[Path, Path]:
    """train.csv has one good row and one row with an invalid CIF."""
    raw_base = tmp_path / "raw"
    norm_base = tmp_path / "normalized"
    (raw_base / "perov-5").mkdir(parents=True)
    norm_base.mkdir()
    _write_csv(raw_base / "perov-5" / "train.csv", [_GOOD_ROW_1, _BAD_ROW])
    return raw_base, norm_base


# ---------------------------------------------------------------------------
# IngestResult properties
# ---------------------------------------------------------------------------

class TestIngestResult:

    def test_success_rate_all_ok(self, perov_raw):
        raw, norm = perov_raw
        r = ingest_dataset("perov-5", raw_base=raw, norm_base=norm)
        assert r.success_rate == pytest.approx(1.0)

    def test_success_rate_zero_total(self):
        r = IngestResult("x", 0, 0, 0, 0.0, Path("."))
        assert r.success_rate == 0.0

    def test_str_representation(self, perov_raw):
        raw, norm = perov_raw
        r = ingest_dataset("perov-5", raw_base=raw, norm_base=norm)
        s = str(r)
        assert "perov-5" in s
        assert "ok" in s


# ---------------------------------------------------------------------------
# ingest_dataset — happy path
# ---------------------------------------------------------------------------

class TestIngestDatasetHappyPath:

    def test_returns_ingest_result(self, perov_raw):
        raw, norm = perov_raw
        r = ingest_dataset("perov-5", raw_base=raw, norm_base=norm)
        assert isinstance(r, IngestResult)

    def test_n_ok(self, perov_raw):
        raw, norm = perov_raw
        # 3 files × 2 rows = 6 total
        r = ingest_dataset("perov-5", raw_base=raw, norm_base=norm)
        assert r.n_ok == 6

    def test_n_err_zero(self, perov_raw):
        raw, norm = perov_raw
        r = ingest_dataset("perov-5", raw_base=raw, norm_base=norm)
        assert r.n_err == 0

    def test_n_total(self, perov_raw):
        raw, norm = perov_raw
        r = ingest_dataset("perov-5", raw_base=raw, norm_base=norm)
        assert r.n_total == 6

    def test_output_dir_created(self, perov_raw):
        raw, norm = perov_raw
        r = ingest_dataset("perov-5", raw_base=raw, norm_base=norm)
        assert r.output_dir.exists()

    def test_structures_jsonl_written(self, perov_raw):
        raw, norm = perov_raw
        r = ingest_dataset("perov-5", raw_base=raw, norm_base=norm)
        jsonl = r.output_dir / "structures.jsonl"
        assert jsonl.exists()
        lines = [l for l in jsonl.read_text().splitlines() if l.strip()]
        assert len(lines) == 6

    def test_manifest_written(self, perov_raw):
        raw, norm = perov_raw
        r = ingest_dataset("perov-5", raw_base=raw, norm_base=norm)
        manifest = json.loads((r.output_dir / "manifest.json").read_text())
        assert manifest["n_structures"] == 6
        assert manifest["dataset_name"] == "perov-5"

    def test_splits_in_manifest(self, perov_raw):
        raw, norm = perov_raw
        r = ingest_dataset("perov-5", raw_base=raw, norm_base=norm)
        manifest = json.loads((r.output_dir / "manifest.json").read_text())
        assert manifest["splits"]["train"] == 2
        assert manifest["splits"]["val"] == 2
        assert manifest["splits"]["test"] == 2

    def test_dataset_name_in_result(self, perov_raw):
        raw, norm = perov_raw
        r = ingest_dataset("perov-5", raw_base=raw, norm_base=norm)
        assert r.dataset_name == "perov-5"

    def test_elapsed_s_positive(self, perov_raw):
        raw, norm = perov_raw
        r = ingest_dataset("perov-5", raw_base=raw, norm_base=norm)
        assert r.elapsed_s >= 0.0


# ---------------------------------------------------------------------------
# ingest_dataset — limit parameter
# ---------------------------------------------------------------------------

class TestIngestDatasetLimit:

    def test_limit_one_row_per_file(self, perov_raw):
        raw, norm = perov_raw
        r = ingest_dataset("perov-5", raw_base=raw, norm_base=norm, limit=1)
        # 3 files × 1 row each
        assert r.n_ok == 3

    def test_limit_zero_reads_no_rows(self, perov_raw):
        raw, norm = perov_raw
        r = ingest_dataset("perov-5", raw_base=raw, norm_base=norm, limit=0)
        assert r.n_ok == 0
        assert r.n_total == 0


# ---------------------------------------------------------------------------
# ingest_dataset — partial raw (some files missing)
# ---------------------------------------------------------------------------

class TestIngestDatasetPartialRaw:

    def test_partial_raw_reads_available_files(self, perov_raw_partial):
        raw, norm = perov_raw_partial
        r = ingest_dataset("perov-5", raw_base=raw, norm_base=norm)
        # Only train.csv is present → 2 records
        assert r.n_ok == 2

    def test_partial_raw_no_errors_for_missing_files(self, perov_raw_partial):
        raw, norm = perov_raw_partial
        r = ingest_dataset("perov-5", raw_base=raw, norm_base=norm)
        assert r.n_err == 0


# ---------------------------------------------------------------------------
# ingest_dataset — bad CIF rows
# ---------------------------------------------------------------------------

class TestIngestDatasetBadRows:

    def test_bad_row_counted_as_error(self, perov_raw_one_bad_row):
        raw, norm = perov_raw_one_bad_row
        r = ingest_dataset("perov-5", raw_base=raw, norm_base=norm)
        assert r.n_err == 1

    def test_good_row_still_ingested(self, perov_raw_one_bad_row):
        raw, norm = perov_raw_one_bad_row
        r = ingest_dataset("perov-5", raw_base=raw, norm_base=norm)
        assert r.n_ok == 1

    def test_error_sample_captured(self, perov_raw_one_bad_row):
        raw, norm = perov_raw_one_bad_row
        r = ingest_dataset("perov-5", raw_base=raw, norm_base=norm)
        assert len(r.error_samples) >= 1

    def test_success_rate_below_one(self, perov_raw_one_bad_row):
        raw, norm = perov_raw_one_bad_row
        r = ingest_dataset("perov-5", raw_base=raw, norm_base=norm)
        assert r.success_rate < 1.0


# ---------------------------------------------------------------------------
# ingest_dataset — no raw files at all
# ---------------------------------------------------------------------------

class TestIngestDatasetNoRaw:

    def test_no_raw_files_zero_ok(self, tmp_path):
        raw = tmp_path / "raw"
        norm = tmp_path / "normalized"
        raw.mkdir()
        norm.mkdir()
        r = ingest_dataset("perov-5", raw_base=raw, norm_base=norm)
        assert r.n_ok == 0

    def test_no_raw_files_zero_err(self, tmp_path):
        raw = tmp_path / "raw"
        norm = tmp_path / "normalized"
        raw.mkdir()
        norm.mkdir()
        r = ingest_dataset("perov-5", raw_base=raw, norm_base=norm)
        assert r.n_err == 0


# ---------------------------------------------------------------------------
# ingest_dataset — unknown dataset
# ---------------------------------------------------------------------------

class TestIngestDatasetUnknown:

    def test_unknown_dataset_raises_key_error(self, tmp_path):
        raw = tmp_path / "raw"
        norm = tmp_path / "normalized"
        raw.mkdir()
        norm.mkdir()
        with pytest.raises(KeyError):
            ingest_dataset("nonexistent", raw_base=raw, norm_base=norm)
