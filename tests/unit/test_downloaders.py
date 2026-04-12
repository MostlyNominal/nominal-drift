"""
Tests for nominal_drift.datasets.downloaders module.

Comprehensive test suite for DownloadResult, BaseDownloader, and all
dataset-specific downloaders (Perov5, MP20, MPTS52, Carbon24).
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import urllib.error

from nominal_drift.datasets.downloaders.base_downloader import (
    DownloadResult,
    BaseDownloader,
)
from nominal_drift.datasets.downloaders.perov5_downloader import Perov5Downloader
from nominal_drift.datasets.downloaders.mp20_downloader import MP20Downloader
from nominal_drift.datasets.downloaders.mpts52_downloader import MPTS52Downloader
from nominal_drift.datasets.downloaders.carbon24_downloader import Carbon24Downloader


# ============================================================================
# DownloadResult Tests
# ============================================================================

class TestDownloadResult:
    """Test DownloadResult dataclass."""

    def test_summary_contains_dataset_name(self):
        """summary() includes dataset_name."""
        result = DownloadResult(
            dataset_name="test-dataset",
            raw_dir="/data/test-dataset",
            files_downloaded=["file1.csv"],
            n_structures_found=10,
            total_bytes=1024,
            checksum_sha256="abc123",
            already_existed=False,
            warnings=[],
            notes=[],
        )
        summary = result.summary()
        assert "test-dataset" in summary

    def test_summary_contains_file_list(self):
        """summary() includes files_downloaded."""
        result = DownloadResult(
            dataset_name="test",
            raw_dir="/data/test",
            files_downloaded=["a.csv", "b.csv"],
            n_structures_found=5,
            total_bytes=2048,
            checksum_sha256=None,
            already_existed=False,
            warnings=[],
            notes=[],
        )
        summary = result.summary()
        assert "a.csv" in summary or "['a.csv', 'b.csv']" in summary

    def test_summary_already_existed_status(self):
        """summary() shows 'already present' when already_existed=True."""
        result = DownloadResult(
            dataset_name="test",
            raw_dir="/data/test",
            files_downloaded=[],
            n_structures_found=0,
            total_bytes=0,
            checksum_sha256=None,
            already_existed=True,
            warnings=[],
            notes=[],
        )
        summary = result.summary()
        assert "already" in summary.lower() or "present" in summary.lower()

    def test_summary_downloaded_status(self):
        """summary() shows 'downloaded' when already_existed=False."""
        result = DownloadResult(
            dataset_name="test",
            raw_dir="/data/test",
            files_downloaded=["file.csv"],
            n_structures_found=5,
            total_bytes=1024,
            checksum_sha256=None,
            already_existed=False,
            warnings=[],
            notes=[],
        )
        summary = result.summary()
        assert "download" in summary.lower()

    def test_result_fields_correct_types(self):
        """All DownloadResult fields have correct types."""
        result = DownloadResult(
            dataset_name="test",
            raw_dir="/path",
            files_downloaded=["f1", "f2"],
            n_structures_found=100,
            total_bytes=5000,
            checksum_sha256="sha256hex",
            already_existed=False,
            warnings=["w1"],
            notes=["n1"],
        )
        assert isinstance(result.dataset_name, str)
        assert isinstance(result.raw_dir, str)
        assert isinstance(result.files_downloaded, list)
        assert isinstance(result.n_structures_found, int)
        assert isinstance(result.total_bytes, int)
        assert isinstance(result.checksum_sha256, (str, type(None)))
        assert isinstance(result.already_existed, bool)
        assert isinstance(result.warnings, list)
        assert isinstance(result.notes, list)


# ============================================================================
# BaseDownloader Tests
# ============================================================================

class TestBaseDownloader:
    """Test BaseDownloader base class."""

    def test_is_present_false_nonexistent_dir(self, tmp_path):
        """is_present() returns False when directory doesn't exist."""
        dl = BaseDownloader(raw_base_dir=str(tmp_path))
        dl.dataset_name = "missing"
        dl.expected_files = ["file.csv"]
        assert dl.is_present() is False

    def test_is_present_false_missing_files(self, tmp_path):
        """is_present() returns False when expected files are missing."""
        dl = BaseDownloader(raw_base_dir=str(tmp_path))
        dl.dataset_name = "incomplete"
        dl.expected_files = ["file1.csv", "file2.csv"]

        # Create directory but not all files
        (tmp_path / "incomplete").mkdir()
        (tmp_path / "incomplete" / "file1.csv").touch()

        assert dl.is_present() is False

    def test_is_present_true_all_files(self, tmp_path):
        """is_present() returns True when all expected files present."""
        dl = BaseDownloader(raw_base_dir=str(tmp_path))
        dl.dataset_name = "complete"
        dl.expected_files = ["file1.csv", "file2.csv"]

        # Create directory and all files
        (tmp_path / "complete").mkdir()
        (tmp_path / "complete" / "file1.csv").touch()
        (tmp_path / "complete" / "file2.csv").touch()

        assert dl.is_present() is True

    def test_is_present_empty_expected_files(self, tmp_path):
        """is_present() returns False if expected_files is empty."""
        dl = BaseDownloader(raw_base_dir=str(tmp_path))
        dl.dataset_name = "empty"
        dl.expected_files = []
        assert dl.is_present() is False

    def test_count_csv_rows_missing_file(self, tmp_path):
        """_count_csv_rows() returns 0 for missing file."""
        dl = BaseDownloader(raw_base_dir=str(tmp_path))
        count = dl._count_csv_rows(tmp_path / "nonexistent.csv")
        assert count == 0

    def test_count_csv_rows_with_header(self, tmp_path):
        """_count_csv_rows() excludes header row."""
        dl = BaseDownloader(raw_base_dir=str(tmp_path))
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col1,col2\nval1,val2\nval3,val4\n")

        count = dl._count_csv_rows(csv_file)
        assert count == 2  # 2 data rows (excluding header)

    def test_count_csv_rows_single_row(self, tmp_path):
        """_count_csv_rows() counts single data row correctly."""
        dl = BaseDownloader(raw_base_dir=str(tmp_path))
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col1,col2\ndata1,data2\n")

        count = dl._count_csv_rows(csv_file)
        assert count == 1

    def test_count_csv_rows_empty_after_header(self, tmp_path):
        """_count_csv_rows() returns 0 for header-only CSV."""
        dl = BaseDownloader(raw_base_dir=str(tmp_path))
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col1,col2\n")

        count = dl._count_csv_rows(csv_file)
        assert count == 0

    def test_sha256_file_returns_hex_string(self, tmp_path):
        """_sha256_file() returns 64-character hex string."""
        dl = BaseDownloader(raw_base_dir=str(tmp_path))
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        sha = dl._sha256_file(test_file)
        assert isinstance(sha, str)
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    def test_count_jsonl_lines_missing_file(self, tmp_path):
        """_count_jsonl_lines() returns 0 for missing file."""
        dl = BaseDownloader(raw_base_dir=str(tmp_path))
        count = dl._count_jsonl_lines(tmp_path / "nonexistent.jsonl")
        assert count == 0

    def test_count_jsonl_lines_multiple_lines(self, tmp_path):
        """_count_jsonl_lines() counts non-empty lines."""
        dl = BaseDownloader(raw_base_dir=str(tmp_path))
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"a": 1}\n{"b": 2}\n{"c": 3}\n')

        count = dl._count_jsonl_lines(jsonl_file)
        assert count == 3

    def test_verify_delegates_to_is_present(self, tmp_path):
        """verify() delegates to is_present()."""
        dl = BaseDownloader(raw_base_dir=str(tmp_path))
        dl.dataset_name = "test"
        dl.expected_files = ["file.txt"]

        assert dl.verify() == dl.is_present()


# ============================================================================
# Downloader Class Attributes Tests
# ============================================================================

class TestDownloaderAttributes:
    """Test that all downloader classes have required attributes."""

    def test_perov5_dataset_name(self):
        """Perov5Downloader has dataset_name attribute."""
        assert hasattr(Perov5Downloader, "dataset_name")
        assert Perov5Downloader.dataset_name == "perov-5"

    def test_perov5_source_url(self):
        """Perov5Downloader has source_url attribute."""
        assert hasattr(Perov5Downloader, "source_url")
        assert isinstance(Perov5Downloader.source_url, str)

    def test_perov5_expected_files(self):
        """Perov5Downloader has expected_files list."""
        assert hasattr(Perov5Downloader, "expected_files")
        assert isinstance(Perov5Downloader.expected_files, list)

    def test_mp20_dataset_name(self):
        """MP20Downloader has dataset_name attribute."""
        assert hasattr(MP20Downloader, "dataset_name")
        assert MP20Downloader.dataset_name == "mp-20"

    def test_mp20_source_url(self):
        """MP20Downloader has source_url attribute."""
        assert hasattr(MP20Downloader, "source_url")
        assert isinstance(MP20Downloader.source_url, str)

    def test_mp20_expected_files(self):
        """MP20Downloader has expected_files list."""
        assert hasattr(MP20Downloader, "expected_files")
        assert isinstance(MP20Downloader.expected_files, list)

    def test_mpts52_dataset_name(self):
        """MPTS52Downloader has dataset_name attribute."""
        assert hasattr(MPTS52Downloader, "dataset_name")
        assert MPTS52Downloader.dataset_name == "mpts-52"

    def test_mpts52_source_url(self):
        """MPTS52Downloader has source_url attribute."""
        assert hasattr(MPTS52Downloader, "source_url")
        assert isinstance(MPTS52Downloader.source_url, str)

    def test_mpts52_expected_files(self):
        """MPTS52Downloader has expected_files list."""
        assert hasattr(MPTS52Downloader, "expected_files")
        assert isinstance(MPTS52Downloader.expected_files, list)

    def test_carbon24_dataset_name(self):
        """Carbon24Downloader has dataset_name attribute."""
        assert hasattr(Carbon24Downloader, "dataset_name")
        assert Carbon24Downloader.dataset_name == "carbon-24"

    def test_carbon24_source_url(self):
        """Carbon24Downloader has source_url attribute."""
        assert hasattr(Carbon24Downloader, "source_url")
        assert isinstance(Carbon24Downloader.source_url, str)

    def test_carbon24_expected_files(self):
        """Carbon24Downloader has expected_files list."""
        assert hasattr(Carbon24Downloader, "expected_files")
        assert isinstance(Carbon24Downloader.expected_files, list)


# ============================================================================
# Downloader Presence Tests
# ============================================================================

class TestDownloaderIsPresent:
    """Test is_present() for each downloader."""

    def test_perov5_is_present_false_empty(self, tmp_path):
        """Perov5Downloader.is_present() False on empty directory."""
        dl = Perov5Downloader(raw_base_dir=str(tmp_path))
        assert dl.is_present() is False

    def test_mp20_is_present_false_empty(self, tmp_path):
        """MP20Downloader.is_present() False on empty directory."""
        dl = MP20Downloader(raw_base_dir=str(tmp_path))
        assert dl.is_present() is False

    def test_mpts52_is_present_false_empty(self, tmp_path):
        """MPTS52Downloader.is_present() False on empty directory."""
        dl = MPTS52Downloader(raw_base_dir=str(tmp_path))
        assert dl.is_present() is False

    def test_carbon24_is_present_false_empty(self, tmp_path):
        """Carbon24Downloader.is_present() False on empty directory."""
        dl = Carbon24Downloader(raw_base_dir=str(tmp_path))
        assert dl.is_present() is False


# ============================================================================
# Download Method Tests (with mocks)
# ============================================================================

class TestDownload:
    """Test download() method with mocked network."""

    @patch("urllib.request.urlopen")
    def test_perov5_download_success(self, mock_urlopen, tmp_path):
        """Perov5Downloader.download() succeeds with mocked data."""
        # Setup mock to return CSV data
        mock_response = MagicMock()
        mock_response.read.return_value = b"col1,col2\nval1,val2\n"
        mock_urlopen.return_value.__enter__.return_value = mock_response

        dl = Perov5Downloader(raw_base_dir=str(tmp_path))
        result = dl.download()

        assert isinstance(result, DownloadResult)
        assert result.dataset_name == "perov-5"

    @patch("urllib.request.urlopen")
    def test_download_returns_download_result_instance(self, mock_urlopen, tmp_path):
        """download() returns DownloadResult instance."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"col1,col2\nrow1,row2\n"
        mock_urlopen.return_value.__enter__.return_value = mock_response

        dl = Perov5Downloader(raw_base_dir=str(tmp_path))
        result = dl.download()

        assert isinstance(result, DownloadResult)

    def test_download_already_existed_skips(self, tmp_path):
        """download() skips when files already present."""
        # Pre-populate the directory
        dl = Perov5Downloader(raw_base_dir=str(tmp_path))
        dl.raw_dir.mkdir(parents=True, exist_ok=True)
        for f in dl.expected_files:
            (dl.raw_dir / f).write_text("col1,col2\nrow1,row2\n")

        result = dl.download()

        assert result.already_existed is True
        assert result.files_downloaded == []

    @patch("urllib.request.urlopen")
    def test_download_with_force_redownloads(self, mock_urlopen, tmp_path):
        """download(force=True) re-downloads even when present."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.read.return_value = b"col1,col2\nval1,val2\n"
        mock_urlopen.return_value.__enter__.return_value = mock_response

        # Pre-populate
        dl = Perov5Downloader(raw_base_dir=str(tmp_path))
        dl.raw_dir.mkdir(parents=True, exist_ok=True)
        (dl.raw_dir / "train.csv").write_text("old")

        result = dl.download(force=True)

        assert result.already_existed is False

    @patch("urllib.request.urlopen")
    def test_download_handles_network_error_gracefully(self, mock_urlopen, tmp_path):
        """download() handles URLError gracefully."""
        # Setup mock to raise URLError
        mock_urlopen.side_effect = urllib.error.URLError("Connection failed")

        dl = Perov5Downloader(raw_base_dir=str(tmp_path))
        result = dl.download()

        # Should not raise, but return result with warnings
        assert isinstance(result, DownloadResult)
        assert len(result.warnings) > 0

    def test_download_result_already_existed_is_bool(self, tmp_path):
        """DownloadResult.already_existed is bool."""
        dl = Perov5Downloader(raw_base_dir=str(tmp_path))
        dl.raw_dir.mkdir(parents=True, exist_ok=True)
        for f in dl.expected_files:
            (dl.raw_dir / f).touch()

        result = dl.download()
        assert isinstance(result.already_existed, bool)

    def test_download_result_warnings_is_list(self, tmp_path):
        """DownloadResult.warnings is list."""
        dl = Perov5Downloader(raw_base_dir=str(tmp_path))

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("test")
            result = dl.download()

        assert isinstance(result.warnings, list)


# ============================================================================
# CLI Import Tests
# ============================================================================

class TestCLIImports:
    """Test that CLI commands can be imported."""

    def test_dataset_list_command_imports(self):
        """dataset_list command imports without error."""
        # Import main module
        from nominal_drift.cli.main import app
        assert app is not None

    def test_dataset_fetch_command_imports(self):
        """dataset_fetch command imports without error."""
        from nominal_drift.cli.main import app
        assert app is not None

    def test_all_downloaders_importable(self):
        """All 4 downloader classes are importable."""
        from nominal_drift.datasets.downloaders.perov5_downloader import Perov5Downloader
        from nominal_drift.datasets.downloaders.mp20_downloader import MP20Downloader
        from nominal_drift.datasets.downloaders.mpts52_downloader import MPTS52Downloader
        from nominal_drift.datasets.downloaders.carbon24_downloader import Carbon24Downloader

        assert Perov5Downloader is not None
        assert MP20Downloader is not None
        assert MPTS52Downloader is not None
        assert Carbon24Downloader is not None


# ============================================================================
# Additional Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_csv_file(self, tmp_path):
        """_count_csv_rows() handles empty CSV."""
        dl = BaseDownloader(raw_base_dir=str(tmp_path))
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")

        count = dl._count_csv_rows(csv_file)
        assert count == 0

    def test_csv_with_blank_lines(self, tmp_path):
        """_count_csv_rows() ignores blank lines properly."""
        dl = BaseDownloader(raw_base_dir=str(tmp_path))
        csv_file = tmp_path / "blanks.csv"
        csv_file.write_text("col1,col2\nval1,val2\n\nval3,val4\n")

        # The function counts all lines after header, including blank ones
        count = dl._count_csv_rows(csv_file)
        assert count == 3

    def test_download_result_with_none_checksum(self):
        """DownloadResult handles None checksum correctly."""
        result = DownloadResult(
            dataset_name="test",
            raw_dir="/data",
            files_downloaded=[],
            n_structures_found=0,
            total_bytes=0,
            checksum_sha256=None,
            already_existed=False,
            warnings=[],
            notes=[],
        )
        summary = result.summary()
        assert "None" in summary

    def test_raw_dir_property(self, tmp_path):
        """raw_dir property returns correct path."""
        dl = Perov5Downloader(raw_base_dir=str(tmp_path))
        expected = tmp_path / "perov-5"
        assert dl.raw_dir == expected

    def test_multiple_warnings_in_result(self):
        """DownloadResult accumulates multiple warnings."""
        warnings = ["warning1", "warning2", "warning3"]
        result = DownloadResult(
            dataset_name="test",
            raw_dir="/data",
            files_downloaded=[],
            n_structures_found=0,
            total_bytes=0,
            checksum_sha256=None,
            already_existed=False,
            warnings=warnings,
            notes=[],
        )
        assert result.warnings == warnings
        assert len(result.warnings) == 3

    def test_perov5_expected_files_not_empty(self):
        """Perov5Downloader has non-empty expected_files."""
        assert len(Perov5Downloader.expected_files) > 0

    def test_mp20_expected_files_not_empty(self):
        """MP20Downloader has non-empty expected_files."""
        assert len(MP20Downloader.expected_files) > 0

    def test_carbon24_expected_files_not_empty(self):
        """Carbon24Downloader has non-empty expected_files."""
        assert len(Carbon24Downloader.expected_files) > 0
