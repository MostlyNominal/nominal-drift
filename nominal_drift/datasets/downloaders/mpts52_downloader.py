"""
Downloader for the MPTS-52 dataset.
https://github.com/ml-evs/mpts
"""
from __future__ import annotations
from pathlib import Path
import urllib.error
import urllib.request

from nominal_drift.datasets.downloaders.base_downloader import (
    BaseDownloader,
    DownloadResult,
)


class MPTS52Downloader(BaseDownloader):
    """Download MPTS-52 dataset from ml-evs repo."""

    dataset_name = "mpts-52"
    source_url = "https://raw.githubusercontent.com/ml-evs/mpts/master/data/mpts_52/"
    expected_files = ["train.csv"]

    def download(self, force: bool = False) -> DownloadResult:
        """Download MPTS-52 dataset."""

        # Check if already present
        if self.is_present() and not force:
            total_rows = sum(
                self._count_csv_rows(self.raw_dir / f)
                for f in self.expected_files
            )
            return DownloadResult(
                dataset_name=self.dataset_name,
                raw_dir=str(self.raw_dir.absolute()),
                files_downloaded=[],
                n_structures_found=total_rows,
                total_bytes=0,
                checksum_sha256=None,
                already_existed=True,
                warnings=[],
                notes=["Dataset already present, skipping download"],
            )

        # Create raw directory
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        files_downloaded = []
        total_bytes = 0
        warnings = []

        # Try to download primary file (train.csv)
        primary_url = f"{self.source_url}train.csv"
        dest = self.raw_dir / "train.csv"

        try:
            bytes_written = self._download_file(primary_url, dest)
            files_downloaded.append("train.csv")
            total_bytes += bytes_written
        except (urllib.error.URLError, TimeoutError, Exception) as exc:
            warnings.append(
                f"MPTS-52 primary URL not available: {exc}. "
                "Place data manually in the raw directory."
            )

        # Try optional files if primary succeeded
        if files_downloaded:
            for optional_file in ["val.csv", "test.csv"]:
                url = f"{self.source_url}{optional_file}"
                dest = self.raw_dir / optional_file
                try:
                    bytes_written = self._download_file(url, dest)
                    files_downloaded.append(optional_file)
                    total_bytes += bytes_written
                except (urllib.error.URLError, TimeoutError, Exception):
                    # Optional files are not critical
                    pass

        # If no files were downloaded, write a placeholder info file
        if not files_downloaded:
            info_path = self.raw_dir / "info.txt"
            info_path.write_text(
                f"MPTS-52 dataset download failed.\n"
                f"Please manually download from:\n"
                f"{self.source_url}\n"
                f"Expected files: {', '.join(self.expected_files)}\n"
            )

        # Count structures from downloaded CSVs
        n_structures = sum(
            self._count_csv_rows(self.raw_dir / f)
            for f in files_downloaded
            if f.endswith(".csv")
        )

        return DownloadResult(
            dataset_name=self.dataset_name,
            raw_dir=str(self.raw_dir.absolute()),
            files_downloaded=files_downloaded,
            n_structures_found=n_structures,
            total_bytes=total_bytes,
            checksum_sha256=None,
            already_existed=False,
            warnings=warnings,
            notes=[],
        )
