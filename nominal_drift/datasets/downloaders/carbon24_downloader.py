"""
Downloader for the Carbon-24 dataset (CDVAE).
https://github.com/txie-93/cdvae
"""
from __future__ import annotations
from pathlib import Path
import urllib.error
import urllib.request

from nominal_drift.datasets.downloaders.base_downloader import (
    BaseDownloader,
    DownloadResult,
)


class Carbon24Downloader(BaseDownloader):
    """Download Carbon-24 dataset from CDVAE repo."""

    dataset_name = "carbon-24"
    source_url = "https://raw.githubusercontent.com/txie-93/cdvae/main/data/carbon_24/"
    expected_files = ["train.csv", "val.csv", "test.csv"]

    def download(self, force: bool = False) -> DownloadResult:
        """Download Carbon-24 dataset."""

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

        # Download each file
        for filename in self.expected_files:
            url = f"{self.source_url}{filename}"
            dest = self.raw_dir / filename

            try:
                bytes_written = self._download_file(url, dest)
                files_downloaded.append(filename)
                total_bytes += bytes_written
            except (urllib.error.URLError, TimeoutError, Exception) as exc:
                warnings.append(f"Failed to download {filename}: {exc}")
                # Continue trying other files

        # If no files were downloaded, write a placeholder info file
        if not files_downloaded:
            info_path = self.raw_dir / "info.txt"
            info_path.write_text(
                f"Carbon-24 dataset download failed.\n"
                f"Please manually download from:\n"
                f"{self.source_url}\n"
                f"Expected files: {', '.join(self.expected_files)}\n"
            )

        # Count structures from downloaded CSVs
        n_structures = sum(
            self._count_csv_rows(self.raw_dir / f)
            for f in files_downloaded
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
