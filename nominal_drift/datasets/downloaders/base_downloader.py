"""
Base classes and shared utilities for Nominal Drift dataset downloaders.
"""
from __future__ import annotations
import hashlib
import json
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import urllib.request


@dataclass
class DownloadResult:
    dataset_name: str
    raw_dir: str                  # absolute path where raw files landed
    files_downloaded: list[str]   # relative file names written
    n_structures_found: int       # count parsed from manifest / file scan
    total_bytes: int              # total bytes downloaded
    checksum_sha256: str | None   # sha256 hex of the primary archive, or None
    already_existed: bool         # True if skipped because data was present
    warnings: list[str]
    notes: list[str]

    def summary(self) -> str:
        status = "skipped (already present)" if self.already_existed else "downloaded"
        return (
            f"[{self.dataset_name}] {status} → {self.raw_dir}\n"
            f"  files: {self.files_downloaded}\n"
            f"  structures: {self.n_structures_found}\n"
            f"  bytes: {self.total_bytes:,}\n"
            f"  sha256: {self.checksum_sha256}\n"
            f"  warnings: {self.warnings}"
        )


class BaseDownloader:
    """Abstract base for dataset downloaders."""
    dataset_name: str = ""
    source_url: str = ""
    expected_files: list[str] = []

    def __init__(self, raw_base_dir: str = "data/datasets/raw"):
        self.raw_base_dir = Path(raw_base_dir)

    @property
    def raw_dir(self) -> Path:
        return self.raw_base_dir / self.dataset_name

    def is_present(self) -> bool:
        """Return True if the expected files are already present."""
        if not self.raw_dir.exists():
            return False
        for f in self.expected_files:
            if not (self.raw_dir / f).exists():
                return False
        return bool(self.expected_files)

    def download(self, force: bool = False) -> DownloadResult:
        raise NotImplementedError

    def verify(self) -> bool:
        """Check expected files are present."""
        return self.is_present()

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _sha256_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _download_file(self, url: str, dest: Path, desc: str = "") -> int:
        """Download url → dest, return bytes written."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return len(data)

    def _extract_archive(self, archive_path: Path, dest_dir: Path) -> list[str]:
        """Extract tar.gz or zip. Returns list of extracted names."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        extracted = []
        if archive_path.name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(dest_dir)
                extracted = tf.getnames()
        elif archive_path.name.endswith(".zip"):
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(dest_dir)
                extracted = zf.namelist()
        return extracted

    def _count_jsonl_lines(self, path: Path) -> int:
        if not path.exists():
            return 0
        count = 0
        with open(path) as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def _count_json_list(self, path: Path) -> int:
        if not path.exists():
            return 0
        try:
            data = json.loads(path.read_text())
            return len(data) if isinstance(data, list) else 0
        except Exception:
            return 0

    def _count_csv_rows(self, path: Path) -> int:
        """Count data rows in CSV (excluding header). Return 0 if file missing."""
        if not path.exists():
            return 0
        count = 0
        with open(path, newline="") as f:
            for i, _ in enumerate(f):
                if i > 0:  # skip header
                    count += 1
        return count
