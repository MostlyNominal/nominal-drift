"""
Downloader for MPTS-52 via mp-time-split (figshare snapshot).

MPTS-52 is NOT distributed as CSV files.  The authoritative source is a
compressed JSON snapshot on figshare, fetched by ``MPTimeSplit.load()``.
This downloader delegates to ``ingest_mpts52_via_mp_time_split()`` so the
GUI's Download button still works for completeness, but the real logic is
that downloading and normalising are one atomic step.
"""
from __future__ import annotations

from nominal_drift.datasets.downloaders.base_downloader import (
    BaseDownloader,
    DownloadResult,
)


class MPTS52Downloader(BaseDownloader):
    """Download + normalise MPTS-52 via figshare / mp-time-split.

    Unlike the CSV-based datasets, MPTS-52 has no separate raw CSV layer.
    Calling ``download()`` invokes ``ingest_mpts52_via_mp_time_split()``
    which uses ``MPTimeSplit.load()`` to fetch the figshare snapshot
    (~150 MB) and writes ``structures.jsonl`` directly.
    """

    dataset_name = "mpts-52"
    source_url = "https://figshare.com/articles/dataset/Materials_Project_Time_Split_Data/19991516"
    expected_files: list[str] = []   # no CSV raw files

    def download(self, force: bool = False) -> DownloadResult:
        """Fetch MPTS-52 from figshare via mp-time-split.

        Parameters
        ----------
        force : bool
            Re-download even if the normalised snapshot already exists.
        """
        from pathlib import Path
        from nominal_drift.datasets.ingest import ingest_mpts52_via_mp_time_split

        # Derive norm_base from raw_base (../normalized)
        norm_base = Path(self.raw_dir).parent.parent / "normalized"

        try:
            result = ingest_mpts52_via_mp_time_split(
                norm_base=norm_base,
                force_download=force,
                verbose=False,
            )
            if result.n_ok > 0:
                return DownloadResult(
                    dataset_name=self.dataset_name,
                    raw_dir=str(self.raw_dir),
                    files_downloaded=["structures.jsonl"],
                    n_structures_found=result.n_ok,
                    total_bytes=0,
                    checksum_sha256=None,
                    already_existed=(result.n_ok > 0 and not force),
                    warnings=[],
                    notes=[
                        f"Downloaded from figshare via mp-time-split. "
                        f"Normalised {result.n_ok:,} structures in "
                        f"{result.elapsed_s:.1f}s."
                    ],
                )
            else:
                errs = "; ".join(result.error_samples[:2]) if result.error_samples else "unknown"
                return DownloadResult(
                    dataset_name=self.dataset_name,
                    raw_dir=str(self.raw_dir),
                    files_downloaded=[],
                    n_structures_found=0,
                    total_bytes=0,
                    checksum_sha256=None,
                    already_existed=False,
                    warnings=[f"Ingestion failed: {errs}"],
                    notes=[],
                )
        except Exception as exc:
            return DownloadResult(
                dataset_name=self.dataset_name,
                raw_dir=str(self.raw_dir),
                files_downloaded=[],
                n_structures_found=0,
                total_bytes=0,
                checksum_sha256=None,
                already_existed=False,
                warnings=[
                    f"mp-time-split download failed: {exc}. "
                    "Ensure mp-time-split is installed: "
                    "pip install 'mp-time-split<0.2'"
                ],
                notes=[],
            )

    def is_present(self) -> bool:
        """MPTS-52 is 'present' when the normalised JSONL exists."""
        from pathlib import Path
        norm_base = Path(self.raw_dir).parent.parent / "normalized"
        jsonl = norm_base / "mpts-52" / "structures.jsonl"
        return jsonl.exists() and jsonl.stat().st_size > 0
