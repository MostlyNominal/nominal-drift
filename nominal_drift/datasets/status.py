"""
nominal_drift.datasets.status
==============================
Dataset status inspection — real paths, real counts, honest reporting.

This module answers the question "what is actually on disk?" without
pretending that data is present when it is not.

Public API
----------
``DATASET_REGISTRY``
    Dict mapping dataset name → ``DatasetInfo`` (static metadata).

``DatasetStatus``
    Dataclass describing the on-disk state of one dataset.

``get_dataset_status(name, raw_base, norm_base)``
    Return a ``DatasetStatus`` for the named dataset.

``get_all_statuses(raw_base, norm_base)``
    Return ``dict[name, DatasetStatus]`` for all registered datasets.

``DatasetInfo``
    Static metadata: expected files, source URL, manual instructions, etc.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Static dataset registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetInfo:
    """Static metadata for one registered dataset."""
    name: str
    description: str
    source_url: str
    expected_raw_files: tuple[str, ...]
    license: str
    citation: str
    auto_downloadable: bool
    manual_instructions: str


DATASET_REGISTRY: dict[str, DatasetInfo] = {
    "perov-5": DatasetInfo(
        name="perov-5",
        description=(
            "18,928 perovskite ABX₃ structures from CDVAE benchmark. "
            "Contains formation energies and atomic sites."
        ),
        source_url=(
            "https://raw.githubusercontent.com/txie-93/cdvae/main/data/perov_5/"
        ),
        expected_raw_files=("train.csv", "val.csv", "test.csv"),
        license="MIT (CDVAE repo)",
        citation=(
            "Xie, T. et al. Crystal Diffusion Variational Autoencoder for "
            "Periodic Material Generation. ICLR 2022."
        ),
        auto_downloadable=True,
        manual_instructions=(
            "If auto-download fails, manually download these 3 files from:\n"
            "  https://raw.githubusercontent.com/txie-93/cdvae/main/data/perov_5/train.csv\n"
            "  https://raw.githubusercontent.com/txie-93/cdvae/main/data/perov_5/val.csv\n"
            "  https://raw.githubusercontent.com/txie-93/cdvae/main/data/perov_5/test.csv\n"
            "Place them at: data/datasets/raw/perov-5/"
        ),
    ),
    "mp-20": DatasetInfo(
        name="mp-20",
        description=(
            "~45,000 Materials Project structures from CDVAE benchmark. "
            "Diverse inorganic crystals; includes formation energy."
        ),
        source_url=(
            "https://raw.githubusercontent.com/txie-93/cdvae/main/data/mp_20/"
        ),
        expected_raw_files=("train.csv", "val.csv", "test.csv"),
        license="Creative Commons (Materials Project data)",
        citation=(
            "Xie, T. et al. Crystal Diffusion Variational Autoencoder for "
            "Periodic Material Generation. ICLR 2022. "
            "Data from: Jain, A. et al. APL Materials 2013."
        ),
        auto_downloadable=True,
        manual_instructions=(
            "If auto-download fails, manually download these 3 files from:\n"
            "  https://raw.githubusercontent.com/txie-93/cdvae/main/data/mp_20/train.csv\n"
            "  https://raw.githubusercontent.com/txie-93/cdvae/main/data/mp_20/val.csv\n"
            "  https://raw.githubusercontent.com/txie-93/cdvae/main/data/mp_20/test.csv\n"
            "Place them at: data/datasets/raw/mp-20/"
        ),
    ),
    "carbon-24": DatasetInfo(
        name="carbon-24",
        description=(
            "10,153 carbon allotropes (graphene, diamond, fullerenes, etc.) "
            "from CDVAE benchmark. Contains energy above hull."
        ),
        source_url=(
            "https://raw.githubusercontent.com/txie-93/cdvae/main/data/carbon_24/"
        ),
        expected_raw_files=("train.csv", "val.csv", "test.csv"),
        license="MIT (CDVAE repo)",
        citation=(
            "Xie, T. et al. Crystal Diffusion Variational Autoencoder for "
            "Periodic Material Generation. ICLR 2022."
        ),
        auto_downloadable=True,
        manual_instructions=(
            "If auto-download fails, manually download these 3 files from:\n"
            "  https://raw.githubusercontent.com/txie-93/cdvae/main/data/carbon_24/train.csv\n"
            "  https://raw.githubusercontent.com/txie-93/cdvae/main/data/carbon_24/val.csv\n"
            "  https://raw.githubusercontent.com/txie-93/cdvae/main/data/carbon_24/test.csv\n"
            "Place them at: data/datasets/raw/carbon-24/"
        ),
    ),
    "mpts-52": DatasetInfo(
        name="mpts-52",
        description=(
            "~40,000 Materials Project structures with temporal train/val/test "
            "splits (MPTS-52 / MPTimeSplit benchmark). Structures sorted by "
            "earliest literature publication year; 5 × TimeSeriesSplit CV folds "
            "plus a held-out test split. Up to 52 atoms per cell. "
            "Downloaded automatically from figshare via mp-time-split."
        ),
        source_url="https://figshare.com/articles/dataset/Materials_Project_Time_Split_Data/19991516",
        expected_raw_files=(),   # no CSV files — figshare snapshot via mp-time-split
        license="Creative Commons (Materials Project data)",
        citation=(
            "Baird, S.G. et al. Matbench Discovery and the MPTS-52 benchmark. "
            "Figshare dataset: https://doi.org/10.6084/m9.figshare.19991516. "
            "mp-time-split: https://github.com/sparks-baird/mp-time-split. "
            "Materials Project: Jain, A. et al. APL Materials 2013."
        ),
        auto_downloadable=True,
        manual_instructions=(
            "MPTS-52 is downloaded automatically via mp-time-split from figshare.\n"
            "Run the normalisation step in the Dataset page, or from the CLI:\n"
            "  python -m nominal_drift.datasets.ingest --name mpts-52\n\n"
            "This downloads ~150 MB from:\n"
            f"  https://figshare.com/ndownloader/files/35592011\n\n"
            "If the download fails, ensure mp-time-split is installed:\n"
            "  pip install 'mp-time-split<0.2'"
        ),
    ),
}

# ---------------------------------------------------------------------------
# On-disk status
# ---------------------------------------------------------------------------

@dataclass
class DatasetStatus:
    """On-disk status of one dataset — always reflects reality."""
    name: str
    info: DatasetInfo

    # Raw layer
    raw_dir: Path
    raw_files_present: list[str]
    raw_files_missing: list[str]
    raw_row_counts: dict[str, int]   # filename → row count

    # Normalised layer
    norm_dir: Path
    norm_jsonl_present: bool
    norm_structure_count: int        # 0 if not normalised
    norm_elements: list[str]         # elements seen, empty if not normalised
    norm_manifest: Optional[dict]    # parsed manifest.json, or None

    @property
    def is_raw_complete(self) -> bool:
        # Datasets with no expected raw files (e.g. MPTS-52 which uses a
        # figshare snapshot via mp-time-split) are considered "raw complete"
        # once they have been normalised (the snapshot is the canonical source).
        if not self.info.expected_raw_files:
            return self.is_normalised
        return len(self.raw_files_missing) == 0 and bool(self.raw_files_present)

    @property
    def is_normalised(self) -> bool:
        return self.norm_jsonl_present and self.norm_structure_count > 0

    @property
    def total_raw_rows(self) -> int:
        return sum(self.raw_row_counts.values())

    @property
    def download_status_label(self) -> str:
        if self.is_normalised:
            return f"✅ normalised ({self.norm_structure_count:,} structures)"
        # Figshare/MPTimeSplit datasets have no CSV raw files
        if not self.info.expected_raw_files:
            return "❌ not downloaded — click Normalise to fetch from figshare"
        if self.is_raw_complete:
            return f"📥 raw present ({self.total_raw_rows:,} rows) — not normalised"
        if self.raw_files_present:
            missing = ", ".join(self.raw_files_missing)
            return f"⚠️ partial ({len(self.raw_files_present)}/{len(self.info.expected_raw_files)} files; missing: {missing})"
        return "❌ not downloaded"


# ---------------------------------------------------------------------------
# Inspector
# ---------------------------------------------------------------------------

def _count_csv_rows(path: Path) -> int:
    """Count data rows in CSV (excluding header). Returns 0 if file missing."""
    if not path.exists():
        return 0
    count = 0
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for i, _ in enumerate(fh):
            if i > 0:
                count += 1
    return count


def _count_jsonl_lines(path: Path) -> int:
    """Count non-empty lines in a .jsonl file."""
    if not path.exists():
        return 0
    count = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def _read_manifest(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def get_dataset_status(
    name: str,
    raw_base: str | Path = "data/datasets/raw",
    norm_base: str | Path = "data/datasets/normalized",
) -> DatasetStatus:
    """Inspect on-disk state of dataset *name*.

    Parameters
    ----------
    name : str
        Dataset name (must be a key in ``DATASET_REGISTRY``).
    raw_base : str | Path
        Base directory for raw downloads.
    norm_base : str | Path
        Base directory for normalised outputs.

    Raises
    ------
    KeyError
        If *name* is not in ``DATASET_REGISTRY``.
    """
    info = DATASET_REGISTRY[name]
    raw_dir = Path(raw_base) / name
    norm_dir = Path(norm_base) / name

    # --- Raw layer ---
    present = []
    missing = []
    row_counts: dict[str, int] = {}
    for fname in info.expected_raw_files:
        fpath = raw_dir / fname
        if fpath.exists():
            present.append(fname)
            row_counts[fname] = _count_csv_rows(fpath)
        else:
            missing.append(fname)

    # --- Normalised layer ---
    jsonl_path = norm_dir / "structures.jsonl"
    manifest_path = norm_dir / "manifest.json"
    manifest = _read_manifest(manifest_path)
    norm_count = _count_jsonl_lines(jsonl_path)
    norm_elements: list[str] = []
    if manifest:
        norm_elements = sorted(manifest.get("elements_present", []))

    return DatasetStatus(
        name=name,
        info=info,
        raw_dir=raw_dir,
        raw_files_present=present,
        raw_files_missing=missing,
        raw_row_counts=row_counts,
        norm_dir=norm_dir,
        norm_jsonl_present=jsonl_path.exists(),
        norm_structure_count=norm_count,
        norm_elements=norm_elements,
        norm_manifest=manifest,
    )


def get_all_statuses(
    raw_base: str | Path = "data/datasets/raw",
    norm_base: str | Path = "data/datasets/normalized",
) -> dict[str, DatasetStatus]:
    """Return status for every registered dataset."""
    return {
        name: get_dataset_status(name, raw_base, norm_base)
        for name in DATASET_REGISTRY
    }
