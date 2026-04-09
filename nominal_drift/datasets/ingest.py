"""
nominal_drift.datasets.ingest
==============================
End-to-end ingestion pipeline: raw CSV → CIF → pymatgen → CrystalRecord
→ structures.jsonl + manifest.json.

This module is the single entry point for converting raw downloaded files
into the normalised Lane B format.  It is resilient: rows that fail CIF
parsing are counted and reported rather than crashing the whole run.

Usage (programmatic)
---------------------
>>> from nominal_drift.datasets.ingest import ingest_dataset
>>> result = ingest_dataset("perov-5")
>>> print(result)
IngestResult(n_ok=18928, n_err=0, ...)

Usage (CLI)
-----------
    python -m nominal_drift.datasets.ingest --name perov-5
    python -m nominal_drift.datasets.ingest --name perov-5 --limit 100 --verbose

Public API
----------
``IngestResult``
    Dataclass summarising an ingestion run.

``ingest_dataset(name, raw_base, norm_base, split_override, limit, verbose)``
    Ingest one dataset end-to-end.

``ingest_all(raw_base, norm_base, verbose)``
    Ingest all registered datasets.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from nominal_drift.datasets.adapters import get_adapter, normalise_records
from nominal_drift.datasets.schema import CrystalRecord
from nominal_drift.datasets.status import DATASET_REGISTRY

# Default paths relative to project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_RAW_BASE_DEFAULT = _PROJECT_ROOT / "data" / "datasets" / "raw"
_NORM_BASE_DEFAULT = _PROJECT_ROOT / "data" / "datasets" / "normalized"


# ---------------------------------------------------------------------------
# IngestResult
# ---------------------------------------------------------------------------

@dataclass
class IngestResult:
    """Summary of one ingestion run."""

    dataset_name: str
    n_ok: int           # records successfully converted
    n_err: int          # records that failed CIF parsing
    n_total: int        # total CSV rows seen (excluding header)
    elapsed_s: float    # wall-clock seconds
    output_dir: Path    # where structures.jsonl was written
    error_samples: list[str] = field(default_factory=list)  # first N error messages

    @property
    def success_rate(self) -> float:
        if self.n_total == 0:
            return 0.0
        return self.n_ok / self.n_total

    def __str__(self) -> str:
        return (
            f"IngestResult({self.dataset_name}: "
            f"{self.n_ok:,} ok / {self.n_err:,} err / {self.n_total:,} total "
            f"in {self.elapsed_s:.1f}s → {self.output_dir})"
        )


# ---------------------------------------------------------------------------
# CSV row iterator
# ---------------------------------------------------------------------------

def _iter_csv_rows(
    csv_path: Path,
    split: str | None,
    limit: int | None = None,
) -> Iterator[tuple[int, dict]]:
    """Yield (source_index, row_dict) from a CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file.
    split : str | None
        Value to inject into the row dict under key ``"split"`` if not
        already present in the CSV.
    limit : int | None
        Maximum number of rows to yield (useful for smoke testing).
    """
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            if "split" not in row or not row["split"]:
                row["split"] = split
            yield i, row


# ---------------------------------------------------------------------------
# Core ingestion function
# ---------------------------------------------------------------------------

def ingest_dataset(
    name: str,
    raw_base: str | Path | None = None,
    norm_base: str | Path | None = None,
    split_override: str | None = None,
    limit: int | None = None,
    verbose: bool = False,
    max_error_samples: int = 5,
) -> IngestResult:
    """Ingest one dataset from raw CSV files to normalised JSONL.

    The raw directory is expected to contain one or more CSV files as
    listed in ``DATASET_REGISTRY[name].expected_raw_files``.  Each CSV
    file is ingested with the split inferred from the filename (``train``,
    ``val``, ``test``) unless *split_override* is given.

    Parameters
    ----------
    name : str
        Dataset name (key in DATASET_REGISTRY).
    raw_base : str | Path | None
        Root of raw data directory (default: ``data/datasets/raw``).
    norm_base : str | Path | None
        Root of normalised output directory (default: ``data/datasets/normalized``).
    split_override : str | None
        Force all records to this split label (useful for datasets with
        only one CSV file).
    limit : int | None
        Maximum rows per CSV file (for smoke testing; None = all rows).
    verbose : bool
        Print progress to stdout.
    max_error_samples : int
        How many error messages to capture in IngestResult.error_samples.

    Returns
    -------
    IngestResult
    """
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {name!r}")

    info = DATASET_REGISTRY[name]
    raw_base = Path(raw_base) if raw_base else _RAW_BASE_DEFAULT
    norm_base = Path(norm_base) if norm_base else _NORM_BASE_DEFAULT
    raw_dir = raw_base / name
    norm_dir = norm_base / name

    adapter = get_adapter(name)

    records: list[CrystalRecord] = []
    n_err = 0
    n_total = 0
    error_samples: list[str] = []
    t0 = time.monotonic()

    for fname in info.expected_raw_files:
        csv_path = raw_dir / fname
        if not csv_path.exists():
            if verbose:
                print(f"  [skip] {fname} — not found at {csv_path}")
            continue

        # Infer split from filename: "train.csv" → "train"
        inferred_split = fname.replace(".csv", "") if not split_override else split_override
        if inferred_split not in ("train", "val", "test"):
            inferred_split = split_override  # may be None

        if verbose:
            print(f"  [read] {fname} (split={inferred_split!r})")

        for source_index, row in _iter_csv_rows(csv_path, split=inferred_split, limit=limit):
            n_total += 1
            try:
                record = adapter.convert(row, source_index)
                records.append(record)
            except Exception as exc:
                n_err += 1
                if len(error_samples) < max_error_samples:
                    error_samples.append(f"{fname}[{source_index}]: {exc}")

        if verbose:
            ok_so_far = len(records)
            print(f"  [done] {fname}: {ok_so_far:,} ok so far, {n_err} errors")

    n_ok = len(records)
    elapsed = time.monotonic() - t0

    if verbose:
        print(f"  Writing {n_ok:,} records → {norm_dir}")

    if records:
        normalise_records(records, str(norm_dir), name)
    else:
        if verbose:
            print(f"  [warn] No records to write for {name}")

    return IngestResult(
        dataset_name=name,
        n_ok=n_ok,
        n_err=n_err,
        n_total=n_total,
        elapsed_s=elapsed,
        output_dir=norm_dir,
        error_samples=error_samples,
    )


def ingest_all(
    raw_base: str | Path | None = None,
    norm_base: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, IngestResult]:
    """Ingest all registered datasets.

    Datasets whose raw files are not present are silently skipped
    (no error — the user may not have downloaded them yet).

    Parameters
    ----------
    raw_base : str | Path | None
    norm_base : str | Path | None
    verbose : bool

    Returns
    -------
    dict[str, IngestResult]
        Results keyed by dataset name; only includes datasets that had
        at least one CSV file present.
    """
    results: dict[str, IngestResult] = {}
    for name in DATASET_REGISTRY:
        result = ingest_dataset(name, raw_base=raw_base, norm_base=norm_base, verbose=verbose)
        results[name] = result
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nominal_drift.datasets.ingest",
        description="Ingest raw CDVAE/DiffCSP datasets into normalised JSONL.",
    )
    p.add_argument(
        "--name",
        choices=list(DATASET_REGISTRY.keys()) + ["all"],
        default="all",
        help="Dataset to ingest (default: all).",
    )
    p.add_argument(
        "--raw-base",
        default=None,
        help="Override raw data root (default: data/datasets/raw).",
    )
    p.add_argument(
        "--norm-base",
        default=None,
        help="Override normalised output root (default: data/datasets/normalized).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max rows per CSV file (for smoke testing).",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-file progress.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    if args.name == "all":
        results = ingest_all(
            raw_base=args.raw_base,
            norm_base=args.norm_base,
            verbose=args.verbose,
        )
    else:
        result = ingest_dataset(
            name=args.name,
            raw_base=args.raw_base,
            norm_base=args.norm_base,
            limit=args.limit,
            verbose=args.verbose,
        )
        results = {args.name: result}

    print("\n=== Ingestion Summary ===")
    for name, r in results.items():
        status = "✅" if r.n_err == 0 else "⚠️"
        print(
            f"{status} {name}: {r.n_ok:,} ok / {r.n_err:,} err "
            f"({r.success_rate:.1%}) in {r.elapsed_s:.1f}s"
        )
        if r.error_samples:
            for sample in r.error_samples:
                print(f"    ERR: {sample}")


if __name__ == "__main__":
    main()
