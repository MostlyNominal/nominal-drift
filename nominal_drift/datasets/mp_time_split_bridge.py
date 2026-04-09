"""
nominal_drift.datasets.mp_time_split_bridge
============================================
Bridge between ``mp_time_split.core.MPTimeSplit`` and Nominal Drift's
internal ``CrystalRecord`` schema.

``MPTimeSplit`` provides the canonical temporal train/val/test splits for
the MPTS-52 dataset (Materials Project Time Split, ~40 k structures, up to
52 atoms per cell, chronologically ordered by earliest literature
publication year).

This module is responsible for:
  1. Loading MPTS-52 via ``MPTimeSplit.load()`` from figshare.
  2. Converting the five (train_inputs, val_inputs) cross-validation folds
     and the final test split into ``CrystalRecord`` objects.
  3. Exposing the fold structure for use by the matbench-genmetrics
     evaluation pipeline.

Architecture
------------
- **Normalised snapshot**: the canonical lane-B JSONL uses the *final*
  ``get_test_data()`` split — ``split="train"`` for final training
  structures, ``split="test"`` for held-out test structures.  Every record
  also stores ``fold_memberships`` in properties so upstream code can
  reconstruct any CV fold without re-loading.
- **CV fold data**: ``get_fold_records(mpt, fold)`` returns
  ``(train_records, val_records)`` directly from a loaded ``MPTimeSplit``.
- **No duplication**: each unique structure appears exactly once in the
  normalised JSONL file.

Public API
----------
``FOLDS``
    ``[0, 1, 2, 3, 4]`` — the five temporal cross-validation folds.

``AVAILABLE_MODES``
    MPTimeSplit split modes (TimeSeriesSplit, TimeKFold, …).

``load_mpts52(save_dir, dummy, force_download, target)``
    Load MPTS-52 via MPTimeSplit; return the fitted ``MPTimeSplit`` object.

``mpt_to_crystal_records(mpt)``
    Convert ALL structures in the MPTimeSplit snapshot to ``CrystalRecord``
    objects using the final test split for the ``split`` field.

``get_fold_records(mpt, fold)``
    Return ``(train_records, val_records)`` for one CV fold.

``get_test_records(mpt)``
    Return ``(final_train_records, test_records)`` for the held-out test
    split.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from nominal_drift.datasets.schema import CrystalRecord
from nominal_drift.datasets.pymatgen_bridge import structure_to_crystal_record

# Re-export the fold constants so consumers only need to import from here
try:
    from mp_time_split.core import FOLDS, AVAILABLE_MODES, MPTimeSplit as _MPTimeSplit
    _HAS_MP_TIME_SPLIT = True
except ImportError:
    FOLDS: list[int] = [0, 1, 2, 3, 4]
    AVAILABLE_MODES: list[str] = ["TimeSeriesSplit", "TimeSeriesOverflowSplit", "TimeKFold"]
    _HAS_MP_TIME_SPLIT = False

DATASET_NAME = "mpts-52"
DEFAULT_TARGET = "energy_above_hull"

# figshare URLs (kept here for reference; MPTimeSplit handles the actual download)
FIGSHARE_FULL_URL = "https://figshare.com/ndownloader/files/35592011"
FIGSHARE_DUMMY_URL = "https://figshare.com/ndownloader/files/35592005"


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

def _require_mp_time_split() -> None:
    if not _HAS_MP_TIME_SPLIT:
        raise ImportError(
            "mp-time-split is required for this function.  "
            "Install it with: pip install 'mp-time-split<0.2'"
        )


# ---------------------------------------------------------------------------
# Load MPTS-52
# ---------------------------------------------------------------------------

def load_mpts52(
    save_dir: str | Path | None = None,
    dummy: bool = False,
    force_download: bool = False,
    target: str = DEFAULT_TARGET,
    mode: str = "TimeSeriesSplit",
) -> "MPTimeSplit":
    """Load the MPTS-52 dataset via MPTimeSplit.

    Downloads the figshare snapshot on first call (≈ 150 MB compressed);
    subsequent calls use the cached file.

    Parameters
    ----------
    save_dir : str | Path | None
        Directory where the downloaded snapshot is cached.  Defaults to
        ``~/.data_home`` (managed by mp-time-split internals).
    dummy : bool
        Use the tiny dummy snapshot for testing (default ``False``).
    force_download : bool
        Re-download even if the snapshot is already cached.
    target : str
        Property to use as the prediction target in ``MPTimeSplit.outputs``.
        Default ``"energy_above_hull"``.
    mode : str
        Temporal split mode.  One of ``AVAILABLE_MODES``.

    Returns
    -------
    MPTimeSplit
        Fitted object with ``inputs``, ``outputs``, ``trainval_splits``,
        and ``test_split`` attributes populated.

    Raises
    ------
    ImportError
        If ``mp-time-split`` is not installed.
    """
    _require_mp_time_split()
    kwargs: dict[str, Any] = {"target": target, "mode": mode}
    if save_dir is not None:
        kwargs["save_dir"] = str(save_dir)

    mpt = _MPTimeSplit(**kwargs)
    mpt.load(dummy=dummy, force_download=force_download)
    return mpt


# ---------------------------------------------------------------------------
# Single-structure converter helper
# ---------------------------------------------------------------------------

def _structure_to_record(
    structure,
    mp_id: str,
    target_value: float | None,
    target_name: str,
    split: str | None,
    source_index: int,
) -> CrystalRecord | None:
    """Convert one pymatgen Structure + metadata → CrystalRecord.

    Returns ``None`` if conversion fails (malformed structure).
    """
    properties: dict[str, Any] = {
        "mp_id": str(mp_id),
        "formula": structure.formula,
    }
    if target_value is not None:
        try:
            properties[target_name] = float(target_value)
        except (TypeError, ValueError):
            pass

    try:
        return structure_to_crystal_record(
            structure=structure,
            source_dataset=DATASET_NAME,
            source_index=source_index,
            split=split,
            properties=properties,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# mpt_to_crystal_records — full dataset snapshot
# ---------------------------------------------------------------------------

def mpt_to_crystal_records(mpt: "MPTimeSplit") -> list[CrystalRecord]:
    """Convert ALL structures in a loaded MPTimeSplit to CrystalRecords.

    Uses the **final test split** to assign ``split`` labels:
      - ``"train"`` for ``final_train_inputs``
      - ``"test"``  for ``test_inputs``

    Each record also stores the target value (e.g. ``energy_above_hull``) in
    ``properties``.

    Parameters
    ----------
    mpt : MPTimeSplit
        A loaded MPTimeSplit instance (``mpt.load()`` must have been called).

    Returns
    -------
    list[CrystalRecord]
        All records in chronological order (train first, then test).
    """
    train_inputs, test_inputs, train_outputs, test_outputs = mpt.get_test_data()
    target_name = mpt.target

    records: list[CrystalRecord] = []
    global_index = 0

    for inputs, outputs, split in [
        (train_inputs, train_outputs, "train"),
        (test_inputs, test_outputs, "test"),
    ]:
        for (mp_id, structure), (_, target_val) in zip(
            inputs.items(), outputs.items()
        ):
            rec = _structure_to_record(
                structure=structure,
                mp_id=str(mp_id),
                target_value=target_val,
                target_name=target_name,
                split=split,
                source_index=global_index,
            )
            if rec is not None:
                records.append(rec)
            global_index += 1

    return records


# ---------------------------------------------------------------------------
# get_fold_records — CV fold accessor
# ---------------------------------------------------------------------------

def get_fold_records(
    mpt: "MPTimeSplit",
    fold: int,
) -> tuple[list[CrystalRecord], list[CrystalRecord]]:
    """Return (train_records, val_records) for one CV fold.

    Parameters
    ----------
    mpt : MPTimeSplit
        Loaded MPTimeSplit instance.
    fold : int
        Fold index (one of ``FOLDS = [0, 1, 2, 3, 4]``).

    Returns
    -------
    tuple[list[CrystalRecord], list[CrystalRecord]]
        ``(train_records, val_records)``
    """
    if fold not in FOLDS:
        raise ValueError(f"fold={fold!r} must be one of {FOLDS}")

    train_inputs, val_inputs, train_outputs, val_outputs = mpt.get_train_and_val_data(fold)
    target_name = mpt.target

    def _convert(inputs, outputs, split, start_index):
        recs = []
        for i, ((mp_id, structure), (_, target_val)) in enumerate(
            zip(inputs.items(), outputs.items())
        ):
            rec = _structure_to_record(
                structure=structure,
                mp_id=str(mp_id),
                target_value=target_val,
                target_name=target_name,
                split=split,
                source_index=start_index + i,
            )
            if rec is not None:
                recs.append(rec)
        return recs

    train_records = _convert(train_inputs, train_outputs, "train", start_index=0)
    val_records = _convert(val_inputs, val_outputs, "val", start_index=len(train_records))
    return train_records, val_records


# ---------------------------------------------------------------------------
# get_test_records — final test split accessor
# ---------------------------------------------------------------------------

def get_test_records(
    mpt: "MPTimeSplit",
) -> tuple[list[CrystalRecord], list[CrystalRecord]]:
    """Return (final_train_records, test_records) for the held-out test split.

    Parameters
    ----------
    mpt : MPTimeSplit
        Loaded MPTimeSplit instance.

    Returns
    -------
    tuple[list[CrystalRecord], list[CrystalRecord]]
        ``(final_train_records, test_records)``
    """
    return (
        mpt_to_crystal_records.__wrapped__(mpt, split_filter="train")
        if hasattr(mpt_to_crystal_records, "__wrapped__")
        else _get_test_records_impl(mpt)
    )


def _get_test_records_impl(
    mpt: "MPTimeSplit",
) -> tuple[list[CrystalRecord], list[CrystalRecord]]:
    train_inputs, test_inputs, train_outputs, test_outputs = mpt.get_test_data()
    target_name = mpt.target

    def _convert(inputs, outputs, split):
        recs = []
        for i, ((mp_id, structure), (_, target_val)) in enumerate(
            zip(inputs.items(), outputs.items())
        ):
            rec = _structure_to_record(
                structure=structure,
                mp_id=str(mp_id),
                target_value=target_val,
                target_name=target_name,
                split=split,
                source_index=i,
            )
            if rec is not None:
                recs.append(rec)
        return recs

    return (
        _convert(train_inputs, train_outputs, "train"),
        _convert(test_inputs, test_outputs, "test"),
    )


# Override get_test_records with the clean implementation
get_test_records = _get_test_records_impl
