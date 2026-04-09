"""
Dataset Import page — real status, real download, honest reporting.

The page audits what is actually on disk and shows it.  It never claims
a dataset is available when it is not.  Downloads run via the existing
downloader classes; if network access fails a clear manual-import path
is shown instead.
"""
from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Resolve paths relative to the project root (not the CWD of the runner)
# ---------------------------------------------------------------------------

from pathlib import Path
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_RAW_BASE  = _PROJECT_ROOT / "data" / "datasets" / "raw"
_NORM_BASE = _PROJECT_ROOT / "data" / "datasets" / "normalized"


def _get_statuses():
    from nominal_drift.datasets.status import get_all_statuses
    return get_all_statuses(raw_base=_RAW_BASE, norm_base=_NORM_BASE)


def _run_download(name: str) -> tuple[bool, str]:
    """Attempt to download *name* using the registered downloader.

    Returns (success: bool, message: str).
    """
    try:
        from nominal_drift.datasets.downloaders.perov5_downloader import Perov5Downloader
        from nominal_drift.datasets.downloaders.mp20_downloader import MP20Downloader
        from nominal_drift.datasets.downloaders.carbon24_downloader import Carbon24Downloader
        from nominal_drift.datasets.downloaders.mpts52_downloader import MPTS52Downloader

        _DOWNLOADER_MAP = {
            "perov-5":   Perov5Downloader,
            "mp-20":     MP20Downloader,
            "carbon-24": Carbon24Downloader,
            "mpts-52":   MPTS52Downloader,
        }
        cls = _DOWNLOADER_MAP[name]
        dl = cls(raw_base_dir=str(_RAW_BASE))
        result = dl.download(force=False)

        if result.already_existed:
            return True, f"Already present — {result.n_structures_found:,} rows found."
        if result.files_downloaded:
            return True, (
                f"Downloaded {len(result.files_downloaded)} file(s): "
                f"{result.files_downloaded}. "
                f"{result.n_structures_found:,} rows. "
                f"{result.total_bytes:,} bytes."
            )
        # Nothing downloaded — likely network error
        warn_txt = "; ".join(result.warnings) if result.warnings else "unknown error"
        return False, f"Download failed: {warn_txt}"
    except Exception as exc:
        return False, f"Downloader error: {exc}"


def _run_normalize(name: str) -> tuple[bool, str]:
    """Normalise raw CSV → structures.jsonl for *name* via the ingest pipeline.

    Returns (success: bool, message: str).
    """
    try:
        from nominal_drift.datasets.ingest import ingest_dataset
        from nominal_drift.datasets.status import get_dataset_status

        status = get_dataset_status(name, raw_base=_RAW_BASE, norm_base=_NORM_BASE)
        if not status.is_raw_complete:
            return False, "Raw files are not complete — download first."

        result = ingest_dataset(
            name=name,
            raw_base=_RAW_BASE,
            norm_base=_NORM_BASE,
            verbose=False,
        )

        if result.n_ok == 0:
            errs = "; ".join(result.error_samples[:3]) if result.error_samples else "unknown"
            return False, f"No records could be parsed. Errors: {errs}"

        msg = (
            f"Normalised {result.n_ok:,} structures in {result.elapsed_s:.1f}s → "
            f"{result.output_dir}."
        )
        if result.n_err:
            msg += f" Skipped {result.n_err:,} malformed rows."
        return True, msg

    except Exception as exc:
        return False, f"Normalisation error: {exc}"


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------

def render():
    st.title("📦 Dataset Import")
    st.caption(
        "Crystal structure datasets · Perov-5 · MP-20 · Carbon-24 · MPTS-52 · "
        "Real download status"
    )

    # -----------------------------------------------------------------------
    # Status overview table
    # -----------------------------------------------------------------------
    st.subheader("Dataset Status")

    statuses = _get_statuses()

    cols = st.columns([2, 4, 2, 2])
    cols[0].markdown("**Dataset**")
    cols[1].markdown("**Status**")
    cols[2].markdown("**Structures**")
    cols[3].markdown("**Elements**")

    for name, s in statuses.items():
        c0, c1, c2, c3 = st.columns([2, 4, 2, 2])
        c0.markdown(f"`{name}`")
        c1.markdown(s.download_status_label)
        if s.is_normalised:
            c2.markdown(f"{s.norm_structure_count:,}")
            c3.markdown(f"{len(s.norm_elements)}")
        elif s.is_raw_complete:
            c2.markdown(f"{s.total_raw_rows:,} rows")
            c3.markdown("—")
        else:
            c2.markdown("—")
            c3.markdown("—")

    st.divider()

    # -----------------------------------------------------------------------
    # Per-dataset actions
    # -----------------------------------------------------------------------
    st.subheader("Manage Dataset")

    dataset = st.selectbox(
        "Select dataset",
        list(DATASET_REGISTRY_NAMES := list(statuses.keys())),
        key="dataset_select",
    )

    s = statuses[dataset]
    info = s.info

    with st.expander("Dataset info", expanded=False):
        st.markdown(f"**Description:** {info.description}")
        st.markdown(f"**Source:** [{info.source_url}]({info.source_url})")
        st.markdown(f"**License:** {info.license}")
        st.markdown(f"**Citation:** {info.citation}")
        st.markdown(f"**Auto-downloadable:** {'Yes' if info.auto_downloadable else 'No'}")

    # Raw layer status
    st.markdown("**Raw layer**")
    st.code(str(s.raw_dir), language=None)
    if s.is_raw_complete:
        for fname, count in s.raw_row_counts.items():
            st.markdown(f"  ✅ `{fname}` — {count:,} rows")
    elif s.raw_files_present:
        for fname in s.raw_files_present:
            st.markdown(f"  ✅ `{fname}` — {s.raw_row_counts[fname]:,} rows")
        for fname in s.raw_files_missing:
            st.markdown(f"  ❌ `{fname}` — missing")
    else:
        st.markdown("  ❌ No files downloaded yet.")

    # Normalised layer status
    st.markdown("**Normalised layer**")
    st.code(str(s.norm_dir), language=None)
    if s.is_normalised:
        st.markdown(
            f"  ✅ `structures.jsonl` — {s.norm_structure_count:,} structures"
        )
        if s.norm_elements:
            st.markdown(
                f"  Elements: `{'`, `'.join(s.norm_elements[:20])}`"
                + (" ..." if len(s.norm_elements) > 20 else "")
            )
    else:
        st.markdown("  ❌ Not yet normalised.")

    # Actions
    col_dl, col_norm = st.columns(2)

    with col_dl:
        if st.button(
            "⬇️ Download raw data",
            disabled=s.is_raw_complete,
            key=f"dl_{dataset}",
        ):
            with st.spinner(f"Downloading {dataset}…"):
                ok, msg = _run_download(dataset)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
                st.markdown("**Manual import instructions:**")
                st.code(info.manual_instructions, language="text")

    with col_norm:
        if st.button(
            "⚙️ Normalise → JSONL",
            disabled=not s.is_raw_complete or s.is_normalised,
            key=f"norm_{dataset}",
        ):
            with st.spinner(f"Normalising {dataset}…"):
                ok, msg = _run_normalize(dataset)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    # Always show manual import path — never hide it
    if not s.is_raw_complete:
        st.divider()
        st.markdown("**Manual import path** (use if download fails):")
        st.code(info.manual_instructions, language="text")
        st.markdown(
            f"After placing files at `{s.raw_dir}`, "
            f"click **Normalise** above to index the dataset."
        )

    # -----------------------------------------------------------------------
    # CLI hint
    # -----------------------------------------------------------------------
    st.divider()
    st.markdown("**CLI equivalent:**")
    st.code(
        f"nominal-drift dataset fetch --name {dataset}\n"
        f"nominal-drift dataset normalise --name {dataset}",
        language="bash",
    )
