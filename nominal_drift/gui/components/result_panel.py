"""Result panel component."""
import streamlit as st


def show_diffusion_result(result) -> None:
    """Render a DiffusionOutput in the right panel."""
    col1, col2 = st.columns(2)
    col1.metric("Min Concentration", f"{result.min_concentration_wt_pct:.2f} wt%")
    if result.depletion_depth_nm:
        col2.metric("Depletion Depth", f"{result.depletion_depth_nm:.1f} nm")
    if result.warnings:
        for w in result.warnings:
            st.warning(w)
