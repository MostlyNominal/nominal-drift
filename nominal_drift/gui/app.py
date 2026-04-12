"""
Nominal Drift GUI — Local browser-based materials engineering workbench.
Run with: streamlit run nominal_drift/gui/app.py
"""
import streamlit as st


def main():
    st.set_page_config(
        page_title="Nominal Drift",
        page_icon="⚗️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar navigation
    st.sidebar.title("⚗️ Nominal Drift")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigate",
        [
            "🔬 Diffusion",
            "⚠️ Mechanism Assessment",
            "🎬 Animation Studio",
            "📦 Dataset Import",
            "📄 Reports",
            "🗄️ Experiment Database",
            "💬 Chat Copilot",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Nominal Drift v0.2.0 · Local · No cloud")

    # Route to page
    from nominal_drift.gui.pages import (
        diffusion_page,
        sensitization_page,
        animation_page,
        dataset_page,
        reports_page,
        experiment_db_page,
        chat_page,
    )

    route = {
        "🔬 Diffusion": diffusion_page.render,
        "⚠️ Mechanism Assessment": sensitization_page.render,
        "🎬 Animation Studio": animation_page.render,
        "📦 Dataset Import": dataset_page.render,
        "📄 Reports": reports_page.render,
        "🗄️ Experiment Database": experiment_db_page.render,
        "💬 Chat Copilot": chat_page.render,
    }
    route[page]()


if __name__ == "__main__":
    main()
