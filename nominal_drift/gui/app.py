"""
Nominal Drift GUI — Local browser-based engineering workbench.
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
            "⚠️ Sensitization",
            "🌡️ Heat Treatment",
            "🎬 Mechanism Animation",
            "📦 Dataset Import",
            "📄 Reports",
            "🗄️ Experiment Database",
            "💬 Chat Copilot",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Nominal Drift v0.1.0 · Local · No cloud")

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
        "⚠️ Sensitization": sensitization_page.render,
        "🌡️ Heat Treatment": diffusion_page.render,
        "🎬 Mechanism Animation": animation_page.render,
        "📦 Dataset Import": dataset_page.render,
        "📄 Reports": reports_page.render,
        "🗄️ Experiment Database": experiment_db_page.render,
        "💬 Chat Copilot": chat_page.render,
    }
    route[page]()


if __name__ == "__main__":
    main()
