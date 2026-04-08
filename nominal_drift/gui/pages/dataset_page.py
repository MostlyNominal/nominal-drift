"""Dataset import page."""
import streamlit as st


def render():
    st.title("📦 Dataset Import")
    st.caption("Import crystal structure datasets · Perov-5, MP-20, Carbon-24, MPTS-52")
    dataset = st.selectbox("Dataset", ["perov-5", "mp-20", "carbon-24", "mpts-52"])
    raw_path = st.text_input("Raw data path", "data/datasets/raw")
    out_path = st.text_input("Output path", "data/datasets/normalized")
    if st.button("Import Dataset"):
        try:
            from nominal_drift.datasets.adapters import get_adapter, normalise_records

            st.info(f"Adapter ready for {dataset}. Place raw data at: {raw_path}/{dataset}/")
            st.success("Use the CLI: nominal-drift dataset fetch --name " + dataset)
        except Exception as e:
            st.error(str(e))
