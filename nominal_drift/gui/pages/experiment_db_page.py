"""Experiment database page."""
import streamlit as st


def render():
    st.title("🗄️ Experiment Database")
    st.caption("Browse and search stored experiments")
    try:
        from nominal_drift.knowledge.experiment_store import list_experiments

        alloy_filter = st.text_input("Filter by alloy (optional)", "")
        rows = list_experiments(alloy_designation=alloy_filter or None, limit=50)
        if rows:
            import pandas as pd

            df = pd.DataFrame(
                [
                    {
                        "ID": str(r.get("experiment_id", ""))[:8],
                        "Alloy": r.get("alloy_designation", "—"),
                        "Element": r.get("element", "—"),
                        "Min Cr": r.get("min_concentration_wt_pct", None),
                        "Depth nm": r.get("depletion_depth_nm", None),
                        "Date": r.get("created_at", "—"),
                    }
                    for r in rows
                ]
            )
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No experiments found. Run a diffusion simulation first.")
    except Exception as e:
        st.error(str(e))
