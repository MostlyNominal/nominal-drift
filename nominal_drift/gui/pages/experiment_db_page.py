"""Experiment database page."""
import streamlit as st


def render():
    st.title("🗄️ Experiment Database")
    st.caption("Browse and search stored diffusion experiments · SQLite local store")

    st.info(
        "**How this works:** Run a diffusion simulation on the Diffusion page, "
        "then click **Save result to Experiment Database** at the bottom of the "
        "results panel.  Results are stored in `data/experiments.db` (SQLite, local)."
    )

    try:
        from nominal_drift.knowledge.experiment_store import list_experiments

        alloy_filter = st.text_input("Filter by alloy designation (optional)", "")
        rows = list_experiments(alloy_designation=alloy_filter or None, limit=50)

        if rows:
            import pandas as pd

            df = pd.DataFrame(
                [
                    {
                        "ID":      str(r.get("experiment_id", ""))[:8],
                        "Alloy":   r.get("alloy_designation", "—"),
                        "Matrix":  r.get("alloy_matrix", "—"),
                        "Element": r.get("element", "—"),
                        "Min conc (wt%)": r.get("min_concentration_wt_pct"),
                        "Depth (nm)":     r.get("depletion_depth_nm"),
                        "Label":   r.get("user_label") or "—",
                        "Date":    r.get("created_at", "—"),
                    }
                    for r in rows
                ]
            )
            st.dataframe(df, use_container_width=True)
            st.caption(f"{len(rows)} experiment(s) stored.")
        else:
            st.info(
                "No experiments found yet.  "
                "Go to **Diffusion** → run a solve → click "
                "**Save result to Experiment Database**."
            )

    except Exception as e:
        st.error(
            f"Could not read experiment database: {e}\n\n"
            "This is expected on first run if `data/experiments.db` does not "
            "exist yet — it is created automatically when you save your first result."
        )
