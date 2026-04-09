"""Diffusion solver page."""
import streamlit as st


def render():
    st.title("🔬 Diffusion Solver")
    st.caption("1D Crank–Nicolson · Fickian diffusion · any supported element")
    from nominal_drift.gui.forms import render_diffusion_form

    with st.form("diffusion_form"):
        template = render_diffusion_form()
        submitted = st.form_submit_button("▶ Run Diffusion")

    if submitted:
        with st.spinner("Solving..."):
            try:
                from nominal_drift.schemas.composition import AlloyComposition
                from nominal_drift.schemas.ht_schedule import HTSchedule, HTStep
                from nominal_drift.science.diffusion_engine import solve_diffusion

                comp = AlloyComposition(
                    alloy_designation=template.alloy_designation,
                    alloy_matrix=template.alloy_matrix,
                    composition_wt_pct=template.composition,
                )
                # Unpack hold_steps dicts into HTStep objects
                steps = []
                for i, s in enumerate(template.hold_steps):
                    steps.append(HTStep(
                        step=int(s["step"]),
                        type=str(s.get("type", "isothermal_hold")),
                        T_hold_C=float(s["T_hold_C"]),
                        hold_min=float(s["hold_min"]),
                    ))
                ht = HTSchedule(steps=steps)
                result = solve_diffusion(
                    comp,
                    ht,
                    element=template.element,
                    matrix=template.diffusion_matrix,
                    C_sink_wt_pct=template.c_sink_wt_pct,
                )
                depth_str = (
                    f"Depletion depth: {result.depletion_depth_nm:.1f} nm"
                    if result.depletion_depth_nm
                    else ""
                )
                st.success(
                    f"Min [{result.element}]: "
                    f"{result.min_concentration_wt_pct:.2f} wt%"
                    f" | {depth_str}" if depth_str else
                    f"Min [{result.element}]: "
                    f"{result.min_concentration_wt_pct:.2f} wt%"
                )
                st.json(
                    {
                        "element": result.element,
                        "min_wt_pct": round(result.min_concentration_wt_pct, 3),
                        "depth_nm": result.depletion_depth_nm,
                        "warnings": result.warnings,
                    }
                )
            except Exception as e:
                st.error(f"Error: {e}")
