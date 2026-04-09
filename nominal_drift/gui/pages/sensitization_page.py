"""Mechanism assessment page — depletion risk classification."""
import streamlit as st


def render():
    st.title("⚠️ Mechanism Assessment")
    st.caption("First-order depletion risk classification · multi-species coupling")
    from nominal_drift.gui.forms import render_sensitization_form

    with st.form("sens_form"):
        template = render_sensitization_form()
        submitted = st.form_submit_button("▶ Assess")

    if submitted:
        with st.spinner("Assessing..."):
            try:
                from nominal_drift.schemas.composition import AlloyComposition
                from nominal_drift.schemas.ht_schedule import HTSchedule, HTStep
                from nominal_drift.science.diffusion_engine import solve_diffusion
                from nominal_drift.science.sensitization_model import evaluate_sensitization

                comp = AlloyComposition(
                    alloy_designation=template.diffusion_template.alloy_designation,
                    alloy_matrix=template.diffusion_template.alloy_matrix,
                    composition_wt_pct=template.diffusion_template.composition,
                )
                # Unpack hold_steps dicts into HTStep objects
                steps = []
                for i, s in enumerate(template.diffusion_template.hold_steps):
                    steps.append(HTStep(
                        step=int(s["step"]),
                        type=str(s.get("type", "isothermal_hold")),
                        T_hold_C=float(s["T_hold_C"]),
                        hold_min=float(s["hold_min"]),
                    ))
                ht = HTSchedule(steps=steps)

                element = template.diffusion_template.element
                result = solve_diffusion(
                    comp,
                    ht,
                    element=element,
                    matrix=template.diffusion_template.diffusion_matrix,
                    C_sink_wt_pct=template.diffusion_template.c_sink_wt_pct,
                )
                assessment = evaluate_sensitization(
                    cr_output=result,
                    c_threshold_wt_pct=template.cr_threshold_wt_pct,
                )
                colour = {"low": "🟢", "moderate": "🟡", "high": "🔴"}.get(
                    assessment.risk_level, "⚪"
                )
                st.metric("Risk Level", f"{colour} {assessment.risk_level.upper()}")
                st.metric(
                    f"Min {element} Concentration",
                    f"{assessment.min_cr_wt_pct:.2f} wt%",
                )
                st.info(f"Mechanism: {assessment.mechanism_label}")
            except Exception as e:
                st.error(f"Error: {e}")
