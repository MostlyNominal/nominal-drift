"""Form renderers that return template objects."""
import streamlit as st
from nominal_drift.templates.run_templates import (
    DiffusionRunTemplate,
    SensitizationRunTemplate,
    MechanismAnimationTemplate,
)


def render_diffusion_form() -> DiffusionRunTemplate:
    """Renders a Streamlit form and returns a populated DiffusionRunTemplate."""
    st.subheader("Alloy")
    designation = st.text_input("Alloy Designation", value="316L")
    cr = st.slider("Cr [wt%]", 10.0, 25.0, 16.5, 0.1)
    c = st.slider("C [wt%]", 0.001, 0.10, 0.02, 0.001)
    n = st.slider("N [wt%]", 0.001, 0.20, 0.07, 0.001)
    ni = st.slider("Ni [wt%]", 5.0, 15.0, 10.0, 0.1)
    fe = 100.0 - cr - c - n - ni

    st.subheader("Heat Treatment")
    T = st.slider("Temperature [°C]", 400, 1100, 700, 10)
    t_min = st.slider("Hold Time [min]", 1, 1440, 120, 1)
    c_sink = st.slider("Cr Sink [wt%]", 9.0, 15.0, 12.0, 0.1)

    return DiffusionRunTemplate(
        alloy_designation=designation,
        composition={"Cr": cr, "C": c, "N": n, "Ni": ni, "Fe": max(fe, 0.0)},
        element="Cr",
        hold_steps=[{"step": 1, "T_hold_C": float(T), "hold_min": float(t_min)}],
        c_sink_wt_pct=c_sink,
    )


def render_sensitization_form() -> SensitizationRunTemplate:
    """Renders sensitization form with nested diffusion form."""
    diffusion_template = render_diffusion_form()
    st.subheader("Sensitization Options")
    cr_thresh = st.slider("Cr Threshold [wt%]", 9.0, 14.0, 12.0, 0.1)
    include_c = st.checkbox("Include C coupling", value=False)
    include_n = st.checkbox("Include N coupling", value=False)
    return SensitizationRunTemplate(
        diffusion_template=diffusion_template,
        cr_threshold_wt_pct=cr_thresh,
        include_c=include_c,
        include_n=include_n,
    )
