"""Mechanism assessment page — depletion risk classification + Layer 2 thermodynamic context."""
import streamlit as st


def render():
    st.title("⚠️ Mechanism Assessment")
    st.caption("First-order depletion risk classification · multi-species coupling")
    from nominal_drift.gui.forms import render_sensitization_form, render_preset_selectbox

    # -----------------------------------------------------------------------
    # Preset selectbox OUTSIDE the form so switching material immediately
    # updates all composition sliders and element options.
    # -----------------------------------------------------------------------
    selected = render_preset_selectbox(form_suffix="_sens")

    with st.form("sens_form"):
        template = render_sensitization_form(selected, form_suffix="_sens")
        # No-fake-data contract: refuse to assess when the underlying
        # diffusion solve is unavailable for this material system.
        run_disabled = template.diffusion_template.element is None
        submitted = st.form_submit_button(
            "▶ Assess",
            disabled=run_disabled,
            help=(
                "Assessment unavailable: no validated Arrhenius constants "
                "for any species in this material system.  "
                "Layer 2 thermodynamic context is still available below."
                if run_disabled
                else None
            ),
        )

    # Note for ferritic / unsupported matrices
    if run_disabled:
        st.info(
            "⚠️ **Layer 1 assessment not available for this material system.**  "
            "Arrhenius constants only exist for `austenite_FeCrNi`.  \n\n"
            "**Layer 2 Thermodynamic Context** below runs independently and "
            "provides phase stability assessment for ferritic alloys."
        )

    if submitted:
        if template.diffusion_template.element is None:
            st.error(
                "Cannot assess: no solver-supported diffusion element for "
                "this material system. Add validated Arrhenius constants "
                "in `nominal_drift/science/constants/arrhenius.json` first."
            )
        else:
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

                    # -----------------------------------------------------------
                    # Runtime proof: Arrhenius constants actually used for this run
                    # -----------------------------------------------------------
                    arrhenius_used = result.metadata.get("arrhenius", {})
                    if arrhenius_used:
                        with st.expander(
                            "📋 Layer 1 provenance — Arrhenius constants used",
                            expanded=False,
                        ):
                            st.caption(
                                f"Source: `nominal_drift/science/constants/arrhenius.json` "
                                f"· matrix: `{result.matrix}`"
                            )
                            for el, vals in arrhenius_used.items():
                                st.write(
                                    f"**{el}**: D₀ = {vals['D0']:.3e} m²/s  |  "
                                    f"Qd = {vals['Qd']/1000:.1f} kJ/mol  |  "
                                    f"D({vals['T_C']} °C) = {vals['D_at_T']:.3e} m²/s"
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

                    # Show assumptions so the user knows what was and wasn't modelled
                    with st.expander("ℹ️ Layer 1 model assumptions and limitations"):
                        for a in assessment.assumptions:
                            st.caption(f"• {a}")
                        if assessment.notes:
                            st.markdown("**Notes:**")
                            for n in assessment.notes:
                                st.caption(f"• {n}")

                    # Auto-compute Layer 2 after successful Layer 1 assessment
                    _compute_and_cache_layer2(template.diffusion_template)

                except Exception as e:
                    st.error(f"Error: {e}")

    # ------------------------------------------------------------------
    # Layer 2 — Thermodynamic Context
    # Always available, including for ferritic alloys.
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("### Layer 2 — Thermodynamic Context")
    st.caption(
        "Independent of the depletion risk classifier. "
        "Provides phase stability screening for all Fe-Cr based alloys."
    )

    if st.button(
        "🔬 Run Thermodynamic Analysis",
        key="sens_layer2_run_btn",
        help="Compute thermodynamic phase stability for the current composition and temperature.",
    ):
        _compute_and_cache_layer2(template.diffusion_template)

    from nominal_drift.gui.components.thermodynamic_panel import render_thermodynamic_panel
    cached_result = st.session_state.get("_last_thermo_result_sens")
    render_thermodynamic_panel(
        cached_result,
        title="🔬 Layer 2 — Thermodynamic Context (Sensitization)",
        expanded=True,
    )


# ---------------------------------------------------------------------------
# Layer 2 helper
# ---------------------------------------------------------------------------

def _compute_and_cache_layer2(diff_template) -> None:
    """
    Compute Layer 2 thermodynamic context from the sensitization form template
    and store in ``st.session_state["_last_thermo_result_sens"]``.
    """
    from nominal_drift.science.thermodynamics import get_thermodynamic_context

    try:
        T_C = 700.0
        if diff_template.hold_steps:
            T_C = float(diff_template.hold_steps[0].get("T_hold_C", 700.0))

        with st.spinner("Computing Layer 2 thermodynamic context…"):
            result = get_thermodynamic_context(
                composition_wt_pct=diff_template.composition,
                T_C=T_C,
                alloy_matrix=diff_template.alloy_matrix,
            )
        st.session_state["_last_thermo_result_sens"] = result

    except Exception as exc:
        st.warning(f"Layer 2 computation failed: {exc}")
        st.session_state["_last_thermo_result_sens"] = None
