"""Diffusion solver page — Layer 1 + Layer 2."""
import streamlit as st


def render():
    st.title("🔬 Diffusion Solver")
    st.caption(
        "1D Crank–Nicolson · Fickian diffusion · "
        "only solver-supported species (Cr, C, N in austenite_FeCrNi)"
    )

    # -----------------------------------------------------------------------
    # Data provenance panel — makes clear what this solver uses vs does not.
    # -----------------------------------------------------------------------
    with st.expander("ℹ️ What drives this solver (data provenance)", expanded=False):
        st.markdown(
            "**Layer 1 — Diffusion solver uses:**\n"
            "- `nominal_drift/science/constants/arrhenius.json` "
            "— D₀ and Qd for Cr, C, N in austenite_FeCrNi (peer-reviewed values)\n"
            "- `nominal_drift/data/presets/*.json` "
            "— material composition and heat-treatment defaults\n\n"
            "**Layer 1 does NOT use:**\n"
            "- The perov-5 / mp-20 / carbon-24 / mpts-52 crystal datasets "
            "— those exist for crystal structure browsing and generative model "
            "evaluation (Dataset Import page), completely separate from the diffusion lane\n"
            "- DFT energetics, CALPHAD thermodynamics, or precipitation kinetics\n"
            "- Any network call or cloud lookup at runtime\n\n"
            "**Layer 2 (below)** adds thermodynamic phase stability context "
            "independently.  It never modifies Layer 1 outputs.\n\n"
            "**After running:** the Arrhenius constants expander shows "
            "the exact D₀, Qd, and D(T) values used for this specific run."
        )

    from nominal_drift.gui.forms import render_diffusion_form, render_preset_selectbox

    # -----------------------------------------------------------------------
    # Preset selectbox OUTSIDE the form.
    # Changing the material here immediately re-renders the page so all
    # composition sliders, element options, and HT ranges update to the new
    # preset's defaults.  If the selectbox were inside the form, those
    # controls would stay stale until the user clicked Run.
    # -----------------------------------------------------------------------
    selected = render_preset_selectbox(form_suffix="")

    with st.form("diffusion_form"):
        template = render_diffusion_form(selected, form_suffix="")
        # Disable the run button if no solver-supported diffusion species
        # exists for the chosen material system. This is the no-fake-data
        # contract: we refuse to run a fake "Cr" simulation on an aluminium
        # alloy just to keep the workflow clickable.
        run_disabled = template.element is None
        submitted = st.form_submit_button(
            "▶ Run Diffusion",
            disabled=run_disabled,
            help=(
                "Diffusion solve is unavailable for this material system — "
                "no validated Arrhenius constants in the solver database. "
                "You can still run Layer 2 thermodynamic analysis below."
                if run_disabled
                else None
            ),
        )

    # Show an informational note for ferritic / unsupported matrices
    if run_disabled:
        st.info(
            "⚠️ **Layer 1 diffusion not available for this material system.**  "
            "The bundled Arrhenius database (`arrhenius.json`) only contains "
            "constants for `austenite_FeCrNi` (Cr, C, N).  "
            "The `ferrite_FeCr` matrix has no validated diffusion constants yet.  \n\n"
            "You can still run **Layer 2 Thermodynamic Context** below — "
            "it works independently for all Fe-Cr based alloys."
        )

    if submitted:
        if template.element is None:
            st.error(
                "Cannot run: no solver-supported diffusion element for this "
                "material system. Add validated Arrhenius constants in "
                "`nominal_drift/science/constants/arrhenius.json` first."
            )
        else:
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

                    # -----------------------------------------------------------
                    # Runtime proof: show which constants the solver actually used.
                    # The ``metadata`` dict is written by solve_diffusion() at each
                    # HT step and contains the exact D0, Qd, and D(T) values looked
                    # up from the Arrhenius JSON database for this run.
                    # -----------------------------------------------------------
                    arrhenius_used = result.metadata.get("arrhenius", {})
                    if arrhenius_used:
                        with st.expander(
                            "📋 Layer 1 provenance — Arrhenius constants used",
                            expanded=True,
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

                    st.json(
                        {
                            "element": result.element,
                            "matrix": result.matrix,
                            "min_wt_pct": round(result.min_concentration_wt_pct, 3),
                            "depth_nm": result.depletion_depth_nm,
                            "C_bulk_wt_pct": result.C_bulk_wt_pct,
                            "C_sink_wt_pct": result.C_sink_wt_pct,
                            "warnings": result.warnings,
                        }
                    )

                    # Store result in session_state so the save button below can
                    # access it even after Streamlit rerenders the page.
                    st.session_state["_last_diffusion_result"] = {
                        "alloy_designation": template.alloy_designation,
                        "alloy_matrix":      template.alloy_matrix,
                        "composition_json":  template.composition,
                        "ht_schedule_json":  {
                            "steps": template.hold_steps,
                        },
                        "element":                   result.element,
                        "matrix":                    result.matrix,
                        "c_bulk_wt_pct":             result.C_bulk_wt_pct,
                        "c_sink_wt_pct":             result.C_sink_wt_pct,
                        "min_concentration_wt_pct":  result.min_concentration_wt_pct,
                        "depletion_depth_nm":        result.depletion_depth_nm,
                        "warnings_json":             result.warnings,
                    }

                    # Auto-compute Layer 2 after a successful Layer 1 run
                    _compute_and_cache_layer2(template)

                except Exception as e:
                    st.error(f"Error: {e}")

    # ------------------------------------------------------------------
    # Save to Experiment Database (outside the form, uses session state)
    # ------------------------------------------------------------------
    if st.session_state.get("_last_diffusion_result"):
        st.divider()
        label = st.text_input(
            "Label (optional)",
            key="diffusion_save_label",
            placeholder="e.g. 316L sensitization test",
        )
        notes = st.text_input(
            "Notes (optional)",
            key="diffusion_save_notes",
            placeholder="e.g. baseline run",
        )
        if st.button("💾 Save result to Experiment Database", key="diffusion_save_btn"):
            try:
                from nominal_drift.knowledge.experiment_store import write_experiment
                rec = dict(st.session_state["_last_diffusion_result"])
                rec["user_label"] = label or None
                rec["user_notes"] = notes or None
                eid = write_experiment(rec)
                st.success(f"Saved — experiment ID: `{eid}`")
            except Exception as e:
                st.error(f"Could not save: {e}")

    # ------------------------------------------------------------------
    # Layer 2 — Thermodynamic Context
    # Always available, even for ferritic alloys where Layer 1 is disabled.
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("### Layer 2 — Thermodynamic Context")
    st.caption(
        "Runs independently of the diffusion solver. "
        "Available for all Fe-Cr based alloys including ferritic (430SS). "
        "Results are additive context — they never modify Layer 1 outputs."
    )

    # Manual trigger button (always visible; also auto-runs after Layer 1 solve)
    if st.button(
        "🔬 Run Thermodynamic Analysis",
        key="layer2_run_btn",
        help="Compute thermodynamic phase stability context for the current composition and temperature.",
    ):
        _compute_and_cache_layer2(template)

    # Render whatever is cached
    from nominal_drift.gui.components.thermodynamic_panel import render_thermodynamic_panel
    cached_result = st.session_state.get("_last_thermo_result")
    render_thermodynamic_panel(cached_result, expanded=True)


# ---------------------------------------------------------------------------
# Layer 2 helper (shared with auto-run after Layer 1 solve)
# ---------------------------------------------------------------------------

def _compute_and_cache_layer2(template) -> None:
    """
    Compute Layer 2 thermodynamic context from the current form template
    and store the result in ``st.session_state["_last_thermo_result"]``.

    Computation path:
      nominal_drift.science.thermodynamics.get_thermodynamic_context()
      ← template.composition (wt% dict)
      ← first HT step temperature (°C)
      ← template.alloy_matrix (matrix string)
    """
    from nominal_drift.science.thermodynamics import get_thermodynamic_context

    try:
        # Extract representative temperature from first hold step
        T_C = 700.0  # fallback
        if template.hold_steps:
            T_C = float(template.hold_steps[0].get("T_hold_C", 700.0))

        with st.spinner("Computing Layer 2 thermodynamic context…"):
            result = get_thermodynamic_context(
                composition_wt_pct=template.composition,
                T_C=T_C,
                alloy_matrix=template.alloy_matrix,
            )
        st.session_state["_last_thermo_result"] = result

    except Exception as exc:
        st.warning(f"Layer 2 computation failed: {exc}")
        st.session_state["_last_thermo_result"] = None
