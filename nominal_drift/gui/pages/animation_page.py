"""Animation page — engineering profile and microstructure-inspired modes."""
import streamlit as st


def render():
    st.title("🎬 Animation Studio")
    st.caption("Visualise diffusion dynamics — engineering profile or microstructure view")
    from nominal_drift.gui.forms import render_diffusion_form, render_preset_selectbox

    # -----------------------------------------------------------------------
    # Mode selector and preset selectbox — OUTSIDE the form so changes here
    # cause an immediate re-render with updated composition sliders and
    # element options.
    # -----------------------------------------------------------------------
    mode = st.radio(
        "Animation Mode",
        [
            "📈 Engineering Profile  (C(x,t) concentration curve — real solver output)",
            "🔬 Microstructure Schematic  (particle colour = depletion field — illustrative)",
        ],
        horizontal=False,
    )

    # Scope of the solver — shown before controls so the user cannot miss it
    with st.expander("ℹ️ What is and is not modelled — data provenance", expanded=False):
        st.markdown(
            "**Implemented (real physics, animated from real solver output):**\n"
            "- 1D Crank–Nicolson Fickian diffusion of the selected element\n"
            "- Arrhenius D(T) = D₀ · exp(−Qd/RT) from "
            "`nominal_drift/science/constants/arrhenius.json`\n"
            "- Boundary condition: fixed sink concentration at x = 0\n"
            "- Material composition from `nominal_drift/data/presets/*.json`\n\n"
            "**Not implemented (do not infer from the animation):**\n"
            "- Precipitation kinetics (no carbide / nitride / γ′ nucleation)\n"
            "- Phase stability or CALPHAD free-energy minimisation\n"
            "- Thermodynamic solubility limits\n"
            "- Multi-element coupled diffusion (each element solved independently)\n"
            "- DFT energetics or atomistic positions\n\n"
            "**What the crystal datasets (perov-5 / mp-20 / carbon-24 / mpts-52) do:**\n"
            "These datasets are for crystal structure browsing and generative model "
            "evaluation only.  They are **not connected to this diffusion solver** "
            "in any way — the animations shown here come entirely from the "
            "Crank–Nicolson solver and the Arrhenius constants file.\n\n"
            "**Engineering Profile** mode shows the actual C(x,t) concentration "
            "curve from the solver — it moves visibly because the solver stores "
            "200+ snapshots.  "
            "**Microstructure Schematic** mode shows particle colours encoding the "
            "same concentration field — a visual aid, not an atomistic simulation."
        )

    # Preset selectbox outside the form — unique key suffix to avoid clashing
    # with the diffusion page's "alloy_preset" widget key in session_state.
    selected = render_preset_selectbox(form_suffix="_anim")

    with st.form("anim_form"):
        template = render_diffusion_form(selected, form_suffix="_anim")
        run_disabled = template.element is None
        submitted = st.form_submit_button(
            "▶ Generate Animation",
            disabled=run_disabled,
            help=(
                "Animation unavailable: no diffusion solve possible for "
                "this material system."
                if run_disabled
                else None
            ),
        )

    if submitted:
        if template.element is None:
            st.error(
                "Cannot animate: no solver-supported diffusion element for "
                "this material system."
            )
            return
        with st.spinner("Generating animation..."):
            try:
                from pathlib import Path
                from nominal_drift.schemas.composition import AlloyComposition
                from nominal_drift.schemas.ht_schedule import HTSchedule, HTStep
                from nominal_drift.science.diffusion_engine import solve_diffusion

                comp = AlloyComposition(
                    alloy_designation=template.alloy_designation,
                    alloy_matrix=template.alloy_matrix,
                    composition_wt_pct=template.composition,
                )
                steps = []
                for s in template.hold_steps:
                    steps.append(HTStep(
                        step=int(s["step"]),
                        type=str(s.get("type", "isothermal_hold")),
                        T_hold_C=float(s["T_hold_C"]),
                        hold_min=float(s["hold_min"]),
                    ))
                ht = HTSchedule(steps=steps)

                # Pass matrix explicitly — never silently use a default that
                # doesn't match the selected material system.
                result = solve_diffusion(
                    comp, ht,
                    element=template.element,
                    matrix=template.diffusion_matrix,
                    C_sink_wt_pct=template.c_sink_wt_pct,
                )

                # Show which constants drove this animation so the user can
                # verify the simulation is grounded in the real database.
                arrhenius_used = result.metadata.get("arrhenius", {})
                if arrhenius_used:
                    with st.expander("📋 Arrhenius constants used", expanded=False):
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

                import tempfile
                out_dir = Path(tempfile.mkdtemp(prefix="nd_anim_"))

                if "Schematic" in mode:
                    from nominal_drift.viz.microstructure_animator import (
                        animate_microstructure,
                    )
                    gif_path = animate_microstructure(
                        output=result,
                        save_path=out_dir / "microstructure.gif",
                    )
                    caption = (
                        f"Microstructure scene — {result.element} in {result.matrix} · "
                        f"particle colours encode solved C(x,t) field · "
                        f"NO precipitation / phase kinetics modelled"
                    )
                else:
                    # Engineering Profile: real C(x,t) concentration curve
                    # animated directly from solver output.  Uses animate_diffusion()
                    # which plots the profile line — NOT dots.  Dots cannot show
                    # diffusion clearly when depletion depth is <5% of domain.
                    from nominal_drift.viz.animator import animate_diffusion
                    gif_path = animate_diffusion(
                        output=result,
                        save_path=out_dir / "profile.gif",
                        max_frames=80,
                    )
                    caption = (
                        f"Engineering profile — {result.element} concentration C(x,t) · "
                        f"1D Crank–Nicolson solver · Arrhenius D(T) from arrhenius.json · "
                        f"matrix: {result.matrix}"
                    )

                st.success(
                    f"Animation complete — {result.element} diffusion, "
                    f"{result.total_time_min:.1f} min at "
                    f"{template.hold_steps[0]['T_hold_C']:.0f} °C"
                )
                with open(gif_path, "rb") as f:
                    gif_bytes = f.read()
                st.image(gif_bytes, caption=caption)

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)

            except Exception as e:
                st.error(f"Error: {e}")
