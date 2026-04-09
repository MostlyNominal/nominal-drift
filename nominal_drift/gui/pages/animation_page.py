"""Animation page — engineering profile and microstructure-inspired modes."""
import streamlit as st


def render():
    st.title("🎬 Animation Studio")
    st.caption("Visualise diffusion dynamics — engineering profile or microstructure view")
    from nominal_drift.gui.forms import render_diffusion_form

    mode = st.radio(
        "Animation Mode",
        ["📈 Engineering Profile", "🔬 Microstructure Scene"],
        horizontal=True,
    )

    with st.form("anim_form"):
        template = render_diffusion_form()
        submitted = st.form_submit_button("▶ Generate Animation")

    if submitted:
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
                result = solve_diffusion(
                    comp, ht,
                    element=template.element,
                    C_sink_wt_pct=template.c_sink_wt_pct,
                )

                import tempfile
                out_dir = Path(tempfile.mkdtemp(prefix="nd_anim_"))

                if "Microstructure" in mode:
                    from nominal_drift.viz.microstructure_animator import (
                        animate_microstructure,
                    )
                    gif_path = animate_microstructure(
                        output=result,
                        save_path=out_dir / "microstructure.gif",
                    )
                else:
                    from nominal_drift.viz.mechanism_animator import (
                        animate_mechanism,
                    )
                    gif_path = animate_mechanism(
                        output=result,
                        save_path=out_dir / "mechanism.gif",
                    )

                st.success(f"Animation saved: {gif_path}")
                with open(gif_path, "rb") as f:
                    gif_bytes = f.read()
                st.image(gif_bytes, caption=f"{result.element} diffusion animation")

            except Exception as e:
                st.error(f"Error: {e}")
