"""Animation page."""
import streamlit as st


def render():
    st.title("🎬 Mechanism Animation")
    st.caption("Animated Cr depletion profile · MP4 / GIF output")
    st.info("Configure your diffusion run first, then generate the animation.")
    from nominal_drift.gui.forms import render_diffusion_form

    with st.form("anim_form"):
        template = render_diffusion_form()
        fmt = st.selectbox("Output Format", ["mp4", "gif"])
        submitted = st.form_submit_button("▶ Generate Animation")

    if submitted:
        st.info(
            "Animation generation requires the viz.animator module. "
            "Results are saved to outputs/animations/."
        )
