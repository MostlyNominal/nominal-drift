"""
nominal_drift.gui.components.thermodynamic_panel
=================================================
Streamlit component that renders the Layer 2 thermodynamic context panel.

Architecture
------------
This component is DISPLAY-ONLY.  It renders a pre-computed ThermodynamicResult
and never calls the diffusion solver, modifies Layer 1 outputs, or reads from
crystal datasets (Layer 3).

Layer boundary contract
-----------------------
* Layer 1 (Arrhenius/Crank-Nicolson) inputs/outputs are NOT touched here.
* Layer 2 results are rendered verbatim — no post-processing that could mask
  computation failures or silently substitute fake values.
* If ThermodynamicResult is None, an explicit "not computed" notice is shown.

Public API
----------
``render_thermodynamic_panel(result, title, expanded)``
    Render the full Layer 2 panel.

``render_thermodynamic_summary_badge(result)``
    Inline compact summary strip (one-liner metric row).
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from nominal_drift.science.thermodynamics import ThermodynamicResult


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

def render_thermodynamic_panel(
    result: Optional[ThermodynamicResult],
    title: str = "🔬 Layer 2 — Thermodynamic Context",
    expanded: bool = False,
) -> None:
    """
    Render the full Layer 2 thermodynamic context panel.

    Parameters
    ----------
    result : ThermodynamicResult or None
        Pre-computed result from ``get_thermodynamic_context()``.
        If None, a "not yet computed" placeholder is shown.
    title : str
        Expander title shown in the UI.
    expanded : bool
        Whether the expander starts open.
    """
    with st.expander(title, expanded=expanded):
        if result is None:
            st.info(
                "Layer 2 thermodynamic context not yet computed. "
                "Run a diffusion solve and it will appear here."
            )
            return

        # ------------------------------------------------------------------
        # Architecture banner
        # ------------------------------------------------------------------
        st.caption(
            "**Layer 2 = thermodynamic phase stability screening.** "
            "Independent of the Layer 1 Crank–Nicolson diffusion solver. "
            "Results are additive context — they do NOT alter diffusion outputs."
        )

        # ------------------------------------------------------------------
        # Summary line
        # ------------------------------------------------------------------
        st.markdown(f"**{result.layer2_summary}**")
        st.divider()

        # ------------------------------------------------------------------
        # Path A — pycalphad result (if available)
        # ------------------------------------------------------------------
        _render_calphad_section(result)

        # ------------------------------------------------------------------
        # Path B — analytical screening
        # ------------------------------------------------------------------
        _render_analytical_section(result)

        # ------------------------------------------------------------------
        # Limitations + provenance
        # ------------------------------------------------------------------
        _render_limitations(result)


# ---------------------------------------------------------------------------
# Sub-renderers
# ---------------------------------------------------------------------------

def _render_calphad_section(result: ThermodynamicResult) -> None:
    """Render the pycalphad equilibrium section (Path A)."""
    st.markdown("#### Path A — pycalphad Equilibrium")

    if result.calphad is None:
        st.info(
            "**pycalphad path not executed.** "
            "No TDB file was found at the default search location "
            "(`data/thermodynamics/`).  "
            "Drop a validated TDB for Fe-Cr-Ni-C-N (austenitic) or "
            "Fe-Cr-C-N (ferritic) there and re-run to enable full CALPHAD "
            "equilibrium calculations."
        )
        st.markdown(
            "_Expected filenames_: `austenitic.tdb` / `ferritic.tdb` / `steel.tdb`.  "
            "Compatible TDB sources: Thermo-Calc TCFE/TCSTEEL (commercial), "
            "OpenCalphad distributions, or user-assessed open databases."
        )
        return

    cal = result.calphad

    if not cal.available:
        # Computation attempted but failed
        st.warning(
            f"⚠ pycalphad computation failed: `{cal.error}`"
        )
        if cal.tdb_path:
            st.caption(f"TDB attempted: `{cal.tdb_path}`")
        _render_tdb_coverage_note(cal.tdb_path or "", result.alloy_matrix)
        return

    # Success — show phases
    st.success("pycalphad equilibrium completed.")

    col1, col2 = st.columns(2)
    col1.metric("Temperature", f"{result.temperature_C:.0f} °C")
    col2.metric("Phases found", len(cal.phases_present))

    if cal.phases_present:
        phase_rows = [
            f"| {ph} | {frac:.4f} |"
            for ph, frac in cal.phases_present
        ]
        st.markdown(
            "| Phase | Mole fraction |\n"
            "|-------|---------------|\n"
            + "\n".join(phase_rows)
        )
    else:
        st.warning("No phases returned with fraction > 0.001.")

    if cal.gm_j_mol is not None:
        st.caption(f"System Gibbs energy: {cal.gm_j_mol:.1f} J mol⁻¹")

    _render_tdb_coverage_note(cal.tdb_path, result.alloy_matrix)

    with st.expander("📋 pycalphad provenance", expanded=False):
        st.markdown(f"- **TDB file**: `{cal.tdb_path}`")
        st.markdown(f"- **Elements used**: {', '.join(cal.elements_used)}")
        st.markdown(f"- **Alloy matrix**: `{result.alloy_matrix}`")
        st.markdown(
            f"- **Composition** (wt%): "
            + ", ".join(
                f"{el}={wt:.3f}" for el, wt in sorted(result.composition_wt_pct.items())
            )
        )
        st.markdown(f"- **pycalphad version**: see `pip show pycalphad`")


def _render_tdb_coverage_note(tdb_path: str, alloy_matrix: str) -> None:
    """Show a coverage warning if the known-incomplete bundled TDB is used."""
    bundled_name = "mc_fecocrnbti.tdb"
    if bundled_name in tdb_path:
        st.warning(
            f"⚠ **Bundled test TDB detected** (`{bundled_name}`).  "
            "This is a **pycalphad test database** (Matcalc 2.060 subset, ODBL) "
            "retained for Fe-Co-Cr-Nb-Ti interactions.  "
            "FCC_A1 (austenite) and M23C6 interaction parameters for the "
            "Fe-Cr-Ni-C-N system are **incomplete or missing**.  "
            "Phase fractions from this database should be treated as "
            "**illustrative only** — not as validated equilibrium results.  "
            "Provide a full TDB (e.g. TCFE, TCSTEEL) for quantitative outputs."
        )


def _render_analytical_section(result: ThermodynamicResult) -> None:
    """Render the analytical screening section (Path B)."""
    a = result.analytical
    st.markdown("#### Path B — Analytical Screening (always computed)")
    st.caption(
        "Semi-analytical models based on published open-literature correlations. "
        "Valid for screening only — see limitations below."
    )

    # Matrix phase prediction
    col1, col2 = st.columns(2)
    col1.metric("Cr-equivalent", f"{a.cr_equivalent_wt_pct:.1f} wt%")
    col2.metric("Ni-equivalent", f"{a.ni_equivalent_wt_pct:.1f} wt%")

    mat_colour = "🟢" if "Austenite" in a.predicted_matrix_phase else (
        "🔵" if "Ferrite" in a.predicted_matrix_phase else "🟡"
    )
    st.markdown(
        f"**Predicted matrix phase**: {mat_colour} {a.predicted_matrix_phase}  \n"
        f"*{a.matrix_phase_note}*"
    )
    st.caption("Source: DeLong (1974) Cr_eq / Ni_eq diagram")
    st.divider()

    # M23C6 sensitization
    m23_icon = "⚠️" if a.m23c6_susceptible else "✅"
    m23_colour = "orange" if a.m23c6_susceptible else "green"
    st.markdown(
        f"**M23C6 sensitization window**: {m23_icon} "
        f"{'Thermodynamically accessible' if a.m23c6_susceptible else 'Not expected'}"
    )
    st.caption(a.m23c6_note)
    st.caption("Source: Hull (1973), Tedmon–Vermilyea–Rosolowski (1971)")

    # Sigma phase
    sig_icon = "⚠️" if a.sigma_susceptible else "✅"
    st.markdown(
        f"**Sigma phase susceptibility**: {sig_icon} "
        f"{'Accessible (slow kinetics)' if a.sigma_susceptible else 'Not expected'}"
    )
    st.caption(a.sigma_note)
    st.caption("Source: Sedriks (1996)")

    # Carbon activity
    st.markdown(f"**Carbon activity** (a_C): `{a.carbon_activity:.4e}` (relative to graphite)")
    st.caption(a.carbon_activity_note)

    # Ms temperature
    if a.ms_temperature_C is not None:
        st.markdown(f"**Martensite-start (Andrews 1965)**: {a.ms_temperature_C:.0f} °C")
    else:
        st.markdown("**Martensite-start**: far below ambient — austenite thermally stable")
    st.caption(a.ms_temperature_note)

    with st.expander("📚 References", expanded=False):
        for ref in a.references:
            st.markdown(f"- {ref}")


def _render_limitations(result: ThermodynamicResult) -> None:
    """Render the limitations and honesty block."""
    with st.expander("⚠ Limitations & what this layer does NOT compute", expanded=False):
        st.error(
            "**Layer 2 does NOT compute:**\n"
            "- Precipitation kinetics (incubation time, growth rate, coarsening)\n"
            "- Grain boundary segregation energetics\n"
            "- Alloy formation kinetics\n"
            "- DFT-backed migration barriers\n"
            "- Validated CALPHAD outputs unless a full TDB is provided\n"
            "- TTT / CCT diagrams\n"
            "- Weld HAZ or cooling-rate effects"
        )
        st.markdown("**Known limitations of this implementation:**")
        for lim in result.limitations:
            st.markdown(f"- {lim}")

        st.markdown(
            "**Layer separation guarantee:** Layer 1 (Crank–Nicolson) outputs "
            "(D₀, Qd, C(x,t) profiles, depletion depth) are computed independently "
            "and are never modified by Layer 2.  "
            "Layer 3 (crystal datasets) is entirely separate — no dataset data "
            "enters Layer 1 or Layer 2 calculations."
        )


# ---------------------------------------------------------------------------
# Compact summary badge
# ---------------------------------------------------------------------------

def render_thermodynamic_summary_badge(
    result: Optional[ThermodynamicResult],
) -> None:
    """
    Render a compact one-line thermodynamic summary strip.

    Suitable for showing after a diffusion solve result, without the full
    Layer 2 panel.
    """
    if result is None:
        return

    a = result.analytical
    flags = []
    if a.m23c6_susceptible:
        flags.append("⚠ M23C6")
    if a.sigma_susceptible:
        flags.append("⚠ Sigma")
    if not flags:
        flags.append("✓ stable phases")

    col1, col2, col3 = st.columns(3)
    col1.metric("Matrix (Cr_eq/Ni_eq)", a.predicted_matrix_phase.split("(")[0].strip())
    col2.metric("Cr_eq / Ni_eq", f"{a.cr_equivalent_wt_pct:.1f} / {a.ni_equivalent_wt_pct:.1f}")
    col3.metric("Phase risk", "  ".join(flags))
