"""Form renderers that return template objects.

Forms are driven by material presets loaded from
``nominal_drift/data/presets/*.json``.  When a preset is selected, the
form populates with that alloy's composition, element list, HT ranges,
and sink/threshold defaults.  When "Custom" is selected, the user enters
values manually with generic defaults.

Element selection is filtered by the Arrhenius constants database via
``nominal_drift.science.supported_elements``.  Elements present in alloy
compositions but absent from the database (e.g. Al, Cu, Ti in Ni/Al
presets) are shown as an informational note and **excluded** from the
diffusion-element selectbox so the workflow cannot fail late.

If a preset has *no* solver-supported diffusion species (e.g. an aluminium
or nickel-base alloy whose Arrhenius constants are not yet in the database),
the form refuses to fall back to an unrelated element.  Instead it
disables the run button and surfaces an explicit "diffusion solve not
available for this material system" message.  This is the no-fake-data
contract: the GUI will never silently substitute Cr (or anything else)
just to keep the workflow looking runnable.

PRESET SELECTBOX PLACEMENT
---------------------------
The alloy-preset selectbox **must** live OUTSIDE the ``st.form`` block in
the calling page.  Inside a Streamlit form, widget changes only take effect
after form submission, so a selectbox inside the form cannot update
composition sliders in real time — the old composition stays on screen until
the user clicks Submit.

The correct call pattern in a page module is::

    selected = st.selectbox("Alloy Preset", list_designations() + ["Custom"],
                            key="alloy_preset")  # OUTSIDE the form

    with st.form("my_form"):
        template = render_diffusion_form(selected)  # INSIDE the form
        submitted = st.form_submit_button("Run")

All widget keys inside ``render_diffusion_form`` are namespaced with the
selected preset so that switching presets gives every widget a fresh key and
Streamlit creates it with the correct default value.
"""
import streamlit as st
from nominal_drift.data.presets.loader import (
    AlloyPreset,
    get_preset,
    list_designations,
)
from nominal_drift.science.supported_elements import (
    DIFFUSION_SUPPORTED,
    filter_to_supported,
    get_supported_for_matrix,
    unsupported_explanation,
)
from nominal_drift.templates.run_templates import (
    DiffusionRunTemplate,
    SensitizationRunTemplate,
)


# ---------------------------------------------------------------------------
# Matrix mapping: preset matrix string → AlloyMatrix Literal value
# ---------------------------------------------------------------------------
# Preset JSON files use Arrhenius-compatible matrix names (e.g.
# "austenite_FeCrNi").  AlloyComposition requires the short AlloyMatrix
# Literal.  This mapping bridges the two.

_MATRIX_TO_SCHEMA: dict[str, str] = {
    # Ferrous — Arrhenius names → schema Literal
    "austenite_FeCrNi": "austenite",
    "austenite": "austenite",
    "ferrite": "ferrite",
    "duplex": "duplex",
    "martensite": "martensite",
    # Non-ferrous — preset matrix names → schema Literal
    "Ni_base_superalloy": "ni_fcc",
    "Al_FCC": "al_fcc",
    "Ti_HCP": "ti_hcp",
    "Ti_BCC": "ti_bcc",
    "Cu_FCC": "cu_fcc",
    "Co_base": "co_hcp",
    "perovskite_oxide": "perovskite",
    "oxide": "oxide",
    # Pass-through for values already in Literal form
    "al_fcc": "al_fcc",
    "ni_fcc": "ni_fcc",
    "cu_fcc": "cu_fcc",
    "co_hcp": "co_hcp",
    "ti_hcp": "ti_hcp",
    "ti_bcc": "ti_bcc",
    "fcc": "fcc",
    "bcc": "bcc",
    "hcp": "hcp",
    "perovskite": "perovskite",
    "carbon": "carbon",
    "generic": "generic",
    "unknown": "unknown",
}


def _schema_matrix(preset_matrix: str) -> str:
    """Map a preset/Arrhenius matrix name to the AlloyMatrix Literal value."""
    return _MATRIX_TO_SCHEMA.get(preset_matrix, "unknown")


# ---------------------------------------------------------------------------
# Fallback defaults for "Custom" mode
# ---------------------------------------------------------------------------

_CUSTOM_DEFAULTS = {
    "designation": "Custom",
    "matrix": "generic",
    "composition": {},
    # Custom mode advertises only what the solver actually supports.
    "diffusion_elements": sorted(DIFFUSION_SUPPORTED),
    "default_element": None,  # no default — user must pick a supported element
    "sink": 0.0,
    "threshold": 0.0,
    "T_min": 100,
    "T_max": 1200,
    "T_default": 600,
    "hold_min": 1,
    "hold_max": 2880,
    "hold_default": 60,
}


# ---------------------------------------------------------------------------
# Internal: render composition sliders from a preset or custom input
# ---------------------------------------------------------------------------

def _render_composition(preset: AlloyPreset | None, pk: str) -> dict[str, float]:
    """Render element sliders and return {element: wt%}.

    Parameters
    ----------
    preset : AlloyPreset | None
        The selected alloy preset, or None for Custom mode.
    pk : str
        Preset key — the selected alloy designation string (e.g. "304L",
        "Custom").  Incorporated into every widget key so that switching
        presets gives all sliders fresh keys with correct default values,
        preventing Streamlit session-state from carrying forward stale
        values from the previously selected material.
    """
    if preset is not None:
        comp = dict(preset.composition_wt_pct)
        balance_el = preset.balance_element
        result = {}
        # Show sliders for each element in the preset composition
        for el, default_val in sorted(comp.items()):
            # Skip the balance element — it will be recomputed from
            # 100 - sum(others) after all sliders are rendered.
            if el == balance_el:
                continue
            elif default_val > 5:
                lo, hi = max(0.1, default_val * 0.5), min(99.0, default_val * 1.5)
            elif default_val > 0.5:
                lo, hi = 0.1, max(5.0, default_val * 3.0)
            else:
                lo, hi = 0.001, max(1.0, default_val * 10.0)
            step = 0.01 if default_val < 1.0 else 0.1
            # Key includes preset name → fresh widget when preset changes
            val = st.slider(
                f"{el} [wt%]",
                float(lo), float(hi), float(default_val), float(step),
                key=f"comp_{pk}_{el}",
            )
            result[el] = val
        # Compute balance element from explicit preset field
        used = sum(result.values())
        if balance_el:
            result[balance_el] = max(0.0, 100.0 - used)
            st.caption(f"{balance_el} (balance): {result[balance_el]:.2f} wt%")
        return result
    else:
        # Custom mode — let user type in elements
        st.caption("Enter composition as element=wt% pairs (one per line)")
        raw = st.text_area(
            "Composition",
            value="Fe=70.0\nCr=18.0\nNi=10.0\nC=0.02",
            height=120,
            key=f"custom_comp_{pk}",
        )
        comp = {}
        for line in raw.strip().split("\n"):
            line = line.strip()
            if "=" in line:
                parts = line.split("=", 1)
                try:
                    comp[parts[0].strip()] = float(parts[1].strip())
                except ValueError:
                    pass
        return comp


def _find_balance_element(comp: dict[str, float]) -> str | None:
    """Find the highest-wt% element (the balance)."""
    if not comp:
        return None
    return max(comp, key=comp.get)


# ---------------------------------------------------------------------------
# Public: preset selectbox (call OUTSIDE the st.form)
# ---------------------------------------------------------------------------

def render_preset_selectbox(form_suffix: str = "") -> str:
    """Render the alloy-preset selectbox and return the selected string.

    This widget MUST be placed OUTSIDE the ``st.form`` block so that
    selecting a different material immediately re-renders the page with
    fresh composition sliders.

    Parameters
    ----------
    form_suffix : str
        Optional suffix appended to the widget key to disambiguate between
        multiple forms on different pages (e.g. ``"_anim"``).

    Returns
    -------
    str
        Selected designation (e.g. ``"304L"``) or ``"Custom"``.
    """
    st.subheader("Material")
    designations = list_designations()
    options = designations + ["Custom"]
    return st.selectbox(
        "Alloy Preset",
        options,
        index=0,
        key=f"alloy_preset{form_suffix}",
    )


# ---------------------------------------------------------------------------
# Public: Diffusion form
# ---------------------------------------------------------------------------

def render_diffusion_form(
    selected: str | None = None,
    form_suffix: str = "",
) -> DiffusionRunTemplate:
    """Render a preset-driven diffusion form and return a template.

    Parameters
    ----------
    selected : str | None
        The alloy designation chosen from the selectbox.  Must be provided
        by the caller from a selectbox rendered OUTSIDE the ``st.form``
        so that changing the preset triggers an immediate re-render.
        If None, falls back to the first listed designation (for backwards
        compatibility in unit tests).
    form_suffix : str
        Optional suffix appended to all widget keys to disambiguate between
        different forms that share this renderer (e.g. animation vs diffusion
        pages).  Must match the suffix used in ``render_preset_selectbox``.
    """
    # Resolve selected preset
    if selected is None:
        designations = list_designations()
        selected = st.session_state.get(
            f"alloy_preset{form_suffix}",
            designations[0] if designations else "Custom",
        )

    preset = get_preset(selected) if selected != "Custom" else None
    # pk (preset key) namespaces every widget key so switching presets creates
    # fresh widgets with the new preset's default values.
    pk = f"{selected}{form_suffix}"

    # Alloy designation (editable, initialised from preset)
    if preset:
        designation = st.text_input(
            "Alloy Designation", value=preset.designation, key=f"designation_{pk}"
        )
    else:
        designation = st.text_input(
            "Alloy Designation", value="Custom", key=f"designation_{pk}"
        )

    # Composition
    st.subheader("Composition")
    composition = _render_composition(preset, pk)

    # ------------------------------------------------------------------
    # Diffusion element — material-aware, no fake fallbacks.
    #
    # Two distinct sets are computed:
    #   1. ``preset_relevant`` — what the alloy *cares about* metallurgically
    #      (driven by the preset JSON ``diffusion_elements`` field).
    #   2. ``solver_supported`` — what the solver can *actually* simulate
    #      for this matrix (driven by Arrhenius DB ``get_supported_for_matrix``).
    #
    # The selectbox is the intersection.  If the intersection is empty
    # the form refuses to silently substitute Cr (or any other element).
    # Instead it displays an explicit "not yet available" message and
    # disables the run path by returning a template with element=None.
    # ------------------------------------------------------------------
    st.subheader("Heat Treatment")
    diffusion_unavailable = False
    if preset:
        preset_relevant = list(preset.diffusion_elements)
        # Solver support is *matrix-specific*. For an Al/Ni-base matrix the
        # Arrhenius DB returns no supported species (empty set), which is
        # the truthful answer.
        solver_supported_for_matrix = get_supported_for_matrix(preset.matrix)
        # Intersection preserves preset order
        element_options = [
            el for el in preset_relevant if el in solver_supported_for_matrix
        ]
        unsupported = [
            el for el in preset_relevant if el not in solver_supported_for_matrix
        ]

        if unsupported:
            st.info(
                f"ℹ️ Preset-relevant elements not yet solver-supported for "
                f"matrix **`{preset.matrix}`**: "
                f"**{', '.join(unsupported)}**. "
                f"These appear in the composition but cannot be simulated "
                f"until validated Arrhenius constants (D₀, Qd) are added "
                f"for this matrix."
            )

        if not element_options:
            # No solver-supported diffusion species for this material system.
            # Do NOT fall back to Cr or any other element — be explicit.
            diffusion_unavailable = True
            st.error(
                f"🚫 **Diffusion solve is not available for "
                f"`{preset.designation}` ({preset.matrix}).**\n\n"
                f"This material system has no validated Arrhenius constants "
                f"in the solver database. The relevant species "
                f"(**{', '.join(preset_relevant)}**) require D₀ and Qd "
                f"from peer-reviewed literature before they can be simulated.\n\n"
                f"Currently the solver supports only: "
                f"**{', '.join(sorted(DIFFUSION_SUPPORTED))}** "
                f"(matrix: `austenite_FeCrNi`).\n\n"
                f"To add support for this matrix, place validated constants "
                f"in `nominal_drift/science/constants/arrhenius.json`."
            )
            element = None
            default_idx = 0
        else:
            # Pick default: prefer preset's default if supported, else first
            default_el = preset.default_diffusing_element
            default_idx = (
                element_options.index(default_el)
                if default_el in element_options
                else 0
            )
            element = st.selectbox(
                "Diffusing Element",
                element_options,
                index=default_idx,
                key=f"diff_element_{pk}",
                help=(
                    f"Solver-supported elements for matrix `{preset.matrix}`: "
                    f"{', '.join(element_options)}."
                ),
            )
    else:
        # Custom mode: only offer supported elements present in composition.
        # If the user typed only unsupported species we surface that, not Cr.
        comp_keys = sorted(composition.keys()) if composition else []
        element_options, comp_unsupported = filter_to_supported(comp_keys)
        if comp_unsupported:
            st.info(
                f"ℹ️ Composition elements without Arrhenius constants: "
                f"**{', '.join(comp_unsupported)}** — present in composition "
                f"but not selectable for diffusion."
            )
        if not element_options:
            diffusion_unavailable = True
            st.error(
                "🚫 **No solver-supported diffusion element in this composition.**\n\n"
                f"Currently supported: **{', '.join(sorted(DIFFUSION_SUPPORTED))}** "
                f"(matrix: `austenite_FeCrNi`).\n\n"
                f"Add at least one of these to the composition, or extend "
                f"`science/constants/arrhenius.json` with validated constants "
                f"for the species you need."
            )
            element = None
        else:
            element = st.selectbox(
                "Diffusing Element",
                element_options,
                index=0,
                key=f"diff_element_{pk}",
                help=(
                    f"Showing only solver-supported elements present in your "
                    f"composition. Full solver-supported set: "
                    f"{', '.join(sorted(DIFFUSION_SUPPORTED))}."
                ),
            )

    # HT parameters from preset ranges
    if preset:
        ht = preset.ht_ranges
        T_min = int(ht.get("T_min_C", 100))
        T_max = int(ht.get("T_max_C", 1200))
        T_default = int(ht.get("T_default_C", 600))
        hold_min = int(ht.get("hold_min_min", 1))
        hold_max = int(ht.get("hold_min_max", 2880))
        hold_default = int(ht.get("hold_min_default", 60))
        sink_default = float(preset.default_sink_wt_pct)
    else:
        T_min, T_max, T_default = 100, 1200, 600
        hold_min, hold_max, hold_default = 1, 2880, 60
        sink_default = 0.0

    T = st.slider(
        "Temperature [°C]", T_min, T_max, T_default, 10, key=f"ht_temp_{pk}"
    )
    t_min = st.slider(
        "Hold Time [min]", hold_min, hold_max, hold_default, 1, key=f"ht_hold_{pk}"
    )

    # Sink concentration — generic label
    sink_lo = max(0.0, sink_default * 0.5) if sink_default > 0 else 0.0
    sink_hi = max(sink_default * 1.5, 1.0) if sink_default > 0 else 50.0
    c_sink = st.slider(
        "Interface Sink [wt%]",
        float(sink_lo),
        float(sink_hi),
        float(sink_default),
        0.1,
        key=f"c_sink_{pk}",
    )

    # Map preset matrix to schema-compatible value; keep full name for Arrhenius
    arrhenius_matrix = preset.matrix if preset else "generic"
    schema_matrix = _schema_matrix(arrhenius_matrix)

    return DiffusionRunTemplate(
        alloy_designation=designation,
        alloy_matrix=schema_matrix,
        composition=composition,
        element=element,
        diffusion_matrix=arrhenius_matrix,
        hold_steps=[{
            "step": 1,
            "type": "isothermal_hold",
            "T_hold_C": float(T),
            "hold_min": float(t_min),
        }],
        c_sink_wt_pct=c_sink,
    )


# ---------------------------------------------------------------------------
# Public: Sensitization form
# ---------------------------------------------------------------------------

def render_sensitization_form(
    selected: str | None = None,
    form_suffix: str = "",
) -> SensitizationRunTemplate:
    """Render preset-driven sensitization/assessment form.

    Parameters
    ----------
    selected : str | None
        Selected alloy designation — must be passed from a selectbox rendered
        OUTSIDE the ``st.form`` (see module docstring).
    form_suffix : str
        Widget key suffix for disambiguation.
    """
    diffusion_template = render_diffusion_form(selected, form_suffix)

    st.subheader("Assessment Options")

    # Threshold from preset — look up by selected designation, not by text
    # input value, to avoid stale-state mismatches.
    pk = f"{selected or 'Custom'}{form_suffix}"
    preset = get_preset(selected) if selected and selected != "Custom" else None
    if preset:
        thresh_default = float(preset.default_depletion_threshold_wt_pct)
    else:
        thresh_default = 0.0

    thresh_lo = max(0.0, thresh_default * 0.5) if thresh_default > 0 else 0.0
    thresh_hi = max(thresh_default * 1.5, 1.0) if thresh_default > 0 else 50.0
    cr_thresh = st.slider(
        "Depletion Threshold [wt%]",
        float(thresh_lo),
        float(thresh_hi),
        float(thresh_default),
        0.1,
        key=f"depletion_threshold_{pk}",
    )

    include_c = st.checkbox("Include C coupling", value=False, key=f"inc_c_{pk}")
    include_n = st.checkbox("Include N coupling", value=False, key=f"inc_n_{pk}")

    return SensitizationRunTemplate(
        diffusion_template=diffusion_template,
        cr_threshold_wt_pct=cr_thresh,
        include_c=include_c,
        include_n=include_n,
    )
