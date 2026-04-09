"""Form renderers that return template objects.

Forms are driven by material presets loaded from
``nominal_drift/data/presets/*.json``.  When a preset is selected, the
form populates with that alloy's composition, element list, HT ranges,
and sink/threshold defaults.  When "Custom" is selected, the user enters
values manually with generic defaults.

Element selection is filtered by the Arrhenius constants database via
``nominal_drift.science.supported_elements``.  Elements present in alloy
compositions but absent from the database (e.g. Al, Cu, Ti in Ni/Al
presets) are shown as an informational note but excluded from the
diffusion-element selectbox so the workflow cannot fail late.
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
    "diffusion_elements": ["Cr", "C", "N"],
    "default_element": "Cr",
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

def _render_composition(preset: AlloyPreset | None) -> dict[str, float]:
    """Render element sliders and return {element: wt%}."""
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
            val = st.slider(
                f"{el} [wt%]",
                float(lo), float(hi), float(default_val), float(step),
                key=f"comp_{el}",
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
            key="custom_comp",
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
# Public: Diffusion form
# ---------------------------------------------------------------------------

def render_diffusion_form() -> DiffusionRunTemplate:
    """Render a preset-driven diffusion form and return a template."""
    # Material selection
    st.subheader("Material")
    designations = list_designations()
    options = designations + ["Custom"]
    selected = st.selectbox("Alloy Preset", options, index=0, key="alloy_preset")

    preset = get_preset(selected) if selected != "Custom" else None

    # Alloy designation (editable)
    if preset:
        designation = st.text_input(
            "Alloy Designation", value=preset.designation, key="designation"
        )
    else:
        designation = st.text_input(
            "Alloy Designation", value="Custom", key="designation"
        )

    # Composition
    st.subheader("Composition")
    composition = _render_composition(preset)

    # Diffusion element — filtered to Arrhenius-supported elements only
    st.subheader("Heat Treatment")
    if preset:
        raw_elements = list(preset.diffusion_elements)
        element_options, unsupported = filter_to_supported(raw_elements)
        if unsupported:
            st.warning(
                f"⚠️ **{', '.join(unsupported)}** "
                f"{'is' if len(unsupported) == 1 else 'are'} not yet supported "
                f"for diffusion simulation (no validated Arrhenius constants). "
                f"Only **{', '.join(sorted(DIFFUSION_SUPPORTED))}** can be simulated. "
                f"These elements are still shown in the composition."
            )
        if not element_options:
            # All preset elements are unsupported — fall back to Cr
            element_options = ["Cr"]
            st.warning(
                "None of this preset's diffusion elements are currently supported. "
                "Defaulting to Cr. Add Arrhenius constants to enable other elements."
            )
        # Pick default: prefer preset's default if supported, else first available
        default_el = preset.default_diffusing_element
        default_idx = (
            element_options.index(default_el)
            if default_el in element_options
            else 0
        )
    else:
        # Custom mode: only offer supported elements present in composition
        comp_keys = sorted(composition.keys()) if composition else []
        element_options, _ = filter_to_supported(comp_keys)
        if not element_options:
            element_options = list(sorted(DIFFUSION_SUPPORTED))
        default_idx = 0

    element = st.selectbox(
        "Diffusing Element",
        element_options,
        index=default_idx,
        key="diff_element",
        help=f"Supported elements: {', '.join(sorted(DIFFUSION_SUPPORTED))}. "
             f"Other elements require validated Arrhenius constants.",
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
        "Temperature [°C]", T_min, T_max, T_default, 10, key="ht_temp"
    )
    t_min = st.slider(
        "Hold Time [min]", hold_min, hold_max, hold_default, 1, key="ht_hold"
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
        key="c_sink",
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

def render_sensitization_form() -> SensitizationRunTemplate:
    """Render preset-driven sensitization/assessment form."""
    diffusion_template = render_diffusion_form()

    st.subheader("Assessment Options")

    # Threshold from preset
    preset = get_preset(diffusion_template.alloy_designation)
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
        key="depletion_threshold",
    )

    include_c = st.checkbox("Include C coupling", value=False, key="inc_c")
    include_n = st.checkbox("Include N coupling", value=False, key="inc_n")

    return SensitizationRunTemplate(
        diffusion_template=diffusion_template,
        cr_threshold_wt_pct=cr_thresh,
        include_c=include_c,
        include_n=include_n,
    )
