"""
nominal_drift.core.tool_router
==============================
Routes chat intents to Nominal Drift science modules.
The LLM never calculates — it only routes. The modules do the work.
"""
from __future__ import annotations

_ROUTE_TABLE = {
    "diffusion_run": "nominal_drift.science.diffusion_engine",
    "sensitization_run": "nominal_drift.science.sensitization_model",
    "mechanism_animation": "nominal_drift.viz.animator",
    "sensitization_experiment": "nominal_drift.knowledge.experiment_store",
    "literature_entry": "nominal_drift.knowledge.literature_store",
    "dataset_import": "nominal_drift.datasets.adapters",
    "report_request": "nominal_drift.reports.report_builder",
    "comparison": "nominal_drift.knowledge.experiment_store",
}


def route_intent(intent: str, raw_prompt: str) -> str | None:
    """
    Given a template_type intent string, return a short informational response
    describing which module would handle it. Returns None if intent is unknown.
    This is a routing stub — actual execution happens in the page forms.
    """
    module = _ROUTE_TABLE.get(intent)
    if not module:
        return None
    return (
        f"🔀 Routing **{intent}** → `{module}`\n\n"
        f"Use the **{intent.replace('_', ' ').title()}** page in the sidebar "
        f"to fill in parameters and run this computation."
    )


def list_routes() -> dict[str, str]:
    """Return the full routing table."""
    return dict(_ROUTE_TABLE)
