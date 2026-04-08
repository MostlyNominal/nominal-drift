"""
nominal_drift.llm.narration
============================
LLM narration layer for NominalDrift Sprint 1.

Takes a completed diffusion result (``DiffusionOutput``) plus the experiment
context (``AlloyComposition``, ``HTSchedule``) and generates a concise
engineering interpretation by rendering a Jinja2 prompt template and sending
it to a local Ollama inference server.

This module is intentionally thin:

* It does **not** orchestrate multi-step workflows.
* It does **not** parse free-form text (that belongs in a future extractors
  sub-package).
* It does **not** export results — narration text is returned as a plain
  ``str`` for the caller to handle.

Public API
----------
``render_narration_prompt(output, composition, ht_schedule, db_record=None)``
    Render the Jinja2 template to a string without calling the LLM.
    Useful for inspection, logging, and unit testing.

``narrate_diffusion_result(output, composition, ht_schedule, ...)``
    Render the prompt and call the local Ollama client; return the
    narration string.

``DEFAULT_SYSTEM_PROMPT``
    Module-level constant used when *system_prompt* is not supplied by the
    caller.  Establishes the assistant persona and scientific honesty rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import jinja2

from nominal_drift.llm.client import OllamaClient, OllamaConnectionError  # noqa: F401 (re-exported)
from nominal_drift.llm.extractor import summarise_ht_schedule
from nominal_drift.schemas.composition import AlloyComposition
from nominal_drift.schemas.diffusion_output import DiffusionOutput
from nominal_drift.schemas.ht_schedule import HTSchedule

# ---------------------------------------------------------------------------
# Prompt template loader
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

_JINJA_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_PROMPTS_DIR)),
    undefined=jinja2.StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,
)

# ---------------------------------------------------------------------------
# Default system prompt
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT: str = (
    "You are a materials engineering assistant specialising in computational "
    "diffusion modelling. You interpret simulation outputs clearly and honestly. "
    "You never invent numerical values — all numbers must come from the context "
    "provided. You always acknowledge model assumptions and limitations. "
    "You do not claim the simulation is atomistic, a molecular-dynamics result, "
    "or a validated precipitation kinetics model unless explicitly stated in the "
    "context. Your interpretations are generic enough to apply across alloy "
    "systems — you do not assume a single specific application domain."
)

# ---------------------------------------------------------------------------
# Template context builder
# ---------------------------------------------------------------------------

def _build_context(
    output: DiffusionOutput,
    composition: AlloyComposition,
    ht_schedule: HTSchedule,
    db_record: Optional[dict] = None,
) -> dict:
    """Assemble the Jinja2 template variable dict from typed inputs.

    Parameters
    ----------
    output : DiffusionOutput
        Validated output from ``solve_diffusion``.
    composition : AlloyComposition
        Validated alloy composition.
    ht_schedule : HTSchedule
        Validated heat-treatment schedule.
    db_record : dict | None
        Optional stored experiment record.  If provided, ``user_label`` and
        ``user_notes`` are extracted for prompt context.

    Returns
    -------
    dict
        Context dict ready for Jinja2 template rendering.
    """
    user_label: Optional[str] = None
    user_notes: Optional[str] = None
    if db_record is not None:
        user_label = db_record.get("user_label")
        user_notes = db_record.get("user_notes")

    return {
        "alloy_designation":        composition.alloy_designation,
        "alloy_matrix":             composition.alloy_matrix,
        "element":                  output.element,
        "ht_summary":               summarise_ht_schedule(ht_schedule),
        "c_bulk_wt_pct":            output.C_bulk_wt_pct,
        "c_sink_wt_pct":            output.C_sink_wt_pct,
        "min_concentration_wt_pct": output.min_concentration_wt_pct,
        "depletion_depth_nm":       output.depletion_depth_nm,
        "warnings":                 list(output.warnings),
        "user_label":               user_label,
        "user_notes":               user_notes,
    }


# ---------------------------------------------------------------------------
# Public: prompt renderer (no LLM call)
# ---------------------------------------------------------------------------

def render_narration_prompt(
    output: DiffusionOutput,
    composition: AlloyComposition,
    ht_schedule: HTSchedule,
    db_record: Optional[dict] = None,
) -> str:
    """Render the narration prompt template without calling the LLM.

    Use this function to inspect prompt content, write it to a log, or
    verify template rendering in unit tests without a live Ollama server.

    Parameters
    ----------
    output : DiffusionOutput
        Validated solver output.
    composition : AlloyComposition
        Validated alloy composition.
    ht_schedule : HTSchedule
        Validated heat-treatment schedule.
    db_record : dict | None
        Optional experiment-store record; ``user_label`` and ``user_notes``
        are extracted if present.

    Returns
    -------
    str
        The fully rendered prompt string.

    Raises
    ------
    jinja2.TemplateNotFound
        If ``narrate_diffusion.j2`` is not found in the prompts directory.
    jinja2.UndefinedError
        If a required template variable is missing from the context.
    """
    template = _JINJA_ENV.get_template("narrate_diffusion.j2")
    ctx = _build_context(output, composition, ht_schedule, db_record)
    return template.render(**ctx)


# ---------------------------------------------------------------------------
# Public: narration entry point
# ---------------------------------------------------------------------------

def narrate_diffusion_result(
    output: DiffusionOutput,
    composition: AlloyComposition,
    ht_schedule: HTSchedule,
    system_prompt: Optional[str] = None,
    db_record: Optional[dict] = None,
) -> str:
    """Generate a concise engineering narration of a diffusion result.

    Renders the ``narrate_diffusion.j2`` Jinja2 template with the supplied
    inputs, then sends the rendered prompt to a locally-running Ollama server
    and returns the model's response as a plain string.

    Parameters
    ----------
    output : DiffusionOutput
        Validated output from ``nominal_drift.science.diffusion_engine.solve_diffusion``.
    composition : AlloyComposition
        Validated alloy composition used in the simulation.
    ht_schedule : HTSchedule
        Validated heat-treatment schedule applied in the simulation.
    system_prompt : str | None
        Override the default system prompt.  Pass ``None`` to use
        ``DEFAULT_SYSTEM_PROMPT`` (recommended for most use-cases).
    db_record : dict | None
        Optional experiment-store record (as returned by
        ``experiment_store.read_experiment``).  If provided, ``user_label``
        and ``user_notes`` are injected into the prompt context.

    Returns
    -------
    str
        The model's narration text, stripped of leading/trailing whitespace.

    Raises
    ------
    OllamaConnectionError
        If the Ollama server is unreachable or returns a non-2xx response.
        The error message includes the original connection details so the
        caller can surface a diagnostic without further unwrapping.
    ValueError
        If the Ollama response cannot be parsed (missing ``response`` key or
        invalid JSON).
    jinja2.TemplateNotFound
        If the prompt template file is missing from the package.
    """
    prompt = render_narration_prompt(output, composition, ht_schedule, db_record)
    effective_system = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT

    client = OllamaClient()
    # OllamaConnectionError and ValueError propagate to the caller unchanged —
    # they already carry actionable diagnostic messages.
    return client.generate(prompt, system_prompt=effective_system)
