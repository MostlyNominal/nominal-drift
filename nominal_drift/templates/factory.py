"""Template factory with chat-driven intent routing."""
from nominal_drift.templates.run_templates import (
    DiffusionRunTemplate,
    SensitizationRunTemplate,
    MechanismAnimationTemplate,
)
from nominal_drift.templates.ingest_templates import (
    SensitizationExperimentTemplate,
    LiteratureEntryTemplate,
    DatasetImportTemplate,
)
from nominal_drift.templates.report_templates import (
    ReportRequestTemplate,
    ComparisonTemplate,
)
from nominal_drift.templates.base import BaseTemplate


ALL_TEMPLATE_TYPES = {
    "diffusion_run": DiffusionRunTemplate,
    "sensitization_run": SensitizationRunTemplate,
    "mechanism_animation": MechanismAnimationTemplate,
    "sensitization_experiment": SensitizationExperimentTemplate,
    "literature_entry": LiteratureEntryTemplate,
    "dataset_import": DatasetImportTemplate,
    "report_request": ReportRequestTemplate,
    "comparison": ComparisonTemplate,
}


def create_template(template_type: str, **kwargs) -> BaseTemplate:
    """Instantiate the correct template class from template_type string."""
    if template_type not in ALL_TEMPLATE_TYPES:
        raise KeyError(f"Unknown template type: {template_type}")
    cls = ALL_TEMPLATE_TYPES[template_type]
    return cls(**kwargs)


def template_from_chat_intent(intent: str) -> str:
    """
    Route a free-text intent string to a template_type.
    Simple keyword matching.
    """
    intent_lower = intent.lower()

    if "literature" in intent_lower or "paper" in intent_lower or "article" in intent_lower:
        return "literature_entry"
    elif "diffusion" in intent_lower:
        return "diffusion_run"
    elif "sensitiz" in intent_lower:
        return "sensitization_run"
    elif "animation" in intent_lower or "animate" in intent_lower:
        return "mechanism_animation"
    elif "dataset" in intent_lower or "import" in intent_lower or "perov" in intent_lower or "carbon" in intent_lower or "mpts" in intent_lower:
        return "dataset_import"
    elif "report" in intent_lower:
        return "report_request"
    elif "compar" in intent_lower:
        return "comparison"
    else:
        return "diffusion_run"
