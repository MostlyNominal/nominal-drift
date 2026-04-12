"""Report templates."""
from nominal_drift.templates.base import BaseTemplate


class ReportRequestTemplate(BaseTemplate):
    """Template for a report generation request."""
    template_type: str = "report_request"
    report_title: str = ""
    report_subtitle: str = ""
    author: str = "Nominal Drift"
    include_sections: list[str] = ["summary", "assumptions", "recommendations"]
    output_format: str = "markdown"
    output_path: str = "outputs/reports"
    experiment_ids: list[str] = []
    user_notes: str = ""


class ComparisonTemplate(BaseTemplate):
    """Template for experiment comparison."""
    template_type: str = "comparison"
    experiment_ids: list[str] = []
    comparison_axis: str = "temperature"
    report_title: str = "Experiment Comparison"
    output_format: str = "markdown"
    user_notes: str = ""
