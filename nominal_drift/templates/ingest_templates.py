"""Ingest templates for experiments and data."""
from nominal_drift.templates.base import BaseTemplate


class SensitizationExperimentTemplate(BaseTemplate):
    """Parameters for a sensitization experiment ingestion."""
    template_type: str = "sensitization_experiment"
    alloy_designation: str = ""
    alloy_matrix: str = "austenite"
    composition: dict[str, float] = {}
    heat_treatment: dict = {}
    test_method: str = ""
    result_notes: str = ""
    cr_min_measured_wt_pct: float | None = None
    user_label: str = ""


class LiteratureEntryTemplate(BaseTemplate):
    """Template for a literature entry."""
    template_type: str = "literature_entry"
    title: str = ""
    authors: str = ""
    year: int | None = None
    journal: str = ""
    doi: str = ""
    tags: list[str] = []
    abstract: str = ""
    key_findings: str = ""


class DatasetImportTemplate(BaseTemplate):
    """Template for dataset import."""
    template_type: str = "dataset_import"
    dataset_name: str = ""
    source_url: str = ""
    raw_data_path: str = ""
    output_dir: str = "data/datasets/normalized"
    max_records: int | None = None
    notes: str = ""
