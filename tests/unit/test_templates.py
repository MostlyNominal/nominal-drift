"""Tests for templates module."""
import json
import tempfile
from pathlib import Path
import pytest
from nominal_drift.templates.base import BaseTemplate
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
from nominal_drift.templates.serializer import (
    template_to_json,
    template_from_json,
    save_template,
    load_template,
)
from nominal_drift.templates.factory import (
    create_template,
    template_from_chat_intent,
    ALL_TEMPLATE_TYPES,
)


class TestDiffusionRunTemplate:
    """Test DiffusionRunTemplate instantiation."""

    def test_instantiate_with_defaults(self):
        """DiffusionRunTemplate instantiates with defaults.

        ``element`` defaults to ``None`` (not ``"Cr"``) so that the form
        layer cannot silently fall back to a steel-only species when the
        user has selected a non-steel material system.
        """
        t = DiffusionRunTemplate()
        assert t.template_type == "diffusion_run"
        assert t.alloy_designation == ""
        assert t.composition == {}
        assert t.element is None

    def test_instantiate_with_custom_composition(self):
        """DiffusionRunTemplate with custom composition."""
        comp = {"Cr": 16.5, "C": 0.02, "N": 0.07, "Ni": 10.0, "Fe": 56.31}
        t = DiffusionRunTemplate(alloy_designation="316L", composition=comp)
        assert t.alloy_designation == "316L"
        assert t.composition == comp
        assert t.composition["Cr"] == 16.5

    def test_has_template_type_field(self):
        """All templates have template_type field."""
        t = DiffusionRunTemplate()
        assert hasattr(t, "template_type")
        assert t.template_type == "diffusion_run"

    def test_has_created_at_field(self):
        """All templates have created_at field (auto-set)."""
        t = DiffusionRunTemplate()
        assert hasattr(t, "created_at")
        assert t.created_at is not None
        assert "T" in t.created_at or "Z" in t.created_at

    def test_has_version_field(self):
        """All templates have version='1.0'."""
        t = DiffusionRunTemplate()
        assert t.version == "1.0"

    def test_is_frozen(self):
        """DiffusionRunTemplate is frozen (immutable)."""
        t = DiffusionRunTemplate(alloy_designation="316L")
        with pytest.raises(Exception):
            t.alloy_designation = "304"


class TestSensitizationRunTemplate:
    """Test SensitizationRunTemplate."""

    def test_instantiate(self):
        """SensitizationRunTemplate instantiates."""
        t = SensitizationRunTemplate()
        assert t.template_type == "sensitization_run"

    def test_contains_diffusion_template(self):
        """SensitizationRunTemplate contains a DiffusionRunTemplate."""
        t = SensitizationRunTemplate()
        assert isinstance(t.diffusion_template, DiffusionRunTemplate)

    def test_has_cr_threshold(self):
        """SensitizationRunTemplate has cr_threshold_wt_pct field."""
        t = SensitizationRunTemplate(cr_threshold_wt_pct=12.0)
        assert t.cr_threshold_wt_pct == 12.0

    def test_is_frozen(self):
        """SensitizationRunTemplate is frozen."""
        t = SensitizationRunTemplate()
        with pytest.raises(Exception):
            t.cr_threshold_wt_pct = 13.0


class TestMechanismAnimationTemplate:
    """Test MechanismAnimationTemplate."""

    def test_instantiate(self):
        """MechanismAnimationTemplate instantiates."""
        t = MechanismAnimationTemplate()
        assert t.template_type == "mechanism_animation"

    def test_has_output_format_field(self):
        """MechanismAnimationTemplate has output_format field."""
        t = MechanismAnimationTemplate(output_format="mp4")
        assert t.output_format == "mp4"

    def test_fps_field(self):
        """MechanismAnimationTemplate has fps field."""
        t = MechanismAnimationTemplate(fps=30)
        assert t.fps == 30


class TestLiteratureEntryTemplate:
    """Test LiteratureEntryTemplate."""

    def test_instantiate(self):
        """LiteratureEntryTemplate instantiates."""
        t = LiteratureEntryTemplate()
        assert t.template_type == "literature_entry"

    def test_has_year_field(self):
        """LiteratureEntryTemplate has year field."""
        t = LiteratureEntryTemplate(year=2023)
        assert t.year == 2023


class TestDatasetImportTemplate:
    """Test DatasetImportTemplate."""

    def test_instantiate(self):
        """DatasetImportTemplate instantiates."""
        t = DatasetImportTemplate()
        assert t.template_type == "dataset_import"

    def test_has_dataset_name_field(self):
        """DatasetImportTemplate has dataset_name field."""
        t = DatasetImportTemplate(dataset_name="perov-5")
        assert t.dataset_name == "perov-5"


class TestReportRequestTemplate:
    """Test ReportRequestTemplate."""

    def test_instantiate(self):
        """ReportRequestTemplate instantiates."""
        t = ReportRequestTemplate()
        assert t.template_type == "report_request"

    def test_has_output_format_field(self):
        """ReportRequestTemplate has output_format field."""
        t = ReportRequestTemplate(output_format="html")
        assert t.output_format == "html"


class TestComparisonTemplate:
    """Test ComparisonTemplate."""

    def test_instantiate(self):
        """ComparisonTemplate instantiates."""
        t = ComparisonTemplate()
        assert t.template_type == "comparison"

    def test_has_experiment_ids_field(self):
        """ComparisonTemplate has experiment_ids field."""
        t = ComparisonTemplate(experiment_ids=["exp1", "exp2"])
        assert t.experiment_ids == ["exp1", "exp2"]


class TestSerializer:
    """Test serialization functions."""

    def test_template_to_json(self):
        """serializer: template_to_json produces valid JSON."""
        t = DiffusionRunTemplate(alloy_designation="316L")
        json_str = template_to_json(t)
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["template_type"] == "diffusion_run"
        assert data["alloy_designation"] == "316L"

    def test_template_from_json_roundtrip(self):
        """serializer: template_from_json round-trips correctly."""
        original = DiffusionRunTemplate(alloy_designation="304L")
        json_str = template_to_json(original)
        restored = template_from_json(json_str, DiffusionRunTemplate)
        assert restored.alloy_designation == original.alloy_designation
        assert restored.template_type == original.template_type

    def test_save_template_json(self):
        """serializer: save_template writes .json file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            t = DiffusionRunTemplate(alloy_designation="316L")
            path = str(Path(tmpdir) / "test.json")
            result_path = save_template(t, path)
            assert Path(result_path).exists()
            assert Path(result_path).suffix == ".json"
            content = json.loads(Path(result_path).read_text())
            assert content["alloy_designation"] == "316L"

    def test_save_template_yaml(self):
        """serializer: save_template writes .yaml file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            t = DiffusionRunTemplate(alloy_designation="316L")
            path = str(Path(tmpdir) / "test.yaml")
            result_path = save_template(t, path)
            assert Path(result_path).exists()
            assert Path(result_path).suffix == ".yaml"

    def test_load_template_json(self):
        """serializer: load_template reads back correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            t = DiffusionRunTemplate(alloy_designation="316L")
            path = str(Path(tmpdir) / "test.json")
            save_template(t, path)
            loaded = load_template(path, DiffusionRunTemplate)
            assert loaded.alloy_designation == "316L"
            assert loaded.template_type == "diffusion_run"


class TestFactory:
    """Test factory functions."""

    def test_create_diffusion_run(self):
        """factory: create_template('diffusion_run') returns DiffusionRunTemplate."""
        t = create_template("diffusion_run", alloy_designation="316L")
        assert isinstance(t, DiffusionRunTemplate)
        assert t.alloy_designation == "316L"

    def test_create_sensitization_run(self):
        """factory: create_template('sensitization_run') returns SensitizationRunTemplate."""
        t = create_template("sensitization_run")
        assert isinstance(t, SensitizationRunTemplate)

    def test_create_mechanism_animation(self):
        """factory: create_template('mechanism_animation') returns MechanismAnimationTemplate."""
        t = create_template("mechanism_animation")
        assert isinstance(t, MechanismAnimationTemplate)

    def test_create_literature_entry(self):
        """factory: create_template('literature_entry') returns LiteratureEntryTemplate."""
        t = create_template("literature_entry")
        assert isinstance(t, LiteratureEntryTemplate)

    def test_create_report_request(self):
        """factory: create_template('report_request') returns ReportRequestTemplate."""
        t = create_template("report_request")
        assert isinstance(t, ReportRequestTemplate)

    def test_create_unknown_type_raises(self):
        """factory: create_template raises KeyError for unknown type."""
        with pytest.raises(KeyError):
            create_template("unknown_type_xyz")

    def test_template_from_chat_intent_diffusion(self):
        """factory: template_from_chat_intent('run diffusion 316L') returns 'diffusion_run'."""
        intent = template_from_chat_intent("run diffusion 316L")
        assert intent == "diffusion_run"

    def test_template_from_chat_intent_sensitization(self):
        """factory: template_from_chat_intent('sensitize') returns 'sensitization_run'."""
        intent = template_from_chat_intent("sensitize this alloy")
        assert intent == "sensitization_run"

    def test_template_from_chat_intent_animation(self):
        """factory: template_from_chat_intent('animate mechanism') returns 'mechanism_animation'."""
        intent = template_from_chat_intent("animate mechanism")
        assert intent == "mechanism_animation"

    def test_template_from_chat_intent_literature(self):
        """factory: template_from_chat_intent('add literature paper') returns 'literature_entry'."""
        intent = template_from_chat_intent("add literature paper on sensitization")
        assert intent == "literature_entry"

    def test_template_from_chat_intent_dataset(self):
        """factory: template_from_chat_intent('import perov-5 dataset') returns 'dataset_import'."""
        intent = template_from_chat_intent("import perov-5 dataset")
        assert intent == "dataset_import"

    def test_template_from_chat_intent_report(self):
        """factory: template_from_chat_intent('generate report') returns 'report_request'."""
        intent = template_from_chat_intent("generate report")
        assert intent == "report_request"

    def test_template_from_chat_intent_comparison(self):
        """factory: template_from_chat_intent('compar') matches 'comparison'."""
        intent = template_from_chat_intent("compare experiments")
        assert intent == "comparison"

    def test_all_template_types_in_factory(self):
        """ALL_TEMPLATE_TYPES contains all expected templates."""
        expected_types = [
            "diffusion_run",
            "sensitization_run",
            "mechanism_animation",
            "sensitization_experiment",
            "literature_entry",
            "dataset_import",
            "report_request",
            "comparison",
        ]
        for t_type in expected_types:
            assert t_type in ALL_TEMPLATE_TYPES
            assert ALL_TEMPLATE_TYPES[t_type] is not None
