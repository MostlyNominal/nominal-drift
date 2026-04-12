"""
tests/unit/test_orchestrator.py
================================
TDD tests for nominal_drift.core.orchestrator (Day 9).

All tests use mocked externals so no live Ollama instance or slow
diffusion solves are required.  The orchestration logic — directory
creation, record assembly, return-dict shape — is the subject under test.

Test categories
---------------
A  Return dict structure — required keys and types
B  Output files — plot and animation are created by the mocked functions
C  Experiment persistence — record written to and readable from DB
D  Narration — mocked LLM returns correct text; fallback on connection error
E  Warning propagation — solver warnings reach the return dict
F  Path overrides — custom base_output_dir and db_path are respected

Run with:
    pytest tests/unit/test_orchestrator.py -v
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nominal_drift.core.orchestrator import run_showcase_workflow
from nominal_drift.knowledge.experiment_store import read_experiment
from nominal_drift.llm.client import OllamaConnectionError
from nominal_drift.schemas.composition import AlloyComposition
from nominal_drift.schemas.diffusion_output import DiffusionOutput
from nominal_drift.schemas.ht_schedule import HTSchedule, HTStep


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_COMP = AlloyComposition(
    alloy_designation="316L",
    alloy_matrix="austenite",
    composition_wt_pct={
        "Fe": 68.88, "Cr": 16.50, "Ni": 10.10,
        "Mo": 2.10,  "Mn": 1.80,  "Si": 0.50,
        "C":  0.02,  "N":  0.07,  "P":  0.03,
    },
)

_SCHED = HTSchedule(steps=[
    HTStep(step=1, type="sensitization_soak",
           T_hold_C=700.0, hold_min=1.0,
           cooling_method="air_cool"),
])


def _fake_output(element: str = "Cr", warnings: list[str] | None = None) -> DiffusionOutput:
    """Minimal valid DiffusionOutput for mocking solve_diffusion."""
    return DiffusionOutput(
        element=element,
        matrix="austenite_FeCrNi",
        x_m=[0.0, 1e-9, 2e-9, 3e-9],
        t_s=[0.0, 60.0],
        concentration_profiles=[
            [16.5, 16.5, 16.5, 16.5],
            [12.0, 14.25, 16.0, 16.5],
        ],
        C_bulk_wt_pct=16.5,
        C_sink_wt_pct=12.0,
        min_concentration_wt_pct=12.0,
        depletion_depth_nm=42.5,
        warnings=warnings or [],
        metadata={"element": element},
    )


def _fake_plot_fn(output, save_path, **kwargs):
    """Mock plot function that writes a stub PNG and returns the path."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG\r\n")
    return str(p)


def _fake_anim_fn(output, save_path, **kwargs):
    """Mock animate function that writes a stub MP4 and returns the path."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"FAKEVIDEO")
    return str(p)


def _run_workflow(
    tmp_path,
    element: str = "Cr",
    warnings: list[str] | None = None,
    narration_text: str = "Mocked narration.",
    user_label: str | None = None,
    user_notes: str | None = None,
) -> dict:
    """Run the workflow with all external dependencies mocked."""
    fake_output = _fake_output(element=element, warnings=warnings)
    with (
        patch("nominal_drift.core.orchestrator.solve_diffusion",
              return_value=fake_output),
        patch("nominal_drift.core.orchestrator.plot_concentration_profile",
              side_effect=_fake_plot_fn),
        patch("nominal_drift.core.orchestrator.animate_diffusion",
              side_effect=_fake_anim_fn),
        patch("nominal_drift.core.orchestrator.narrate_diffusion_result",
              return_value=narration_text),
    ):
        return run_showcase_workflow(
            _COMP, _SCHED,
            element=element,
            base_output_dir=str(tmp_path / "outputs"),
            db_path=str(tmp_path / "test.db"),
            user_label=user_label,
            user_notes=user_notes,
        )


# ---------------------------------------------------------------------------
# A. Return dict structure
# ---------------------------------------------------------------------------

class TestReturnDictStructure:
    """run_showcase_workflow must return a dict with the required keys."""

    _REQUIRED_KEYS = {
        "experiment_id",
        "output_dir",
        "plot_path",
        "animation_path",
        "narration",
        "min_concentration_wt_pct",
        "depletion_depth_nm",
        "warnings",
        "element",
        "alloy_designation",
    }

    def test_returns_dict(self, tmp_path):
        result = _run_workflow(tmp_path)
        assert isinstance(result, dict)

    def test_all_required_keys_present(self, tmp_path):
        result = _run_workflow(tmp_path)
        missing = self._REQUIRED_KEYS - result.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_experiment_id_is_valid_uuid(self, tmp_path):
        result = _run_workflow(tmp_path)
        parsed = uuid.UUID(result["experiment_id"])
        assert str(parsed) == result["experiment_id"]

    def test_output_dir_is_string(self, tmp_path):
        result = _run_workflow(tmp_path)
        assert isinstance(result["output_dir"], str)

    def test_plot_path_is_string(self, tmp_path):
        result = _run_workflow(tmp_path)
        assert isinstance(result["plot_path"], str)

    def test_animation_path_is_string(self, tmp_path):
        result = _run_workflow(tmp_path)
        assert isinstance(result["animation_path"], str)

    def test_narration_is_string(self, tmp_path):
        result = _run_workflow(tmp_path)
        assert isinstance(result["narration"], str)

    def test_warnings_is_list(self, tmp_path):
        result = _run_workflow(tmp_path)
        assert isinstance(result["warnings"], list)

    def test_element_matches_input(self, tmp_path):
        result = _run_workflow(tmp_path, element="Cr")
        assert result["element"] == "Cr"

    def test_alloy_designation_matches_input(self, tmp_path):
        result = _run_workflow(tmp_path)
        assert result["alloy_designation"] == "316L"

    def test_min_concentration_is_float(self, tmp_path):
        result = _run_workflow(tmp_path)
        assert isinstance(result["min_concentration_wt_pct"], float)

    def test_depletion_depth_is_float_or_none(self, tmp_path):
        result = _run_workflow(tmp_path)
        assert result["depletion_depth_nm"] is None or isinstance(
            result["depletion_depth_nm"], float
        )


# ---------------------------------------------------------------------------
# B. Output files
# ---------------------------------------------------------------------------

class TestOutputFiles:
    """Stub files written by mocked viz functions must be present on disk."""

    def test_plot_file_exists(self, tmp_path):
        result = _run_workflow(tmp_path)
        assert Path(result["plot_path"]).exists()

    def test_animation_file_exists(self, tmp_path):
        result = _run_workflow(tmp_path)
        assert Path(result["animation_path"]).exists()

    def test_plot_is_under_output_dir(self, tmp_path):
        result = _run_workflow(tmp_path)
        assert result["plot_path"].startswith(result["output_dir"])

    def test_animation_is_under_output_dir(self, tmp_path):
        result = _run_workflow(tmp_path)
        assert result["animation_path"].startswith(result["output_dir"])

    def test_output_dir_exists(self, tmp_path):
        result = _run_workflow(tmp_path)
        assert Path(result["output_dir"]).is_dir()

    def test_plot_filename_contains_element(self, tmp_path):
        result = _run_workflow(tmp_path, element="Cr")
        assert "cr" in Path(result["plot_path"]).name.lower()

    def test_animation_filename_contains_element(self, tmp_path):
        result = _run_workflow(tmp_path, element="Cr")
        assert "cr" in Path(result["animation_path"]).name.lower()


# ---------------------------------------------------------------------------
# C. Experiment persistence
# ---------------------------------------------------------------------------

class TestExperimentPersistence:
    """The experiment record must be readable back from the DB after the run."""

    def test_experiment_id_readable_from_db(self, tmp_path):
        db = str(tmp_path / "test.db")
        result = _run_workflow(tmp_path)
        # Use the same db path the workflow wrote to
        db_path = str(tmp_path / "test.db")
        stored = read_experiment(result["experiment_id"], db_path=db_path)
        assert stored["experiment_id"] == result["experiment_id"]

    def test_alloy_designation_persisted(self, tmp_path):
        result = _run_workflow(tmp_path)
        db_path = str(tmp_path / "test.db")
        stored = read_experiment(result["experiment_id"], db_path=db_path)
        assert stored["alloy_designation"] == "316L"

    def test_element_persisted(self, tmp_path):
        result = _run_workflow(tmp_path)
        db_path = str(tmp_path / "test.db")
        stored = read_experiment(result["experiment_id"], db_path=db_path)
        assert stored["element"] == "Cr"

    def test_plot_path_persisted(self, tmp_path):
        result = _run_workflow(tmp_path)
        db_path = str(tmp_path / "test.db")
        stored = read_experiment(result["experiment_id"], db_path=db_path)
        assert stored["plot_path"] == result["plot_path"]

    def test_animation_path_persisted(self, tmp_path):
        result = _run_workflow(tmp_path)
        db_path = str(tmp_path / "test.db")
        stored = read_experiment(result["experiment_id"], db_path=db_path)
        assert stored["animation_path"] == result["animation_path"]

    def test_user_label_persisted(self, tmp_path):
        result = _run_workflow(tmp_path, user_label="my-test-run")
        db_path = str(tmp_path / "test.db")
        stored = read_experiment(result["experiment_id"], db_path=db_path)
        assert stored["user_label"] == "my-test-run"

    def test_user_notes_persisted(self, tmp_path):
        result = _run_workflow(tmp_path, user_notes="Reference run.")
        db_path = str(tmp_path / "test.db")
        stored = read_experiment(result["experiment_id"], db_path=db_path)
        assert stored["user_notes"] == "Reference run."


# ---------------------------------------------------------------------------
# D. Narration
# ---------------------------------------------------------------------------

class TestNarration:
    """Narration text must be returned; Ollama failure must not abort workflow."""

    def test_narration_text_returned(self, tmp_path):
        result = _run_workflow(tmp_path, narration_text="Cr depletion detected.")
        assert result["narration"] == "Cr depletion detected."

    def test_narration_is_nonempty_string(self, tmp_path):
        result = _run_workflow(tmp_path)
        assert isinstance(result["narration"], str)
        assert len(result["narration"]) > 0

    def test_ollama_connection_error_returns_fallback(self, tmp_path):
        fake_output = _fake_output()
        with (
            patch("nominal_drift.core.orchestrator.solve_diffusion",
                  return_value=fake_output),
            patch("nominal_drift.core.orchestrator.plot_concentration_profile",
                  side_effect=_fake_plot_fn),
            patch("nominal_drift.core.orchestrator.animate_diffusion",
                  side_effect=_fake_anim_fn),
            patch("nominal_drift.core.orchestrator.narrate_diffusion_result",
                  side_effect=OllamaConnectionError("server down")),
        ):
            result = run_showcase_workflow(
                _COMP, _SCHED,
                base_output_dir=str(tmp_path / "outputs"),
                db_path=str(tmp_path / "test.db"),
            )
        # Must not raise — narration fallback message returned instead
        assert "narration unavailable" in result["narration"].lower()

    def test_ollama_failure_does_not_prevent_file_creation(self, tmp_path):
        fake_output = _fake_output()
        with (
            patch("nominal_drift.core.orchestrator.solve_diffusion",
                  return_value=fake_output),
            patch("nominal_drift.core.orchestrator.plot_concentration_profile",
                  side_effect=_fake_plot_fn),
            patch("nominal_drift.core.orchestrator.animate_diffusion",
                  side_effect=_fake_anim_fn),
            patch("nominal_drift.core.orchestrator.narrate_diffusion_result",
                  side_effect=OllamaConnectionError("server down")),
        ):
            result = run_showcase_workflow(
                _COMP, _SCHED,
                base_output_dir=str(tmp_path / "outputs"),
                db_path=str(tmp_path / "test.db"),
            )
        assert Path(result["plot_path"]).exists()
        assert Path(result["animation_path"]).exists()

    def test_ollama_failure_does_not_prevent_db_write(self, tmp_path):
        fake_output = _fake_output()
        db = str(tmp_path / "test.db")
        with (
            patch("nominal_drift.core.orchestrator.solve_diffusion",
                  return_value=fake_output),
            patch("nominal_drift.core.orchestrator.plot_concentration_profile",
                  side_effect=_fake_plot_fn),
            patch("nominal_drift.core.orchestrator.animate_diffusion",
                  side_effect=_fake_anim_fn),
            patch("nominal_drift.core.orchestrator.narrate_diffusion_result",
                  side_effect=OllamaConnectionError("server down")),
        ):
            result = run_showcase_workflow(
                _COMP, _SCHED,
                base_output_dir=str(tmp_path / "outputs"),
                db_path=db,
            )
        stored = read_experiment(result["experiment_id"], db_path=db)
        assert stored["experiment_id"] == result["experiment_id"]


# ---------------------------------------------------------------------------
# E. Warning propagation
# ---------------------------------------------------------------------------

class TestWarningPropagation:
    """Solver warnings must flow through to the return dict."""

    def test_empty_warnings_returns_empty_list(self, tmp_path):
        result = _run_workflow(tmp_path, warnings=[])
        assert result["warnings"] == []

    def test_single_warning_propagated(self, tmp_path):
        result = _run_workflow(tmp_path, warnings=["domain boundary approaching"])
        assert "domain boundary approaching" in result["warnings"]

    def test_multiple_warnings_all_propagated(self, tmp_path):
        warns = ["boundary warning", "check x_max_m"]
        result = _run_workflow(tmp_path, warnings=warns)
        for w in warns:
            assert w in result["warnings"]

    def test_warnings_is_list_type(self, tmp_path):
        result = _run_workflow(tmp_path, warnings=["some warning"])
        assert isinstance(result["warnings"], list)


# ---------------------------------------------------------------------------
# F. Path overrides
# ---------------------------------------------------------------------------

class TestPathOverrides:
    """Custom base_output_dir and db_path must be respected."""

    def test_custom_base_output_dir_used(self, tmp_path):
        custom_dir = str(tmp_path / "my_custom_outputs")
        result = _run_workflow(tmp_path)
        # The _run_workflow helper always passes tmp_path/outputs — check it's there
        assert str(tmp_path / "outputs") in result["output_dir"]

    def test_run_dir_is_under_base_output_dir(self, tmp_path):
        base = str(tmp_path / "outputs")
        result = _run_workflow(tmp_path)
        assert result["output_dir"].startswith(base)

    def test_custom_db_path_writes_to_correct_location(self, tmp_path):
        db = str(tmp_path / "custom_db" / "runs.db")
        fake_output = _fake_output()
        with (
            patch("nominal_drift.core.orchestrator.solve_diffusion",
                  return_value=fake_output),
            patch("nominal_drift.core.orchestrator.plot_concentration_profile",
                  side_effect=_fake_plot_fn),
            patch("nominal_drift.core.orchestrator.animate_diffusion",
                  side_effect=_fake_anim_fn),
            patch("nominal_drift.core.orchestrator.narrate_diffusion_result",
                  return_value="ok"),
        ):
            result = run_showcase_workflow(
                _COMP, _SCHED,
                base_output_dir=str(tmp_path / "outputs"),
                db_path=db,
            )
        # The DB file must exist at the custom path
        from pathlib import Path as _P
        assert _P(db).exists()
        stored = read_experiment(result["experiment_id"], db_path=db)
        assert stored["experiment_id"] == result["experiment_id"]

    def test_two_runs_produce_different_output_dirs(self, tmp_path):
        result1 = _run_workflow(tmp_path)
        result2 = _run_workflow(tmp_path)
        assert result1["output_dir"] != result2["output_dir"]

    def test_two_runs_produce_different_experiment_ids(self, tmp_path):
        result1 = _run_workflow(tmp_path)
        result2 = _run_workflow(tmp_path)
        assert result1["experiment_id"] != result2["experiment_id"]
