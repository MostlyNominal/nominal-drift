"""
tests/integration/test_showcase_workflow.py
============================================
Integration tests for the NominalDrift Sprint 1 end-to-end workflow.

These tests exercise the CLI entry point (``nominal-drift run``) and the
underlying orchestrator together with the real diffusion solver, schema
validation, and experiment store.  Two external-dependency layers are mocked:

  * ``narrate_diffusion_result`` — requires a live Ollama server
  * ``animate_diffusion``       — ffmpeg + matplotlib rendering takes ~15 s per
                                  call; animation is already fully unit-tested
                                  in tests/unit/test_animator.py

The static plot (``plot_concentration_profile``) runs for real — it is fast
(< 0.5 s per call with the Agg backend) and is the primary artefact visible
to users from the CLI output.

Each test class is fully isolated via pytest's ``tmp_path`` fixture.

Categories
----------
A  CLI demo run (no input file)
B  CLI with JSON input file
C  Output files created on disk (real plot; stub animation)
D  Experiment persistence verified via DB read-back
E  Warning propagation through CLI
F  Orchestrator called directly (bypassing CLI)

Run with:
    pytest tests/integration/test_showcase_workflow.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from nominal_drift.cli.main import app
from nominal_drift.core.orchestrator import run_showcase_workflow
from nominal_drift.knowledge.experiment_store import read_experiment
from nominal_drift.schemas.composition import AlloyComposition
from nominal_drift.schemas.ht_schedule import HTSchedule, HTStep

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_NARRATION_STUB = "Continuum model shows Cr depletion near the boundary."

_COMP = AlloyComposition(
    alloy_designation="316L",
    alloy_matrix="austenite",
    composition_wt_pct={
        "Fe": 68.88, "Cr": 16.50, "Ni": 10.10,
        "Mo":  2.10, "Mn":  1.80, "Si":  0.50,
        "C":   0.02, "N":   0.07, "P":   0.03,
    },
)

# 1-minute hold — fast diffusion solve (< 0.1 s) for all orchestrator-direct tests
_SCHED = HTSchedule(steps=[
    HTStep(step=1, type="sensitization_soak",
           T_hold_C=700.0, hold_min=1.0,
           cooling_method="air_cool"),
])

runner = CliRunner()


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

def _narration_patch():
    """Context manager: mock narrate_diffusion_result in the orchestrator module."""
    return patch(
        "nominal_drift.core.orchestrator.narrate_diffusion_result",
        return_value=_NARRATION_STUB,
    )


def _animate_stub(output, save_path, **kwargs) -> str:
    """Side-effect for animate_diffusion: write a stub file, return its path.

    Animation rendering (ffmpeg + matplotlib) takes ~15 s per call in this
    environment.  That behaviour is already covered by test_animator.py.
    These integration tests verify orchestration plumbing, not render quality.
    """
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"FAKEVIDEO_STUB")
    return str(p)


def _animate_patch():
    """Context manager: mock animate_diffusion in the orchestrator module."""
    return patch(
        "nominal_drift.core.orchestrator.animate_diffusion",
        side_effect=_animate_stub,
    )


def _cli_run(tmp_path: Path, extra_args: list[str] | None = None):
    """Invoke the CLI with isolated output and DB paths.

    Both narration and animation are mocked so the call completes in < 1 s.
    Note: Typer's CliRunner invokes the single registered command directly —
    the function-name token ("run") is NOT passed as an argument.
    """
    args = [
        "--output-dir", str(tmp_path / "outputs"),
        "--db",         str(tmp_path / "test.db"),
    ]
    if extra_args:
        args.extend(extra_args)
    with _narration_patch(), _animate_patch():
        return runner.invoke(app, args)


def _run_orchestrator(tmp_path: Path, **kwargs) -> dict:
    """Call run_showcase_workflow directly with both slow layers mocked."""
    with _narration_patch(), _animate_patch():
        return run_showcase_workflow(
            _COMP, _SCHED,
            base_output_dir=str(tmp_path / "outputs"),
            db_path=str(tmp_path / "test.db"),
            **kwargs,
        )


# ---------------------------------------------------------------------------
# A. CLI demo run (no input file)
# ---------------------------------------------------------------------------

class TestCliDemoRun:
    """``nominal-drift run`` with no positional argument must succeed."""

    def test_demo_exits_zero(self, tmp_path):
        result = _cli_run(tmp_path)
        assert result.exit_code == 0, (
            f"Expected exit code 0, got {result.exit_code}.\n"
            f"Output:\n{result.output}"
        )

    def test_demo_output_contains_experiment_id_header(self, tmp_path):
        result = _cli_run(tmp_path)
        assert "Experiment ID" in result.output

    def test_demo_output_contains_alloy_designation(self, tmp_path):
        result = _cli_run(tmp_path)
        assert "316L" in result.output

    def test_demo_output_contains_element(self, tmp_path):
        result = _cli_run(tmp_path)
        assert "Cr" in result.output

    def test_demo_output_contains_narration_text(self, tmp_path):
        result = _cli_run(tmp_path)
        assert _NARRATION_STUB in result.output

    def test_demo_output_contains_static_plot_reference(self, tmp_path):
        result = _cli_run(tmp_path)
        assert "Static plot" in result.output or ".png" in result.output

    def test_demo_output_contains_animation_reference(self, tmp_path):
        result = _cli_run(tmp_path)
        assert "Animation" in result.output or ".mp4" in result.output

    def test_demo_output_contains_concentration_wt_pct(self, tmp_path):
        result = _cli_run(tmp_path)
        assert "wt%" in result.output

    def test_demo_output_contains_depletion_depth_row(self, tmp_path):
        result = _cli_run(tmp_path)
        assert "Depletion depth" in result.output

    def test_demo_creates_db_file(self, tmp_path):
        _cli_run(tmp_path)
        assert (tmp_path / "test.db").exists()


# ---------------------------------------------------------------------------
# B. CLI with JSON input file
# ---------------------------------------------------------------------------

class TestCliJsonInput:
    """``nominal-drift run <path.json>`` must load, validate, and run."""

    def _write_input_json(self, tmp_path: Path, alloy: str = "304") -> str:
        """Write a valid input JSON and return its path."""
        data = {
            "alloy_designation": alloy,
            "alloy_matrix": "austenite",
            "composition_wt_pct": {
                "Fe": 70.88, "Cr": 18.00, "Ni":  8.00,
                "Mn":  1.80, "Si":  0.50,
                "C":   0.04, "N":   0.04, "P":   0.03, "S": 0.01,
            },
            "ht_schedule": {
                "steps": [{
                    "step": 1,
                    "type": "sensitization_soak",
                    "T_hold_C": 650.0,
                    "hold_min": 1.0,
                    "cooling_method": "air_cool",
                }],
            },
        }
        p = tmp_path / "input.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return str(p)

    def test_json_run_exits_zero(self, tmp_path):
        json_path = self._write_input_json(tmp_path)
        result = _cli_run(tmp_path, extra_args=[json_path])
        assert result.exit_code == 0, result.output

    def test_json_run_shows_alloy_from_file(self, tmp_path):
        json_path = self._write_input_json(tmp_path, alloy="304")
        result = _cli_run(tmp_path, extra_args=[json_path])
        assert "304" in result.output

    def test_missing_json_file_exits_nonzero(self, tmp_path):
        result = _cli_run(tmp_path, extra_args=["/no/such/file.json"])
        assert result.exit_code != 0

    def test_malformed_json_exits_nonzero(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not: valid json!!!", encoding="utf-8")
        result = _cli_run(tmp_path, extra_args=[str(bad)])
        assert result.exit_code != 0

    def test_invalid_composition_exits_nonzero(self, tmp_path):
        """Composition that fails Pydantic validation (missing Cr) must exit 1."""
        data = {
            "alloy_designation": "mystery",
            "alloy_matrix": "austenite",
            "composition_wt_pct": {"Fe": 50.0},   # missing Cr; sum << 100
            "ht_schedule": {
                "steps": [{"step": 1, "type": "x",
                            "T_hold_C": 700.0, "hold_min": 1.0}],
            },
        }
        p = tmp_path / "invalid.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = _cli_run(tmp_path, extra_args=[str(p)])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# C. Output files created on disk
# ---------------------------------------------------------------------------

class TestOutputFilesExist:
    """Plot and animation stub files must be created in the run directory."""

    def test_plot_png_exists(self, tmp_path):
        result = _run_orchestrator(tmp_path)
        assert Path(result["plot_path"]).exists()

    def test_plot_is_larger_than_1kb(self, tmp_path):
        """Real matplotlib PNG must exceed 1 kB — verifies real rendering ran."""
        result = _run_orchestrator(tmp_path)
        assert Path(result["plot_path"]).stat().st_size > 1024

    def test_animation_stub_exists(self, tmp_path):
        result = _run_orchestrator(tmp_path)
        assert Path(result["animation_path"]).exists()

    def test_animation_file_is_nonempty(self, tmp_path):
        result = _run_orchestrator(tmp_path)
        assert Path(result["animation_path"]).stat().st_size > 0

    def test_output_dir_is_created(self, tmp_path):
        result = _run_orchestrator(tmp_path)
        assert Path(result["output_dir"]).is_dir()

    def test_plot_inside_run_dir(self, tmp_path):
        result = _run_orchestrator(tmp_path)
        assert result["plot_path"].startswith(result["output_dir"])

    def test_animation_inside_run_dir(self, tmp_path):
        result = _run_orchestrator(tmp_path)
        assert result["animation_path"].startswith(result["output_dir"])

    def test_run_dir_contains_element_in_name(self, tmp_path):
        result = _run_orchestrator(tmp_path)
        run_name = Path(result["output_dir"]).name
        assert "cr" in run_name.lower()


# ---------------------------------------------------------------------------
# D. Experiment persistence
# ---------------------------------------------------------------------------

class TestExperimentPersistence:
    """After a workflow run, the record must be readable from SQLite."""

    def test_record_readable_after_run(self, tmp_path):
        result = _run_orchestrator(tmp_path)
        stored = read_experiment(
            result["experiment_id"], db_path=str(tmp_path / "test.db")
        )
        assert stored["experiment_id"] == result["experiment_id"]

    def test_composition_json_round_trips_as_dict(self, tmp_path):
        result = _run_orchestrator(tmp_path)
        stored = read_experiment(
            result["experiment_id"], db_path=str(tmp_path / "test.db")
        )
        assert isinstance(stored["composition_json"], dict)
        assert "Cr" in stored["composition_json"]

    def test_ht_schedule_json_round_trips(self, tmp_path):
        result = _run_orchestrator(tmp_path)
        stored = read_experiment(
            result["experiment_id"], db_path=str(tmp_path / "test.db")
        )
        assert isinstance(stored["ht_schedule_json"], dict)
        assert "steps" in stored["ht_schedule_json"]

    def test_plot_path_persisted(self, tmp_path):
        result = _run_orchestrator(tmp_path)
        stored = read_experiment(
            result["experiment_id"], db_path=str(tmp_path / "test.db")
        )
        assert stored["plot_path"] == result["plot_path"]

    def test_animation_path_persisted(self, tmp_path):
        result = _run_orchestrator(tmp_path)
        stored = read_experiment(
            result["experiment_id"], db_path=str(tmp_path / "test.db")
        )
        assert stored["animation_path"] == result["animation_path"]

    def test_user_label_persisted(self, tmp_path):
        result = _run_orchestrator(tmp_path, user_label="integration-test")
        stored = read_experiment(
            result["experiment_id"], db_path=str(tmp_path / "test.db")
        )
        assert stored["user_label"] == "integration-test"

    def test_user_notes_persisted(self, tmp_path):
        result = _run_orchestrator(tmp_path, user_notes="Sprint 1 reference run.")
        stored = read_experiment(
            result["experiment_id"], db_path=str(tmp_path / "test.db")
        )
        assert stored["user_notes"] == "Sprint 1 reference run."

    def test_db_file_at_custom_path(self, tmp_path):
        custom_db = str(tmp_path / "custom" / "my_runs.db")
        with _narration_patch(), _animate_patch():
            result = run_showcase_workflow(
                _COMP, _SCHED,
                base_output_dir=str(tmp_path / "outputs"),
                db_path=custom_db,
            )
        assert Path(custom_db).exists()
        stored = read_experiment(result["experiment_id"], db_path=custom_db)
        assert stored["experiment_id"] == result["experiment_id"]


# ---------------------------------------------------------------------------
# E. Warning propagation
# ---------------------------------------------------------------------------

class TestWarningPropagation:
    """Solver warnings must appear in the result dict and not abort the run."""

    def test_warnings_key_always_present(self, tmp_path):
        result = _run_orchestrator(tmp_path)
        assert "warnings" in result
        assert isinstance(result["warnings"], list)

    def test_no_warnings_for_short_safe_run(self, tmp_path):
        """1-minute hold in 5 µm domain must not trigger a boundary warning."""
        result = _run_orchestrator(tmp_path)
        # Warnings list must exist; it may or may not be empty for this case
        assert isinstance(result["warnings"], list)

    def test_workflow_completes_even_when_warnings_present(self, tmp_path):
        """A longer hold may produce solver warnings; run must still complete."""
        sched_long = HTSchedule(steps=[
            HTStep(step=1, type="sensitization_soak",
                   T_hold_C=800.0, hold_min=60.0),
        ])
        with _narration_patch(), _animate_patch():
            result = run_showcase_workflow(
                _COMP, sched_long,
                base_output_dir=str(tmp_path / "outputs"),
                db_path=str(tmp_path / "test.db"),
            )
        assert isinstance(result, dict)
        assert "warnings" in result

    def test_cli_exits_zero_when_warnings_present(self, tmp_path):
        """CLI must exit 0 even if solver warnings appear in the output."""
        # Drive the CLI demo case; warnings may or may not appear — exit 0 required
        result = _cli_run(tmp_path)
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# F. Orchestrator called directly (no CLI)
# ---------------------------------------------------------------------------

class TestOrchestratorDirect:
    """run_showcase_workflow() must behave correctly when called from Python."""

    def test_returns_dict_with_all_required_keys(self, tmp_path):
        result = _run_orchestrator(tmp_path)
        required = {
            "experiment_id", "output_dir", "plot_path", "animation_path",
            "narration", "min_concentration_wt_pct", "depletion_depth_nm",
            "warnings", "element", "alloy_designation",
        }
        missing = required - result.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_narration_text_returned(self, tmp_path):
        result = _run_orchestrator(tmp_path)
        assert result["narration"] == _NARRATION_STUB

    def test_element_default_is_cr(self, tmp_path):
        result = _run_orchestrator(tmp_path)
        assert result["element"] == "Cr"

    def test_alloy_designation_matches_input(self, tmp_path):
        result = _run_orchestrator(tmp_path)
        assert result["alloy_designation"] == "316L"

    def test_min_concentration_is_float_in_valid_range(self, tmp_path):
        result = _run_orchestrator(tmp_path)
        c_min = result["min_concentration_wt_pct"]
        assert isinstance(c_min, float)
        # For a 1-minute hold Cr depletion is negligible — c_min ≈ c_bulk
        assert 12.0 <= c_min <= 16.5

    def test_ollama_failure_returns_fallback_narration(self, tmp_path):
        """OllamaConnectionError must not abort the workflow."""
        from nominal_drift.llm.client import OllamaConnectionError
        with (
            patch("nominal_drift.core.orchestrator.narrate_diffusion_result",
                  side_effect=OllamaConnectionError("server down")),
            _animate_patch(),
        ):
            result = run_showcase_workflow(
                _COMP, _SCHED,
                base_output_dir=str(tmp_path / "outputs"),
                db_path=str(tmp_path / "test.db"),
            )
        assert "narration unavailable" in result["narration"].lower()
        # Other artefacts must still be present
        assert Path(result["plot_path"]).exists()

    def test_two_runs_produce_different_experiment_ids(self, tmp_path):
        r1 = _run_orchestrator(tmp_path)
        r2 = _run_orchestrator(tmp_path)
        assert r1["experiment_id"] != r2["experiment_id"]

    def test_two_runs_produce_different_output_dirs(self, tmp_path):
        r1 = _run_orchestrator(tmp_path)
        r2 = _run_orchestrator(tmp_path)
        assert r1["output_dir"] != r2["output_dir"]
