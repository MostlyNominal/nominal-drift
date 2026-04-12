"""
tests/unit/test_narration.py
============================
TDD tests for nominal_drift.llm.narration (Day 8).

Test categories
---------------
A  Prompt rendering — render_narration_prompt() produces correct content
B  narrate_diffusion_result() — returns str; correct system-prompt handling
C  Warnings in prompt context
D  Assumptions in prompt context
E  Optional db_record handling
F  Multi-species compatibility (Cr, N, C)
G  Ollama client failure surfaces cleanly
H  HT schedule extractor (summarise_ht_schedule)

All LLM calls are mocked.  No live Ollama instance is required.

Run with:
    pytest tests/unit/test_narration.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

from nominal_drift.llm.client import OllamaConnectionError
from nominal_drift.llm import narration
from nominal_drift.llm.extractor import summarise_ht_schedule
from nominal_drift.schemas.composition import AlloyComposition
from nominal_drift.schemas.diffusion_output import DiffusionOutput
from nominal_drift.schemas.ht_schedule import HTSchedule, HTStep


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_output(
    element: str = "Cr",
    c_bulk: float = 16.5,
    c_sink: float = 12.0,
    c_min: float = 12.0,
    depletion_depth_nm: float | None = 42.5,
    warnings: list[str] | None = None,
) -> DiffusionOutput:
    """Build a minimal valid DiffusionOutput for the given element."""
    return DiffusionOutput(
        element=element,
        matrix="austenite_FeCrNi",
        x_m=[0.0, 1.0e-9, 2.0e-9, 3.0e-9],
        t_s=[0.0, 3600.0],
        concentration_profiles=[
            [c_bulk, c_bulk, c_bulk, c_bulk],
            [c_sink, (c_sink + c_bulk) / 2, c_bulk * 0.99, c_bulk],
        ],
        C_bulk_wt_pct=c_bulk,
        C_sink_wt_pct=c_sink,
        min_concentration_wt_pct=c_min,
        depletion_depth_nm=depletion_depth_nm,
        warnings=warnings or [],
        metadata={"element": element, "matrix": "austenite_FeCrNi"},
    )


def _make_composition(designation: str = "316L") -> AlloyComposition:
    return AlloyComposition(
        alloy_designation=designation,
        alloy_matrix="austenite",
        composition_wt_pct={
            "Fe": 68.88, "Cr": 16.50, "Ni": 10.10,
            "Mo": 2.10,  "Mn": 1.80,  "Si": 0.50,
            "C":  0.02,  "N":  0.07,  "P":  0.03,
        },
    )


def _make_schedule(
    T_C: float = 700.0,
    hold_min: float = 60.0,
    step_type: str = "sensitization_soak",
    cooling: str | None = "air_cool",
) -> HTSchedule:
    return HTSchedule(steps=[
        HTStep(
            step=1,
            type=step_type,
            T_hold_C=T_C,
            hold_min=hold_min,
            cooling_method=cooling,
        )
    ])


# ---------------------------------------------------------------------------
# Module-scoped fixtures (immutable objects shared across all tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_output() -> DiffusionOutput:
    return _make_output()


@pytest.fixture(scope="module")
def composition() -> AlloyComposition:
    return _make_composition()


@pytest.fixture(scope="module")
def ht_schedule() -> HTSchedule:
    return _make_schedule()


# ---------------------------------------------------------------------------
# A. Prompt rendering
# ---------------------------------------------------------------------------

class TestPromptRendering:
    """render_narration_prompt() must produce well-formed, populated strings."""

    def test_render_returns_nonempty_string(self, sample_output, composition, ht_schedule):
        result = narration.render_narration_prompt(sample_output, composition, ht_schedule)
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_render_contains_alloy_designation(self, sample_output, composition, ht_schedule):
        prompt = narration.render_narration_prompt(sample_output, composition, ht_schedule)
        assert "316L" in prompt

    def test_render_contains_alloy_matrix(self, sample_output, composition, ht_schedule):
        prompt = narration.render_narration_prompt(sample_output, composition, ht_schedule)
        assert "austenite" in prompt

    def test_render_contains_element_symbol(self, sample_output, composition, ht_schedule):
        prompt = narration.render_narration_prompt(sample_output, composition, ht_schedule)
        assert "Cr" in prompt

    def test_render_contains_c_bulk(self, sample_output, composition, ht_schedule):
        prompt = narration.render_narration_prompt(sample_output, composition, ht_schedule)
        # 16.5 should appear in the numerical results block
        assert "16.5" in prompt

    def test_render_contains_c_sink(self, sample_output, composition, ht_schedule):
        prompt = narration.render_narration_prompt(sample_output, composition, ht_schedule)
        assert "12.0" in prompt

    def test_render_contains_ht_summary(self, sample_output, composition, ht_schedule):
        prompt = narration.render_narration_prompt(sample_output, composition, ht_schedule)
        # Temperature and type must appear somewhere in the rendered text
        assert "700" in prompt
        assert "sensitization_soak" in prompt

    def test_render_contains_depletion_depth_when_set(self, composition, ht_schedule):
        output = _make_output(depletion_depth_nm=42.5)
        prompt = narration.render_narration_prompt(output, composition, ht_schedule)
        assert "42.5" in prompt

    def test_render_shows_not_determined_when_depth_is_none(self, composition, ht_schedule):
        output = _make_output(depletion_depth_nm=None)
        prompt = narration.render_narration_prompt(output, composition, ht_schedule)
        assert "not determined" in prompt.lower()

    def test_render_without_db_record_does_not_raise(
        self, sample_output, composition, ht_schedule
    ):
        # Must not raise even when db_record is absent
        result = narration.render_narration_prompt(
            sample_output, composition, ht_schedule, db_record=None
        )
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# B. narrate_diffusion_result() — client mocked
# ---------------------------------------------------------------------------

class TestNarrationFunction:
    """narrate_diffusion_result() must call the Ollama client and return str."""

    def _mocked_client(self, return_text: str = "Mocked narration.") -> MagicMock:
        mock_client = MagicMock()
        mock_client.generate.return_value = return_text
        return mock_client

    def test_returns_string(self, sample_output, composition, ht_schedule):
        mock_client = self._mocked_client()
        with patch("nominal_drift.llm.narration.OllamaClient", return_value=mock_client):
            result = narration.narrate_diffusion_result(
                sample_output, composition, ht_schedule
            )
        assert isinstance(result, str)

    def test_returns_client_response(self, sample_output, composition, ht_schedule):
        mock_client = self._mocked_client("Domain shows significant Cr depletion.")
        with patch("nominal_drift.llm.narration.OllamaClient", return_value=mock_client):
            result = narration.narrate_diffusion_result(
                sample_output, composition, ht_schedule
            )
        assert result == "Domain shows significant Cr depletion."

    def test_uses_default_system_prompt_when_none(
        self, sample_output, composition, ht_schedule
    ):
        mock_client = self._mocked_client()
        with patch("nominal_drift.llm.narration.OllamaClient", return_value=mock_client):
            narration.narrate_diffusion_result(
                sample_output, composition, ht_schedule, system_prompt=None
            )
        _, kwargs = mock_client.generate.call_args
        assert kwargs.get("system_prompt") == narration.DEFAULT_SYSTEM_PROMPT

    def test_uses_custom_system_prompt_when_supplied(
        self, sample_output, composition, ht_schedule
    ):
        mock_client = self._mocked_client()
        custom_sp = "You are a terse metallurgical reporter."
        with patch("nominal_drift.llm.narration.OllamaClient", return_value=mock_client):
            narration.narrate_diffusion_result(
                sample_output, composition, ht_schedule, system_prompt=custom_sp
            )
        _, kwargs = mock_client.generate.call_args
        assert kwargs.get("system_prompt") == custom_sp

    def test_prompt_passed_to_generate_contains_element(
        self, sample_output, composition, ht_schedule
    ):
        mock_client = self._mocked_client()
        with patch("nominal_drift.llm.narration.OllamaClient", return_value=mock_client):
            narration.narrate_diffusion_result(sample_output, composition, ht_schedule)
        positional_prompt = mock_client.generate.call_args[0][0]
        assert "Cr" in positional_prompt

    def test_prompt_passed_to_generate_is_nonempty(
        self, sample_output, composition, ht_schedule
    ):
        mock_client = self._mocked_client()
        with patch("nominal_drift.llm.narration.OllamaClient", return_value=mock_client):
            narration.narrate_diffusion_result(sample_output, composition, ht_schedule)
        positional_prompt = mock_client.generate.call_args[0][0]
        assert len(positional_prompt.strip()) > 100  # not a trivial stub


# ---------------------------------------------------------------------------
# C. Warnings in prompt context
# ---------------------------------------------------------------------------

class TestWarningsInPrompt:
    """Solver warnings must appear in the rendered prompt."""

    def test_single_warning_in_prompt(self, composition, ht_schedule):
        output = _make_output(warnings=["domain boundary approaching"])
        prompt = narration.render_narration_prompt(output, composition, ht_schedule)
        assert "domain boundary approaching" in prompt

    def test_multiple_warnings_all_in_prompt(self, composition, ht_schedule):
        output = _make_output(warnings=["boundary warning", "check x_max_m"])
        prompt = narration.render_narration_prompt(output, composition, ht_schedule)
        assert "boundary warning" in prompt
        assert "check x_max_m" in prompt

    def test_empty_warnings_no_warnings_section(self, composition, ht_schedule):
        output = _make_output(warnings=[])
        prompt = narration.render_narration_prompt(output, composition, ht_schedule)
        # The SOLVER WARNINGS header should not appear when there are no warnings
        assert "SOLVER WARNINGS" not in prompt

    def test_nonempty_warnings_have_section_header(self, composition, ht_schedule):
        output = _make_output(warnings=["some warning"])
        prompt = narration.render_narration_prompt(output, composition, ht_schedule)
        assert "SOLVER WARNINGS" in prompt

    def test_warning_rule_added_to_task_when_warnings_present(
        self, composition, ht_schedule
    ):
        output = _make_output(warnings=["domain boundary approaching"])
        prompt = narration.render_narration_prompt(output, composition, ht_schedule)
        # The template adds an extra task rule when warnings are present
        assert "solver warning" in prompt.lower()


# ---------------------------------------------------------------------------
# D. Assumptions in prompt context
# ---------------------------------------------------------------------------

class TestAssumptionsInPrompt:
    """Model assumptions must always appear in the rendered prompt."""

    def test_continuum_model_disclaimer_present(self, sample_output, composition, ht_schedule):
        prompt = narration.render_narration_prompt(sample_output, composition, ht_schedule)
        assert "continuum" in prompt.lower()

    def test_not_atomistic_disclaimer_present(self, sample_output, composition, ht_schedule):
        prompt = narration.render_narration_prompt(sample_output, composition, ht_schedule)
        assert "NOT" in prompt and "atomistic" in prompt.lower()

    def test_not_precipitation_kinetics_disclaimer_present(
        self, sample_output, composition, ht_schedule
    ):
        prompt = narration.render_narration_prompt(sample_output, composition, ht_schedule)
        assert "precipitation" in prompt.lower()

    def test_crank_nicolson_mentioned(self, sample_output, composition, ht_schedule):
        prompt = narration.render_narration_prompt(sample_output, composition, ht_schedule)
        assert "Crank" in prompt or "crank" in prompt.lower()

    def test_arrhenius_mentioned(self, sample_output, composition, ht_schedule):
        prompt = narration.render_narration_prompt(sample_output, composition, ht_schedule)
        assert "Arrhenius" in prompt

    def test_dirichlet_boundary_mentioned(self, sample_output, composition, ht_schedule):
        prompt = narration.render_narration_prompt(sample_output, composition, ht_schedule)
        assert "Dirichlet" in prompt

    def test_do_not_invent_numbers_instruction_present(
        self, sample_output, composition, ht_schedule
    ):
        prompt = narration.render_narration_prompt(sample_output, composition, ht_schedule)
        assert "NOT invent" in prompt or "not invent" in prompt.lower()


# ---------------------------------------------------------------------------
# E. Optional db_record handling
# ---------------------------------------------------------------------------

class TestDbRecordHandling:
    """db_record is optional; if supplied, user_label and user_notes are used."""

    def test_user_label_in_prompt_when_db_record_supplied(
        self, sample_output, composition, ht_schedule
    ):
        db_record = {"user_label": "sprint1-ref", "user_notes": None}
        prompt = narration.render_narration_prompt(
            sample_output, composition, ht_schedule, db_record=db_record
        )
        assert "sprint1-ref" in prompt

    def test_user_notes_in_prompt_when_db_record_supplied(
        self, sample_output, composition, ht_schedule
    ):
        db_record = {"user_label": None, "user_notes": "Reference run at 700 °C."}
        prompt = narration.render_narration_prompt(
            sample_output, composition, ht_schedule, db_record=db_record
        )
        assert "Reference run at 700" in prompt

    def test_none_db_record_does_not_insert_label(
        self, sample_output, composition, ht_schedule
    ):
        prompt = narration.render_narration_prompt(
            sample_output, composition, ht_schedule, db_record=None
        )
        assert "Run label" not in prompt

    def test_db_record_with_none_label_does_not_raise(
        self, sample_output, composition, ht_schedule
    ):
        db_record = {"user_label": None, "user_notes": None}
        result = narration.render_narration_prompt(
            sample_output, composition, ht_schedule, db_record=db_record
        )
        assert isinstance(result, str)

    def test_narrate_with_db_record_returns_string(
        self, sample_output, composition, ht_schedule
    ):
        db_record = {"user_label": "my-run", "user_notes": "check depletion"}
        mock_client = MagicMock()
        mock_client.generate.return_value = "Narration with db_record."
        with patch("nominal_drift.llm.narration.OllamaClient", return_value=mock_client):
            result = narration.narrate_diffusion_result(
                sample_output, composition, ht_schedule, db_record=db_record
            )
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# F. Multi-species compatibility
# ---------------------------------------------------------------------------

class TestMultiSpeciesCompatibility:
    """Prompt must render correctly for Cr, N, and C without code changes."""

    @pytest.mark.parametrize("element,c_bulk,c_sink,c_min", [
        ("Cr", 16.5, 12.0, 12.0),
        ("N",  0.07,  0.01, 0.01),
        ("C",  0.02,  0.005, 0.005),
    ])
    def test_element_symbol_appears_in_prompt(
        self, element, c_bulk, c_sink, c_min, composition, ht_schedule
    ):
        output = _make_output(element=element, c_bulk=c_bulk, c_sink=c_sink, c_min=c_min)
        prompt = narration.render_narration_prompt(output, composition, ht_schedule)
        assert element in prompt

    @pytest.mark.parametrize("element,c_bulk,c_sink,c_min", [
        ("Cr", 16.5, 12.0, 12.0),
        ("N",  0.07,  0.01, 0.01),
        ("C",  0.02,  0.005, 0.005),
    ])
    def test_bulk_concentration_appears_in_prompt(
        self, element, c_bulk, c_sink, c_min, composition, ht_schedule
    ):
        output = _make_output(element=element, c_bulk=c_bulk, c_sink=c_sink, c_min=c_min)
        prompt = narration.render_narration_prompt(output, composition, ht_schedule)
        # The rounded bulk value should appear
        assert str(round(c_bulk, 4)) in prompt or f"{c_bulk}" in prompt

    @pytest.mark.parametrize("element,c_bulk,c_sink,c_min", [
        ("Cr", 16.5, 12.0, 12.0),
        ("N",  0.07,  0.01, 0.01),
        ("C",  0.02,  0.005, 0.005),
    ])
    def test_narrate_returns_string_for_each_species(
        self, element, c_bulk, c_sink, c_min, composition, ht_schedule
    ):
        output = _make_output(element=element, c_bulk=c_bulk, c_sink=c_sink, c_min=c_min)
        mock_client = MagicMock()
        mock_client.generate.return_value = f"Narration for {element}."
        with patch("nominal_drift.llm.narration.OllamaClient", return_value=mock_client):
            result = narration.narrate_diffusion_result(output, composition, ht_schedule)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# G. Ollama client failure surfaces cleanly
# ---------------------------------------------------------------------------

class TestClientFailureSurfaces:
    """OllamaConnectionError must propagate from narrate_diffusion_result."""

    def test_connection_error_propagates(
        self, sample_output, composition, ht_schedule
    ):
        mock_client = MagicMock()
        mock_client.generate.side_effect = OllamaConnectionError(
            "Cannot connect to Ollama at http://localhost:11434."
        )
        with patch("nominal_drift.llm.narration.OllamaClient", return_value=mock_client):
            with pytest.raises(OllamaConnectionError):
                narration.narrate_diffusion_result(sample_output, composition, ht_schedule)

    def test_connection_error_message_preserved(
        self, sample_output, composition, ht_schedule
    ):
        original_msg = "Cannot connect to Ollama at http://localhost:11434."
        mock_client = MagicMock()
        mock_client.generate.side_effect = OllamaConnectionError(original_msg)
        with patch("nominal_drift.llm.narration.OllamaClient", return_value=mock_client):
            with pytest.raises(OllamaConnectionError, match="Cannot connect"):
                narration.narrate_diffusion_result(sample_output, composition, ht_schedule)

    def test_value_error_from_bad_response_propagates(
        self, sample_output, composition, ht_schedule
    ):
        mock_client = MagicMock()
        mock_client.generate.side_effect = ValueError(
            "Ollama response JSON is missing the 'response' key."
        )
        with patch("nominal_drift.llm.narration.OllamaClient", return_value=mock_client):
            with pytest.raises(ValueError):
                narration.narrate_diffusion_result(sample_output, composition, ht_schedule)

    def test_template_render_still_runs_before_client_call(
        self, sample_output, composition, ht_schedule
    ):
        """Template rendering must complete before the client call; a client
        failure must not mask a template error."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = OllamaConnectionError("down")
        with patch("nominal_drift.llm.narration.OllamaClient", return_value=mock_client):
            with pytest.raises(OllamaConnectionError):
                narration.narrate_diffusion_result(sample_output, composition, ht_schedule)
        # generate() must have been called exactly once (template rendered OK)
        mock_client.generate.assert_called_once()


# ---------------------------------------------------------------------------
# H. HT schedule extractor
# ---------------------------------------------------------------------------

class TestHTScheduleExtractor:
    """summarise_ht_schedule() must produce human-readable, accurate strings."""

    def test_single_step_contains_temperature(self):
        s = _make_schedule(T_C=700.0)
        summary = summarise_ht_schedule(s)
        assert "700" in summary

    def test_single_step_contains_step_type(self):
        s = _make_schedule(step_type="homogenisation")
        summary = summarise_ht_schedule(s)
        assert "homogenisation" in summary

    def test_minutes_shown_for_sub_hour_hold(self):
        s = _make_schedule(hold_min=30.0)
        summary = summarise_ht_schedule(s)
        assert "min" in summary

    def test_hours_shown_for_exact_hour_hold(self):
        s = _make_schedule(hold_min=120.0)
        summary = summarise_ht_schedule(s)
        assert " h" in summary
        assert "2" in summary

    def test_decimal_hours_shown_for_non_integer_hours(self):
        s = _make_schedule(hold_min=90.0)
        summary = summarise_ht_schedule(s)
        assert "1.5 h" in summary

    def test_cooling_method_included_when_present(self):
        s = _make_schedule(cooling="water_quench")
        summary = summarise_ht_schedule(s)
        assert "water_quench" in summary

    def test_cooling_method_absent_when_none(self):
        s = _make_schedule(cooling=None)
        summary = summarise_ht_schedule(s)
        assert "None" not in summary
        assert "water_quench" not in summary

    def test_two_step_schedule_uses_arrow_separator(self):
        sched = HTSchedule(steps=[
            HTStep(step=1, type="solution_anneal",   T_hold_C=1080.0, hold_min=30.0,
                   cooling_method="water_quench"),
            HTStep(step=2, type="sensitization_soak", T_hold_C=700.0,  hold_min=60.0,
                   cooling_method="air_cool"),
        ])
        summary = summarise_ht_schedule(sched)
        assert " → " in summary

    def test_two_step_schedule_contains_both_temperatures(self):
        sched = HTSchedule(steps=[
            HTStep(step=1, type="solution_anneal",   T_hold_C=1080.0, hold_min=30.0),
            HTStep(step=2, type="sensitization_soak", T_hold_C=700.0,  hold_min=60.0),
        ])
        summary = summarise_ht_schedule(sched)
        assert "1080" in summary
        assert "700" in summary

    def test_steps_ordered_by_step_number(self):
        """Steps should appear in ascending step-number order in the summary."""
        sched = HTSchedule(steps=[
            HTStep(step=1, type="solution_anneal",    T_hold_C=1080.0, hold_min=30.0),
            HTStep(step=2, type="sensitization_soak", T_hold_C=700.0,  hold_min=60.0),
        ])
        summary = summarise_ht_schedule(sched)
        idx_1080 = summary.index("1080")
        idx_700  = summary.index("700")
        assert idx_1080 < idx_700, "Step 1 (1080 °C) must appear before Step 2 (700 °C)"
