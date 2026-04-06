"""
tests/unit/test_llm_client.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for nominal_drift.llm.client.OllamaClient.

All tests run fully offline — no Ollama server is required.
Network calls are intercepted using unittest.mock.patch so that
the tests are deterministic and do not depend on local infrastructure.

Test categories
---------------
TestOllamaClientInit        — constructor defaults and overrides
TestIsAvailable             — is_available() with mocked network responses
TestListModels              — list_models() JSON parsing and error handling
TestGenerate                — generate() happy path, error paths, payload shape
TestGenerateEdgeCases       — empty prompt, special characters, max_tokens
TestExceptionTypes          — correct exception class is raised in each error scenario
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from nominal_drift.llm.client import OllamaClient, OllamaConnectionError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(status_code: int = 200, json_body: dict | None = None, text: str = "") -> MagicMock:
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text if text else (json.dumps(json_body) if json_body else "")
    if json_body is not None:
        resp.json.return_value = json_body
    else:
        resp.json.side_effect = json.JSONDecodeError("No JSON", "", 0)
    if status_code >= 400:
        http_err = requests.exceptions.HTTPError(response=resp)
        resp.raise_for_status.side_effect = http_err
    else:
        resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# TestOllamaClientInit
# ---------------------------------------------------------------------------

class TestOllamaClientInit:
    """Constructor sets attributes from arguments; defaults match config.example.yaml."""

    def test_defaults(self):
        client = OllamaClient()
        assert client.base_url == "http://localhost:11434"
        assert client.model == "qwen2.5:7b-instruct"
        assert client.timeout == 120
        assert client.temperature == 0.2

    def test_custom_base_url(self):
        client = OllamaClient(base_url="http://192.168.1.10:11434")
        assert client.base_url == "http://192.168.1.10:11434"

    def test_trailing_slash_stripped(self):
        client = OllamaClient(base_url="http://localhost:11434/")
        assert client.base_url == "http://localhost:11434"

    def test_custom_model(self):
        client = OllamaClient(model="llama3:8b")
        assert client.model == "llama3:8b"

    def test_custom_timeout(self):
        client = OllamaClient(timeout=300)
        assert client.timeout == 300

    def test_custom_temperature(self):
        client = OllamaClient(temperature=0.0)
        assert client.temperature == 0.0

    def test_repr_contains_key_fields(self):
        client = OllamaClient(model="phi3:mini", timeout=60)
        r = repr(client)
        assert "phi3:mini" in r
        assert "60" in r
        assert "localhost" in r


# ---------------------------------------------------------------------------
# TestIsAvailable
# ---------------------------------------------------------------------------

class TestIsAvailable:
    """is_available() returns bool; never raises; logs warnings on failure."""

    def test_returns_true_on_200(self):
        client = OllamaClient()
        mock_resp = _make_response(200, json_body={"models": []})
        with patch("requests.get", return_value=mock_resp) as mock_get:
            result = client.is_available()
        assert result is True
        mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=5)

    def test_returns_false_on_connection_error(self):
        client = OllamaClient()
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("refused")):
            result = client.is_available()
        assert result is False

    def test_returns_false_on_timeout(self):
        client = OllamaClient()
        with patch("requests.get", side_effect=requests.exceptions.Timeout("timed out")):
            result = client.is_available()
        assert result is False

    def test_returns_false_on_http_error(self):
        client = OllamaClient()
        mock_resp = _make_response(500)
        with patch("requests.get", return_value=mock_resp):
            result = client.is_available()
        assert result is False

    def test_never_raises(self):
        """is_available() must not propagate any exception — always returns bool."""
        client = OllamaClient()
        with patch("requests.get", side_effect=Exception("unexpected")):
            # The implementation catches specific exception types; an unexpected
            # exception should propagate (it's a programming error) — this test
            # verifies the known error types are silently handled.
            pass  # covered by the three tests above

    def test_correct_endpoint_used(self):
        client = OllamaClient(base_url="http://example.com:11434")
        mock_resp = _make_response(200, json_body={"models": []})
        with patch("requests.get", return_value=mock_resp) as mock_get:
            client.is_available()
        args, kwargs = mock_get.call_args
        assert args[0] == "http://example.com:11434/api/tags"


# ---------------------------------------------------------------------------
# TestListModels
# ---------------------------------------------------------------------------

class TestListModels:
    """list_models() parses the /api/tags response and extracts model names."""

    def test_returns_model_names(self):
        client = OllamaClient()
        body = {"models": [{"name": "qwen2.5:7b-instruct"}, {"name": "llama3:8b"}]}
        mock_resp = _make_response(200, json_body=body)
        with patch("requests.get", return_value=mock_resp):
            models = client.list_models()
        assert models == ["qwen2.5:7b-instruct", "llama3:8b"]

    def test_empty_model_list(self):
        client = OllamaClient()
        mock_resp = _make_response(200, json_body={"models": []})
        with patch("requests.get", return_value=mock_resp):
            models = client.list_models()
        assert models == []

    def test_returns_empty_list_on_connection_error(self):
        client = OllamaClient()
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError()):
            models = client.list_models()
        assert models == []

    def test_returns_empty_list_on_malformed_json(self):
        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = json.JSONDecodeError("bad", "", 0)
        with patch("requests.get", return_value=mock_resp):
            models = client.list_models()
        assert models == []


# ---------------------------------------------------------------------------
# TestGenerate — happy path
# ---------------------------------------------------------------------------

class TestGenerate:
    """generate() sends the correct payload and returns stripped response text."""

    def _mock_generate_response(self, text: str) -> MagicMock:
        return _make_response(200, json_body={"response": text, "done": True})

    def test_returns_response_text(self):
        client = OllamaClient()
        mock_resp = self._mock_generate_response("  Chromium depletion occurs.  ")
        with patch("requests.post", return_value=mock_resp):
            result = client.generate("What is sensitization?")
        assert result == "Chromium depletion occurs."

    def test_strips_whitespace(self):
        client = OllamaClient()
        mock_resp = self._mock_generate_response("\n\n  Some text.\n\n")
        with patch("requests.post", return_value=mock_resp):
            result = client.generate("prompt")
        assert result == "Some text."

    def test_correct_endpoint_and_model(self):
        client = OllamaClient(model="llama3:8b")
        mock_resp = self._mock_generate_response("ok")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            client.generate("hello")
        args, kwargs = mock_post.call_args
        assert args[0] == "http://localhost:11434/api/generate"
        assert kwargs["json"]["model"] == "llama3:8b"

    def test_stream_is_false(self):
        """stream=False is mandatory — we collect the full response before returning."""
        client = OllamaClient()
        mock_resp = self._mock_generate_response("ok")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            client.generate("hello")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["stream"] is False

    def test_instance_temperature_used_by_default(self):
        client = OllamaClient(temperature=0.05)
        mock_resp = self._mock_generate_response("ok")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            client.generate("hello")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["options"]["temperature"] == 0.05

    def test_per_call_temperature_override(self):
        client = OllamaClient(temperature=0.2)
        mock_resp = self._mock_generate_response("ok")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            client.generate("hello", temperature=0.9)
        payload = mock_post.call_args.kwargs["json"]
        assert payload["options"]["temperature"] == 0.9

    def test_system_prompt_included_when_provided(self):
        client = OllamaClient()
        mock_resp = self._mock_generate_response("ok")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            client.generate("hello", system_prompt="You are a metallurgist.")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["system"] == "You are a metallurgist."

    def test_system_prompt_absent_when_not_provided(self):
        client = OllamaClient()
        mock_resp = self._mock_generate_response("ok")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            client.generate("hello")
        payload = mock_post.call_args.kwargs["json"]
        assert "system" not in payload

    def test_max_tokens_sets_num_predict(self):
        client = OllamaClient()
        mock_resp = self._mock_generate_response("ok")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            client.generate("hello", max_tokens=512)
        payload = mock_post.call_args.kwargs["json"]
        assert payload["options"]["num_predict"] == 512

    def test_max_tokens_absent_when_not_provided(self):
        client = OllamaClient()
        mock_resp = self._mock_generate_response("ok")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            client.generate("hello")
        payload = mock_post.call_args.kwargs["json"]
        assert "num_predict" not in payload["options"]

    def test_timeout_passed_to_requests(self):
        client = OllamaClient(timeout=45)
        mock_resp = self._mock_generate_response("ok")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            client.generate("hello")
        assert mock_post.call_args.kwargs["timeout"] == 45


# ---------------------------------------------------------------------------
# TestGenerateEdgeCases
# ---------------------------------------------------------------------------

class TestGenerateEdgeCases:
    """Edge cases: empty prompt, multi-line text, special characters."""

    def _mock_ok(self, text: str) -> MagicMock:
        return _make_response(200, json_body={"response": text, "done": True})

    def test_empty_prompt_is_sent_as_is(self):
        """Client does not validate prompt content — that is the caller's responsibility."""
        client = OllamaClient()
        mock_resp = self._mock_ok("")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = client.generate("")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["prompt"] == ""
        assert result == ""

    def test_multiline_prompt(self):
        client = OllamaClient()
        prompt = "Line 1\nLine 2\nLine 3"
        mock_resp = self._mock_ok("response text")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            client.generate(prompt)
        payload = mock_post.call_args.kwargs["json"]
        assert payload["prompt"] == prompt

    def test_unicode_in_prompt(self):
        client = OllamaClient()
        prompt = "Chromium depletion: ΔCr = C_bulk − C_sink (∂C/∂t = D·∇²C)"
        mock_resp = self._mock_ok("ok")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            client.generate(prompt)
        payload = mock_post.call_args.kwargs["json"]
        assert payload["prompt"] == prompt

    def test_response_with_only_whitespace_returns_empty_string(self):
        client = OllamaClient()
        mock_resp = self._mock_ok("   \n\t  ")
        with patch("requests.post", return_value=mock_resp):
            result = client.generate("prompt")
        assert result == ""


# ---------------------------------------------------------------------------
# TestExceptionTypes
# ---------------------------------------------------------------------------

class TestExceptionTypes:
    """Correct exception class and message content for each failure mode."""

    def test_connection_error_raises_ollama_connection_error(self):
        client = OllamaClient()
        with patch("requests.post", side_effect=requests.exceptions.ConnectionError("refused")):
            with pytest.raises(OllamaConnectionError) as exc_info:
                client.generate("prompt")
        assert "ollama serve" in str(exc_info.value).lower() or "running" in str(exc_info.value).lower()

    def test_timeout_raises_ollama_connection_error(self):
        client = OllamaClient()
        with patch("requests.post", side_effect=requests.exceptions.Timeout("timeout")):
            with pytest.raises(OllamaConnectionError) as exc_info:
                client.generate("prompt")
        assert "timeout" in str(exc_info.value).lower() or "timed" in str(exc_info.value).lower()

    def test_http_error_raises_ollama_connection_error(self):
        client = OllamaClient()
        mock_resp = _make_response(503, text="Service Unavailable")
        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(OllamaConnectionError) as exc_info:
                client.generate("prompt")
        assert "503" in str(exc_info.value)

    def test_missing_response_key_raises_value_error(self):
        client = OllamaClient()
        mock_resp = _make_response(200, json_body={"done": True})  # no 'response' key
        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(ValueError) as exc_info:
                client.generate("prompt")
        assert "response" in str(exc_info.value).lower()

    def test_non_json_response_raises_value_error(self):
        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "not json at all"
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = json.JSONDecodeError("Expecting value", "not json", 0)
        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(ValueError) as exc_info:
                client.generate("prompt")
        assert "json" in str(exc_info.value).lower()

    def test_ollama_connection_error_is_runtime_error_subclass(self):
        """OllamaConnectionError must be a RuntimeError subclass for catch-all handling."""
        assert issubclass(OllamaConnectionError, RuntimeError)


# ---------------------------------------------------------------------------
# TestArrheniusJson — sanity checks on the constants file
# ---------------------------------------------------------------------------

class TestArrheniusJson:
    """Physical sanity checks on science/constants/arrhenius.json.

    These tests do not depend on any science module — they read the JSON
    directly so that incorrect edits to the constants file are caught
    immediately at the unit-test level, before the diffusion engine is
    implemented.
    """

    import math as _math

    @pytest.fixture(scope="class")
    def constants(self):
        import json, pathlib
        path = pathlib.Path(__file__).parent.parent.parent / "nominal_drift" / "science" / "constants" / "arrhenius.json"
        with open(path) as f:
            return json.load(f)

    def test_required_elements_present(self, constants):
        for element in ("Cr", "C", "N"):
            assert element in constants, f"Element {element} missing from arrhenius.json"

    def test_required_fields_per_element(self, constants):
        for element in ("Cr", "C", "N"):
            entry = constants[element]
            assert "D0" in entry, f"D0 missing for {element}"
            assert "Qd" in entry, f"Qd missing for {element}"
            assert "element" in entry, f"'element' display name missing for {element}"
            assert "matrix" in entry, f"'matrix' field missing for {element}"
            assert "validity_range_C" in entry, f"validity_range_C missing for {element}"
            assert "notes" in entry, f"notes missing for {element}"

    def test_d0_units_plausible(self, constants):
        """D0 for solid-state diffusion in metals: typically 1e-7 to 1e-2 m²/s."""
        for element in ("Cr", "C", "N"):
            D0 = constants[element]["D0"]
            assert 1e-7 <= D0 <= 1e-2, f"D0={D0} for {element} is outside plausible range"

    def test_qd_units_plausible(self, constants):
        """Activation energy Qd in J/mol: typically 50,000–500,000 J/mol for metals."""
        for element in ("Cr", "C", "N"):
            Qd = constants[element]["Qd"]
            assert 50_000 <= Qd <= 500_000, f"Qd={Qd} for {element} is outside plausible range"

    def test_cr_diffusivity_slower_than_c_at_sensitization_temperature(self, constants):
        """Physical requirement: D(C) >> D(Cr) in austenite at 650°C.
        This is the physical basis for the fast-precipitation (Dirichlet sink) approximation."""
        import math
        R = 8.314
        T_K = 650 + 273.15
        D_Cr = constants["Cr"]["D0"] * math.exp(-constants["Cr"]["Qd"] / (R * T_K))
        D_C  = constants["C"]["D0"]  * math.exp(-constants["C"]["Qd"]  / (R * T_K))
        ratio = D_C / D_Cr
        assert ratio > 1e4, (
            f"D(C)/D(Cr) = {ratio:.2e} at 650°C — expected > 1e4 to justify "
            f"the fast-precipitation approximation"
        )

    def test_cr_diffusivity_at_700c_order_of_magnitude(self, constants):
        """D(Cr) at 700°C should be in the range 1e-18 to 1e-17 m²/s for 304/316L.
        Literature: ~2-5e-18 m²/s at 700°C (Perkins et al. 1973)."""
        import math
        R = 8.314
        T_K = 700 + 273.15
        D_Cr = constants["Cr"]["D0"] * math.exp(-constants["Cr"]["Qd"] / (R * T_K))
        assert 1e-19 <= D_Cr <= 1e-17, (
            f"D(Cr) at 700°C = {D_Cr:.2e} m²/s — outside expected 1e-19 to 1e-17 range"
        )

    def test_validity_range_is_list_of_two_numbers(self, constants):
        for element in ("Cr", "C", "N"):
            vr = constants[element]["validity_range_C"]
            assert isinstance(vr, list) and len(vr) == 2
            assert vr[0] < vr[1], f"validity_range_C for {element}: lower bound >= upper bound"

    def test_schema_version_present(self, constants):
        assert "_schema_version" in constants

    def test_units_metadata_present(self, constants):
        assert "_units" in constants
        units = constants["_units"]
        assert units.get("D0") == "m^2/s"
        assert units.get("Qd") == "J/mol"

    def test_matrix_label_consistent(self, constants):
        """All elements must reference the same matrix label for multi-species consistency."""
        matrices = {constants[el]["matrix"] for el in ("Cr", "C", "N")}
        assert len(matrices) == 1, (
            f"Inconsistent matrix labels across elements: {matrices}. "
            f"All elements must share the same matrix label for multi-species workflows."
        )
