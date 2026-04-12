"""
nominal_drift.llm.client
~~~~~~~~~~~~~~~~~~~~~
Thin wrapper around a locally-running Ollama server.

Responsibilities
----------------
* Check whether the Ollama server is reachable (is_available)
* Send a prompt and stream or collect the response text (generate)
* Handle network/timeout errors gracefully with clear messages

The OllamaClient is intentionally stateless and side-effect-free.
It does NOT hold conversation history — each call to generate() is
a fresh, single-turn inference. Session state is managed upstream by
the orchestrator.

Typical usage
-------------
>>> from nominal_drift.llm.client import OllamaClient
>>> client = OllamaClient()
>>> if client.is_available():
...     text = client.generate("Explain sensitization in stainless steel briefly.")
...     print(text)

Configuration
-------------
All parameters have sensible defaults that match config.example.yaml.
Pass overrides explicitly or construct from a loaded config dict.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Default constants matching config.example.yaml
_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = "qwen2.5:7b-instruct"
_DEFAULT_TIMEOUT_S = 120  # seconds; generous to allow for first-token latency on cold model load
_DEFAULT_TEMPERATURE = 0.2  # low temperature: metallurgical narration should be factual, not creative


class OllamaConnectionError(RuntimeError):
    """Raised when the Ollama server is unreachable or returns an unexpected error."""


class OllamaClient:
    """Client for a locally-running Ollama inference server.

    Parameters
    ----------
    base_url:
        Root URL of the Ollama server. Defaults to ``http://localhost:11434``.
    model:
        Ollama model tag to use for generation. Defaults to ``qwen2.5:7b-instruct``.
        Any model that has been pulled via ``ollama pull <model>`` is valid.
    timeout:
        HTTP request timeout in seconds. Increase for large models or slow hardware.
    temperature:
        Sampling temperature passed to the model. Lower = more deterministic.
        Use 0.0–0.3 for factual metallurgical narration; higher values introduce
        creative variability not appropriate for engineering reports.
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        model: str = _DEFAULT_MODEL,
        timeout: int = _DEFAULT_TIMEOUT_S,
        temperature: float = _DEFAULT_TEMPERATURE,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the Ollama server is reachable and responding.

        Performs a lightweight GET to the ``/api/tags`` endpoint.  This
        endpoint lists installed models and is always available when the
        server is running — no model load is required.

        Returns
        -------
        bool
            True if the server returned HTTP 200; False for any connection
            or HTTP error.
        """
        url = f"{self.base_url}/api/tags"
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            return True
        except requests.exceptions.ConnectionError:
            logger.warning(
                "Ollama server not reachable at %s. "
                "Start it with: ollama serve",
                self.base_url,
            )
            return False
        except requests.exceptions.Timeout:
            logger.warning("Ollama availability check timed out at %s", self.base_url)
            return False
        except requests.exceptions.HTTPError as exc:
            logger.warning("Ollama returned unexpected HTTP status: %s", exc)
            return False

    def list_models(self) -> list[str]:
        """Return the names of all locally available models.

        Calls ``GET /api/tags`` and extracts the ``name`` field from each
        model entry.

        Returns
        -------
        list[str]
            Model names (e.g. ``['qwen2.5:7b-instruct', 'llama3:8b']``).
            Returns an empty list if the server is unreachable.
        """
        url = f"{self.base_url}/api/tags"
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Could not retrieve model list: %s", exc)
            return []

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a single-turn prompt and return the complete response as a string.

        Uses the ``POST /api/generate`` endpoint with ``stream=False`` so the
        entire response is collected before returning.  This simplifies error
        handling and is appropriate for the narration use-case where the full
        text is needed before displaying it to the user.

        Parameters
        ----------
        prompt:
            The user-side prompt text (required).
        system_prompt:
            Optional system prompt.  If provided, it is sent as the
            ``system`` field in the Ollama request body.  Use this to set the
            assistant's persona and enforce the mandatory scientific caveats.
        temperature:
            Override the instance-level temperature for this call.
        max_tokens:
            If provided, sets ``num_predict`` in the Ollama options dict.

        Returns
        -------
        str
            The model's response text, stripped of leading/trailing whitespace.

        Raises
        ------
        OllamaConnectionError
            If the server is unreachable or returns a non-2xx HTTP status.
        ValueError
            If the response body cannot be decoded as JSON or is missing the
            expected ``response`` field.
        """
        url = f"{self.base_url}/api/generate"

        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
            },
        }

        if system_prompt is not None:
            payload["system"] = system_prompt

        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Is the server running? (ollama serve)\nDetails: {exc}"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaConnectionError(
                f"Request to Ollama timed out after {self.timeout}s. "
                f"The model may still be loading. Try increasing the timeout.\nDetails: {exc}"
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise OllamaConnectionError(
                f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc

        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Ollama response was not valid JSON. Raw response (first 500 chars): "
                f"{resp.text[:500]}"
            ) from exc

        if "response" not in data:
            raise ValueError(
                f"Ollama response JSON is missing the 'response' key. Keys present: "
                f"{list(data.keys())}"
            )

        return data["response"].strip()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"OllamaClient(base_url={self.base_url!r}, model={self.model!r}, "
            f"timeout={self.timeout}s, temperature={self.temperature})"
        )
