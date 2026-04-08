"""
tests/unit/test_animator.py
============================
TDD tests for nominal_drift.viz.animator.

Test categories:
  1. Animation file creation — file exists, non-trivial size.
  2. Return value — always a plain str; matches the actual saved path.
  3. Writer selection — MP4 when ffmpeg available; GIF fallback when mocked
     unavailable; explicit .gif request always uses PillowWriter.
  4. Function arguments — title, threshold_wt_pct, fps, max_frames.
  5. Frame subsampling — max_frames respected; first/last frames preserved.
  6. Multi-element interface — Cr, N, C all produce valid output files.

Fixtures use synthetic erf-shaped profiles (same helper as test_profile_plotter)
so tests are fast and completely isolated from the diffusion engine.

Two fixture sizes are provided:
  tiny_output   — 5 stored frames, 20 spatial nodes  (fast; used for writer tests)
  sample_output — 12 stored frames, 50 spatial nodes (realistic; used for
                  argument and element tests)

Run with:
    pytest tests/unit/test_animator.py -v
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")

import pytest

from nominal_drift.schemas.diffusion_output import DiffusionOutput


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _erf_profile(
    x_m: list[float],
    t_s: float,
    C_sink: float,
    C_bulk: float,
    D: float = 1.5e-19,
) -> list[float]:
    """Synthetic Cr-depletion profile via the analytical erf solution."""
    profile = []
    for x in x_m:
        if x == 0.0 or t_s <= 0.0:
            c = C_sink if x == 0.0 else C_bulk
        else:
            arg = x / (2.0 * math.sqrt(D * t_s))
            c = C_sink + (C_bulk - C_sink) * math.erf(arg)
            c = max(C_sink, min(C_bulk, c))
        profile.append(c)
    profile[0] = C_sink
    profile[-1] = C_bulk
    return profile


def _make_output(
    element: str = "Cr",
    C_bulk: float = 16.5,
    C_sink: float = 12.0,
    n_x: int = 20,
    n_t: int = 5,
    total_s: float = 7200.0,
    D: float = 1.5e-19,
) -> DiffusionOutput:
    """Build a minimal but valid DiffusionOutput for animator tests."""
    x_max_m = 5e-6
    x_m = [x_max_m * i / (n_x - 1) for i in range(n_x)]
    t_vals = [total_s * i / (n_t - 1) for i in range(n_t)]
    profiles = [_erf_profile(x_m, t, C_sink, C_bulk, D) for t in t_vals]
    return DiffusionOutput(
        element=element,
        matrix="austenite_FeCrNi",
        x_m=x_m,
        t_s=t_vals,
        concentration_profiles=profiles,
        C_bulk_wt_pct=C_bulk,
        C_sink_wt_pct=C_sink,
        min_concentration_wt_pct=C_sink,
        depletion_depth_nm=40.0 if C_sink < C_bulk else None,
        warnings=[],
        metadata={"arrhenius": {}, "solver": {}},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_output() -> DiffusionOutput:
    """5 frames × 20 nodes — fast for writer-selection tests."""
    return _make_output(n_x=20, n_t=5)


@pytest.fixture(scope="module")
def sample_output() -> DiffusionOutput:
    """12 frames × 50 nodes — richer for argument and element tests."""
    return _make_output(n_x=50, n_t=12)


@pytest.fixture(scope="module")
def dense_output() -> DiffusionOutput:
    """400 frames × 20 nodes — triggers max_frames subsampling."""
    return _make_output(n_x=20, n_t=400)


# ---------------------------------------------------------------------------
# Import helper
# ---------------------------------------------------------------------------

def _import_animator():
    from nominal_drift.viz import animator
    return animator


# ---------------------------------------------------------------------------
# 1. Animation file creation
# ---------------------------------------------------------------------------

class TestAnimationFileCreation:
    """Smoke tests: does the function produce a file?"""

    def test_gif_file_is_created(self, tiny_output, tmp_path):
        mod = _import_animator()
        save_path = tmp_path / "test_anim.gif"
        mod.animate_diffusion(tiny_output, save_path, fps=5)
        assert save_path.exists(), f"Expected GIF at {save_path}"

    def test_mp4_file_is_created_when_ffmpeg_available(self, tiny_output, tmp_path):
        from matplotlib.animation import FFMpegWriter
        if not FFMpegWriter.isAvailable():
            pytest.skip("ffmpeg not available in this environment")
        mod = _import_animator()
        save_path = tmp_path / "test_anim.mp4"
        result = mod.animate_diffusion(tiny_output, save_path, fps=5)
        assert Path(result).exists()

    def test_file_size_is_nontrivial_gif(self, tiny_output, tmp_path):
        mod = _import_animator()
        save_path = tmp_path / "size_check.gif"
        mod.animate_diffusion(tiny_output, save_path, fps=5)
        size = os.path.getsize(save_path)
        assert size > 1024, (
            f"GIF file suspiciously small: {size} bytes. Expected > 1 kB."
        )

    def test_file_size_is_nontrivial_mp4(self, tiny_output, tmp_path):
        from matplotlib.animation import FFMpegWriter
        if not FFMpegWriter.isAvailable():
            pytest.skip("ffmpeg not available")
        mod = _import_animator()
        save_path = tmp_path / "size_check.mp4"
        result = mod.animate_diffusion(tiny_output, save_path, fps=5)
        assert os.path.getsize(result) > 1024


# ---------------------------------------------------------------------------
# 2. Return value
# ---------------------------------------------------------------------------

class TestReturnValue:
    """Function must return the actual saved path as a plain str."""

    def test_returns_str_for_gif(self, tiny_output, tmp_path):
        mod = _import_animator()
        result = mod.animate_diffusion(tiny_output, tmp_path / "r.gif", fps=5)
        assert isinstance(result, str), f"Expected str, got {type(result).__name__}"

    def test_returns_str_for_mp4(self, tiny_output, tmp_path):
        from matplotlib.animation import FFMpegWriter
        if not FFMpegWriter.isAvailable():
            pytest.skip("ffmpeg not available")
        mod = _import_animator()
        result = mod.animate_diffusion(tiny_output, tmp_path / "r.mp4", fps=5)
        assert isinstance(result, str)

    def test_gif_returned_path_matches_save_path(self, tiny_output, tmp_path):
        mod = _import_animator()
        save_path = tmp_path / "match.gif"
        result = mod.animate_diffusion(tiny_output, save_path, fps=5)
        assert Path(result) == save_path

    def test_mp4_returned_path_matches_save_path(self, tiny_output, tmp_path):
        from matplotlib.animation import FFMpegWriter
        if not FFMpegWriter.isAvailable():
            pytest.skip("ffmpeg not available")
        mod = _import_animator()
        save_path = tmp_path / "match.mp4"
        result = mod.animate_diffusion(tiny_output, save_path, fps=5)
        assert Path(result) == save_path

    def test_path_object_input_returns_str(self, tiny_output, tmp_path):
        mod = _import_animator()
        result = mod.animate_diffusion(tiny_output, tmp_path / "obj.gif", fps=5)
        assert isinstance(result, str)

    def test_str_input_returns_str(self, tiny_output, tmp_path):
        mod = _import_animator()
        save_path = str(tmp_path / "str.gif")
        result = mod.animate_diffusion(tiny_output, save_path, fps=5)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 3. Writer selection
# ---------------------------------------------------------------------------

class TestWriterSelection:
    """Correct writer chosen based on extension and ffmpeg availability."""

    def test_gif_extension_always_uses_pillow(self, tiny_output, tmp_path):
        """Requesting .gif must use PillowWriter even when ffmpeg is present."""
        mod = _import_animator()
        save_path = tmp_path / "pillow.gif"
        result = mod.animate_diffusion(tiny_output, save_path, fps=5)
        assert result.endswith(".gif")
        assert Path(result).exists()

    def test_gif_fallback_when_ffmpeg_mocked_unavailable(self, tiny_output, tmp_path):
        """
        When FFMpegWriter is mocked as unavailable, a .mp4 request must fall
        back to a .gif file using PillowWriter.  The returned path reflects
        the actual saved extension (.gif).
        """
        mod = _import_animator()
        save_path = tmp_path / "fallback.mp4"

        with patch(
            "nominal_drift.viz.animator.FFMpegWriter.isAvailable",
            return_value=False,
        ):
            result = mod.animate_diffusion(tiny_output, save_path, fps=5)

        assert result.endswith(".gif"), (
            f"Expected .gif fallback path, got: {result}"
        )
        assert Path(result).exists(), f"Fallback GIF not found at {result}"
        assert os.path.getsize(result) > 1024

    def test_gif_fallback_file_different_from_mp4_path(self, tiny_output, tmp_path):
        """Fallback path changes the extension from .mp4 to .gif."""
        mod = _import_animator()
        mp4_path = tmp_path / "ext_check.mp4"

        with patch(
            "nominal_drift.viz.animator.FFMpegWriter.isAvailable",
            return_value=False,
        ):
            result = mod.animate_diffusion(tiny_output, mp4_path, fps=5)

        assert not result.endswith(".mp4"), (
            "When ffmpeg is unavailable the result must not be .mp4"
        )

    def test_mp4_used_when_ffmpeg_available(self, tiny_output, tmp_path):
        from matplotlib.animation import FFMpegWriter
        if not FFMpegWriter.isAvailable():
            pytest.skip("ffmpeg not available")
        mod = _import_animator()
        save_path = tmp_path / "real_mp4.mp4"
        result = mod.animate_diffusion(tiny_output, save_path, fps=5)
        assert result.endswith(".mp4")
        assert Path(result).exists()


# ---------------------------------------------------------------------------
# 4. Function arguments
# ---------------------------------------------------------------------------

class TestFunctionArguments:
    """All optional arguments are accepted without raising."""

    def test_default_call_gif(self, sample_output, tmp_path):
        mod = _import_animator()
        mod.animate_diffusion(sample_output, tmp_path / "default.gif", fps=5)

    def test_with_title(self, sample_output, tmp_path):
        mod = _import_animator()
        mod.animate_diffusion(
            sample_output,
            tmp_path / "titled.gif",
            title="316L sensitisation at 650°C",
            fps=5,
        )

    def test_with_none_title(self, sample_output, tmp_path):
        mod = _import_animator()
        mod.animate_diffusion(
            sample_output,
            tmp_path / "no_title.gif",
            title=None,
            fps=5,
        )

    def test_with_threshold_line(self, sample_output, tmp_path):
        mod = _import_animator()
        mod.animate_diffusion(
            sample_output,
            tmp_path / "threshold.gif",
            threshold_wt_pct=12.0,
            fps=5,
        )

    def test_with_none_threshold(self, sample_output, tmp_path):
        mod = _import_animator()
        mod.animate_diffusion(
            sample_output,
            tmp_path / "no_threshold.gif",
            threshold_wt_pct=None,
            fps=5,
        )

    def test_all_kwargs_combined(self, sample_output, tmp_path):
        mod = _import_animator()
        mod.animate_diffusion(
            sample_output,
            tmp_path / "all_kwargs.gif",
            title="Full kwargs test",
            threshold_wt_pct=13.0,
            fps=5,
            max_frames=50,
        )

    def test_low_fps(self, tiny_output, tmp_path):
        """fps=1 must not raise."""
        mod = _import_animator()
        mod.animate_diffusion(tiny_output, tmp_path / "lowfps.gif", fps=1)

    def test_high_fps(self, tiny_output, tmp_path):
        """fps=30 must not raise."""
        mod = _import_animator()
        mod.animate_diffusion(tiny_output, tmp_path / "highfps.gif", fps=30)


# ---------------------------------------------------------------------------
# 5. Frame subsampling
# ---------------------------------------------------------------------------

class TestFrameSubsampling:
    """max_frames respected; first and last frames always included."""

    def test_dense_output_does_not_raise(self, dense_output, tmp_path):
        """400-frame output with max_frames=10 must complete without error."""
        mod = _import_animator()
        mod.animate_diffusion(
            dense_output,
            tmp_path / "subsampled.gif",
            fps=5,
            max_frames=10,
        )

    def test_subsampled_file_is_created(self, dense_output, tmp_path):
        mod = _import_animator()
        save_path = tmp_path / "sub_created.gif"
        mod.animate_diffusion(dense_output, save_path, fps=5, max_frames=10)
        assert save_path.exists()

    def test_subsampled_file_nontrivial(self, dense_output, tmp_path):
        mod = _import_animator()
        save_path = tmp_path / "sub_size.gif"
        mod.animate_diffusion(dense_output, save_path, fps=5, max_frames=10)
        assert os.path.getsize(save_path) > 1024

    def test_fewer_frames_than_max_animates_all(self, tiny_output, tmp_path):
        """When n_stored < max_frames all frames are used — must not raise."""
        mod = _import_animator()
        mod.animate_diffusion(
            tiny_output,
            tmp_path / "all_frames.gif",
            fps=5,
            max_frames=300,
        )

    def test_select_frame_indices_internal(self):
        """Unit-test the internal frame-selection helper directly."""
        mod = _import_animator()
        # All frames returned when n <= max
        assert mod._select_frame_indices(5, 10) == list(range(5))
        # First and last always included
        indices = mod._select_frame_indices(100, 10)
        assert indices[0] == 0
        assert indices[-1] == 99
        # Length capped at max_frames (may be slightly less due to de-dup)
        assert len(indices) <= 10
        # Single-frame edge case
        assert mod._select_frame_indices(1, 10) == [0]


# ---------------------------------------------------------------------------
# 6. Multi-element interface
# ---------------------------------------------------------------------------

class TestMultiElementInterface:
    """Animator must work for any diffusing species without hardcoding."""

    @pytest.mark.parametrize("element,C_bulk,C_sink,D", [
        ("Cr", 16.5,  12.0,  1.5e-19),
        ("N",  0.07,   0.0,  8.7e-14),
        ("C",  0.02,   0.0,  4.6e-13),
    ])
    def test_animate_for_element(self, element, C_bulk, C_sink, D, tmp_path):
        output = _make_output(
            element=element, C_bulk=C_bulk, C_sink=C_sink,
            n_x=20, n_t=5, D=D,
        )
        mod = _import_animator()
        save_path = tmp_path / f"anim_{element}.gif"
        result = mod.animate_diffusion(output, save_path, fps=5)
        assert Path(result).exists()
        assert os.path.getsize(result) > 1024
