"""Tests for nominal_drift.viz.microstructure_animator."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nominal_drift.schemas.diffusion_output import DiffusionOutput
from nominal_drift.viz.microstructure_animator import (
    DISCLAIMER,
    MicrostructureConfig,
    _blend_primary,
    _hex_to_rgb,
    _infer_matrix_element,
    _infer_secondary_element,
    _make_rgba,
    animate_microstructure,
)
from nominal_drift.viz.species_styles import SpeciesStyle


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


def _make_output(
    element: str = "Cr",
    matrix: str = "austenite_FeCrNi",
    C_bulk: float = 16.5,
    C_sink: float = 12.0,
    n_spatial: int = 50,
    n_frames: int = 5,
) -> DiffusionOutput:
    """Create a minimal DiffusionOutput for testing."""
    x_m = np.linspace(0.0, 5e-6, n_spatial).tolist()
    t_s = np.linspace(0.0, 3600.0, n_frames).tolist()

    profiles = []
    for i, t in enumerate(t_s):
        frac = i / max(1, n_frames - 1)
        x_arr = np.array(x_m)
        C = C_sink + (C_bulk - C_sink) * (1.0 - np.exp(-x_arr / 1e-6) * frac)
        profiles.append(C.tolist())

    return DiffusionOutput(
        element=element,
        matrix=matrix,
        x_m=x_m,
        t_s=t_s,
        concentration_profiles=profiles,
        C_bulk_wt_pct=C_bulk,
        C_sink_wt_pct=C_sink,
        min_concentration_wt_pct=C_sink,
        depletion_depth_nm=50.0,
        warnings=[],
        metadata={
            "element": element,
            "matrix": matrix,
            "ht_schedule_summary": [{"T_hold_C": 700, "hold_min": 60}],
        },
    )


@pytest.fixture
def cr_output():
    return _make_output()


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# -----------------------------------------------------------------------
# MicrostructureConfig
# -----------------------------------------------------------------------


class TestMicrostructureConfig:
    def test_default_construction(self):
        cfg = MicrostructureConfig()
        assert cfg.n_primary == 500
        assert cfg.n_matrix == 300
        assert cfg.n_secondary == 100
        assert cfg.depletion_zone_fraction == 0.25

    def test_frozen(self):
        cfg = MicrostructureConfig()
        with pytest.raises(AttributeError):
            cfg.n_primary = 999

    def test_custom_values(self):
        cfg = MicrostructureConfig(
            n_primary=100,
            boundary_label="Phase\nBoundary",
            domain_label="Matrix",
        )
        assert cfg.n_primary == 100
        assert "Phase" in cfg.boundary_label

    def test_show_legend_default_true(self):
        assert MicrostructureConfig().show_legend is True


# -----------------------------------------------------------------------
# Colour helpers
# -----------------------------------------------------------------------


class TestColourHelpers:
    def test_hex_to_rgb_black(self):
        rgb = _hex_to_rgb("#000000")
        np.testing.assert_array_almost_equal(rgb, [0.0, 0.0, 0.0])

    def test_hex_to_rgb_white(self):
        rgb = _hex_to_rgb("#FFFFFF")
        np.testing.assert_array_almost_equal(rgb, [1.0, 1.0, 1.0])

    def test_hex_to_rgb_red(self):
        rgb = _hex_to_rgb("#FF0000")
        np.testing.assert_array_almost_equal(rgb, [1.0, 0.0, 0.0])

    def test_hex_to_rgb_no_hash(self):
        rgb = _hex_to_rgb("00FF00")
        np.testing.assert_array_almost_equal(rgb, [0.0, 1.0, 0.0])

    def test_make_rgba_1d(self):
        rgb = np.array([0.5, 0.5, 0.5])
        rgba = _make_rgba(rgb, 0.8)
        assert len(rgba) == 4
        assert rgba[3] == pytest.approx(0.8)

    def test_make_rgba_2d(self):
        rgb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        rgba = _make_rgba(rgb, 0.5)
        assert rgba.shape == (2, 4)
        assert rgba[0, 3] == pytest.approx(0.5)


# -----------------------------------------------------------------------
# _blend_primary
# -----------------------------------------------------------------------


class TestBlendPrimary:
    def test_zero_depletion_returns_full_colour(self):
        full = np.array([0.0, 0.0, 1.0])
        dep = np.array([0.5, 0.5, 0.5])
        d = np.array([0.0, 0.0])
        rgba = _blend_primary(d, full, dep)
        assert rgba.shape == (2, 4)
        np.testing.assert_array_almost_equal(rgba[0, :3], full)

    def test_full_depletion_returns_depleted_colour(self):
        full = np.array([0.0, 0.0, 1.0])
        dep = np.array([0.5, 0.5, 0.5])
        d = np.array([1.0])
        rgba = _blend_primary(d, full, dep)
        np.testing.assert_array_almost_equal(rgba[0, :3], dep)

    def test_alpha_decreases_with_depletion(self):
        full = np.array([1.0, 0.0, 0.0])
        dep = np.array([0.5, 0.5, 0.5])
        d = np.array([0.0, 0.5, 1.0])
        rgba = _blend_primary(d, full, dep)
        assert rgba[0, 3] > rgba[1, 3] > rgba[2, 3]


# -----------------------------------------------------------------------
# Inference helpers
# -----------------------------------------------------------------------


class TestInferMatrixElement:
    def test_austenite_returns_fe(self):
        out = _make_output(matrix="austenite_FeCrNi")
        assert _infer_matrix_element(out) == "Fe"

    def test_nickel_base_returns_ni(self):
        out = _make_output(matrix="Ni_base_superalloy")
        assert _infer_matrix_element(out) == "Ni"

    def test_aluminium_returns_al(self):
        out = _make_output(matrix="Al_7075")
        assert _infer_matrix_element(out) == "Al"

    def test_oxide_returns_o(self):
        out = _make_output(matrix="perovskite_oxide_BaTiO3")
        assert _infer_matrix_element(out) == "O"

    def test_unknown_defaults_to_fe(self):
        out = _make_output(matrix="unknown_matrix_xyz")
        assert _infer_matrix_element(out) == "Fe"


class TestInferSecondaryElement:
    def test_cr_pairs_with_c(self):
        assert _infer_secondary_element("Cr") == "C"

    def test_c_pairs_with_cr(self):
        assert _infer_secondary_element("C") == "Cr"

    def test_n_pairs_with_cr(self):
        assert _infer_secondary_element("N") == "Cr"

    def test_ni_pairs_with_al(self):
        assert _infer_secondary_element("Ni") == "Al"

    def test_unknown_returns_none(self):
        assert _infer_secondary_element("Zz") is None


# -----------------------------------------------------------------------
# DISCLAIMER
# -----------------------------------------------------------------------


class TestDisclaimer:
    def test_disclaimer_is_string(self):
        assert isinstance(DISCLAIMER, str)

    def test_disclaimer_mentions_continuum(self):
        assert "continuum" in DISCLAIMER.lower() or "Continuum" in DISCLAIMER

    def test_disclaimer_mentions_not_atomistic(self):
        assert "NOT" in DISCLAIMER or "not" in DISCLAIMER.lower()

    def test_disclaimer_mentions_illustrative(self):
        assert "illustrat" in DISCLAIMER.lower()


# -----------------------------------------------------------------------
# animate_microstructure — integration tests
# -----------------------------------------------------------------------


class TestAnimateMicrostructure:
    def test_creates_gif_file(self, cr_output, tmp_dir):
        path = animate_microstructure(
            cr_output, tmp_dir / "test.gif",
            config=MicrostructureConfig(n_primary=30, n_matrix=20, n_secondary=10),
            max_frames=3, fps=5,
        )
        assert Path(path).exists()
        assert path.endswith(".gif")

    def test_gif_file_is_nonempty(self, cr_output, tmp_dir):
        path = animate_microstructure(
            cr_output, tmp_dir / "test.gif",
            config=MicrostructureConfig(n_primary=30, n_matrix=20, n_secondary=10),
            max_frames=3, fps=5,
        )
        assert os.path.getsize(path) > 0

    def test_normalises_extension_to_gif(self, cr_output, tmp_dir):
        path = animate_microstructure(
            cr_output, tmp_dir / "test.mp4",
            config=MicrostructureConfig(n_primary=20, n_matrix=10, n_secondary=5),
            max_frames=2, fps=5,
        )
        assert path.endswith(".gif")

    def test_accepts_config_as_dict(self, cr_output, tmp_dir):
        path = animate_microstructure(
            cr_output, tmp_dir / "test.gif",
            config={"n_primary": 20, "n_matrix": 10, "n_secondary": 5},
            max_frames=2, fps=5,
        )
        assert Path(path).exists()

    def test_accepts_none_config(self, cr_output, tmp_dir):
        path = animate_microstructure(
            cr_output, tmp_dir / "test.gif",
            config=None,
            max_frames=2, fps=5,
        )
        assert Path(path).exists()

    def test_custom_title(self, cr_output, tmp_dir):
        path = animate_microstructure(
            cr_output, tmp_dir / "test.gif",
            config=MicrostructureConfig(n_primary=20, n_matrix=10, n_secondary=5),
            title="Test Title",
            max_frames=2, fps=5,
        )
        assert Path(path).exists()

    def test_with_custom_species_style_map(self, cr_output, tmp_dir):
        custom = {"Cr": SpeciesStyle(colour="#FF0000", radius=2.0, label="CustomCr")}
        path = animate_microstructure(
            cr_output, tmp_dir / "test.gif",
            config=MicrostructureConfig(n_primary=20, n_matrix=10, n_secondary=5),
            species_style_map=custom,
            max_frames=2, fps=5,
        )
        assert Path(path).exists()

    def test_with_explicit_matrix_element(self, cr_output, tmp_dir):
        path = animate_microstructure(
            cr_output, tmp_dir / "test.gif",
            config=MicrostructureConfig(n_primary=20, n_matrix=10, n_secondary=5),
            matrix_element="Ni",
            max_frames=2, fps=5,
        )
        assert Path(path).exists()

    def test_with_explicit_secondary_element(self, cr_output, tmp_dir):
        path = animate_microstructure(
            cr_output, tmp_dir / "test.gif",
            config=MicrostructureConfig(n_primary=20, n_matrix=10, n_secondary=5),
            secondary_element="N",
            max_frames=2, fps=5,
        )
        assert Path(path).exists()

    def test_no_secondary_when_none(self, cr_output, tmp_dir):
        path = animate_microstructure(
            cr_output, tmp_dir / "test.gif",
            config=MicrostructureConfig(n_primary=20, n_matrix=10, n_secondary=0),
            secondary_element=None,
            max_frames=2, fps=5,
        )
        assert Path(path).exists()

    def test_creates_parent_directories(self, cr_output, tmp_dir):
        deep_path = tmp_dir / "a" / "b" / "c" / "test.gif"
        path = animate_microstructure(
            cr_output, deep_path,
            config=MicrostructureConfig(n_primary=20, n_matrix=10, n_secondary=5),
            max_frames=2, fps=5,
        )
        assert Path(path).exists()

    def test_no_legend_mode(self, cr_output, tmp_dir):
        path = animate_microstructure(
            cr_output, tmp_dir / "test.gif",
            config=MicrostructureConfig(
                n_primary=20, n_matrix=10, n_secondary=5,
                show_legend=False,
            ),
            max_frames=2, fps=5,
        )
        assert Path(path).exists()

    def test_no_depletion_zone(self, cr_output, tmp_dir):
        path = animate_microstructure(
            cr_output, tmp_dir / "test.gif",
            config=MicrostructureConfig(
                n_primary=20, n_matrix=10, n_secondary=5,
                depletion_zone_fraction=0.0,
            ),
            max_frames=2, fps=5,
        )
        assert Path(path).exists()

    def test_non_steel_material(self, tmp_dir):
        """Test with a non-steel material (Ni-base alloy)."""
        out = _make_output(
            element="Al",
            matrix="Ni_base_superalloy",
            C_bulk=6.0,
            C_sink=2.0,
        )
        path = animate_microstructure(
            out, tmp_dir / "ni_base.gif",
            config=MicrostructureConfig(
                n_primary=20, n_matrix=10, n_secondary=5,
                boundary_label="Phase\nBoundary",
                domain_label="Gamma Matrix",
            ),
            max_frames=2, fps=5,
        )
        assert Path(path).exists()

    def test_oxide_material(self, tmp_dir):
        """Test with an oxide / perovskite material."""
        out = _make_output(
            element="O",
            matrix="perovskite_oxide_BaTiO3",
            C_bulk=50.0,
            C_sink=45.0,
        )
        path = animate_microstructure(
            out, tmp_dir / "oxide.gif",
            config=MicrostructureConfig(
                n_primary=20, n_matrix=10, n_secondary=5,
                boundary_label="Surface",
                domain_label="Bulk",
            ),
            max_frames=2, fps=5,
        )
        assert Path(path).exists()

    def test_structure_record_extension_point(self, cr_output, tmp_dir):
        """Verify structure_record kwarg is accepted (future CIF extension)."""
        path = animate_microstructure(
            cr_output, tmp_dir / "test.gif",
            config=MicrostructureConfig(n_primary=20, n_matrix=10, n_secondary=5),
            structure_record={"dummy": "crystal_record"},
            max_frames=2, fps=5,
        )
        assert Path(path).exists()

    def test_cif_path_extension_point(self, cr_output, tmp_dir):
        """Verify cif_path kwarg is accepted (future CIF extension)."""
        path = animate_microstructure(
            cr_output, tmp_dir / "test.gif",
            config=MicrostructureConfig(n_primary=20, n_matrix=10, n_secondary=5),
            cif_path="/dummy/path/to/structure.cif",
            max_frames=2, fps=5,
        )
        assert Path(path).exists()

    def test_zero_driving_force(self, tmp_dir):
        """C_bulk == C_sink should not crash."""
        out = _make_output(C_bulk=12.0, C_sink=12.0)
        path = animate_microstructure(
            out, tmp_dir / "flat.gif",
            config=MicrostructureConfig(n_primary=20, n_matrix=10, n_secondary=5),
            max_frames=2, fps=5,
        )
        assert Path(path).exists()

    def test_single_frame(self, tmp_dir):
        """Single stored profile should still produce a valid GIF."""
        out = _make_output(n_frames=1)
        path = animate_microstructure(
            out, tmp_dir / "single.gif",
            config=MicrostructureConfig(n_primary=20, n_matrix=10, n_secondary=5),
            max_frames=1, fps=5,
        )
        assert Path(path).exists()
