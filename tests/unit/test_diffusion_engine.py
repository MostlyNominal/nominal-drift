"""
tests/unit/test_diffusion_engine.py
=====================================
TDD tests for nominal_drift.science.diffusion_engine.

All five test categories specified in the Day 3-4 requirements:

  A. Analytical validation against the erf solution
     C(x,t) = C_sink + (C_bulk - C_sink) * erf(x / (2*sqrt(D*t)))
     Maximum absolute error < 1 wt% across the profile.

  B. No-driving-force: flat profile when C_sink == C_bulk.

  C. Depletion front warning: "domain_boundary" warning fires when the
     depletion front exceeds 80% of x_max.

  D. Monotonic sink behaviour: minimum concentration at x=0 must never
     increase once the Dirichlet sink is active.

  E. Nitrogen diffusion sanity: D(N) > D(Cr) at the same temperature,
     confirming the fast-precipitation hierarchy (C >> N >> Cr).

Additional structural tests:
  F. DiffusionOutput schema compliance: engine returns a valid, validated
     DiffusionOutput object for a well-posed 316L input.
  G. Multi-element interface: engine accepts element="N" and produces a
     physically distinct result from element="Cr".
  H. Solver reproducibility: two identical calls return byte-for-byte
     identical results.

Run with:
    pytest tests/unit/test_diffusion_engine.py -v
"""

from __future__ import annotations

import math

import pytest

from nominal_drift.schemas.composition import AlloyComposition
from nominal_drift.schemas.diffusion_output import DiffusionOutput
from nominal_drift.schemas.ht_schedule import HTSchedule, HTStep

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def composition_316L() -> AlloyComposition:
    """Nominal 316L austenitic stainless steel composition."""
    return AlloyComposition(
        alloy_designation="316L",
        alloy_matrix="austenite",
        composition_wt_pct={
            "Fe": 68.88,  # increased so elements sum to 100.00 wt%
            "Cr": 16.50,
            "Ni": 10.10,
            "Mo": 2.10,
            "Mn": 1.80,
            "Si": 0.50,
            "C":  0.02,
            "N":  0.07,
            "P":  0.03,
        },
    )


@pytest.fixture(scope="module")
def single_step_650C_120min() -> HTSchedule:
    """Single isothermal hold at 650°C for 120 minutes — peak sensitization."""
    return HTSchedule(steps=[
        HTStep(
            step=1,
            type="sensitization_soak",
            T_hold_C=650.0,
            hold_min=120.0,
            cooling_method="air_cool",
        )
    ])


@pytest.fixture(scope="module")
def single_step_700C_60min() -> HTSchedule:
    """Single isothermal hold at 700°C for 60 minutes."""
    return HTSchedule(steps=[
        HTStep(
            step=1,
            type="sensitization_soak",
            T_hold_C=700.0,
            hold_min=60.0,
        )
    ])


# ---------------------------------------------------------------------------
# Helper: import engine lazily so import errors produce clear test failures
# ---------------------------------------------------------------------------

def _import_engine():
    from nominal_drift.science import diffusion_engine
    return diffusion_engine


def _solve(**kwargs):
    """Call solve_diffusion with defaults filled in."""
    eng = _import_engine()
    return eng.solve_diffusion(**kwargs)


# ---------------------------------------------------------------------------
# A. Analytical validation
# ---------------------------------------------------------------------------

class TestAnalyticalValidation:
    """
    Compare Crank-Nicolson output against the exact erf solution for a
    constant-temperature, constant-D scenario.

    Exact solution (Crank, 1975, §2.3 eq. 2.14):
        C(x,t) = C_sink + (C_bulk - C_sink) * erf(x / (2 * sqrt(D * t)))

    This is valid for a semi-infinite domain with:
        C(0, t) = C_sink   (Dirichlet)
        C(∞, t) = C_bulk   (far-field)
        C(x, 0) = C_bulk   (uniform IC)

    The numerical domain is finite (x_max = 5 µm) so we use a moderate
    time t=3600s at T=700°C, where the depletion front is well within
    the domain and boundary effects are negligible.
    """

    # Physical parameters for the validation case
    T_C = 700.0        # temperature [°C]
    t_s = 3600.0       # hold time [s]
    C_bulk = 18.0      # wt% Cr
    C_sink = 12.0      # wt% Cr
    element = "Cr"
    matrix = "austenite_FeCrNi"
    n_spatial = 400    # fine grid for low numerical error
    x_max_m = 5e-6     # 5 µm domain

    # Tolerance: max absolute error < 1 wt% across the profile
    TOLERANCE_WTP = 1.0

    @pytest.fixture(scope="class")
    def validation_schedule(self) -> HTSchedule:
        return HTSchedule(steps=[
            HTStep(step=1, type="isothermal_hold", T_hold_C=self.T_C, hold_min=self.t_s / 60.0)
        ])

    @pytest.fixture(scope="class")
    def validation_composition(self) -> AlloyComposition:
        return AlloyComposition(
            alloy_designation="ValidationAlloy",
            alloy_matrix="austenite",
            composition_wt_pct={"Fe": 82.0, "Cr": self.C_bulk, "Ni": 0.0},
        )

    @pytest.fixture(scope="class")
    def result(self, validation_composition, validation_schedule) -> DiffusionOutput:
        return _solve(
            composition=validation_composition,
            ht_schedule=validation_schedule,
            element=self.element,
            matrix=self.matrix,
            n_spatial=self.n_spatial,
            x_max_m=self.x_max_m,
            C_sink_wt_pct=self.C_sink,
        )

    def _exact(self, x_m: float, D: float) -> float:
        """Exact erf solution at position x_m and time self.t_s."""
        arg = x_m / (2.0 * math.sqrt(D * self.t_s))
        return self.C_sink + (self.C_bulk - self.C_sink) * math.erf(arg)

    def _D_at_T(self) -> float:
        """Compute D(Cr) at self.T_C using the arrhenius constants."""
        eng = _import_engine()
        constants = eng.load_arrhenius_constants()
        return eng.arrhenius_D(self.T_C, self.element, self.matrix, constants)

    def test_returns_diffusion_output(self, result):
        """Engine must return a validated DiffusionOutput object."""
        assert isinstance(result, DiffusionOutput)

    def test_element_and_matrix_recorded(self, result):
        assert result.element == self.element
        assert result.matrix == self.matrix

    def test_final_profile_matches_analytical_within_tolerance(self, result):
        """
        Max absolute deviation between numerical and analytical solution
        must be < TOLERANCE_WTP wt% at every spatial node.
        """
        D = self._D_at_T()
        numerical = result.final_profile
        x_values = result.x_m

        errors = []
        for x, c_num in zip(x_values, numerical):
            c_exact = self._exact(x, D)
            errors.append(abs(c_num - c_exact))

        max_err = max(errors)
        assert max_err < self.TOLERANCE_WTP, (
            f"Max absolute error = {max_err:.4f} wt% exceeds tolerance "
            f"of {self.TOLERANCE_WTP} wt%.  "
            f"Check Crank-Nicolson implementation and boundary conditions."
        )

    def test_boundary_conditions_enforced(self, result):
        """Left BC = C_sink exactly; right BC = C_bulk exactly."""
        final = result.final_profile
        assert abs(final[0] - self.C_sink) < 1e-6, (
            f"Left BC violation: final_profile[0] = {final[0]}, expected {self.C_sink}"
        )
        assert abs(final[-1] - self.C_bulk) < 1e-3, (
            f"Right BC violation: final_profile[-1] = {final[-1]}, expected {self.C_bulk}"
        )

    def test_initial_profile_is_uniform_at_C_bulk(self, result):
        """t=0 profile must be uniform at C_bulk (fully annealed initial condition)."""
        initial = result.initial_profile
        # The left node is immediately set to C_sink at t=0+, so skip index 0
        # All interior and right nodes must equal C_bulk
        for i, c in enumerate(initial[1:], start=1):
            assert abs(c - self.C_bulk) < 1e-6, (
                f"Initial profile non-uniform at node {i}: C={c}, expected {self.C_bulk}"
            )

    def test_profile_is_monotonically_increasing_from_boundary(self, result):
        """
        At any stored time step, the profile must increase monotonically
        from x=0 to x=x_max (depletion geometry: depleted at GB, bulk
        concentration in grain interior).
        """
        for t_idx, profile in enumerate(result.concentration_profiles):
            if t_idx == 0:
                continue  # t=0 is flat — skip
            for i in range(len(profile) - 1):
                assert profile[i] <= profile[i + 1] + 1e-9, (
                    f"Profile not monotonically increasing at t_idx={t_idx}, "
                    f"node {i}: C[{i}]={profile[i]:.6f} > C[{i+1}]={profile[i+1]:.6f}"
                )


# ---------------------------------------------------------------------------
# B. No driving force — flat profile
# ---------------------------------------------------------------------------

class TestNoDrivingForce:
    """When C_sink == C_bulk there is no concentration gradient and the
    profile must remain flat throughout the simulation."""

    def test_flat_profile_when_sink_equals_bulk(self, composition_316L, single_step_650C_120min):
        C_bulk = composition_316L.bulk_Cr_wt_pct  # 16.5 wt%
        result = _solve(
            composition=composition_316L,
            ht_schedule=single_step_650C_120min,
            element="Cr",
            matrix="austenite_FeCrNi",
            n_spatial=100,
            x_max_m=5e-6,
            C_sink_wt_pct=C_bulk,  # no driving force
        )
        for t_idx, profile in enumerate(result.concentration_profiles):
            for j, c in enumerate(profile):
                assert abs(c - C_bulk) < 1e-6, (
                    f"Profile not flat at t_idx={t_idx}, node {j}: "
                    f"C={c:.8f}, expected {C_bulk}"
                )

    def test_no_depletion_depth_reported_when_no_driving_force(
        self, composition_316L, single_step_650C_120min
    ):
        C_bulk = composition_316L.bulk_Cr_wt_pct
        result = _solve(
            composition=composition_316L,
            ht_schedule=single_step_650C_120min,
            element="Cr",
            matrix="austenite_FeCrNi",
            n_spatial=100,
            x_max_m=5e-6,
            C_sink_wt_pct=C_bulk,
        )
        assert result.depletion_depth_nm is None


# ---------------------------------------------------------------------------
# C. Depletion front warning
# ---------------------------------------------------------------------------

class TestDepletionFrontWarning:
    """When the depletion front exceeds 80% of x_max, a warning must appear
    in result.warnings.  We force this by using a tiny domain, high temperature,
    and long hold time so depletion saturates the domain."""

    def test_domain_boundary_warning_fires(self):
        """Use a 500 nm domain at 800°C for 24 h — depletion will saturate."""
        composition = AlloyComposition(
            alloy_designation="304",
            alloy_matrix="austenite",
            composition_wt_pct={"Fe": 72.0, "Cr": 18.0, "Ni": 10.0},
        )
        schedule = HTSchedule(steps=[
            HTStep(step=1, type="sensitization_soak", T_hold_C=800.0, hold_min=1440.0)
        ])
        result = _solve(
            composition=composition,
            ht_schedule=schedule,
            element="Cr",
            matrix="austenite_FeCrNi",
            n_spatial=100,
            x_max_m=5e-7,  # tiny 500 nm domain
            C_sink_wt_pct=12.0,
        )
        warning_text = " ".join(result.warnings).lower()
        assert "domain" in warning_text or "boundary" in warning_text, (
            f"Expected domain/boundary warning; got warnings: {result.warnings}"
        )


# ---------------------------------------------------------------------------
# D. Monotonic sink behaviour
# ---------------------------------------------------------------------------

class TestMonotonicSinkBehaviour:
    """
    During a constant-temperature hold the concentration at x=0 (grain
    boundary) is pinned to C_sink by the Dirichlet BC.  It must never
    increase above C_sink at any stored time step.
    """

    def test_minimum_at_x0_never_increases(self, composition_316L, single_step_650C_120min):
        result = _solve(
            composition=composition_316L,
            ht_schedule=single_step_650C_120min,
            element="Cr",
            matrix="austenite_FeCrNi",
            n_spatial=200,
            x_max_m=5e-6,
            C_sink_wt_pct=12.0,
        )
        # The concentration at x=0 must stay at C_sink (±floating-point noise)
        for t_idx, profile in enumerate(result.concentration_profiles):
            c_at_gb = profile[0]
            assert c_at_gb <= 12.0 + 1e-6, (
                f"GB concentration rose above C_sink at t_idx={t_idx}: "
                f"C(x=0)={c_at_gb:.8f} wt%, C_sink=12.0 wt%"
            )

    def test_min_concentration_does_not_decrease_below_sink(
        self, composition_316L, single_step_650C_120min
    ):
        """The Dirichlet BC should prevent concentration from falling below C_sink."""
        result = _solve(
            composition=composition_316L,
            ht_schedule=single_step_650C_120min,
            element="Cr",
            matrix="austenite_FeCrNi",
            n_spatial=200,
            x_max_m=5e-6,
            C_sink_wt_pct=12.0,
        )
        assert result.min_concentration_wt_pct >= 12.0 - 1e-6, (
            f"min_concentration_wt_pct={result.min_concentration_wt_pct} "
            f"is below C_sink=12.0 wt%"
        )


# ---------------------------------------------------------------------------
# E. Nitrogen diffusion sanity
# ---------------------------------------------------------------------------

class TestNitrogenDiffusionSanity:
    """
    D(N) must be > D(Cr) at any temperature in the sensitization range.
    D(C) must be > D(N) at any temperature in the sensitization range.
    This hierarchy (C >> N >> Cr) is the physical basis for the
    fast-precipitation approximation and must be verified whenever
    Arrhenius constants are modified.
    """

    TEMPERATURES_C = [550, 650, 700, 750, 850]

    @pytest.fixture(scope="class")
    def constants(self):
        eng = _import_engine()
        return eng.load_arrhenius_constants()

    def test_D_nitrogen_greater_than_D_chromium(self, constants):
        eng = _import_engine()
        matrix = "austenite_FeCrNi"
        for T in self.TEMPERATURES_C:
            D_N = eng.arrhenius_D(T, "N", matrix, constants)
            D_Cr = eng.arrhenius_D(T, "Cr", matrix, constants)
            assert D_N > D_Cr, (
                f"D(N) <= D(Cr) at {T}°C: D_N={D_N:.3e}, D_Cr={D_Cr:.3e}.  "
                f"Check arrhenius.json — N should diffuse faster than Cr in austenite."
            )

    def test_D_carbon_greater_than_D_nitrogen(self, constants):
        eng = _import_engine()
        matrix = "austenite_FeCrNi"
        for T in self.TEMPERATURES_C:
            D_C = eng.arrhenius_D(T, "C", matrix, constants)
            D_N = eng.arrhenius_D(T, "N", matrix, constants)
            assert D_C > D_N, (
                f"D(C) <= D(N) at {T}°C: D_C={D_C:.3e}, D_N={D_N:.3e}.  "
                f"Check arrhenius.json — C should diffuse faster than N in austenite."
            )

    def test_nitrogen_diffusion_result_differs_from_chromium(
        self, composition_316L, single_step_700C_60min
    ):
        """
        Solving for N under the same conditions as Cr must produce a
        physically different (faster-diffusing) profile — the N depletion
        zone must be wider than the Cr zone.
        """
        result_Cr = _solve(
            composition=composition_316L,
            ht_schedule=single_step_700C_60min,
            element="Cr",
            matrix="austenite_FeCrNi",
            n_spatial=200,
            x_max_m=5e-6,
            C_sink_wt_pct=12.0,
        )
        # For N we use the actual bulk N content (0.07 wt%) and a plausible
        # sink value — here 0.0 wt% (all N consumed at GB).
        result_N = _solve(
            composition=composition_316L,
            ht_schedule=single_step_700C_60min,
            element="N",
            matrix="austenite_FeCrNi",
            n_spatial=200,
            x_max_m=5e-6,
            C_sink_wt_pct=0.0,
        )
        # N profile should have a wider depletion zone than Cr
        # Proxy: compare concentration at node 10 (interior)
        # N profile should be more depleted (lower relative to bulk) there
        C_N_interior = result_N.final_profile[10]
        C_Cr_interior = result_Cr.final_profile[10]
        N_bulk = composition_316L.composition_wt_pct.get("N", 0.07)
        Cr_bulk = composition_316L.bulk_Cr_wt_pct

        N_relative_depletion = (N_bulk - C_N_interior) / N_bulk if N_bulk > 0 else 0
        Cr_relative_depletion = (Cr_bulk - C_Cr_interior) / Cr_bulk

        assert N_relative_depletion > Cr_relative_depletion, (
            f"Expected N to show greater relative depletion than Cr at node 10 "
            f"(N diffuses faster).  "
            f"N rel. depletion={N_relative_depletion:.4f}, "
            f"Cr rel. depletion={Cr_relative_depletion:.4f}"
        )


# ---------------------------------------------------------------------------
# F. DiffusionOutput schema compliance
# ---------------------------------------------------------------------------

class TestSchemaCompliance:
    """Engine output must be a fully validated DiffusionOutput object."""

    def test_output_is_validated_pydantic_model(self, composition_316L, single_step_650C_120min):
        result = _solve(
            composition=composition_316L,
            ht_schedule=single_step_650C_120min,
        )
        assert isinstance(result, DiffusionOutput)

    def test_output_x_m_length_matches_n_spatial(self, composition_316L, single_step_650C_120min):
        n = 150
        result = _solve(
            composition=composition_316L,
            ht_schedule=single_step_650C_120min,
            n_spatial=n,
        )
        assert len(result.x_m) == n

    def test_output_t_s_starts_at_zero(self, composition_316L, single_step_650C_120min):
        result = _solve(
            composition=composition_316L,
            ht_schedule=single_step_650C_120min,
        )
        assert result.t_s[0] == pytest.approx(0.0)

    def test_output_t_s_ends_at_total_hold(self, composition_316L, single_step_650C_120min):
        result = _solve(
            composition=composition_316L,
            ht_schedule=single_step_650C_120min,
        )
        expected_total_s = single_step_650C_120min.total_hold_s
        assert result.t_s[-1] == pytest.approx(expected_total_s, rel=1e-3)

    def test_metadata_contains_arrhenius(self, composition_316L, single_step_650C_120min):
        result = _solve(
            composition=composition_316L,
            ht_schedule=single_step_650C_120min,
        )
        assert "arrhenius" in result.metadata

    def test_metadata_contains_solver(self, composition_316L, single_step_650C_120min):
        result = _solve(
            composition=composition_316L,
            ht_schedule=single_step_650C_120min,
        )
        assert "solver" in result.metadata

    def test_c_bulk_matches_alloy_composition(self, composition_316L, single_step_650C_120min):
        result = _solve(
            composition=composition_316L,
            ht_schedule=single_step_650C_120min,
            element="Cr",
        )
        assert result.C_bulk_wt_pct == pytest.approx(composition_316L.bulk_Cr_wt_pct)


# ---------------------------------------------------------------------------
# G. Multi-element interface
# ---------------------------------------------------------------------------

class TestMultiElementInterface:
    """Engine must accept any element present in arrhenius.json and produce
    a physically distinct result.  This is the extensibility gateway for the
    multi-species framework."""

    def test_element_field_recorded_correctly_for_cr(
        self, composition_316L, single_step_700C_60min
    ):
        result = _solve(
            composition=composition_316L,
            ht_schedule=single_step_700C_60min,
            element="Cr",
        )
        assert result.element == "Cr"

    def test_element_field_recorded_correctly_for_n(
        self, composition_316L, single_step_700C_60min
    ):
        result = _solve(
            composition=composition_316L,
            ht_schedule=single_step_700C_60min,
            element="N",
            C_sink_wt_pct=0.0,
        )
        assert result.element == "N"

    def test_unknown_element_raises(self, composition_316L, single_step_700C_60min):
        with pytest.raises((KeyError, ValueError)):
            _solve(
                composition=composition_316L,
                ht_schedule=single_step_700C_60min,
                element="Au",  # not in arrhenius.json
            )


# ---------------------------------------------------------------------------
# H. Solver reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Two identical solve_diffusion() calls must return identical results."""

    def test_identical_calls_return_identical_profiles(
        self, composition_316L, single_step_650C_120min
    ):
        kwargs = dict(
            composition=composition_316L,
            ht_schedule=single_step_650C_120min,
            element="Cr",
            matrix="austenite_FeCrNi",
            n_spatial=200,
            x_max_m=5e-6,
            C_sink_wt_pct=12.0,
        )
        r1 = _solve(**kwargs)
        r2 = _solve(**kwargs)
        assert r1.final_profile == r2.final_profile
        assert r1.min_concentration_wt_pct == r2.min_concentration_wt_pct
