"""Run templates for simulations."""
from typing import Optional

from nominal_drift.templates.base import BaseTemplate


class DiffusionRunTemplate(BaseTemplate):
    """Parameters for a diffusion simulation.

    ``element`` is ``Optional[str]``: when no solver-supported diffusion
    species exists for the chosen material system the form returns
    ``element=None`` to mark the run as unavailable rather than silently
    falling back to a default like ``"Cr"``.  Consumers must check this
    before invoking the solver.
    """
    template_type: str = "diffusion_run"
    alloy_designation: str = ""
    alloy_matrix: str = "austenite"
    composition: dict[str, float] = {}
    element: Optional[str] = None
    diffusion_matrix: str = "austenite_FeCrNi"
    hold_steps: list[dict] = []
    c_sink_wt_pct: float = 12.0
    n_spatial: int = 200
    x_max_m: float = 5e-6
    user_notes: str = ""


class SensitizationRunTemplate(BaseTemplate):
    """Parameters for a sensitization assessment run."""
    template_type: str = "sensitization_run"
    diffusion_template: DiffusionRunTemplate = DiffusionRunTemplate()
    cr_threshold_wt_pct: float = 12.0
    include_c: bool = False
    include_n: bool = False
    c_bulk_wt_pct: float = 0.02
    n_bulk_wt_pct: float = 0.07
    user_notes: str = ""


class MechanismAnimationTemplate(BaseTemplate):
    """Parameters for mechanism animation generation."""
    template_type: str = "mechanism_animation"
    diffusion_template: DiffusionRunTemplate = DiffusionRunTemplate()
    output_format: str = "mp4"
    fps: int = 15
    output_dir: str = "outputs/animations"
    user_notes: str = ""
