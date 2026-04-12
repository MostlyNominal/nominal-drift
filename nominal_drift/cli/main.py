"""
nominal_drift.cli.main
=======================
Typer-based CLI entry point for NominalDrift Sprint 1.

Entry point (defined in pyproject.toml):
    nominal-drift run [OPTIONS] [INPUT_JSON]

Commands
--------
run
    Execute the full Sprint 1 showcase workflow:
    diffusion solve → static plot → animation → experiment store → narration.

    If INPUT_JSON is omitted, the built-in 316L / 700 °C / 60 min demo case
    is used.  If INPUT_JSON is provided, composition and HT schedule are
    loaded from the file and validated against the existing Pydantic schemas.

JSON input format
-----------------
{
    "alloy_designation": "316L",
    "alloy_matrix": "austenite",
    "composition_wt_pct": {
        "Fe": 68.88, "Cr": 16.50, "Ni": 10.10,
        "Mo": 2.10,  "Mn": 1.80,  "Si": 0.50,
        "C":  0.02,  "N":  0.07,  "P":  0.03
    },
    "ht_schedule": {
        "steps": [
            {
                "step": 1,
                "type": "sensitization_soak",
                "T_hold_C": 700.0,
                "hold_min": 60.0,
                "cooling_method": "air_cool"
            }
        ]
    }
}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from nominal_drift.core.orchestrator import run_showcase_workflow
from nominal_drift.schemas.composition import AlloyComposition
from nominal_drift.schemas.ht_schedule import HTSchedule, HTStep

# ---------------------------------------------------------------------------
# Typer app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="nominal-drift",
    help=(
        "NominalDrift — local scientific AI workbench for materials engineering.\n\n"
        "Sprint 1: 1-D continuum diffusion modelling with LLM narration.\n"
        "Continuum model only — not an atomistic simulation."
    ),
    add_completion=False,
    rich_markup_mode="rich",
    invoke_without_command=True,
)

console = Console()

# ---------------------------------------------------------------------------
# Built-in demo composition and schedule (316L, 700 °C × 60 min)
# ---------------------------------------------------------------------------

_DEMO_COMPOSITION = AlloyComposition(
    alloy_designation="316L",
    alloy_matrix="austenite",
    composition_wt_pct={
        "Fe": 68.88,
        "Cr": 16.50,
        "Ni": 10.10,
        "Mo":  2.10,
        "Mn":  1.80,
        "Si":  0.50,
        "C":   0.02,
        "N":   0.07,
        "P":   0.03,
    },
)

_DEMO_SCHEDULE = HTSchedule(steps=[
    HTStep(
        step=1,
        type="sensitization_soak",
        T_hold_C=700.0,
        hold_min=60.0,
        cooling_method="air_cool",
    )
])

# ---------------------------------------------------------------------------
# JSON loader
# ---------------------------------------------------------------------------

def _load_input_json(path: str) -> tuple[AlloyComposition, HTSchedule]:
    """Load and validate composition + HT schedule from a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON input file.

    Returns
    -------
    tuple[AlloyComposition, HTSchedule]
        Validated schema objects.

    Raises
    ------
    typer.Exit
        On file-not-found, JSON decode error, or Pydantic validation failure.
    """
    p = Path(path)
    if not p.exists():
        console.print(f"[bold red]Error:[/bold red] Input file not found: {path}")
        raise typer.Exit(code=1)

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        console.print(f"[bold red]Error:[/bold red] Could not parse JSON: {exc}")
        raise typer.Exit(code=1) from exc

    try:
        composition = AlloyComposition(
            alloy_designation=data["alloy_designation"],
            alloy_matrix=data["alloy_matrix"],
            composition_wt_pct=data["composition_wt_pct"],
        )
    except Exception as exc:  # pydantic.ValidationError or KeyError
        console.print(f"[bold red]Error:[/bold red] Invalid composition: {exc}")
        raise typer.Exit(code=1) from exc

    try:
        ht_raw = data["ht_schedule"]
        schedule = HTSchedule(steps=[HTStep(**s) for s in ht_raw["steps"]])
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] Invalid HT schedule: {exc}")
        raise typer.Exit(code=1) from exc

    return composition, schedule


# ---------------------------------------------------------------------------
# Rich terminal summary
# ---------------------------------------------------------------------------

def _print_summary(result: dict, composition: AlloyComposition) -> None:
    """Print a formatted run summary to the terminal using Rich."""

    # ---- header panel ----
    console.print()
    console.print(Panel(
        f"[bold cyan]NominalDrift[/bold cyan]  Sprint 1 — run complete\n"
        f"[dim]Continuum diffusion model · not an atomistic simulation[/dim]",
        box=box.ROUNDED,
        expand=False,
    ))

    # ---- run metadata table ----
    meta = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    meta.add_column("Key",   style="bold", min_width=26)
    meta.add_column("Value", style="white")

    meta.add_row("Experiment ID",     result["experiment_id"])
    meta.add_row("Alloy designation", result["alloy_designation"])
    meta.add_row("Matrix",            composition.alloy_matrix)
    meta.add_row("Element (species)", result["element"])
    meta.add_row("Output directory",  result["output_dir"])
    meta.add_row("Static plot",       result["plot_path"])
    meta.add_row("Animation",         result["animation_path"])

    console.print(meta)

    # ---- results table ----
    res = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    res.add_column("Key",   style="bold", min_width=26)
    res.add_column("Value", style="white")

    res.add_row(
        "Min [{}] in domain".format(result["element"]),
        f"{result['min_concentration_wt_pct']:.4f} wt%",
    )
    depth = result["depletion_depth_nm"]
    res.add_row(
        "Depletion depth",
        f"{depth:.1f} nm" if depth is not None else "not determined",
    )

    if result["warnings"]:
        for i, w in enumerate(result["warnings"]):
            label = "Warning" if i == 0 else ""
            res.add_row(label, f"[yellow]{w}[/yellow]")

    console.print(res)

    # ---- narration panel ----
    narration = result["narration"]
    console.print(Panel(
        narration,
        title="[bold]LLM Engineering Narration[/bold]",
        border_style="dim",
        expand=True,
    ))
    console.print()


# ---------------------------------------------------------------------------
# CLI command: run
# ---------------------------------------------------------------------------

@app.callback()
def run(
    ctx: typer.Context,
    input_json: Optional[str] = typer.Argument(
        None,
        help=(
            "Path to a JSON file containing alloy composition and HT schedule.  "
            "If omitted, the built-in 316L / 700 °C / 60 min demo case is used."
        ),
    ),
    element: str = typer.Option(
        "Cr",
        "--element", "-e",
        help="Diffusing species symbol, e.g. Cr, N, C.",
    ),
    matrix: str = typer.Option(
        "austenite_FeCrNi",
        "--matrix", "-m",
        help="Arrhenius diffusivity matrix key.",
    ),
    c_sink_wt_pct: float = typer.Option(
        12.0,
        "--c-sink",
        help="Grain-boundary sink concentration [wt%].",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir", "-o",
        help="Base directory for run outputs. Defaults to <repo>/outputs/.",
    ),
    db_path: Optional[str] = typer.Option(
        None,
        "--db",
        help="SQLite DB path. Defaults to <repo>/data/experiments.db.",
    ),
    user_label: Optional[str] = typer.Option(
        None,
        "--label",
        help="Short human-readable label stored with the experiment record.",
    ),
    user_notes: Optional[str] = typer.Option(
        None,
        "--notes",
        help="Free-form notes stored with the experiment record.",
    ),
) -> None:
    """Run the Sprint 1 showcase workflow (diffusion → plot → animation → store → narrate).

    \b
    Examples
    --------
      # Built-in demo (316L, 700 °C, 60 min, Cr):
      nominal-drift

      # Custom input from a JSON file:
      nominal-drift my_experiment.json --element Cr --c-sink 12.0

      # With labels and custom output location:
      nominal-drift --label "ref-run-001" --output-dir ./results

      # Dataset commands:
      nominal-drift dataset fetch --name perov-5
      nominal-drift dataset list
    """
    # If a sub-command was invoked (e.g. "dataset fetch"), do not run the workflow.
    if ctx.invoked_subcommand is not None:
        return

    # ---- load inputs ----
    if input_json is None:
        console.print(
            "[dim]No input file supplied — using built-in 316L / 700 °C / 60 min demo.[/dim]"
        )
        composition = _DEMO_COMPOSITION
        schedule    = _DEMO_SCHEDULE
    else:
        console.print(f"[dim]Loading input from:[/dim] {input_json}")
        composition, schedule = _load_input_json(input_json)

    # ---- status banner ----
    console.print(
        f"\n[bold]Running workflow[/bold]  "
        f"[cyan]{composition.alloy_designation}[/cyan] · "
        f"[cyan]{element}[/cyan] · "
        f"{composition.alloy_matrix}\n"
    )

    # ---- execute ----
    with console.status("[bold green]Running showcase workflow…[/bold green]"):
        try:
            result = run_showcase_workflow(
                composition,
                schedule,
                element=element,
                matrix=matrix,
                c_sink_wt_pct=c_sink_wt_pct,
                user_label=user_label,
                user_notes=user_notes,
                base_output_dir=output_dir,
                db_path=db_path,
            )
        except Exception as exc:
            console.print(f"\n[bold red]Workflow error:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc

    # ---- summary ----
    _print_summary(result, composition)


# ---------------------------------------------------------------------------
# dataset sub-app
# ---------------------------------------------------------------------------
dataset_app = typer.Typer(name="dataset", help="Dataset management commands.")
app.add_typer(dataset_app)


@dataset_app.command("fetch")
def dataset_fetch(
    name: str = typer.Option(None, "--name", "-n", help="Dataset name (perov-5, mp-20, mpts-52, carbon-24)"),
    all_datasets: bool = typer.Option(False, "--all", "-a", help="Fetch all datasets"),
    raw_dir: str = typer.Option("data/datasets/raw", "--raw-dir", help="Raw data directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-download even if present"),
):
    """Download and verify local crystal structure datasets."""
    from nominal_drift.datasets.downloaders.perov5_downloader import Perov5Downloader
    from nominal_drift.datasets.downloaders.mp20_downloader import MP20Downloader
    from nominal_drift.datasets.downloaders.mpts52_downloader import MPTS52Downloader
    from nominal_drift.datasets.downloaders.carbon24_downloader import Carbon24Downloader

    registry = {
        "perov-5":    Perov5Downloader,
        "mp-20":      MP20Downloader,
        "mpts-52":    MPTS52Downloader,
        "carbon-24":  Carbon24Downloader,
    }

    targets = list(registry.keys()) if all_datasets else ([name] if name else [])
    if not targets:
        typer.echo("Specify --name <dataset> or --all", err=True)
        raise typer.Exit(1)

    for ds_name in targets:
        if ds_name not in registry:
            typer.echo(f"Unknown dataset: {ds_name}. Choose from {list(registry)}", err=True)
            continue
        typer.echo(f"→ Fetching {ds_name}...")
        downloader = registry[ds_name](raw_base_dir=raw_dir)
        result = downloader.download(force=force)
        typer.echo(result.summary())


@dataset_app.command("list")
def dataset_list(
    raw_dir: str = typer.Option("data/datasets/raw", "--raw-dir"),
):
    """List locally available datasets."""
    from nominal_drift.datasets.downloaders.perov5_downloader import Perov5Downloader
    from nominal_drift.datasets.downloaders.mp20_downloader import MP20Downloader
    from nominal_drift.datasets.downloaders.mpts52_downloader import MPTS52Downloader
    from nominal_drift.datasets.downloaders.carbon24_downloader import Carbon24Downloader

    registry = {
        "perov-5":   Perov5Downloader,
        "mp-20":     MP20Downloader,
        "mpts-52":   MPTS52Downloader,
        "carbon-24": Carbon24Downloader,
    }
    for name, cls in registry.items():
        dl = cls(raw_base_dir=raw_dir)
        status = "✓ present" if dl.is_present() else "✗ not downloaded"
        typer.echo(f"  {name:15s}  {status}")
