"""
nominal_drift.reports.report_builder
====================================

Automatic engineering report generation in Markdown and HTML.
Integrates plots, assumptions, warnings, retrieval context, experiment recommendations.

All models are Pydantic v2 **frozen** — immutable after construction and
fully JSON-serialisable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# ReportSection
# ---------------------------------------------------------------------------

class ReportSection(BaseModel, frozen=True):
    """One section of a report.

    Attributes
    ----------
    title : str
        Section heading.
    body : str
        Markdown-formatted text content.
    figure_paths : list[str]
        Relative or absolute paths to PNG/SVG figures (may be empty).
    """

    title: str
    body: str
    figure_paths: list[str] = []


# ---------------------------------------------------------------------------
# ReportSpec
# ---------------------------------------------------------------------------

class ReportSpec(BaseModel, frozen=True):
    """Complete specification for a report.

    Attributes
    ----------
    title : str
        Main report title.
    subtitle : str
        Subtitle or tagline.
    author : str
        Author name.
    created_at : str
        ISO 8601 datetime string.
    sections : list[ReportSection]
        Ordered list of report sections.
    metadata : dict[str, str]
        Key-value pairs (alloy, element, matrix, etc.).
    """

    title: str
    subtitle: str
    author: str
    created_at: str
    sections: list[ReportSection]
    metadata: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Markdown Report Generation
# ---------------------------------------------------------------------------

def build_markdown_report(spec: ReportSpec) -> str:
    """Render ReportSpec to Markdown string.

    Format:
        # {title}
        **{subtitle}**
        *Author: {author}  |  Date: {created_at}*
        ---
        *Metadata: key: val, ...*
        ---
        ## {section.title}
        {section.body}
        [Figure: {path}]  ← one line per figure path
        ---   ← between sections

    Parameters
    ----------
    spec : ReportSpec
        Report specification.

    Returns
    -------
    str
        Markdown string.
    """
    lines = []

    # Header
    lines.append(f"# {spec.title}")
    lines.append(f"**{spec.subtitle}**")
    lines.append(f"*Author: {spec.author}  |  Date: {spec.created_at}*")
    lines.append("---")

    # Metadata
    if spec.metadata:
        meta_parts = [f"{k}: {v}" for k, v in spec.metadata.items()]
        lines.append(f"*Metadata: {', '.join(meta_parts)}*")
    else:
        lines.append("*Metadata: (none)*")
    lines.append("---")

    # Sections
    for i, section in enumerate(spec.sections):
        lines.append(f"## {section.title}")
        lines.append(section.body)
        for fig_path in section.figure_paths:
            lines.append(f"[Figure: {fig_path}]")
        if i < len(spec.sections) - 1:
            lines.append("---")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML Report Generation
# ---------------------------------------------------------------------------

def build_html_report(spec: ReportSpec) -> str:
    """Render ReportSpec to minimal self-contained HTML string.

    Uses inline CSS only (no external dependencies).

    Parameters
    ----------
    spec : ReportSpec
        Report specification.

    Returns
    -------
    str
        HTML string.
    """
    css = """
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }
        hr {
            border: none;
            border-top: 1px solid #ecf0f1;
            margin: 20px 0;
        }
        p {
            margin: 10px 0;
        }
        .meta {
            font-style: italic;
            color: #7f8c8d;
        }
        .metadata-box {
            background-color: #ecf0f1;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-size: 0.95em;
        }
        .section-body {
            margin: 15px 0;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 3px;
        }
        .figure {
            font-style: italic;
            color: #7f8c8d;
            margin: 10px 0;
        }
    </style>
    """

    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        f"<title>{spec.title}</title>",
        css,
        "</head>",
        "<body>",
    ]

    # Header
    html_parts.append(f"<h1>{spec.title}</h1>")
    html_parts.append(f"<h2>{spec.subtitle}</h2>")
    html_parts.append(
        f"<p class='meta'>Author: {spec.author}  |  Date: {spec.created_at}</p>"
    )
    html_parts.append("<hr>")

    # Metadata
    if spec.metadata:
        meta_items = [f"{k}: {v}" for k, v in spec.metadata.items()]
        html_parts.append(
            f"<div class='metadata-box'>Metadata: {', '.join(meta_items)}</div>"
        )
    html_parts.append("<hr>")

    # Sections
    for section in spec.sections:
        html_parts.append(f"<h2>{section.title}</h2>")
        # Convert markdown newlines to <br>
        body_html = section.body.replace("\n", "<br>")
        html_parts.append(f"<div class='section-body'>{body_html}</div>")
        for fig_path in section.figure_paths:
            html_parts.append(f"<p class='figure'>Figure: {fig_path}</p>")

    html_parts.append("</body>")
    html_parts.append("</html>")

    return "\n".join(html_parts)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_report_spec(
    title: str,
    sections: list[ReportSection],
    subtitle: str = "",
    author: str = "Nominal Drift",
    metadata: dict | None = None,
) -> ReportSpec:
    """Factory: creates ReportSpec with auto-timestamp.

    Parameters
    ----------
    title : str
        Report title.
    sections : list[ReportSection]
        List of report sections.
    subtitle : str, optional
        Subtitle (default: "").
    author : str, optional
        Author name (default: "Nominal Drift").
    metadata : dict | None, optional
        Metadata dictionary (default: None → empty dict).

    Returns
    -------
    ReportSpec
        Frozen report specification.
    """
    if metadata is None:
        metadata = {}

    created_at = datetime.now(tz=timezone.utc).isoformat()

    return ReportSpec(
        title=title,
        subtitle=subtitle,
        author=author,
        created_at=created_at,
        sections=sections,
        metadata=metadata,
    )


def build_diffusion_report(
    composition_label: str,
    element: str,
    matrix: str,
    min_cr_wt_pct: float,
    depletion_depth_nm: float | None,
    risk_level: str,
    mechanism_label: str,
    assumptions: list[str],
    warnings: list[str],
    notes: list[str],
    figure_paths: list[str] | None = None,
) -> ReportSpec:
    """Convenience factory: builds a standard single-run diffusion report.

    Creates 3 sections:
        1. "Executive Summary" — composition, element, risk level, mechanism
        2. "Assumptions & Model Limitations" — all assumptions + warnings
        3. "Recommendations & Notes" — all notes

    Parameters
    ----------
    composition_label : str
        Alloy composition designation (e.g., "316L").
    element : str
        Diffusing element (e.g., "Cr").
    matrix : str
        Matrix composition (e.g., "austenite").
    min_cr_wt_pct : float
        Minimum chromium content in wt.%.
    depletion_depth_nm : float | None
        Depletion depth in nanometers (may be None).
    risk_level : str
        Risk assessment (e.g., "HIGH", "MODERATE", "LOW").
    mechanism_label : str
        Diffusion mechanism (e.g., "grain boundary diffusion").
    assumptions : list[str]
        Model assumptions.
    warnings : list[str]
        Warnings and caveats.
    notes : list[str]
        Recommendations and notes.
    figure_paths : list[str] | None, optional
        Paths to figures (default: None → empty list).

    Returns
    -------
    ReportSpec
        Frozen report specification.
    """
    if figure_paths is None:
        figure_paths = []

    # Executive Summary
    depth_str = (
        f"{depletion_depth_nm:.1f} nm"
        if depletion_depth_nm is not None
        else "not calculated"
    )
    exec_body = f"""
**Composition:** {composition_label}
**Diffusing Element:** {element}
**Matrix:** {matrix}
**Minimum Cr Content:** {min_cr_wt_pct:.2f} wt.%
**Depletion Depth:** {depth_str}
**Risk Level:** {risk_level}
**Mechanism:** {mechanism_label}
""".strip()

    exec_section = ReportSection(
        title="Executive Summary",
        body=exec_body,
        figure_paths=figure_paths,
    )

    # Assumptions & Limitations
    assumptions_text = "\n".join([f"- {a}" for a in assumptions])
    warnings_text = "\n".join([f"⚠ {w}" for w in warnings])
    assumptions_body = f"**Assumptions:**\n{assumptions_text}\n\n**Warnings:**\n{warnings_text}"

    assumptions_section = ReportSection(
        title="Assumptions & Model Limitations",
        body=assumptions_body,
    )

    # Recommendations & Notes
    notes_text = "\n".join([f"- {n}" for n in notes])
    notes_section = ReportSection(
        title="Recommendations & Notes",
        body=notes_text,
    )

    metadata = {
        "composition": composition_label,
        "element": element,
        "matrix": matrix,
        "risk_level": risk_level,
    }

    return make_report_spec(
        title=f"Diffusion Analysis: {composition_label} ({element})",
        subtitle=f"Risk Assessment: {risk_level}",
        sections=[exec_section, assumptions_section, notes_section],
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def save_report(
    content: str,
    path: str,
    fmt: str = "md",
) -> str:
    """Write report content to path. Create parent directories.

    Parameters
    ----------
    content : str
        Report content (Markdown or HTML string).
    path : str
        Absolute or relative file path.
    fmt : str, optional
        Format specifier: "md" or "html" (default: "md").

    Returns
    -------
    str
        Absolute path to the written file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return str(p.absolute())
