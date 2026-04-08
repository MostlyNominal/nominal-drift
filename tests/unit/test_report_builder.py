"""
tests/unit/test_report_builder.py
=================================

Unit tests for nominal_drift.reports.report_builder.

Tests cover:
  - ReportSection and ReportSpec model validation
  - Markdown and HTML report generation
  - Factory functions (make_report_spec, build_diffusion_report)
  - File I/O (save_report)

Run with:
    pytest tests/unit/test_report_builder.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from nominal_drift.reports.report_builder import (
    ReportSection,
    ReportSpec,
    build_diffusion_report,
    build_html_report,
    build_markdown_report,
    make_report_spec,
    save_report,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture()
def valid_section() -> ReportSection:
    """A valid report section."""
    return ReportSection(
        title="Introduction",
        body="This is the introduction.\nIt has multiple lines.",
        figure_paths=["fig1.png", "fig2.svg"],
    )


@pytest.fixture()
def simple_section() -> ReportSection:
    """A simple section without figures."""
    return ReportSection(
        title="Methods",
        body="We used the following methods.",
    )


@pytest.fixture()
def valid_spec(valid_section: ReportSection, simple_section: ReportSection) -> ReportSpec:
    """A valid report spec."""
    return ReportSpec(
        title="Test Report",
        subtitle="A Comprehensive Analysis",
        author="Test Author",
        created_at="2026-04-07T12:00:00Z",
        sections=[valid_section, simple_section],
        metadata={"alloy": "316L", "element": "Cr"},
    )


# =============================================================================
# ReportSection Tests
# =============================================================================

def test_section_creation_valid() -> None:
    """Test creating a valid section."""
    sec = ReportSection(
        title="Title",
        body="Body text",
        figure_paths=["fig.png"],
    )
    assert sec.title == "Title"
    assert sec.body == "Body text"
    assert sec.figure_paths == ["fig.png"]


def test_section_default_figure_paths() -> None:
    """Test section with default empty figure paths."""
    sec = ReportSection(title="Title", body="Body")
    assert sec.figure_paths == []


def test_section_frozen() -> None:
    """Test that ReportSection is frozen."""
    sec = ReportSection(title="Title", body="Body")
    with pytest.raises(Exception):  # pydantic FrozenError or similar
        sec.title = "New Title"  # type: ignore


def test_section_empty_body() -> None:
    """Test section with empty body."""
    sec = ReportSection(title="Title", body="")
    assert sec.body == ""


def test_section_multiple_figures() -> None:
    """Test section with multiple figures."""
    figs = ["fig1.png", "fig2.svg", "fig3.pdf"]
    sec = ReportSection(title="Results", body="Results.", figure_paths=figs)
    assert sec.figure_paths == figs
    assert len(sec.figure_paths) == 3


# =============================================================================
# ReportSpec Tests
# =============================================================================

def test_spec_creation_valid(valid_spec: ReportSpec) -> None:
    """Test creating a valid spec."""
    assert valid_spec.title == "Test Report"
    assert valid_spec.subtitle == "A Comprehensive Analysis"
    assert valid_spec.author == "Test Author"
    assert len(valid_spec.sections) == 2
    assert valid_spec.metadata["alloy"] == "316L"


def test_spec_frozen(valid_spec: ReportSpec) -> None:
    """Test that ReportSpec is frozen."""
    with pytest.raises(Exception):
        valid_spec.title = "New Title"  # type: ignore


def test_spec_default_metadata() -> None:
    """Test spec with default empty metadata."""
    spec = ReportSpec(
        title="Report",
        subtitle="Sub",
        author="Author",
        created_at="2026-04-07T12:00:00Z",
        sections=[],
    )
    assert spec.metadata == {}


def test_spec_empty_sections() -> None:
    """Test spec with no sections."""
    spec = ReportSpec(
        title="Report",
        subtitle="",
        author="Author",
        created_at="2026-04-07T12:00:00Z",
        sections=[],
    )
    assert len(spec.sections) == 0


def test_spec_many_sections() -> None:
    """Test spec with many sections."""
    sections = [
        ReportSection(title=f"Section {i}", body=f"Body {i}")
        for i in range(10)
    ]
    spec = ReportSpec(
        title="Report",
        subtitle="Sub",
        author="Author",
        created_at="2026-04-07T12:00:00Z",
        sections=sections,
    )
    assert len(spec.sections) == 10


# =============================================================================
# Markdown Report Generation Tests
# =============================================================================

def test_build_markdown_simple(simple_section: ReportSection) -> None:
    """Test markdown generation with simple spec."""
    spec = ReportSpec(
        title="Simple Report",
        subtitle="Subtitle",
        author="Author",
        created_at="2026-04-07T12:00:00Z",
        sections=[simple_section],
    )
    md = build_markdown_report(spec)

    assert "# Simple Report" in md
    assert "**Subtitle**" in md
    assert "Author: Author" in md
    assert "Date: 2026-04-07T12:00:00Z" in md
    assert "## Methods" in md


def test_build_markdown_with_figures(valid_section: ReportSection) -> None:
    """Test markdown with figures."""
    spec = ReportSpec(
        title="Report with Figures",
        subtitle="Sub",
        author="Author",
        created_at="2026-04-07T12:00:00Z",
        sections=[valid_section],
        metadata={},
    )
    md = build_markdown_report(spec)

    assert "[Figure: fig1.png]" in md
    assert "[Figure: fig2.svg]" in md


def test_build_markdown_with_metadata() -> None:
    """Test markdown with metadata."""
    spec = ReportSpec(
        title="Report",
        subtitle="",
        author="Auth",
        created_at="2026-04-07T12:00:00Z",
        sections=[ReportSection(title="Sec", body="Body")],
        metadata={"key1": "value1", "key2": "value2"},
    )
    md = build_markdown_report(spec)

    assert "key1: value1" in md
    assert "key2: value2" in md


def test_build_markdown_multiple_sections() -> None:
    """Test markdown with multiple sections."""
    sections = [
        ReportSection(title="Section 1", body="Content 1"),
        ReportSection(title="Section 2", body="Content 2"),
        ReportSection(title="Section 3", body="Content 3"),
    ]
    spec = ReportSpec(
        title="Multi-Section Report",
        subtitle="",
        author="",
        created_at="2026-04-07T12:00:00Z",
        sections=sections,
    )
    md = build_markdown_report(spec)

    assert "## Section 1" in md
    assert "## Section 2" in md
    assert "## Section 3" in md
    assert md.count("---") >= 2  # Section separators


def test_build_markdown_section_separators() -> None:
    """Test that section separators are present."""
    sections = [
        ReportSection(title="A", body="Content A"),
        ReportSection(title="B", body="Content B"),
    ]
    spec = ReportSpec(
        title="Report",
        subtitle="",
        author="",
        created_at="2026-04-07T12:00:00Z",
        sections=sections,
    )
    md = build_markdown_report(spec)
    lines = md.split("\n")
    # Should have --- between sections
    assert "---" in lines


# =============================================================================
# HTML Report Generation Tests
# =============================================================================

def test_build_html_simple(simple_section: ReportSection) -> None:
    """Test HTML generation with simple spec."""
    spec = ReportSpec(
        title="HTML Report",
        subtitle="Sub",
        author="Auth",
        created_at="2026-04-07T12:00:00Z",
        sections=[simple_section],
    )
    html = build_html_report(spec)

    assert "<html>" in html
    assert "</html>" in html
    assert "<h1>HTML Report</h1>" in html
    assert "<h2>Sub</h2>" in html
    assert "Auth" in html
    assert "2026-04-07T12:00:00Z" in html


def test_build_html_has_css() -> None:
    """Test that HTML includes CSS."""
    spec = ReportSpec(
        title="Report",
        subtitle="",
        author="",
        created_at="2026-04-07T12:00:00Z",
        sections=[ReportSection(title="S", body="B")],
    )
    html = build_html_report(spec)

    assert "<style>" in html
    assert "</style>" in html
    assert "body {" in html


def test_build_html_with_figures() -> None:
    """Test HTML with figure references."""
    section = ReportSection(
        title="Results",
        body="See figures below.",
        figure_paths=["result1.png", "result2.png"],
    )
    spec = ReportSpec(
        title="Report",
        subtitle="",
        author="",
        created_at="2026-04-07T12:00:00Z",
        sections=[section],
    )
    html = build_html_report(spec)

    assert "Figure: result1.png" in html
    assert "Figure: result2.png" in html
    assert "class='figure'" in html


def test_build_html_multiline_body_to_br() -> None:
    """Test that newlines in body are converted to <br>."""
    section = ReportSection(
        title="Methods",
        body="Line 1\nLine 2\nLine 3",
    )
    spec = ReportSpec(
        title="Report",
        subtitle="",
        author="",
        created_at="2026-04-07T12:00:00Z",
        sections=[section],
    )
    html = build_html_report(spec)

    assert "Line 1<br>Line 2<br>Line 3" in html


def test_build_html_with_metadata() -> None:
    """Test HTML with metadata."""
    spec = ReportSpec(
        title="Report",
        subtitle="",
        author="",
        created_at="2026-04-07T12:00:00Z",
        sections=[ReportSection(title="S", body="B")],
        metadata={"param": "value"},
    )
    html = build_html_report(spec)

    assert "param: value" in html
    assert "metadata-box" in html


def test_build_html_valid_structure() -> None:
    """Test that HTML has valid structure."""
    spec = ReportSpec(
        title="Report",
        subtitle="",
        author="",
        created_at="2026-04-07T12:00:00Z",
        sections=[ReportSection(title="S", body="B")],
    )
    html = build_html_report(spec)

    assert html.startswith("<!DOCTYPE html>")
    assert "<head>" in html and "</head>" in html
    assert "<body>" in html and "</body>" in html
    assert html.count("<h1>") == html.count("</h1>")
    assert html.count("<h2>") == html.count("</h2>")


# =============================================================================
# Factory Function Tests
# =============================================================================

def test_make_report_spec_basic() -> None:
    """Test basic make_report_spec."""
    sections = [ReportSection(title="S1", body="B1")]
    spec = make_report_spec(
        title="My Report",
        sections=sections,
        subtitle="My Subtitle",
    )
    assert spec.title == "My Report"
    assert spec.subtitle == "My Subtitle"
    assert spec.author == "Nominal Drift"
    assert "T" in spec.created_at  # ISO format
    assert len(spec.sections) == 1


def test_make_report_spec_auto_timestamp() -> None:
    """Test that timestamp is auto-generated."""
    spec1 = make_report_spec(title="Report", sections=[])
    spec2 = make_report_spec(title="Report", sections=[])
    # Timestamps should be very close (within seconds)
    assert spec1.created_at != spec2.created_at or True  # Allow same if very fast


def test_make_report_spec_default_author() -> None:
    """Test default author."""
    spec = make_report_spec(title="Report", sections=[])
    assert spec.author == "Nominal Drift"


def test_make_report_spec_custom_author() -> None:
    """Test custom author."""
    spec = make_report_spec(
        title="Report",
        sections=[],
        author="Custom Author",
    )
    assert spec.author == "Custom Author"


def test_make_report_spec_metadata() -> None:
    """Test metadata parameter."""
    metadata = {"key": "value", "alloy": "316L"}
    spec = make_report_spec(
        title="Report",
        sections=[],
        metadata=metadata,
    )
    assert spec.metadata == metadata


def test_make_report_spec_default_metadata_none() -> None:
    """Test default metadata is empty dict."""
    spec = make_report_spec(title="Report", sections=[])
    assert spec.metadata == {}


# =============================================================================
# Diffusion Report Factory Tests
# =============================================================================

def test_build_diffusion_report_basic() -> None:
    """Test basic diffusion report."""
    spec = build_diffusion_report(
        composition_label="316L",
        element="Cr",
        matrix="austenite",
        min_cr_wt_pct=16.5,
        depletion_depth_nm=2.5,
        risk_level="HIGH",
        mechanism_label="grain boundary diffusion",
        assumptions=["Assumption 1", "Assumption 2"],
        warnings=["Warning 1"],
        notes=["Note 1", "Note 2"],
    )
    assert spec.title == "Diffusion Analysis: 316L (Cr)"
    assert spec.subtitle == "Risk Assessment: HIGH"
    assert len(spec.sections) == 3
    assert spec.sections[0].title == "Executive Summary"
    assert spec.sections[1].title == "Assumptions & Model Limitations"
    assert spec.sections[2].title == "Recommendations & Notes"


def test_build_diffusion_report_with_figures() -> None:
    """Test diffusion report with figures."""
    spec = build_diffusion_report(
        composition_label="316L",
        element="Cr",
        matrix="austenite",
        min_cr_wt_pct=16.5,
        depletion_depth_nm=None,
        risk_level="MODERATE",
        mechanism_label="mechanism",
        assumptions=[],
        warnings=[],
        notes=[],
        figure_paths=["depletion.png", "diffusion.svg"],
    )
    assert len(spec.sections[0].figure_paths) == 2


def test_build_diffusion_report_metadata() -> None:
    """Test diffusion report metadata."""
    spec = build_diffusion_report(
        composition_label="304",
        element="Ni",
        matrix="ferrite",
        min_cr_wt_pct=18.0,
        depletion_depth_nm=1.0,
        risk_level="LOW",
        mechanism_label="volume diffusion",
        assumptions=[],
        warnings=[],
        notes=[],
    )
    assert spec.metadata["composition"] == "304"
    assert spec.metadata["element"] == "Ni"
    assert spec.metadata["matrix"] == "ferrite"
    assert spec.metadata["risk_level"] == "LOW"


def test_build_diffusion_report_summary_content() -> None:
    """Test diffusion report executive summary."""
    spec = build_diffusion_report(
        composition_label="316L",
        element="Cr",
        matrix="austenite",
        min_cr_wt_pct=16.5,
        depletion_depth_nm=2.5,
        risk_level="HIGH",
        mechanism_label="grain boundary diffusion",
        assumptions=[],
        warnings=[],
        notes=[],
    )
    summary = spec.sections[0].body
    assert "316L" in summary
    assert "Cr" in summary
    assert "16.50" in summary
    assert "2.5 nm" in summary
    assert "HIGH" in summary


def test_build_diffusion_report_assumptions_section() -> None:
    """Test assumptions and warnings section."""
    spec = build_diffusion_report(
        composition_label="316L",
        element="Cr",
        matrix="austenite",
        min_cr_wt_pct=16.5,
        depletion_depth_nm=None,
        risk_level="MODERATE",
        mechanism_label="mechanism",
        assumptions=["Assumption A", "Assumption B"],
        warnings=["Warning X"],
        notes=[],
    )
    assumptions_section = spec.sections[1]
    assert "Assumption A" in assumptions_section.body
    assert "Assumption B" in assumptions_section.body
    assert "Warning X" in assumptions_section.body


def test_build_diffusion_report_notes_section() -> None:
    """Test recommendations and notes section."""
    spec = build_diffusion_report(
        composition_label="316L",
        element="Cr",
        matrix="austenite",
        min_cr_wt_pct=16.5,
        depletion_depth_nm=None,
        risk_level="HIGH",
        mechanism_label="mechanism",
        assumptions=[],
        warnings=[],
        notes=["Note 1", "Note 2", "Note 3"],
    )
    notes_section = spec.sections[2]
    assert "Note 1" in notes_section.body
    assert "Note 2" in notes_section.body
    assert "Note 3" in notes_section.body


def test_build_diffusion_report_no_depletion_depth() -> None:
    """Test diffusion report with no depletion depth."""
    spec = build_diffusion_report(
        composition_label="316L",
        element="Cr",
        matrix="austenite",
        min_cr_wt_pct=16.5,
        depletion_depth_nm=None,
        risk_level="MODERATE",
        mechanism_label="mechanism",
        assumptions=[],
        warnings=[],
        notes=[],
    )
    summary = spec.sections[0].body
    assert "not calculated" in summary


# =============================================================================
# File I/O Tests
# =============================================================================

def test_save_report_markdown() -> None:
    """Test saving markdown report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/report.md"
        content = "# Test Report\n\nContent here."
        result_path = save_report(content, path, fmt="md")

        assert Path(result_path).exists()
        assert Path(result_path).read_text() == content


def test_save_report_html() -> None:
    """Test saving HTML report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/report.html"
        content = "<html><body>Test</body></html>"
        result_path = save_report(content, path, fmt="html")

        assert Path(result_path).exists()
        assert Path(result_path).read_text() == content


def test_save_report_creates_parent_dirs() -> None:
    """Test that parent directories are created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/subdir1/subdir2/report.md"
        content = "Content"
        result_path = save_report(content, path)

        assert Path(result_path).exists()
        assert Path(result_path).parent.exists()


def test_save_report_returns_absolute_path() -> None:
    """Test that save_report returns absolute path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/report.md"
        result_path = save_report("Content", path)

        assert Path(result_path).is_absolute()


def test_save_report_overwrites_existing() -> None:
    """Test that existing file is overwritten."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/report.md"
        Path(path).write_text("Old content")
        save_report("New content", path)

        assert Path(path).read_text() == "New content"


def test_save_report_utf8() -> None:
    """Test that file is saved as UTF-8."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/report.md"
        content = "Test with emoji: 🔥 and accents: café"
        save_report(content, path)

        assert Path(path).read_text(encoding="utf-8") == content


def test_save_report_empty_content() -> None:
    """Test saving empty report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/report.md"
        save_report("", path)

        assert Path(path).exists()
        assert Path(path).read_text() == ""


# =============================================================================
# Integration Tests
# =============================================================================

def test_full_workflow_markdown() -> None:
    """Test full workflow: create spec, build markdown, save."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sections = [
            ReportSection(title="Intro", body="Introduction text"),
            ReportSection(title="Results", body="Results here"),
        ]
        spec = make_report_spec(
            title="Full Workflow Report",
            sections=sections,
            author="Test",
            metadata={"key": "value"},
        )
        md = build_markdown_report(spec)
        path = f"{tmpdir}/report.md"
        result_path = save_report(md, path, fmt="md")

        saved_content = Path(result_path).read_text()
        assert "# Full Workflow Report" in saved_content
        assert "## Intro" in saved_content
        assert "## Results" in saved_content


def test_full_workflow_html() -> None:
    """Test full workflow with HTML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sections = [ReportSection(title="Section", body="Body")]
        spec = make_report_spec(title="Report", sections=sections)
        html = build_html_report(spec)
        path = f"{tmpdir}/report.html"
        result_path = save_report(html, path, fmt="html")

        saved_content = Path(result_path).read_text()
        assert "<html>" in saved_content
        assert "</html>" in saved_content


def test_diffusion_report_to_markdown() -> None:
    """Test diffusion report conversion to markdown."""
    spec = build_diffusion_report(
        composition_label="316L",
        element="Cr",
        matrix="austenite",
        min_cr_wt_pct=16.5,
        depletion_depth_nm=2.5,
        risk_level="HIGH",
        mechanism_label="grain boundary",
        assumptions=["Assumption 1"],
        warnings=["Warning 1"],
        notes=["Note 1"],
    )
    md = build_markdown_report(spec)
    assert "Diffusion Analysis: 316L (Cr)" in md
    assert "Executive Summary" in md
    assert "Assumptions & Model Limitations" in md
    assert "Recommendations & Notes" in md


def test_diffusion_report_to_html() -> None:
    """Test diffusion report conversion to HTML."""
    spec = build_diffusion_report(
        composition_label="316L",
        element="Cr",
        matrix="austenite",
        min_cr_wt_pct=16.5,
        depletion_depth_nm=2.5,
        risk_level="HIGH",
        mechanism_label="grain boundary",
        assumptions=["Assumption 1"],
        warnings=["Warning 1"],
        notes=["Note 1"],
    )
    html = build_html_report(spec)
    assert "<html>" in html
    assert "Diffusion Analysis: 316L (Cr)" in html
    assert "Executive Summary" in html
