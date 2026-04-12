"""Reports page."""
import streamlit as st


def render():
    st.title("📄 Reports")
    st.caption("Generate engineering reports · Markdown · HTML")
    title = st.text_input("Report Title", "Diffusion Analysis Report")
    author = st.text_input("Author", "Nominal Drift")
    fmt = st.selectbox("Format", ["markdown", "html"])
    notes = st.text_area("Notes / Findings", height=150)
    if st.button("Generate Report"):
        try:
            from nominal_drift.reports.report_builder import (
                make_report_spec,
                ReportSection,
                build_markdown_report,
                build_html_report,
            )

            spec = make_report_spec(
                title=title,
                sections=[ReportSection(title="Notes", body=notes, figure_paths=[])],
                author=author,
            )
            content = (
                build_markdown_report(spec)
                if fmt == "markdown"
                else build_html_report(spec)
            )
            st.code(content, language="markdown" if fmt == "markdown" else "html")
            st.download_button(
                "⬇ Download",
                content,
                file_name=f"report.{('md' if fmt=='markdown' else 'html')}",
            )
        except Exception as e:
            st.error(str(e))
