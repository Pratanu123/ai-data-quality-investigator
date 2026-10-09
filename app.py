"""Streamlit workspace for investigating and cleaning Amazon catalog data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from langchain_core.exceptions import OutputParserException
from openai import APIError

from src.cleaning import apply_cleaning_proposals, propose_cleaning
from src.llm import (
    OpenRouterConfigurationError,
    getllm,
    is_openrouter_configured,
)
from src.prompts_manager import (
    MAX_EVIDENCE_ISSUES,
    FindingExplanation,
    PromptsManager,
    QualityReportSummary,
)
from src.quality_engine import (
    QualityFinding,
    QualityIssue,
    QualityReport,
    VALIDATION_RULES,
    analyze_dataset,
    summarize_dataset,
)

DATA_DIRECTORY = Path(__file__).resolve().parent / "data"
DETAILS_PER_PAGE = 12
WORKFLOW_STAGES = ("Overview", "Investigate", "Clean data", "Export")


def _render_issue_card(finding: QualityFinding, issue: QualityIssue) -> None:
    finding_rule = finding.rule
    with st.container(border=True):
        st.markdown(f"**CSV row {issue.source_row:,}**")
        identity = [
            str(value)
            for value in (
                issue.source_values.get("product_id"),
                issue.source_values.get("product_name"),
            )
            if value is not None and not pd.isna(value)
        ]
        if identity:
            st.caption(" · ".join(identity))
        st.markdown(f"**Why it matters:** {issue.reason}")
        if finding.columns:
            st.markdown("**Inspected values**")
            for column in finding.columns:
                st.write(f"{column}: {issue.source_values.get(column)!r}")

        if finding_rule == "Missing and blank values":
            affected_fields = [
                column
                for column, value in issue.source_values.items()
                if pd.isna(value) or (isinstance(value, str) and not value.strip())
            ]
            for field in affected_fields:
                value = issue.source_values[field]
                label = "Missing" if pd.isna(value) else "Blank"
                st.markdown(f"**Field:** `{field}` · {label}")


def _run_quality_checks(data: pd.DataFrame) -> None:
    with st.status("Running Amazon product quality checks...", expanded=True) as status:
        report = analyze_dataset(data)
        st.session_state["quality_report"] = report
        st.session_state["cleaning_proposals"] = propose_cleaning(data)
        for key in (
            "approved_cleaned_dataset",
            "approved_cleaning_count",
            "approved_proposal_indices",
            "cleaning_preview_editor",
            "quality_ai_summary",
        ):
            st.session_state.pop(key, None)
        status.update(label="Quality checks complete", state="complete", expanded=False)


def _ensure_quality_report() -> QualityReport | None:
    report = st.session_state.get("quality_report")
    if isinstance(report, QualityReport):
        return report
    st.info("Run the Amazon product quality engine from Overview first.")
    return None


def _render_ai_summary(report: QualityReport, openrouter_ready: bool) -> None:
    st.caption(
        "AI summaries use aggregate rule counts; selected finding explanations include "
        f"up to {MAX_EVIDENCE_ISSUES} relevant examples."
    )
    if st.button(
        "Summarize report with AI",
        disabled=not openrouter_ready,
        key="summarize_report_with_ai",
    ):
        try:
            with st.spinner("Generating an evidence-based report summary..."):
                st.session_state["quality_ai_summary"] = PromptsManager.summarize_report(
                    getllm(), report
                )
        except OpenRouterConfigurationError as exc:
            st.error(str(exc))
        except (APIError, OutputParserException) as exc:
            st.error(f"OpenRouter report summary failed: {exc}")

    summary = st.session_state.get("quality_ai_summary")
    if isinstance(summary, QualityReportSummary):
        with st.container(border=True):
            st.markdown("#### AI report summary")
            st.write(summary.overall_summary)
            left, right = st.columns(2)
            with left:
                st.markdown("**Key risks**")
                for risk in summary.key_risks:
                    st.markdown(f"- {risk}")
            with right:
                st.markdown("**Recommended priorities**")
                for priority in summary.recommended_priorities:
                    st.markdown(f"- {priority}")


def _render_investigation(report: QualityReport, openrouter_ready: bool) -> None:
    findings_with_issues = [finding for finding in report.findings if finding.issues]
    rows_flagged = {
        issue.source_row for finding in findings_with_issues for issue in finding.issues
    }
    st.subheader("Investigation workspace")
    st.caption("Select a rule to understand the finding and inspect source records.")
    _render_ai_summary(report, openrouter_ready)
    summary_column, records_column = st.columns(2)
    summary_column.metric("Records flagged", f"{len(rows_flagged):,}")
    summary_column.metric(
        "Rule findings",
        f"{sum(finding.affected_rows for finding in report.findings):,}",
    )
    records_column.metric(
        "Rules with findings",
        f"{sum(bool(finding.issues) for finding in report.findings)}",
    )
    records_column.metric(
        "Exact duplicate rows", f"{report.summary.exact_duplicate_rows:,}"
    )

    finding_by_rule = {finding.rule: finding for finding in report.findings}
    rules_with_findings = [finding.rule for finding in findings_with_issues]
    selected_rule = st.session_state.get("selected_quality_rule")
    if selected_rule not in rules_with_findings:
        selected_rule = rules_with_findings[0] if rules_with_findings else None

    findings_column, details_column = st.columns([0.95, 1.25], gap="large")
    with findings_column:
        st.markdown("#### Findings")
        for finding in report.findings:
            selected = finding.rule == selected_rule
            if st.button(
                finding.rule,
                help=f"{finding.affected_rows:,} affected rows. {finding.details}",
                type="primary" if selected else "secondary",
                use_container_width=True,
                key=f"select_finding_{finding.rule}",
            ):
                selected_rule = finding.rule
                st.session_state["selected_quality_rule"] = selected_rule
                st.rerun()
            st.caption(f"{finding.affected_rows:,} affected")

    with details_column:
        with st.container(border=True):
            if selected_rule is None:
                st.markdown("#### All clear")
                st.write("No records were flagged by the configured validation rules.")
                return

            selected_finding = finding_by_rule[selected_rule]
            st.markdown(f"#### {selected_finding.rule}")
            st.caption(selected_finding.details)
            explanation_key = f"quality_ai_finding_{selected_finding.rule}"

            if selected_finding.issues and st.button(
                "Explain finding and recommend a fix",
                disabled=not openrouter_ready,
                key=f"explain_{selected_finding.rule}",
            ):
                try:
                    with st.spinner("Explaining this finding with OpenRouter..."):
                        st.session_state[explanation_key] = (
                            PromptsManager.explain_finding(getllm(), selected_finding)
                        )
                except OpenRouterConfigurationError as exc:
                    st.error(str(exc))
                except (APIError, OutputParserException) as exc:
                    st.error(f"OpenRouter finding explanation failed: {exc}")

            explanation = st.session_state.get(explanation_key)
            if isinstance(explanation, FindingExplanation):
                with st.container(border=True):
                    priority_class = f"ai-pill-{explanation.priority.lower()}"
                    st.markdown(
                        '<div class="ai-pills">'
                        f'<span class="ai-pill {priority_class}">'
                        f"{explanation.priority.upper()} PRIORITY"
                        "</span>"
                        '<span class="ai-pill">AI-ASSISTED</span>'
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Why it matters**")
                    st.write(explanation.explanation)
                    st.markdown(
                        '<span class="ai-pill ai-pill-fix">RECOMMENDED FIX</span>',
                        unsafe_allow_html=True,
                    )
                    st.write(explanation.recommended_fix)

            if not selected_finding.issues:
                st.success("No records were flagged by this rule.")
                return

            pages = (
                len(selected_finding.issues) + DETAILS_PER_PAGE - 1
            ) // DETAILS_PER_PAGE
            page = 1
            if pages > 1:
                page = st.number_input(
                    "Issue page",
                    min_value=1,
                    max_value=pages,
                    value=1,
                    step=1,
                    key=f"issue_page_{selected_finding.rule}",
                )
            start = (page - 1) * DETAILS_PER_PAGE
            current_issues = selected_finding.issues[start : start + DETAILS_PER_PAGE]
            st.caption(
                f"Records {start + 1:,}–{start + len(current_issues):,} "
                f"of {len(selected_finding.issues):,}"
            )
            for issue in current_issues:
                _render_issue_card(selected_finding, issue)


def _render_cleaning(
    data: pd.DataFrame,
) -> None:
    st.subheader("Clean data")
    st.caption(
        "Preview and explicitly approve deterministic changes. Blank values are not "
        "filled, numeric issues become missing values, and IDs are never rewritten."
    )

    proposals = st.session_state.get("cleaning_proposals")
    if not isinstance(proposals, tuple):
        proposals = propose_cleaning(data)
        st.session_state["cleaning_proposals"] = proposals

    if not proposals:
        st.success("No deterministic cleaning changes to preview.")
    else:
        proposal_preview = pd.DataFrame(
            [
                {
                    "Approve": False,
                    "CSV row": proposal.source_row,
                    "Field": proposal.column,
                    "Current value": str(proposal.before),
                    "Proposed value": (
                        "Missing (blank in CSV)"
                        if proposal.after is None
                        else str(proposal.after)
                    ),
                    "Reason": proposal.reason,
                }
                for proposal in proposals
            ]
        )
        edited = st.data_editor(
            proposal_preview,
            width="stretch",
            hide_index=True,
            num_rows="fixed",
            disabled=["CSV row", "Field", "Current value", "Proposed value", "Reason"],
            column_config={
                "Approve": st.column_config.CheckboxColumn(
                    "Approve",
                    help="Check a proposal to approve it.",
                    default=False,
                )
            },
            key="cleaning_preview_editor",
        )
        st.caption("Unchecked proposals are rejected and will not be applied.")
        selected_indices = tuple(
            index for index, approved in enumerate(edited["Approve"]) if approved
        )
        previous_indices = st.session_state.get("approved_proposal_indices")
        if previous_indices is not None and selected_indices != previous_indices:
            for key in (
                "approved_cleaned_dataset",
                "approved_cleaning_count",
                "approved_proposal_indices",
            ):
                st.session_state.pop(key, None)
            st.info("Approval selection changed. Apply it again to update the cleaned copy.")

        if st.button("Apply approved changes", type="primary", key="apply_cleaning"):
            approved = tuple(proposals[index] for index in selected_indices)
            if not approved:
                for key in (
                    "approved_cleaned_dataset",
                    "approved_cleaning_count",
                    "approved_proposal_indices",
                ):
                    st.session_state.pop(key, None)
                st.info("No changes approved; the source dataset remains unchanged.")
            else:
                try:
                    st.session_state["approved_cleaned_dataset"] = (
                        apply_cleaning_proposals(data, approved)
                    )
                    st.session_state["approved_cleaning_count"] = len(approved)
                    st.session_state["approved_proposal_indices"] = selected_indices
                    st.success(
                        f"{len(approved):,} changes applied to a separate copy. "
                        "The source data remains unchanged."
                    )
                except ValueError as exc:
                    st.error(f"Could not apply approved changes: {exc}")


def _render_export(report: QualityReport | None, dataset_name: str) -> None:
    st.subheader("Export")
    if report is None:
        st.info("Run the quality engine before exporting the findings report.")
        return
    findings = pd.DataFrame(
        [
            {
                "Validation": finding.rule,
                "Affected rows": finding.affected_rows,
                "Details": finding.details,
                "Source rows": ", ".join(str(row) for row in finding.rows),
            }
            for finding in report.findings
        ]
    )
    st.write("Download a findings report, or the separately approved cleaned dataset.")
    st.download_button(
        "Download findings report",
        data=findings.to_csv(index=False).encode("utf-8"),
        file_name=f"{Path(dataset_name).stem}_quality_findings.csv",
        mime="text/csv",
        key="download_findings_csv",
    )

    cleaned = st.session_state.get("approved_cleaned_dataset")
    approved_count = st.session_state.get("approved_cleaning_count", 0)
    if isinstance(cleaned, pd.DataFrame) and approved_count:
        st.download_button(
            "Download approved cleaned CSV",
            data=cleaned.to_csv(index=False).encode("utf-8"),
            file_name=f"{Path(dataset_name).stem}_cleaned.csv",
            mime="text/csv",
            key="download_approved_cleaned_csv",
        )
    else:
        st.caption("A cleaned CSV is available only after you approve changes in Clean data.")


st.set_page_config(
    page_title="Data Quality Investigator",
    page_icon="DQ",
    layout="wide",
)
st.markdown(
    """
    <style>
    .block-container { max-width: 1380px; padding-top: 2.5rem; padding-bottom: 3rem; }
    [data-testid="stMetric"] {
        min-height: 6.3rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 0.7rem;
        padding: 0.8rem 1rem;
    }
    [data-testid="stMetricLabel"] { opacity: 0.78; }
    .workspace-eyebrow {
        margin: 0 0 0.2rem;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        opacity: 0.68;
    }
    .ai-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.15rem 0 1rem;
    }
    .ai-pill {
        display: inline-flex;
        align-items: center;
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 999px;
        padding: 0.25rem 0.65rem;
        font-size: 0.72rem;
        font-weight: 650;
        line-height: 1.2;
        letter-spacing: 0.025em;
        background: rgba(148, 163, 184, 0.12);
    }
    .ai-pill-critical, .ai-pill-high {
        color: #ffb4ab;
        border-color: rgba(255, 118, 105, 0.42);
        background: rgba(255, 85, 75, 0.13);
    }
    .ai-pill-medium {
        color: #ffd28a;
        border-color: rgba(242, 183, 86, 0.4);
        background: rgba(242, 183, 86, 0.12);
    }
    .ai-pill-low {
        color: #a7dfbb;
        border-color: rgba(99, 190, 130, 0.38);
        background: rgba(99, 190, 130, 0.12);
    }
    .ai-pill-fix {
        color: #c5caff;
        border-color: rgba(137, 146, 255, 0.38);
        background: rgba(117, 129, 255, 0.12);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

dataset_path = DATA_DIRECTORY / "amazon.csv"
try:
    dataset = pd.read_csv(dataset_path)
except (
    OSError,
    ValueError,
    pd.errors.ParserError,
    pd.errors.EmptyDataError,
    UnicodeDecodeError,
) as exc:
    st.error(f"Could not load the Amazon product dataset: {exc}")
    st.stop()

dataset_name = dataset_path.name
summary = summarize_dataset(dataset)
openrouter_ready = is_openrouter_configured()
report = st.session_state.get("quality_report")
if not isinstance(report, QualityReport):
    report = None

header, status_column = st.columns([5, 1])
with header:
    st.markdown(
        '<div class="workspace-eyebrow">DATA QUALITY WORKSPACE</div>',
        unsafe_allow_html=True,
    )
    st.title("Data Quality Investigator")
    st.caption(f"`{dataset_name}` · {len(dataset):,} rows · {len(dataset.columns)} columns")
with status_column:
    st.markdown("")
    if report is None:
        st.markdown("**● Ready**")
    else:
        st.markdown("**● In review**")

stage = st.segmented_control(
    "Workflow stage",
    WORKFLOW_STAGES,
    default="Overview",
    key="workflow_stage",
    label_visibility="collapsed",
    width="stretch",
)

flagged_records = (
    len(
        {
            issue.source_row
            for finding in report.findings
            for issue in finding.issues
        }
    )
    if report
    else 0
)
metric_columns = st.columns(2)
metric_columns[0].metric(
    "Quality score",
    "—",
    help="A quality score is not shown until a defensible scoring formula is defined.",
)
metric_columns[1].metric("Records flagged", f"{flagged_records:,}")
metric_columns = st.columns(2)
metric_columns[0].metric(
    "Missing cells", f"{summary.missing_cells + summary.blank_cells:,}"
)
metric_columns[1].metric("Exact duplicates", f"{summary.exact_duplicate_rows:,}")

if stage == "Overview":
    st.subheader("Dataset overview")
    st.caption(
        "This workspace runs the built-in Amazon product checks on the bundled dataset. "
        "No upload or AI-generated check plan is used."
    )
    overview, scope = st.columns([1.15, 0.85], gap="large")
    with overview:
        with st.container(border=True):
            st.markdown("#### Source dataset")
            st.caption(f"{len(dataset):,} records · {len(dataset.columns)} fields")
            st.dataframe(
                dataset.head(8),
                width="stretch",
                hide_index=True,
            )
    with scope:
        with st.container(border=True):
            st.markdown("#### Column profile")
            profile = pd.DataFrame(
                [
                    {
                        "Column": str(column),
                        "Type": str(dataset[column].dtype),
                        "Present": int(dataset[column].notna().sum()),
                        "Distinct": int(dataset[column].nunique(dropna=True)),
                    }
                    for column in dataset.columns
                ]
            )
            st.dataframe(profile, width="stretch", hide_index=True)

    st.markdown("#### Built-in quality checks")
    st.caption(
        "The local engine runs these Amazon-specific rules directly. "
        "No LLM call is made to choose checks."
    )
    for rule in VALIDATION_RULES:
        st.markdown(f"- {rule}")

    if st.button(
        "Run Amazon product quality checks",
        type="primary",
        key="run_quality_checks",
    ):
        with st.spinner("Running deterministic local quality checks..."):
            _run_quality_checks(dataset)

    if report is not None:
        st.success("Amazon product quality checks complete. Continue to Investigate or Clean data.")

elif stage == "Investigate":
    report = _ensure_quality_report()
    if report is not None:
        _render_investigation(report, openrouter_ready)

elif stage == "Clean data":
    report = _ensure_quality_report()
    if report is not None:
        _render_cleaning(dataset)

elif stage == "Export":
    report = _ensure_quality_report()
    _render_export(report, dataset_name)
