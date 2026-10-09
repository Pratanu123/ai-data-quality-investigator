"""LangChain runnable-chain construction for quality prompt workflows."""

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable

from src.prompt_schemas import (
    FindingExplanation,
    QualityCheckPlan,
    QualityReportSummary,
)
from src.prompt_templates import (
    PROMPT_VERSION,
    dataset_quality_checks_prompt,
    finding_explanation_prompt,
    report_summary_prompt,
)


def validation_plan_chain(
    llm: BaseChatModel,
) -> Runnable[dict[str, str], QualityCheckPlan]:
    """Build a chain that proposes checks for a newly selected/uploaded dataset."""
    parser = PydanticOutputParser(pydantic_object=QualityCheckPlan)
    prompt = dataset_quality_checks_prompt(parser.get_format_instructions())
    json_model = llm.bind(
        response_format={"type": "json_object"},
        max_tokens=1200,
        extra_body={"reasoning": {"effort": "minimal"}},
    )
    return (prompt | json_model | parser).with_config(
        {"run_name": f"dataset-quality-checks-v{PROMPT_VERSION}"}
    )


def finding_explanation_chain(
    llm: BaseChatModel,
) -> Runnable[dict[str, str], FindingExplanation]:
    """Build an LLM chain that returns a validated finding explanation."""
    parser = PydanticOutputParser(pydantic_object=FindingExplanation)
    prompt = finding_explanation_prompt(parser.get_format_instructions())
    return (prompt | llm | parser).with_config(
        {"run_name": f"finding-explanation-v{PROMPT_VERSION}"}
    )


def report_summary_chain(
    llm: BaseChatModel,
) -> Runnable[dict[str, str], QualityReportSummary]:
    """Build an LLM chain that returns a validated aggregate report summary."""
    parser = PydanticOutputParser(pydantic_object=QualityReportSummary)
    prompt = report_summary_prompt(parser.get_format_instructions())
    return (prompt | llm | parser).with_config(
        {"run_name": f"report-summary-v{PROMPT_VERSION}"}
    )
