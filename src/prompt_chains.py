"""LangChain runnable-chain construction for quality prompt workflows."""

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable

from src.prompt_schemas import (
    FindingExplanation,
    QualityReportSummary,
)
from src.prompt_templates import (
    PROMPT_VERSION,
    finding_explanation_prompt,
    report_summary_prompt,
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
