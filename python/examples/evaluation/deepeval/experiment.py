import asyncio
import json
import logging
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import AnyMessage, AssistantMessage, ToolMessage
from beeai_framework.evaluation.adapters import DeepEvalLLM
from deepeval import evaluate
from deepeval.dataset import Golden
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ArgumentCorrectnessMetric,
    ContextualRecallMetric,
    ExactMatchMetric,
    FaithfulnessMetric,
    ToolCorrectnessMetric,
)
from deepeval.test_case import LLMTestCase, ToolCall
from evaluation._utils import create_dataset
from examples.evaluation.agent import create_agent
from examples.evaluation.dataset import load_items
from examples.evaluation.deepeval.answer_llm_judge_metric import AnswerLLMJudgeMetric
from examples.evaluation.deepeval.facts_similarity_metric import FactsSimilarityMetric
from examples.evaluation.deepeval.tool_usage_metric import ToolUsageMetric

load_dotenv()

logger = logging.getLogger(__name__)


def count_tool_usage(messages: list[AnyMessage]) -> dict[str, int]:
    tool_counter = Counter()
    for msg in messages:
        if isinstance(msg, ToolMessage):
            for item in msg.content:
                tool_name = getattr(item, "tool_name", None)
                if tool_name and tool_name != "final_answer":
                    tool_counter[tool_name] += 1
    return dict(tool_counter)


def extract_real_tool_calls(messages: list[AnyMessage]) -> list[ToolCall]:
    """Derive ToolCalls with the arguments the agent actually sent, from the run's own
    AssistantMessage tool-call content — instead of the agent's self-reported JSON summary,
    which never includes call arguments (see JSON_SCHEMA_STRING in examples/evaluation/agent.py)."""
    tool_calls = []
    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        for call in msg.get_tool_calls():
            if call.tool_name == "final_answer":
                continue
            try:
                input_parameters = json.loads(call.args)
            except (json.JSONDecodeError, TypeError):
                input_parameters = {}
            tool_calls.append(ToolCall(name=call.tool_name, input_parameters=input_parameters))
    return tool_calls


def rag_goldens() -> list[Golden]:
    """Build the 'expected' half of each test case — the Golden — from the shared dataset."""
    goldens = []
    for item in load_items():
        supporting_titles = item["supporting_titles"]
        goldens.append(
            Golden(
                input=item["question"],
                expected_output=item["answer"],
                context=item["relevant_sentences"],
                expected_tools=[
                    ToolCall(name="Wikipedia", input_parameters={"query": name}) for name in supporting_titles
                ],
                additional_metadata={
                    "expected_facts": item["relevant_sentences"],
                    "expected_tool_usage": {"Wikipedia": item["wiki_times"]},
                    "supporting_titles": supporting_titles,
                },
            )
        )
    return goldens


async def run_rag_agent(agent: RequirementAgent, test_case: LLMTestCase) -> None:
    """`agent_run` callback for `evaluation._utils.create_dataset`: runs the agent and fills
    in the 'actual' half of test_case (actual_output, retrieval_context, tools_called) by
    parsing the agent's multi-hop QA JSON response — same parsing logic as before, just
    moved into the shape create_dataset expects instead of a hand-rolled loop."""
    question = test_case.input
    expected_tool_usage = (test_case.additional_metadata or {}).get("expected_tool_usage", {})

    try:
        response = await agent.run(question)
        memory = response.state.memory.messages
        actual_output = response.last_message.text
        agent_tool_usage_times = count_tool_usage(memory)

        try:
            agent_response_json = json.loads(actual_output)
        except (json.JSONDecodeError, TypeError):
            agent_response_json = {}

        agent_final_answer = (
            agent_response_json.get("answer") or agent_response_json.get("final_answer") or actual_output
        )
        agent_supporting_sentences = agent_response_json.get("supporting_sentences", [])

        # Real tool calls with real arguments, taken from the agent's own run — not the
        # self-reported `tool_used` JSON field, which never includes call arguments.
        agent_tools_used = extract_real_tool_calls(memory)

    except Exception as exc:
        logger.error("Agent failed on question: %r — %s", question, exc)
        agent_final_answer = ""
        agent_supporting_sentences = []
        agent_tools_used = []
        agent_tool_usage_times = {}

    test_case.actual_output = agent_final_answer
    test_case.retrieval_context = agent_supporting_sentences
    test_case.tools_called = agent_tools_used
    if test_case.additional_metadata is not None:
        test_case.additional_metadata["tool_usage"] = agent_tool_usage_times

    logger.info(
        "Test case — Question: %s | Expected: %s | Actual: %s | Expected tools: %s | Actual tools: %s",
        question,
        test_case.expected_output,
        agent_final_answer,
        expected_tool_usage,
        agent_tool_usage_times,
    )


async def main() -> None:
    dataset = await create_dataset(
        name="rag_multi_hop",
        agent_factory=create_agent,
        agent_run=run_rag_agent,
        goldens=rag_goldens(),
    )
    test_cases = dataset.test_cases

    eval_model_name = os.environ.get("EVAL_CHAT_MODEL_NAME", "ollama:llama3.1:8b")
    os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "60")
    eval_model = DeepEvalLLM.from_name(eval_model_name)

    metrics = [
        # Final answer
        ExactMatchMetric(threshold=1.0),
        AnswerLLMJudgeMetric(model=eval_model, threshold=0.7),
        AnswerRelevancyMetric(model=eval_model, threshold=0.7),
        ContextualRecallMetric(model=eval_model, threshold=0.7),
        FaithfulnessMetric(model=eval_model, threshold=0.7),
        # Tools
        ToolCorrectnessMetric(model=eval_model, include_reason=False),
        ToolUsageMetric(),
        ArgumentCorrectnessMetric(threshold=0.7, model=eval_model, include_reason=True),
        # Facts
        FactsSimilarityMetric(model=eval_model),
    ]

    eval_results = evaluate(test_cases=test_cases, metrics=metrics)  # pyrefly: ignore [not-callable]

    try:
        raw_path = Path(__file__).parent / "eval_results_raw.pkl"
        with raw_path.open("wb") as f:
            pickle.dump(eval_results, f)
        logger.info("Saved raw eval results to %s", raw_path)
    except Exception as exc:
        logger.warning("Failed to persist eval results: %s", exc)

    # Pass/fail summary table
    metric_names = [getattr(m, "__name__", None) or m.__class__.__name__ for m in metrics]
    per_test_results = (
        getattr(eval_results, "results", None)
        or getattr(eval_results, "test_results", None)
        or (eval_results if isinstance(eval_results, list) else [])
    )

    rows = []
    success_counts = Counter({name: 0 for name in metric_names})
    total_cases = len(per_test_results)

    for idx, test_res in enumerate(per_test_results):
        metrics_data = getattr(test_res, "metrics_data", None) or getattr(test_res, "metrics_results", None) or []
        metric_success_map = {
            (getattr(md, "metric_name", None) or getattr(md, "name", None) or md.__class__.__name__): getattr(
                md, "success", False
            )
            for md in metrics_data
        }
        row = [f"Test case {idx + 1}"]
        for name in metric_names:
            passed = metric_success_map.get(name, False)
            row.append("V" if passed else "X")
            if passed:
                success_counts[name] += 1
        rows.append(row)  # pyrefly: ignore [bad-argument-type]

    footer = ["Success %"] + [
        f"{(success_counts[n] / total_cases * 100):.0f}%" if total_cases else "0%" for n in metric_names
    ]

    all_rows = [["Test case", *metric_names], *rows, footer]
    col_widths = [max(len(str(c)) for c in col) for col in zip(*all_rows, strict=False)]

    def fmt_row(r: list[Any]) -> str:
        return " | ".join(str(c).ljust(w) for c, w in zip(r, col_widths, strict=False))

    logger.info("\n=== Evaluation Results Table ===")
    logger.info(fmt_row(all_rows[0]))
    logger.info("-+-".join("-" * w for w in col_widths))
    for row in all_rows[1:-1]:
        logger.info(fmt_row(row))
    logger.info("-+-".join("-" * w for w in col_widths))
    logger.info(fmt_row(all_rows[-1]))
    logger.info("=== End Table ===\n")


if __name__ == "__main__":
    asyncio.run(main())
