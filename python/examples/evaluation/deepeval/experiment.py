import json
import logging
import os
import pickle
from collections import Counter
from pathlib import Path
import asyncio

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ExactMatchMetric,
    ToolCorrectnessMetric,
    ArgumentCorrectnessMetric,
)
from deepeval.test_case import LLMTestCase, ToolCall

load_dotenv()

logger = logging.getLogger(__name__)

from examples.evaluation.agent import create_agent
from examples.evaluation.dataset import load_items
from evaluation.adapters import DeepEvalLLM
from beeai_framework.backend import ToolMessage

from AnswerLLMJudgeMetric import AnswerLLMJudgeMetric
from ToolUsageMetric import ToolUsageMetric
from FactsSimilarityMetric import FactsSimilarityMetric


def count_tool_usage(messages):
    tool_counter = Counter()
    for msg in messages:
        if isinstance(msg, ToolMessage):
            for item in msg.content:
                tool_name = getattr(item, "tool_name", None)
                if tool_name and tool_name != "final_answer":
                    tool_counter[tool_name] += 1
    return dict(tool_counter)


async def create_rag_test_cases():
    agent = create_agent()
    test_cases = []
    test_data = load_items()

    for item in test_data:
        question = item["question"]
        expected_output = item["answer"]
        context = item["relevant_sentences"]
        expected_tools = {"Wikipedia": item["wiki_times"]}
        supporting_titles = item["supporting_titles"]
        tools_used = [
            ToolCall(name="Wikipedia", input_parameters={"query": name})
            for name in supporting_titles
        ]

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
                agent_response_json.get("answer")
                or agent_response_json.get("final_answer")
                or actual_output
            )
            agent_supporting_sentences = agent_response_json.get("supporting_sentences", [])

            tool_used_field = agent_response_json.get("tool_used", [])
            agent_tools_used = []

            if isinstance(tool_used_field, str):
                agent_tools_used.append(ToolCall(name=tool_used_field, input_parameters={}))
            elif isinstance(tool_used_field, list):
                for entry in tool_used_field:
                    tool_name = entry.get("tool") if isinstance(entry, dict) else None
                    times_used = entry.get("times_used", 1) if isinstance(entry, dict) else 1
                    titles = entry.get("titles", []) if isinstance(entry, dict) else []
                    if tool_name:
                        agent_tools_used.append(
                            ToolCall(name=tool_name, input_parameters={"titles": titles} if titles else {})
                        )
                        if times_used:
                            agent_tool_usage_times[tool_name] = times_used

            if not agent_tools_used and agent_tool_usage_times:
                for tool_name in agent_tool_usage_times:
                    agent_tools_used.append(ToolCall(name=tool_name, input_parameters={}))

        except Exception as exc:
            logger.error("Agent failed on question: %r — %s", question, exc)
            agent_final_answer = ""
            agent_supporting_sentences = []
            agent_tools_used = []
            agent_tool_usage_times = {}

        test_case = LLMTestCase(
            input=question,
            actual_output=agent_final_answer,
            expected_output=expected_output,
            retrieval_context=agent_supporting_sentences,
            context=context,
            tools_called=agent_tools_used,
            expected_tools=tools_used,
            additional_metadata={
                "expected_facts": context,
                "tool_usage": agent_tool_usage_times,
                "expected_tool_usage": expected_tools,
                "supporting_titles": supporting_titles,
            }
        )

        logger.info(
            "Test case — Question: %s | Expected: %s | Actual: %s | Expected tools: %s | Actual tools: %s",
            question, expected_output, agent_final_answer, expected_tools, agent_tool_usage_times,
        )

        test_cases.append(test_case)

    return test_cases


async def main() -> None:
    test_cases = await create_rag_test_cases()

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

    eval_results = evaluate(test_cases=test_cases, metrics=metrics)

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
        metrics_data = (
            getattr(test_res, "metrics_data", None)
            or getattr(test_res, "metrics_results", None)
            or []
        )
        metric_success_map = {
            (getattr(md, "metric_name", None) or getattr(md, "name", None) or md.__class__.__name__):
            getattr(md, "success", False)
            for md in metrics_data
        }
        row = [f"Test case {idx + 1}"]
        for name in metric_names:
            passed = metric_success_map.get(name, False)
            row.append("V" if passed else "X")
            if passed:
                success_counts[name] += 1
        rows.append(row)

    footer = ["Success %"] + [
        f"{(success_counts[n] / total_cases * 100):.0f}%" if total_cases else "0%"
        for n in metric_names
    ]

    all_rows = [["Test case"] + metric_names] + rows + [footer]
    col_widths = [max(len(str(c)) for c in col) for col in zip(*all_rows)]

    def fmt_row(r):
        return " | ".join(str(c).ljust(w) for c, w in zip(r, col_widths))

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
