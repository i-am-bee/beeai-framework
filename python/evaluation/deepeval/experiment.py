import json
import os
import sys
import pickle
from pathlib import Path
import asyncio
from typing import Counter, List

import pytest
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

# Add python/ root to path for shared modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# Add this folder to path for local metric imports
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.agent import create_agent
from evaluation.dataset import load_items
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
        HotpotQA_expected_output = item["answer"]
        HotpotQA_context = item["relevant_sentences"]
        HotpotQA_expected_tools = {"Wikipedia": item["wiki_times"]}
        supporting_titles = item["supporting_titles"]
        HotpotQA_tools_used = [
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
            print(f"[ERROR] Agent failed on question: {question!r} — {exc}")
            agent_final_answer = ""
            agent_supporting_sentences = []
            agent_tools_used = []
            agent_tool_usage_times = {}

        test_case = LLMTestCase(
            input=question,
            actual_output=agent_final_answer,
            expected_output=HotpotQA_expected_output,
            retrieval_context=agent_supporting_sentences,
            context=HotpotQA_context,
            tools_called=agent_tools_used,
            expected_tools=HotpotQA_tools_used,
            additional_metadata={
                "expected_facts": HotpotQA_context,
                "tool_usage": agent_tool_usage_times,
                "expected_tool_usage": HotpotQA_expected_tools,
                "supporting_titles": supporting_titles,
            }
        )

        print("----- TEST CASE -----")
        print(f"Question: {question}")
        print(f"Expected answer: {HotpotQA_expected_output}")
        print(f"Actual answer: {agent_final_answer}")
        print(f"Expected tools: {HotpotQA_expected_tools}")
        print(f"Actual tools: {agent_tool_usage_times}")
        print("---------------------")

        test_cases.append(test_case)

    return test_cases


@pytest.mark.asyncio
async def test_rag() -> None:
    test_cases = await create_rag_test_cases()

    eval_model_name = os.environ.get("EVAL_CHAT_MODEL_NAME", "vertexai:gemini-2.0-flash-lite-001")
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
        print(f"Saved raw eval results to {raw_path}")
    except Exception as exc:
        print(f"Warning: failed to persist eval results: {exc}")

    # Pass/fail summary table
    def _metric_name(m):
        return getattr(m, "__name__", None) or m.__class__.__name__

    metric_names = [_metric_name(m) for m in metrics]
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

    print("\n=== Evaluation Results Table ===")
    print(fmt_row(all_rows[0]))
    print("-+-".join("-" * w for w in col_widths))
    for row in all_rows[1:-1]:
        print(fmt_row(row))
    print("-+-".join("-" * w for w in col_widths))
    print(fmt_row(all_rows[-1]))
    print("=== End Table ===\n")


if __name__ == "__main__":
    asyncio.run(test_rag())
