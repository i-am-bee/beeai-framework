import sys
import warnings
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Suppress multiprocess resource tracker warnings on Windows - must be before other imports
# TODO: remove once multiprocess fixes ResourceTracker.__del__ on Windows.
# At interpreter shutdown, ResourceTracker.__del__ calls self._stop(), which raises
# AttributeError: '_thread.RLock' object has no attribute '_recursion_count' — a known,
# reproducible symptom of a Windows-specific teardown-ordering bug in the third-party
# `multiprocess` package's resource_tracker module (not Python's stdlib multiprocessing).
# No matching issue was found filed against uqfoundation/multiprocess at the time of
# writing; this comment documents the actual failure so it can be re-checked later.
if sys.platform == "win32":
    warnings.filterwarnings("ignore", category=ResourceWarning)
    os.environ["PYTHONWARNINGS"] = "ignore::ResourceWarning"
    try:
        import multiprocess.resource_tracker
        original_del = multiprocess.resource_tracker.ResourceTracker.__del__
        def patched_del(self):
            try:
                original_del(self)
            except AttributeError:
                pass
        multiprocess.resource_tracker.ResourceTracker.__del__ = patched_del
    except Exception:
        pass

import asyncio
import atexit
import json
import multiprocessing
import pickle
import re

from ragas import experiment, Dataset
from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from ragas.metrics.collections import AnswerAccuracy, ContextPrecision, ContextRecall, ExactMatch, ToolCallAccuracy

from examples.evaluation.agent import create_agent
from evaluation.adapters import InstructorRagasLLM
from examples.evaluation.ragas.facts_similarity_metric import FactsSimilarityMetric

_script_dir = Path(__file__).parent
_data_dir = _script_dir / "data"

_ragas_judge_llm: InstructorRagasLLM | None = None


def get_ragas_judge_llm() -> InstructorRagasLLM:
    """Lazily construct (and cache) the judge LLM on first use, instead of at import time."""
    global _ragas_judge_llm
    if _ragas_judge_llm is None:
        _ragas_judge_llm = InstructorRagasLLM.from_name(
            model_name=os.environ.get("EVAL_CHAT_MODEL_NAME", "ollama:llama3.1:8b")
        )
    return _ragas_judge_llm


def extract_json(text):
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    return text


# Define your experiment
@experiment()
async def my_experiment(row):
    try:
        response = await create_agent().run(row["question"])
    except Exception as exc:
        logger.error("Agent failed on question: %r — %s", row['question'], exc)
        return {
            **row,
            "answer": "",
            "tool_used": [],
            "supporting_titles": [],
            "supporting_sentences": [],
            "reasoning_explanation": [],
            "experiment_name": "baseline_v1",
            "error": str(exc)
        }

    output_text = response.last_message.text
    json_text = extract_json(output_text)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.error("JSON parsing error: %s\nRaw output: %s", e, output_text)
        return {
            **row,
            "answer": "",
            "tool_used": [],
            "supporting_titles": [],
            "supporting_sentences": [],
            "reasoning_explanation": [],
            "experiment_name": "baseline_v1",
            "error": str(e)
        }

    answer_text = data.get("answer", "")
    tool_used = data.get("tool_used", [])
    supporting_titles = data.get("supporting_titles", [])
    supporting_sentences = data.get("supporting_sentences", [])

    logger.info("Agent response: %s", response.last_message.text)

    judge_llm = get_ragas_judge_llm()

    context_precision_result = await ContextPrecision(llm=judge_llm).ascore(
        user_input=row["question"],
        reference=row["answer"],
        retrieved_contexts=supporting_sentences
    )
    context_recall_result = await ContextRecall(llm=judge_llm).ascore(
        user_input=row["question"],
        reference=row["answer"],
        retrieved_contexts=supporting_sentences
    )
    reference_answer = row["answer"]
    if isinstance(reference_answer, list):
        reference_answer = reference_answer[0]
    exact_match_result = await ExactMatch().ascore(
        reference=reference_answer,
        response=answer_text
    )
    answer_accuracy_result = await AnswerAccuracy(llm=judge_llm).ascore(
        user_input=row["question"],
        response=answer_text,
        reference=row["answer"]
    )

    expected_facts = row.get("contexts", [])
    facts_similarity_result = await FactsSimilarityMetric(llm=judge_llm).ascore(
        actual_facts=supporting_sentences,
        expected_facts=expected_facts
    )

    # Build tool messages for ToolCallAccuracy
    tool_messages = []
    for tool_info in tool_used:
        if isinstance(tool_info, dict):
            tool_name = tool_info.get("tool", "")
            times_used = tool_info.get("times_used", 1)
            tool_calls = [ToolCall(name=tool_name, args={}) for _ in range(times_used)]
            tool_messages.append(AIMessage(content="Called tool", tool_calls=tool_calls))

    # Derive reference tool calls from supporting_titles
    reference_tool_calls = [
        ToolCall(name="Wikipedia", args={}) for _ in supporting_titles
    ]

    tool_call_accuracy_result = await ToolCallAccuracy().ascore(
        user_input=tool_messages,
        reference_tool_calls=reference_tool_calls
    )

    return {
        **row,
        "ContextPrecision": context_precision_result.value,
        "ContextRecall": context_recall_result.value,
        "ExactMatch": exact_match_result.value,
        "ToolCallAccuracy": tool_call_accuracy_result.value,
        "AnswerAccuracy": answer_accuracy_result.value,
        "FactsSimilarity": facts_similarity_result.value,
    }


async def main():
    dataset = Dataset.load(name="my_evaluation", backend="local/csv", root_dir=str(_data_dir))

    # Pass an explicit `name` so Ragas writes a single, stable CSV at
    # data/experiments/my_results.csv (overwritten on every run). Without
    # this, Ragas generates a new random filename per run.
    results = await my_experiment.arun(dataset, name="my_results")

    experiments_dir = _data_dir / "experiments"
    results_pkl = experiments_dir / "my_results.pkl"
    with open(results_pkl, "wb") as f:
        pickle.dump(results, f)
    logger.info("Saved to %s", results_pkl)

    logger.info("Saved to %s", experiments_dir / "my_results.csv")
    logger.info("\n%s", results.to_pandas())


if __name__ == "__main__":
    def cleanup():
        try:
            multiprocessing.util._exit_function()
        except Exception:
            pass

    atexit.register(cleanup)
    asyncio.run(main())
