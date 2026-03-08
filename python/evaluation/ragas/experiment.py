import sys
import warnings
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add python/ root to path for shared evaluation module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# Add this folder to path for local metric imports
sys.path.insert(0, str(Path(__file__).parent))

# Suppress multiprocess resource tracker warnings on Windows - must be before other imports
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

from evaluation.agent import create_agent
from evaluation.adapters import InstructorRagasLLM
from FactsSimilarityMetric import FactsSimilarityMetric

# Load dataset
_script_dir = Path(__file__).parent
_data_dir = _script_dir / "data"
dataset = Dataset.load(name="my_evaluation", backend="local/csv", root_dir=str(_data_dir))


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
    response = await create_agent().run(row["question"])

    output_text = response.last_message.text
    json_text = extract_json(output_text)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw output: {output_text}")
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
    reasoning_explanation = data.get("reasoning_explanation", [])

    ragas_judge_llm = InstructorRagasLLM.from_name(
        model_name=os.environ.get("EVAL_CHAT_MODEL_NAME", "vertexai:gemini-2.0-flash-lite-001")
    )
    ragas_judge_llm.is_async = True

    print("agent response:", response.last_message.text)

    ContextPrecision_result = await ContextPrecision(llm=ragas_judge_llm).ascore(
        user_input=row["question"],
        reference=row["answer"],
        retrieved_contexts=supporting_sentences
    )
    ContextRecall_result = await ContextRecall(llm=ragas_judge_llm).ascore(
        user_input=row["question"],
        reference=row["answer"],
        retrieved_contexts=supporting_sentences
    )
    reference_answer = row["answer"]
    if isinstance(reference_answer, list):
        reference_answer = reference_answer[0]
    elif isinstance(reference_answer, str) and reference_answer.startswith("["):
        import ast
        try:
            parsed = ast.literal_eval(reference_answer)
            if isinstance(parsed, list):
                reference_answer = parsed[0]
        except (ValueError, SyntaxError):
            pass
    ExactMatch_result = await ExactMatch().ascore(
        reference=reference_answer,
        response=answer_text
    )
    AnswerAccuracy_result = await AnswerAccuracy(llm=ragas_judge_llm).ascore(
        user_input=row["question"],
        response=answer_text,
        reference=row["answer"]
    )

    expected_facts = row.get("contexts", [])
    FactsSimilarity_result = await FactsSimilarityMetric(llm=ragas_judge_llm).ascore(
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

    ToolCallAccuracy_result = await ToolCallAccuracy().ascore(
        user_input=tool_messages,
        reference_tool_calls=[ToolCall(name="Wikipedia", args={}), ToolCall(name="Wikipedia", args={})]
    )

    return {
        **row,
        "ContextPrecision": ContextPrecision_result.value,
        "ContextRecall": ContextRecall_result.value,
        "ExactMatch": ExactMatch_result.value,
        "ToolCallAccuracy": ToolCallAccuracy_result.value,
        "AnswerAccuracy": AnswerAccuracy_result.value,
        "FactsSimilarity": FactsSimilarity_result.value,
    }


async def main():
    results = await my_experiment.arun(dataset)

    with open(_script_dir / "my_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("✅ Saved to my_results.pkl")

    df = results.to_pandas()
    df.to_csv(_script_dir / "my_results.csv", index=False, encoding="utf-8-sig")
    print("✅ Saved to my_results.csv")
    print(df)


if __name__ == "__main__":
    def cleanup():
        try:
            multiprocessing.util._exit_function()
        except Exception:
            pass

    atexit.register(cleanup)
    asyncio.run(main())
