import json
import os
import sys
from pathlib import Path
import asyncio
import traceback
import tempfile
from typing import Counter, List
from beeai_framework.tools.tool import Tool
import pytest
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ExactMatchMetric,
)

from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from AnswerLLMJudgeMetric import AnswerLLMJudgeMetric
from ToolUsageMetric import ToolUsageMetric
from FactsSimilarityMetric import FactsSimilarityMetric
load_dotenv()

# Add the examples directory to sys.path to import setup_vector_store
examples_path = Path(__file__).parent.parent.parent.parent / "examples" / "agents" / "experimental" / "requirement"
sys.path.insert(0, str(examples_path))

from examples.agents.experimental.requirement.rag import setup_vector_store

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.backend import ChatModel, ToolMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.retrieval import VectorStoreSearchTool
from beeai_framework.adapters.ollama import OllamaChatModel
from beeai_framework.errors import FrameworkError
from beeai_framework.tools.search.wikipedia import (WikipediaTool)
from beeai_framework.tools.weather import OpenMeteoTool

from beeai_framework.tools.code import PythonTool, LocalPythonStorage

from eval.model import DeepEvalLLM
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric, ArgumentCorrectnessMetric
import pickle
import pandas as pd

def count_tool_usage(messages):
    tool_counter = Counter()

    for msg in messages:
        if isinstance(msg, ToolMessage):
            for item in msg.content:
                tool_name = getattr(item, "tool_name", None)
                if tool_name and tool_name != "final_answer":
                    tool_counter[tool_name] += 1

    return dict(tool_counter)

def create_calculator_tool() -> Tool:
    """
    Create a PythonTool configured for mathematical calculations.
    """
    storage = LocalPythonStorage(
        local_working_dir=tempfile.mkdtemp("code_interpreter_source"),
        # CODE_INTERPRETER_TMPDIR should point to where code interpreter stores it's files
        interpreter_working_dir=os.getenv("CODE_INTERPRETER_TMPDIR", "./tmp/code_interpreter_target"),
    )

    python_tool = PythonTool(
        code_interpreter_url=os.getenv("CODE_INTERPRETER_URL", "http://127.0.0.1:50081"),
        storage=storage,
    )
    return python_tool

async def create_agent() -> RequirementAgent:
    """
    Create a RequirementAgent with RAG and Wikipedia capabilities.
    """
    #vector_store = await setup_vector_store()
    #need it?
    vector_store = True
    if vector_store is None:
        raise FileNotFoundError(
            "Failed to instantiate Vector Store. "
            "Either set POPULATE_VECTOR_DB=True in your .env file, or ensure the database file exists."
        )
    search_tool = VectorStoreSearchTool(vector_store=vector_store)

    wiki_tool = WikipediaTool() 
    calculator_tool = create_calculator_tool()

    # Use local Ollama without relying on environment variables
    # Allow overriding the agent model; default aligns with eval model naming
    model_name = os.environ.get("AGENT_CHAT_MODEL_NAME", os.environ.get("EVAL_CHAT_MODEL_NAME", "ollama:llama3.1:8b"))
    llm = ChatModel.from_name(
        model_name,
        {"allow_parallel_tool_calls": True},
    )

    # Create RequirementAgent with multiple tools
    # tools: WikipediaTool for general knowledge, PythonTool for calculations, OpenMeteoTool for weather data

    #Format in Jason:
    #Final answer 
    #List of supporting sentences
    #explanation of reasoning for each sentence by its number
    #tool that was used
    #
    JSON_SCHEMA_STRING = """{
        "answer": "<concise, specific answer only (e.g., 'Delhi')>",
        "tool_used": [{"tool": "...", "times_used": 1}],
        "supporting_titles": ["<title 1>", "<title 2>"],
        "supporting_sentences": ["<sentence 1>", "<sentence 2>"],
        "reasoning_explanation": [{"step": 1, "logic": "The reasoning step"}]
    }"""
    
    agent = RequirementAgent(
        llm=llm, 
        tools=[wiki_tool,OpenMeteoTool(), calculator_tool],
        memory=UnconstrainedMemory(),
        role="You are an expert Multi-hop Question Answering (QA) agent. Your primary role is to extract and combine information from the provided context to answer the user's question. Answer in jason format only.",
        instructions=[
            "RULES and CONSTRAINTS:",
            "1. SOURCE ADHERENCE (NO HALLUCINATION): Your final answer MUST be based ONLY on the context you retrieve from the provided tools (VectorStoreSearchTool or WikipediaTool). Do not use external knowledge.",
            "2. MULTI-HOP: You must perform multi-step reasoning or use multiple tools/retrievals if the question requires it.",
            "3. FINAL FORMAT: Your ONLY final output MUST be a single, valid JSON object adhering strictly to the required keys: answer, tool_used, supporting_titles, supporting_sentences, reasoning_explanation. The final_answer must be concise and specific (e.g., just 'Delhi', not a full sentence). Do not include any text outside the JSON block.",
            "4. THE JSON SCHEMA STRING: " + JSON_SCHEMA_STRING
        ],

    )
    return agent

def extract_retrieval_context(messages) -> List[str]:
    """
    Extract retrieval context from tool messages in the message history.
    Looks for ToolMessage with VectorStoreSearch tool_name and extracts document descriptions.
    """
    retrieval_context = []
    
    for message in messages:
        if isinstance(message, ToolMessage) and message.content and len(message.content) > 0:
            if hasattr(message.content[0], 'tool_name') and message.content[0].tool_name == "VectorStoreSearch":
                try:
                    # Extract the tool result from the message content
                    for content_item in message.content:
                        if hasattr(content_item, 'result') and content_item.result:
                            # Parse the JSON result
                            result_data = json.loads(content_item.result) if isinstance(content_item.result, str) else content_item.result
                            
                            # Extract descriptions from each document
                            if isinstance(result_data, list):
                                for doc in result_data:
                                    if isinstance(doc, dict) and 'description' in doc:
                                        retrieval_context.append(doc['description'])
                except (json.JSONDecodeError, AttributeError, KeyError) as e:
                    # If parsing fails, skip this message
                    print(f"Warning: Failed to parse retrieval context: {e}")
                    continue
    
    return retrieval_context

async def create_rag_test_cases():
    """
    Create RAG test cases by directly invoking the agent and extracting retrieval context.
    """
    agent = await create_agent()
    
    test_cases = []

    dataset_path = Path(__file__).parent / "evaluation_dataset_2_clean.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Per-question stubbed responses (can diverge from ground truth)
    # Stubbed responses now mirror the dataset entries exactly
    # stub_map = {
    #     "Which magazine was started first Arthur's Magazine or First for Women?": {
    #         "answer": "Arthur's Magazine",
    #         "titles": ["Arthur's Magazine", "First for Women"],
    #         "sentences": [
    #             "Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.",
    #             "First for Women is a woman's magazine published by Bauer Media Group in the USA.",
    #         ],
    #         "times_used": 2,
    #     },
    #     "The Oberoi family is part of a hotel company that has a head office in what city?": {
    #         "answer": "Delhi",
    #         "titles": ["Oberoi family", "The Oberoi Group"],
    #         "sentences": [
    #             "The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.",
    #             "The Oberoi Group is a hotel company with its head office in Delhi.",
    #         ],
    #         "times_used": 2,
    #     },
    # }

    for item in test_data:
        question = item["question"]
        
        HotpotQA_expected_output = item["answer"]
        HotpotQA_context = item["relevant_sentences"]
        HotpotQA_expected_tools = {"Wikipedia": item["wiki_times"]}
        supporting_titles = item["supporting_titles"]
        HotpotQA_tools_used = []
        for name in supporting_titles:
            HotpotQA_tools_used.append(ToolCall(name="Wikipedia", input_parameters={'query': name}))

        # Run the agent
        response = await agent.run(question)
        memory = response.memory.messages
        actual_output = response.result.text
        agent_tool_usage_times = count_tool_usage(memory)

        # print(actual_output)
        
        # # Use a stubbed agent response (per question) to avoid waiting for live model calls.
        # stub = stub_map.get(question, {})
        # stub_answer = stub.get("answer", HotpotQA_expected_output)
        # stub_titles = stub.get("titles", supporting_titles)
        # stub_sentences = stub.get("sentences", HotpotQA_context)
        # stub_times = stub.get("times_used", item.get("wiki_times", 1))

        # actual_output = json.dumps({
        #     "answer": stub_answer,
        #     "tool_used": [
        #         {
        #             "tool": "Wikipedia",
        #             "times_used": stub_times,
        #             "titles": stub_titles,
        #         }
        #     ],
        #     "supporting_titles": stub_titles,
        #     "supporting_sentences": stub_sentences,
        #     "reasoning_explanation": [
        #         {
        #             "step": 1,
        #             "logic": f"Wikipedia shows evidence related to: {stub_answer}."
        #         }
        #     ],
        # })
        # agent_tool_usage_times = {"Wikipedia": stub_times}




        # Parse the agent JSON output to fill fields
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
        agent_supporting_titles = agent_response_json.get("supporting_titles", []) or agent_response_json.get(
            "wikipedia_titles_used", []
        )

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
                        ToolCall(
                            name=tool_name,
                            input_parameters={"titles": titles} if titles else {},
                        )
                    )
                    # prefer explicit times_used if provided
                    if times_used:
                        agent_tool_usage_times[tool_name] = times_used
        # If parsing failed to yield tool calls, fall back to counted usage
        if not agent_tools_used and agent_tool_usage_times:
            for tool_name, times_used in agent_tool_usage_times.items():
                agent_tools_used.append(ToolCall(name=tool_name, input_parameters={}))
                
        
        test_case = LLMTestCase(
            input=question,
            actual_output=agent_final_answer,                
            expected_output=HotpotQA_expected_output,                
            retrieval_context=agent_supporting_sentences,  
            context= HotpotQA_context,
            tools_called= agent_tools_used,
            expected_tools= HotpotQA_tools_used,
            additional_metadata={
                "expected_facts": HotpotQA_context,
                "tool_usage":  agent_tool_usage_times,
                "expected_tool_usage": HotpotQA_expected_tools,
                "supporting_titles": supporting_titles, 
            }
            
        )

        # Debug logging for each constructed test case
        print("----- TEST CASE -----")
        print(f"Question: {question}")
        print(f"Expected answer: {HotpotQA_expected_output}")
        print(f"Actual answer: {agent_final_answer}")
        print(f"Expected tools: {HotpotQA_expected_tools}")
        print(f"Actual tools: {agent_tool_usage_times}")
        print(f"Expected facts: {HotpotQA_context}")
        print(f"Actual facts: {agent_supporting_sentences}")
        print(f"Expected tools detail: {HotpotQA_tools_used}")
        print(f"Actual tools detail: {agent_tools_used}")
        print("---------------------")

        test_cases.append(test_case)

    return test_cases



@pytest.mark.asyncio
async def test_rag() -> None:
    # Run evaluation and get test cases
    test_cases = await create_rag_test_cases()
    # Use local Ollama model for evaluation by default (no env key required)
    eval_model_name = os.environ.get("EVAL_CHAT_MODEL_NAME", "ollama:llama3.1:8b")

    # Increase DeepEval per-task timeout for local models (in seconds)
    os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "60")


    eval_model = DeepEvalLLM.from_name(eval_model_name)
    # RAG-specific metrics
    contextual_relevancy = FaithfulnessMetric(
        model = eval_model,
        threshold=0.7
    )
    contextual_precision = AnswerRelevancyMetric(
        model = eval_model,
        threshold=0.7
    )
    tool_correctness_metric = ToolCorrectnessMetric(
        model=eval_model,
        include_reason=False,
    )
    ######### final answer
    # Metric 1: Ensure the final answer exactly matches the expected answer
    answer_exact_match_metric = ExactMatchMetric(threshold=1.0)

    # Metric 2: Ensure the final answer with llm as a judge
    answer_llm_judge_metric = AnswerLLMJudgeMetric(
        model=eval_model,
        threshold=0.7,
    )

    ######### tools
    # Metric 3: Compare tool usage and count vs expected tool usage and count
    tool_usage_metric = ToolUsageMetric()

    # Metric 4: Compare tool arguments
    argument_metric = ArgumentCorrectnessMetric(
        threshold=0.7,
        model = eval_model,
        include_reason=True
    )

    ######### supporting facts

    # Metric 5: Compare retrieved supporting sentences with expected facts - llm as a judge
    facts_metric = FactsSimilarityMetric(
        model=eval_model
    )    

    # Metric 6: measures how much of the truly relevant context (expected_facts / ground-truth evidence) the retrieved context covers.
    contextual_recall_metric = ContextualRecallMetric(
        model = eval_model,
        threshold=0.7
    )
    
    

    # Collect metrics to run (enable all for full table output)
    # Ordered by category:
    # Final answer metrics first, then tool metrics, then facts/context.
    metrics = [
        # Final answer
        answer_exact_match_metric,
        answer_llm_judge_metric,
        contextual_precision,
        contextual_recall_metric,
        contextual_relevancy,
        # Tools
        tool_correctness_metric,
        tool_usage_metric,
        argument_metric,
        # Facts / context
        facts_metric,
    ]

    # Evaluate using DeepEval
    eval_results = evaluate(test_cases=test_cases, metrics=metrics)
    
    # Persist raw eval results (before any table processing)
    try:
        raw_path = Path("eval_results_raw.pkl")
        with raw_path.open("wb") as f:
            pickle.dump(eval_results, f)
        print(f"Saved raw eval results to {raw_path}")
    except Exception as exc:
        print(f"Warning: failed to persist eval results: {exc}")


    # Build a pass/fail table for debugging
    def _metric_name(metric_obj):
        return getattr(metric_obj, "__name__", None) or metric_obj.__class__.__name__

    metric_names = [_metric_name(m) for m in metrics]

    # Try to get per-test results from eval_results; be robust to structure changes
    per_test_results = (
        getattr(eval_results, "results", None)
        or getattr(eval_results, "test_results", None)
        or []
    )

    # If eval_results itself is a list, use it directly
    if isinstance(eval_results, list):
        per_test_results = eval_results

    rows = []
    success_counts = Counter({name: 0 for name in metric_names})
    total_cases = len(per_test_results)

    for idx, test_res in enumerate(per_test_results):
        # Each test result should have metrics_data / metrics_results
        metrics_data = (
            getattr(test_res, "metrics_data", None)
            or getattr(test_res, "metrics_results", None)
            or []
        )
        metric_success_map = {}
        for md in metrics_data:
            md_name = (
                getattr(md, "metric_name", None)
                or getattr(md, "name", None)
                or getattr(md, "__name__", None)
                or md.__class__.__name__
            )
            md_success = getattr(md, "success", False)
            metric_success_map[md_name] = md_success

        row = [f"Test case {idx + 1}"]
        for name in metric_names:
            passed = metric_success_map.get(name, False)
            row.append("V" if passed else "X")
            if passed:
                success_counts[name] += 1
        rows.append(row)

    # Footer with success percentages per metric
    footer = ["Success %"]
    for name in metric_names:
        pct = (success_counts[name] / total_cases * 100) if total_cases else 0
        footer.append(f"{pct:.0f}%")

    # Pretty-print the table
    all_rows = [ ["Test case"] + metric_names ] + rows + [footer]
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*all_rows)]

    def fmt_row(row):
        return " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))

    print("\n=== Evaluation Results Table ===")
    print(fmt_row(all_rows[0]))
    print("-+-".join("-" * w for w in col_widths))
    for row in all_rows[1:-1]:
        print(fmt_row(row))
    print("-+-".join("-" * w for w in col_widths))
    print(fmt_row(all_rows[-1]))
    print("=== End Table ===\n")
    
    
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_rag())


