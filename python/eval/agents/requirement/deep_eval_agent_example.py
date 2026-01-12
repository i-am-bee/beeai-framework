# --- Standard Library Imports ---
import asyncio
import json
import logging
import os
import pickle
import sys
import tempfile
import traceback
from pathlib import Path
from collections import Counter
from typing import Any, List

# --- Environment Setup ---
from dotenv import load_dotenv
load_dotenv()



# --- Logger Configuration ---
def setup_logger():
    logger = logging.getLogger("DeepEvalAgentExample")
    logger.setLevel(logging.INFO)
    
    # Create file handler
    log_file = Path("evaluation.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()


# --- Path Configuration (Must run before local imports) ---
# Ensure the monorepo's python package root is importable
CURRENT_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = CURRENT_DIR.parent.parent.parent # points to .../python
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

# Ensure current directory is in path for local imports like ToolUsageMetric, _utils
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# --- Third-Party Library Imports ---
import pytest
from deepeval import evaluate
from deepeval.evaluate import DisplayConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ArgumentCorrectnessMetric,
    BaseMetric,
    ContextualRecallMetric,
    ExactMatchMetric,
    FaithfulnessMetric,
    GEval,
    ToolCorrectnessMetric,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall

# --- Framework Specific Imports (beeai-framework) ---
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel, ToolMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.search.retrieval import VectorStoreSearchTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.weather import OpenMeteoTool
from beeai_framework.tools.code import PythonTool, LocalPythonStorage
from beeai_framework.adapters.ollama import OllamaChatModel
from beeai_framework.errors import FrameworkError

# --- Local Project Imports ---
from eval.deep_eval import (
    DeepEvalLLM,
    create_evaluation_table,
)
from eval._utils import (
    EvaluationRow,
    EvaluationTable,
    print_evaluation_table,
)

test_cases_num = 50

class FactsSimilarityMetric(BaseMetric):
    # Default so DeepEval's MetricData.success sees a proper boolean
    success: bool = False

    def __init__(self, model: DeepEvalLLM | None = None, threshold: float = 0.5):
        super().__init__()
        # DeepEval expects model to be a DeepEvalBaseLLM; we use our wrapper.
        self.model: DeepEvalLLM = model or DeepEvalLLM.from_name("ollama:llama3.1:8b")
        self.threshold = threshold
        # Let DeepEval use async execution path (a_measure)
        self.async_mode = True

    def _get_expected(self, test_case: LLMTestCase) -> list[str]:
        if hasattr(test_case, "expected_facts"):
            return getattr(test_case, "expected_facts")
        metadata = getattr(test_case, "additional_metadata", None) or {}
        return metadata.get("expected_facts", [])

    async def a_measure(self, test_case: LLMTestCase) -> float:
        actual_facts = getattr(test_case, "retrieval_context", [])
        expected_facts = self._get_expected(test_case)

        if not expected_facts:
            return 1.0 if not actual_facts else 0.0
        if not actual_facts:
            self.score = 0.0
            return 0.0

        prompt = (
            "Role: Expert Information Auditor\n"
            "Task: Evaluate the coverage of 'Expected Facts' within the 'Retrieved Context'.\n\n"
            f"Expected Facts (Ground Truth):\n{expected_facts}\n\n"
            f"Retrieved Context (Agent Output):\n{actual_facts}\n\n"
            "Instructions:\n"
            "1. Break down the Expected Facts into core independent claims.\n"
            "2. For each claim, check if it is supported by the Retrieved Context.\n"
            "3. Calculation: (Number of supported claims) / (Total number of expected claims).\n\n"
            "Final Score: Output ONLY the numerical score between 0.0 and 1.0."
        )

        text = await self.model.a_generate(prompt)
        
        # חילוץ מספר נקי
        import re
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(text))
        score = float(numbers[0]) if numbers else 0.0

        self.score = max(0.0, min(1.0, score))
        self.success = self.score >= self.threshold
        return self.score

    def measure(self, test_case: LLMTestCase) -> float:
        """Synchronous wrapper for environments that call measure() instead of a_measure()."""
        import asyncio

        return asyncio.run(self.a_measure(test_case))

    def is_successful(self) -> bool:
        return getattr(self, "success", False)

    @property
    def __name__(self):
        return "FactsSimilarityMetric"

class AnswerLLMJudgeMetric(BaseMetric):
    """
    Uses an LLM as a judge to compare the actual answer vs the expected answer.
    Returns a semantic similarity score between 0 and 1.
    """

    success: bool = False  # ensure MetricData.success is always a bool

    def __init__(self, model: DeepEvalLLM | None = None, threshold: float = 0.5):
        super().__init__()
        # DeepEval expects model to be a DeepEvalBaseLLM; we use our wrapper.
        self.model: DeepEvalLLM = model or DeepEvalLLM.from_name("ollama:llama3.1:8b")
        self.threshold = threshold
        self.async_mode = True  # DeepEval will call a_measure

    async def a_measure(self, test_case: LLMTestCase) -> float:
        actual = (test_case.actual_output or "").strip()
        expected = (test_case.expected_output or "").strip()

        if not expected:
            return 1.0 if not actual else 0.0

        prompt = (
            "You are an expert evaluator. Your goal is to determine if the Model Answer is semantically identical to the Expected Answer.\n\n"
            f"Question: {test_case.input}\n"
            f"Expected Answer: {expected}\n"
            f"Model Answer: {actual}\n\n"
            "Evaluation Criteria:\n"
            "1. If the answers share the same core meaning (e.g., 'Messi' vs 'Lionel Messi'), give 1.0.\n"
            "2. If the answer is partially correct but missing key info, give 0.5.\n"
            "3. If the answer is wrong or contradicts the expected, give 0.0.\n\n"
            "Instructions: Provide your reasoning in one sentence, then on a new line provide the score as: 'Score: <number>'"
        )

        text = await self.model.a_generate(prompt)
        
        # חילוץ הציון מתוך הטקסט (מטפל במקרים שהמודל חופר)
        try:
            import re
            # מחפש מספר אחרי המילה Score
            match = re.search(r"Score:\s*([\d\.]+)", text)
            if match:
                score = float(match.group(1))
            else:
                # Fallback למקרה שרק החזיר מספר
                score = float(str(text).strip())
        except:
            score = 0.0

        self.score = max(0.0, min(1.0, score))
        self.success = self.score >= self.threshold
        return self.score

    def measure(self, test_case: LLMTestCase) -> float:
        """Sync wrapper in case something calls measure() directly."""
        import asyncio
        return asyncio.run(self.a_measure(test_case))

    def is_successful(self) -> bool:
        return getattr(self, "success", False)

    @property
    def __name__(self) -> str:
        return "AnswerLLMJudgeMetric"

class ToolUsageMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.score = 0.0
        self.success = False

    def measure(self, test_case: LLMTestCase) -> float:
        # חילוץ הנתונים מה-TestCase
        expected_tools = getattr(test_case, "expected_tools", []) or \
                         (test_case.additional_metadata.get("expected_tools_detail") if test_case.additional_metadata else [])
        
        actual_tools = getattr(test_case, "tools_called", []) or \
                       (test_case.additional_metadata.get("actual_tools_detail") if test_case.additional_metadata else [])
        
        if not expected_tools:
            self.score = 1.0 if not actual_tools else 0.0
            self.success = self.score >= self.threshold
            return self.score

        matches = 0
        used_actual_indices = set()

        for expected in expected_tools:
            exp_name = expected.name
            exp_query = str(expected.input_parameters.get("query", "")).lower()
            
            for i, actual in enumerate(actual_tools):
                if i in used_actual_indices:
                    continue
                
                act_name = actual.name
                act_query = str(actual.input_parameters.get("query", "")).lower()
                
                if exp_name == act_name and exp_query in act_query:
                    matches += 1
                    used_actual_indices.add(i)
                    break
        
        self.score = matches / len(expected_tools)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "ToolUsageMetric"

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
        {
            "allow_parallel_tool_calls": True,
            # "tool_choice_support": set(), # <--- השורה הזו מונעת את הקריסה!
        },
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
        "supporting_sentences": ["<sentence 1>", "<sentence 2>"],
    }"""
    
    agent = RequirementAgent(
        llm=llm, 
        tools=[wiki_tool,OpenMeteoTool(), calculator_tool, ThinkTool()],
        memory=UnconstrainedMemory(),
        role="You are an expert Multi-hop Question Answering (QA) agent. Your primary role is to query the available data sources, extract relevant information and combine information from the provided context to answer the user's question. Answer in JSON format only.",
        instructions=[
            "RULES and CONSTRAINTS:",
            "1. SOURCE ADHERENCE (NO HALLUCINATION): Your final answer MUST be based ONLY on the context you retrieve from the provided tools. Do not use external knowledge.",
            "2. Wikipedia tool accepts short terms as the query rather than long queries, e.g. John Doe.",
            "3. MULTI-HOP: You must perform multi-step reasoning or use multiple tools/retrievals if the question requires it.",
            "4. ALWAYS RESPOND WITH JSON",
            "5. THE RESPONSE JSON SCHEMA: " + JSON_SCHEMA_STRING
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

async def create_rag_test_cases(num_rows: int = 50):
    """
    Create RAG test cases by directly invoking the agent and extracting retrieval context.
    """
    
    test_cases = []

    dataset_path = Path(__file__).parent / "evaluation_dataset_50_clean.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Load only requested number of rows (capped at 50)
    test_data = test_data[:min(num_rows, 50)]


    for i, item in enumerate(test_data):
        agent = await create_agent()
        question = item["question"]
        logger.info(f"Running agent for test case {i+1}/{len(test_data)}: {question[:50]}...")
        
        HotpotQA_expected_output = item["answer"]
        HotpotQA_context = item["relevant_sentences"]
        HotpotQA_expected_tools = {"Wikipedia": item["wiki_times"]}
        supporting_titles = item["supporting_titles"]
        HotpotQA_tools_used = []
        for name in supporting_titles:
            HotpotQA_tools_used.append(ToolCall(name="Wikipedia", input_parameters={'query': name}))

        # Run the agent
        response = await agent.run(question)
        state = response.state
        memory = state.memory.messages
        actual_output = response.last_message.text

        actual_tool_calls_count = 0
        agent_tools_list = []
        agent_supporting_sentences = []

        for msg in memory:
            msg_data = msg.to_json_safe()
            role = msg_data.get("role")
            content_list = msg_data.get("content", [])

            # 1. חילוץ ארגומנטים מה-Assistant (מטפל ב-args כ-String)
            if role == "assistant":
                for item in content_list:
                    if item.get("type") == "tool-call" and item.get("tool_name") != "final_answer":
                        actual_tool_calls_count += 1
                        tool_name = item.get("tool_name")
                        
                        # ה-BeeAI שומר את ה-args כסטרינג, אנחנו צריכים לעשות לו parse
                        raw_args = item.get("args", "{}")
                        try:
                            parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        except:
                            parsed_args = {"query": str(raw_args)}

                        agent_tools_list.append(ToolCall(
                            name=tool_name,
                            input_parameters=parsed_args
                        ))

# 2. חילוץ תוצאות מה-Tool (עבור ה-Contextual Recall ו-Facts)
            elif role == "tool":
                for item in content_list:
                    raw_result = item.get("result") or item.get("text", "")
                    
                    # ניקוי המבנה של BeeAI - חילוץ התיאור בלבד
                    fact_text = ""
                    try:
                        # אם התוצאה היא מחרוזת של JSON, נהפוך אותה לאובייקט
                        data = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
                        
                        # אם זה רשימת תוצאות (כמו בויקיפדיה), ניקח את התיאור של התוצאה הראשונה
                        if isinstance(data, list) and len(data) > 0:
                            fact_text = data[0].get('description', str(data[0]))
                        else:
                            fact_text = str(data)
                    except:
                        fact_text = str(raw_result)

                    # סינון הודעות שגיאה וטקסט קצר מדי
                    clean_fact = fact_text.strip()
                    if clean_fact and "no results" not in clean_fact.lower() and len(clean_fact) > 20:
                        if clean_fact not in agent_supporting_sentences:
                            agent_supporting_sentences.append(clean_fact[:500]) # הגבלה ל-500 תווים

        # עדכון המילון הדינמי
        from collections import Counter
        agent_tool_usage_dict = dict(Counter([tc.name for tc in agent_tools_list]))

        try:
            loaded_data = json.loads(actual_output)
            agent_response_json = loaded_data if isinstance(loaded_data, dict) else {}
        except (json.JSONDecodeError, TypeError):
            agent_response_json = {}

        agent_final_answer = (
            agent_response_json.get("answer")
            or agent_response_json.get("final_answer")
            or actual_output
        )
        supporting_sentences_from_agent = agent_response_json.get("supporting_sentences", None)
        agent_supporting_sentences = supporting_sentences_from_agent if isinstance(supporting_sentences_from_agent, list) else agent_supporting_sentences
    

                
        
        test_case = LLMTestCase(
            input=question,
            actual_output=agent_final_answer,                
            expected_output=HotpotQA_expected_output,                
            retrieval_context=agent_supporting_sentences,  
            context= HotpotQA_context,
            tools_called= agent_tools_list,
            expected_tools= HotpotQA_tools_used,
            additional_metadata={
                "expected_facts": HotpotQA_context,
                "tool_usage":  agent_tool_usage_dict,
                "expected_tool_usage": HotpotQA_expected_tools,
                "supporting_titles": supporting_titles, 
            }
            
        )

        # עדכון ה-Prints לדיבאג
        print("----- TEST CASE -----")
        print(f"Question: {question}")
        print(f"Expected answer: {HotpotQA_expected_output}")
        print(f"Actual answer: {agent_final_answer}")
        print(f"Expected tools: {HotpotQA_expected_tools}")
        print(f"Actual tools: {agent_tool_usage_dict}") # עודכן
        print(f"Expected facts: {HotpotQA_context}")
        print(f"Actual facts: {agent_supporting_sentences}")
        print(f"Expected tools detail: {HotpotQA_tools_used}")
        print(f"Actual tools detail: {agent_tools_list}") # עודכן
        print("---------------------")

        test_cases.append(test_case)

    return test_cases



@pytest.mark.asyncio
async def test_rag() -> None:
    # Run evaluation and get test cases
    global test_cases_num
    test_cases = await create_rag_test_cases(test_cases_num) #number beqtween 1 and 50
    # Use local Ollama model for evaluation by default (no env key required)
    eval_model_name = os.environ.get("EVAL_CHAT_MODEL_NAME", "ollama:llama3.1:8b")
    # Increase DeepEval per-task timeout for local models (in seconds)
    os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "1000")
    eval_model = DeepEvalLLM.from_name(eval_model_name)


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


    ######### supporting facts
    # Metric 4: Compare retrieved supporting sentences with expected facts - llm as a judge
    facts_metric = FactsSimilarityMetric(
        model=eval_model
    )    

    # RAG-specific metrics
    # Metric 5: measures how much of the truly relevant context (expected_facts / ground-truth evidence) the retrieved context covers.
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
        # Tools
        tool_usage_metric,
        # Facts / context
        facts_metric,
        #RAG
        contextual_recall_metric,
    ]

    # Evaluate using DeepEval incrementally
    pkl_path = Path(__file__).parent / "eval_results_raw.pkl"

    all_test_results = []
    for i, test_case in enumerate(test_cases):
        logger.info(f"Evaluating test case {i+1}/{len(test_cases)}...")
        try:
            # Run evaluation for a single test case
            res = evaluate(
                test_cases=[test_case], 
                metrics=metrics,
                display_config=DisplayConfig(
                    show_indicator=False, 
                    print_results=False, 
                    verbose_mode=False
                )
            )
            
            # Extract results and add to our collection
            step_results = (
                getattr(res, "results", None)
                or getattr(res, "test_results", None)
                or []
            )

            for result in step_results:
                print(f"\n--- METRIC SCORES FOR TEST CASE {i} ---")
                for metric_data in result.metrics_data:
                    # מדפיס את שם המטריקה, הציון (0.0 עד 1.0) והסיבה (אם יש)
                    status = "✅" if metric_data.success else "❌"
                    print(f"{status} {metric_data.name}: {metric_data.score:.2f}")
                    if metric_data.reason:
                        print(f"   Reason: {metric_data.reason}")
                print("---------------------------------------\n")
            all_test_results.extend(step_results)
            
                
        except Exception as eval_exc:
            logger.error(f"Error evaluating test case {i+1}: {eval_exc}")
            traceback.print_exc()

                    # Pickle the accumulated results after each test case
        finally:
            if all_test_results:
                try:
                    with open(pkl_path, "wb") as f:
                        pickle.dump(all_test_results, f)
                    logger.info(f"Progress saved to {pkl_path} (Total results: {len(all_test_results)})")
                except Exception as p_err:
                    logger.error(f"Critical: Could not write PKL file: {p_err}")

    # Build and print the evaluation results table
    table = create_evaluation_table(all_test_results, metrics)
    print_evaluation_table(table)
    
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_rag())


