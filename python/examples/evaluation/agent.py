import os
import tempfile

from dotenv import load_dotenv

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.code import LocalPythonStorage, PythonTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.weather import OpenMeteoTool

load_dotenv()

JSON_SCHEMA_STRING = (
    '{"answer": "<concise, specific answer only (e.g., \'Delhi\')>",'
    '"tool_used": [{"tool": "...", "times_used": 1}],'
    '"supporting_titles": ["<title 1>", "<title 2>"],'
    '"supporting_sentences": ["<sentence 1>", "<sentence 2>"],'
    '"reasoning_explanation": [{"step": 1, "logic": "The reasoning step"}]}'
)

AGENT_ROLE = (
    "You are an expert Multi-hop Question Answering (QA) agent. "
    "Your primary role is to extract and combine information from the provided context "
    "to answer the user's question. Answer in json format only."
)

AGENT_INSTRUCTIONS = [
    "RULES and CONSTRAINTS:",
    "1. SOURCE ADHERENCE (NO HALLUCINATION): Your final answer MUST be based ONLY on the "
    "context you retrieve from the provided tools (WikipediaTool). Do not use external knowledge.",
    "2. MULTI-HOP: You must perform multi-step reasoning or use multiple tools/retrievals "
    "if the question requires it.",
    "3. FINAL FORMAT: Your ONLY final output MUST be a single, valid JSON object adhering "
    "strictly to the required keys: answer, tool_used, supporting_titles, supporting_sentences, "
    "reasoning_explanation. The final_answer must be concise and specific (e.g., just 'Delhi', "
    "not a full sentence). Do not include any text outside the JSON block.",
    "4. THE JSON SCHEMA STRING: " + JSON_SCHEMA_STRING,
]


def create_calculator_tool() -> PythonTool:
    storage = LocalPythonStorage(
        local_working_dir=tempfile.mkdtemp("code_interpreter_source"),
        interpreter_working_dir=os.getenv("CODE_INTERPRETER_TMPDIR", "./tmp/code_interpreter_target"),
    )
    return PythonTool(
        code_interpreter_url=os.getenv("CODE_INTERPRETER_URL", "http://127.0.0.1:50081"),
        storage=storage,
    )


def create_agent(model_name: str | None = None) -> RequirementAgent:
    """
    Create a shared RequirementAgent for multi-hop QA evaluation.

    Args:
        model_name: ChatModel identifier (e.g. 'ollama:llama3.1:8b' or
                    'vertexai:gemini-2.5-flash'). Defaults to the AGENT_CHAT_MODEL_NAME
                    env var, falling back to EVAL_CHAT_MODEL_NAME, then
                    'ollama:llama3.1:8b'.
    """
    if model_name is None:
        model_name = os.environ.get(
            "AGENT_CHAT_MODEL_NAME",
            os.environ.get("EVAL_CHAT_MODEL_NAME", "ollama:llama3.1:8b"),
        )

    options: dict = {"allow_parallel_tool_calls": True}
    if model_name.startswith("vertexai:"):
        options["allow_prompt_caching"] = False

    llm = ChatModel.from_name(model_name, options)

    return RequirementAgent(
        llm=llm,
        tools=[WikipediaTool(), OpenMeteoTool(), create_calculator_tool()],
        memory=UnconstrainedMemory(),
        role=AGENT_ROLE,
        instructions=AGENT_INSTRUCTIONS,
    )