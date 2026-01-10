import os
import tempfile

from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.weather import OpenMeteoTool
from beeai_framework.tools.code import PythonTool, LocalPythonStorage
from beeai_framework.tools.tool import Tool
from dotenv import load_dotenv
from beeai_framework.adapters.gemini import GeminiChatModel
from beeai_framework.agents.react import ReActAgent
from beeai_framework.adapters.vertexai import VertexAIChatModel
from beeai_framework.agents.requirement import RequirementAgent


load_dotenv()

def create_calculator_tool() -> Tool:
    """
    Create a PythonTool configured for mathematical calculations.
    """
    storage = LocalPythonStorage(
        local_working_dir=tempfile.mkdtemp("code_interpreter_source"),
        interpreter_working_dir=os.getenv("CODE_INTERPRETER_TMPDIR", "./tmp/code_interpreter_target"),
    )

    python_tool = PythonTool(
        code_interpreter_url=os.getenv("CODE_INTERPRETER_URL", "http://127.0.0.1:50081"),
        storage=storage,
    )
    return python_tool

def create_agent() ->  RequirementAgent:
    """
    Create a RequirementAgent with Wikipedia capabilities.
    """
    wiki_tool = WikipediaTool() 
    calculator_tool = create_calculator_tool()

   
    #model_name = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
    #llm = GeminiChatModel(model_name=model_name, ApiKey=os.environ.get("GEMINI_API_KEY"), allow_parallel_tool_calls=True )
    llm = VertexAIChatModel(model_id="gemini-2.5-flash",allow_prompt_caching=False)
    
    JSON_SCHEMA_STRING = """{"answer": "<concise, specific answer only (e.g., 'Delhi')>","tool_used": [{"tool": "...", "times_used": 1}],"supporting_titles": ["<title 1>", "<title 2>"],"supporting_sentences": ["<sentence 1>", "<sentence 2>"],"reasoning_explanation": [{"step": 1, "logic": "The reasoning step"}]}"""
    
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