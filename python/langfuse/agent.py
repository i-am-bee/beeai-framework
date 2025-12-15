import os
import tempfile
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.weather import OpenMeteoTool
from beeai_framework.tools.code import PythonTool, LocalPythonStorage
from beeai_framework.tools.tool import Tool
from dotenv import load_dotenv
from beeai_framework.adapters.gemini import GeminiChatModel

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

def create_agent() -> RequirementAgent:
    """
    Create a RequirementAgent with Wikipedia capabilities.
    """
    wiki_tool = WikipediaTool() 
    calculator_tool = create_calculator_tool()

   
    #model_name = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
    #llm = GeminiChatModel(model_name=model_name, ApiKey=os.environ.get("GEMINI_API_KEY"), allow_parallel_tool_calls=True )
    llm = ChatModel.from_name("groq:llama-3.3-70b-versatile")
    
    
    agent = RequirementAgent(
        llm=llm, 
        tools=[wiki_tool, OpenMeteoTool(), calculator_tool],
        memory=UnconstrainedMemory(),
        role="You are an expert Multi-hop Question Answering (QA) agent. Your primary role is to extract and combine information from the provided context to answer the user's question. Answer in json format only.",
        instructions=[]
           
         
    )
    return agent