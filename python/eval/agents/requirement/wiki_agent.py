import json
import os
import sys
from pathlib import Path
import asyncio
import traceback
import tempfile
from typing import List
from beeai_framework.tools.tool import Tool
import pytest
from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
load_dotenv()

# Add the examples directory to sys.path to import setup_vector_store
examples_path = Path(__file__).parent.parent.parent.parent / "examples" / "agents" / "experimental" / "requirement"
sys.path.insert(0, str(examples_path))

from examples.agents.experimental.requirement.rag import setup_vector_store

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.backend import ChatModel, ToolMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.retrieval import VectorStoreSearchTool
from beeai_framework.adapters.gemini import GeminiChatModel
from beeai_framework.errors import FrameworkError
from beeai_framework.tools.search.wikipedia import (WikipediaTool)
from beeai_framework.adapters.gemini import GeminiChatModel
from beeai_framework.tools.weather import OpenMeteoTool

from beeai_framework.tools.code import PythonTool, LocalPythonStorage

from eval.model import DeepEvalLLM


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

    model_name = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
    llm = GeminiChatModel(model_name=model_name, ApiKey=os.environ.get("GEMINI_API_KEY"), allow_parallel_tool_calls=True )

    # Create RequirementAgent with multiple tools
    # tools: WikipediaTool for general knowledge, PythonTool for calculations, OpenMeteoTool for weather data

    #Format in Jason:
    #Final answer 
    #List of supporting sentences
    #explanation of reasoning for each sentence by its number
    #tool that was used
    #
    JSON_SCHEMA_STRING = """{"final_answer": "...","tool_used": "...","supporting_sentences": ["<sentence 1>", "<sentence 2>"],"reasoning_explanation": [{"step": 1, "logic": "The reasoning step"}]}"""
    agent = RequirementAgent(
        llm=llm, 
        tools=[wiki_tool,OpenMeteoTool(), calculator_tool],
        memory=UnconstrainedMemory(),
        role="You are an expert Multi-hop Question Answering (QA) agent. Your primary role is to extract and combine information from the provided context to answer the user's question. Answer in jason format only.",
        instructions=[
            "RULES and CONSTRAINTS:",
            "1. SOURCE ADHERENCE (NO HALLUCINATION): Your final answer MUST be based ONLY on the context you retrieve from the provided tools (VectorStoreSearchTool or WikipediaTool). Do not use external knowledge.",
            "2. MULTI-HOP: You must perform multi-step reasoning or use multiple tools/retrievals if the question requires it.",
            "3. FINAL FORMAT: Your ONLY final output MUST be a single, valid JSON object adhering strictly to the required keys. Do not include any text outside the JSON block.",
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
    
    # Define test questions and expected outputs
    test_data = [
        (
            "Which magazine was started first Arthur's Magazine or First for Women?",
            "Arthur's Magazine"
        ),
        (
            "The Oberoi family is part of a hotel company that has a head office in what city?",
            "New Delhi"
        )
        # (
        #     "What tools can be used with BeeAI agents?",
        #     "BeeAI agents can use various tools including Search tools (DuckDuckGoSearchTool), Weather tools (OpenMeteoTool), Knowledge tools (LangChainWikipediaTool), and many more available in the beeai_framework.tools module. Tools enhance the agent's capabilities by allowing interaction with external systems."
        # ),
        # (
        #     "What memory types are available for agents?",
        #     "Several memory types are available for different use cases: UnconstrainedMemory for unlimited storage, SlidingMemory for keeping only the most recent messages, TokenMemory for managing token limits, and SummarizeMemory for summarizing previous conversations."
        # ),
        # (
        #     "How can I customize agent behavior in BeeAI Framework?",
        #     "You can customize agent behavior in five ways: 1) Setting execution policy to control retries, timeouts, and iteration limits, 2) Overriding prompt templates including system prompts, 3) Adding tools to enhance capabilities, 4) Configuring memory for context management, and 5) Event observation to monitor execution and implement custom logging."
        # )
    ]
    
    for question, expected_output in test_data:
        # Run the agent
        response = await agent.run(question)
        
        #actual_output = response.result.text + extract_tool_usage_and_facts_trace(response.memory.messages)
        actual_output = response.result.text

        # Extract retrieval context from message history
        retrieval_context = extract_retrieval_context(response.memory.messages)
        
        # Create test case
        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context
        )
        test_cases.append(test_case)

        #TODO trajectory check
    
    return test_cases


@pytest.mark.asyncio
async def test_rag() -> None:
    # Run evaluation and get test cases
    test_cases = await create_rag_test_cases()
     # Get the evaluation model name from the environment, with a safe default
    eval_model_name = os.environ.get("EVAL_CHAT_MODEL_NAME", "google:gemini-2.5-flash")

    eval_model = DeepEvalLLM.from_name(eval_model_name)
    # RAG-specific metrics
    contextual_recall = ContextualRecallMetric(
        model = eval_model,#DeepEvalLLM.from_name(os.environ["EVAL_CHAT_MODEL_NAME"]),
        threshold=0.7
    )
    contextual_relevancy = ContextualRelevancyMetric(
        model = eval_model,#DeepEvalLLM.from_name(os.environ["EVAL_CHAT_MODEL_NAME"]),
        threshold=0.7
    )
    contextual_precision = AnswerRelevancyMetric(
        model = eval_model,#DeepEvalLLM.from_name(os.environ["EVAL_CHAT_MODEL_NAME"]),
        threshold=0.7
    )
    
    # Evaluate using DeepEval
    eval_results = evaluate(
        test_cases=test_cases,
        metrics=[contextual_precision, contextual_recall, contextual_relevancy]
    )
    print(eval_results)
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_rag())