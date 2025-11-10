# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
from pathlib import Path
from typing import List

import pytest
from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# Add the python directory to sys.path so we can import from eval and examples
python_path = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(python_path))

# Import setup_vector_store from examples
from examples.agents.experimental.requirement.rag import setup_vector_store

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.backend import ChatModel, ToolMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.retrieval import VectorStoreSearchTool
from eval.model import DeepEvalLLM


async def create_agent() -> RequirementAgent:
    """
    Create a RequirementAgent with RAG capabilities using VectorStoreSearchTool.
    """
    # Setup vector store using the reusable function from examples
    vector_store = await setup_vector_store()
    if vector_store is None:
        raise FileNotFoundError(
            "Failed to instantiate Vector Store. "
            "Either set POPULATE_VECTOR_DB=True to create a new one, or ensure the database file exists."
        )
    
    # Create the vector store search tool
    search_tool = VectorStoreSearchTool(vector_store=vector_store)
    
    return RequirementAgent(
        llm=ChatModel.from_name("ollama:granite3.3:2b"),
        tools=[search_tool],
        memory=UnconstrainedMemory(),
        instructions=(
            "You are a helpful assistant that answers questions about the BeeAI framework. "
            "Use the vector store search tool to find relevant information from the documentation "
            "before providing your answer. Always search for information first, then provide a "
            "comprehensive response based on what you found."
        ),
    )


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
            "What types of agents are available in BeeAI Framework?",
            "BeeAI Framework provides several agent implementations: ReAct Agent (implements the ReAct pattern for reasoning and acting), Tool Calling Agent (optimized for scenarios where tool usage is the primary focus), Custom Agent (for advanced use cases by extending BaseAgent class), and the new experimental RequirementAgent that combines LLMs, tools, and requirements in a declarative interface."
        ),
        (
            "How does the ReAct Agent work?",
            "The ReActAgent implements the ReAct (Reasoning and Acting) pattern, which structures agent behavior into a cyclical process of reasoning, action, and observation. The agent reasons about a task, takes actions using tools, observes results, and continues reasoning until reaching a conclusion."
        ),
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
    
    for i, (question, expected_output) in enumerate(test_data, 1):
        print(f"\n[{i}/{len(test_data)}] Processing question: {question}")
        # Run the agent
        print("Running agent...")
        response = await agent.run(question)
        print("Agent response received")
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
    
    return test_cases


@pytest.mark.asyncio
async def test_rag() -> None:
    print("Starting RAG evaluation...")
    # Run evaluation and get test cases
    print("Creating test cases...")
    test_cases = await create_rag_test_cases()
    print(f"Created {len(test_cases)} test cases")
    
    # RAG-specific metrics - use the same model as the agent
    model_name = "ollama:granite3.3:2b"
    print(f"\nSetting up evaluation metrics with model: {model_name}")
    deepeval_model = DeepEvalLLM.from_name(model_name)
    contextual_recall = ContextualRecallMetric(
        model=deepeval_model,
        threshold=0.7
    )
    contextual_relevancy = ContextualRelevancyMetric(
        model=deepeval_model,
        threshold=0.7
    )
    contextual_precision = AnswerRelevancyMetric(
        model=deepeval_model,
        threshold=0.7
    )
    
    # Evaluate using DeepEval
    print("\nRunning evaluation...")
    eval_results = evaluate(
        test_cases=test_cases,
        metrics=[contextual_precision, contextual_recall, contextual_relevancy]
    )
    print("\nEvaluation Results:")
    print(eval_results)
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_rag())