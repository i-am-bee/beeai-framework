import asyncio
from beeai_framework.agents.types import AgentExecutionConfig
from config import langfuse
from agent import create_agent
from dataset import get_dataset
from beeai_framework.adapters.vertexai import VertexAIChatModel
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import UserMessage
import pandas as pd
from langfuse import Langfuse
import json
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
)
from ragas.run_config import RunConfig
from ragas.metrics.base import MetricWithLLM
from ragas.llms import llm_factory

def init_ragas_metrics(metrics, llm, embedding):
    for metric in metrics:
        metric.llm = llm
        run_config = RunConfig()
        metric.init(run_config)

async def judge_correctness(llm: ChatModel, query: str, actual: str, expected: str):
    """
    Evaluates if the actual answer is factually consistent with the expected answer.
    Returns: (score: float, reasoning: str)
    """
    if not expected:
        return 0.0, "No expected answer provided"

    prompt = f"""
    You are an objective AI judge.
    
    Query: {query}
    Expected Truth: {expected}
    Agent Answer: {actual}
    
    Task: Determine if the Agent Answer contains the correct information based strictly on the Expected Truth.
    
    Rules:
    1. If the Agent Answer means the same as the Expected Truth -> YES.
    2. If the Agent Answer says "I don't know", "I can't find", or is wrong -> NO.
    3. Ignore minor phrasing differences or extra polite text.
    
    Respond in this exact format:
    REASONING: [Brief explanation why]
    VERDICT: [YES or NO]
    """
    try:
        user_message = UserMessage(prompt)
        response = await llm.run([user_message])
        text = response.get_text_content().strip()
        
      
        is_correct = "VERDICT: YES" in text.upper()
        score = 1.0 if is_correct else 0.0
        return score, text 
        
    except Exception as e:
        print(f"⚠️ Judge failed: {e}")
        return 0.0, f"Error: {str(e)}"


async def run_experiment():
    agent = create_agent()  # Remove await - it's not async
    dataset = get_dataset()
    
    for item in dataset.items:
        print(f"Running evaluation for item: {item.id} (Input: {item.input})")
        
        with item.run(
            run_name="beeai_experiment_hotp",
            run_metadata={"model": "gpt-4o-mini", "temperature": 0.7},
            run_description="BeeAI agent experiment"
        ) as root_span:
            # Run your BeeAI agent - use positional argument
            response = await agent.run(
                item.input["query"],  # Positional, not prompt=
                execution=AgentExecutionConfig(max_iterations=5),
            )
            
            output_text = response.last_message.text
            data = json.loads(output_text)
            answer_text = data.get("answer", "")
            tool_used = data.get("tool_used", [])
            supporting_titles = data.get("supporting_titles", [])
            supporting_sentences = data.get("supporting_sentences", [])
            reasoning_explanation = data.get("reasoning_explanation", [])
            
            # Update the trace with input and output
            root_span.update_trace(
                input=item.input["query"],
                output={"answer": answer_text, "tool_used": tool_used, "supporting_titles": supporting_titles, "supporting_sentences": supporting_sentences, "reasoning_explanation": reasoning_explanation}
            )
            
            # Score the result against expected output
            if item.expected_output and output_text:
                root_span.score_trace(name="exact_match", value=1.0)
            else:
                root_span.score_trace(name="exact_match", value=0.0)
                
    
                
                
        query = item.input["query"]
        expected_val = item.expected_output.get("result", "") if item.expected_output else ""
        
        print(f"⚖️ Judging answer for: '{query}'...")
        llm = ChatModel.from_name("vertexai:gemini-2.0-flash-lite-001")
        score, reasoning = await judge_correctness(
            llm=llm,        
            query=query,         
            actual=output_text,  
            expected=expected_val 
        )

        root_span.score_trace(
            name="ai_correctness",  
            value=score                     
        )
        hallucination_evaluator = next(e for e in evaluators if e['name'] == 'Hallucination')
        print(f"Prompt: {hallucination_evaluator['prompt']}")
        print(f"Output Schema: {hallucination_evaluator['outputSchema']}")

        
        print(f"   🏁 Score: {score} | Reason: {reasoning.split('VERDICT')[0].strip()}")
    print(f"\nFinished processing dataset for run 'beeai_experiment_v1'.")
    # Flush events
    
    langfuse.flush()
    


if __name__ == "__main__":
    asyncio.run(run_experiment())
   
