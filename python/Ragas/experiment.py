import sys
import warnings
import os

# Suppress multiprocess resource tracker warnings on Windows - must be before any imports
if sys.platform == "win32":
    warnings.filterwarnings("ignore", category=ResourceWarning)
    # Patch the multiprocess resource tracker to avoid the RLock error
    os.environ["PYTHONWARNINGS"] = "ignore::ResourceWarning"
    
    # Monkey-patch the problematic __del__ method
    try:
        import multiprocess.resource_tracker
        original_del = multiprocess.resource_tracker.ResourceTracker.__del__
        def patched_del(self):
            try:
                original_del(self)
            except AttributeError:
                pass  # Ignore the RLock error
        multiprocess.resource_tracker.ResourceTracker.__del__ = patched_del
    except:
        pass

import json
from ragas import experiment, Dataset
from agent import create_agent
# Load your dataset
_script_dir = os.path.dirname(os.path.abspath(__file__))
dataset = Dataset.load(name="my_evaluation", backend="local/csv", root_dir=os.path.join(_script_dir, "data"))
from InstructorRagasLLM import InstructorRagasLLM
from ragas.metrics.collections import ContextPrecision, ContextRecall, ExactMatch, ToolCallAccuracy, AnswerAccuracy
import asyncio
import re
from ragas.messages import HumanMessage, ToolCall, ToolMessage, AIMessage
from FactsSimilarityMetric import FactsSimilarityMetric
def extract_json(text):
    """Extract JSON from text, handling markdown code blocks and extra text."""
    # Try to find JSON in code blocks first
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    # Try to find raw JSON object
    json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    return text

# Define your experiment
@experiment()
async def my_experiment(row):
    # Process the input through your AI system
    response = await create_agent().run(row["question"])
    
    ##parse json
    output_text = response.last_message.text
    
    # Extract and clean JSON
    json_text = extract_json(output_text)
    
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw output: {output_text}")
        # Return default values on error
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
    
    ragas_judge_llm = InstructorRagasLLM.from_name(model_name="vertexai:gemini-2.0-flash-lite-001")
    ragas_judge_llm.is_async = True
    ContextPrecision_metric = ContextPrecision(llm=ragas_judge_llm)
    ContextRecall_metric = ContextRecall(llm=ragas_judge_llm)
    # Evaluate  
    print("agent response:", response.last_message.text)
    ContextPrecision_result = await ContextPrecision_metric.ascore(
        user_input=row["question"],
        reference=row["answer"],
        retrieved_contexts=supporting_sentences
    )
    
    ContextRecall_result = await ContextRecall_metric.ascore(
        user_input=row["question"],
        reference=row["answer"],
        retrieved_contexts=supporting_sentences
    )
    
    ExactMatch_metric = ExactMatch()
    ExactMatch_result = await ExactMatch_metric.ascore(
        reference=row["answer"],
        response=answer_text
    )
    
    AnswerAccuracy_metric = AnswerAccuracy(llm=ragas_judge_llm)
    AnswerAccuracy_result = await AnswerAccuracy_metric.ascore(
        user_input=row["question"],
        response=answer_text,
        reference=row["answer"]
    )
    
    # Prepare expected facts from contexts
    expected_facts = row.get('contexts', [])
    
    FactsSimilarityMetric_metric = FactsSimilarityMetric(llm=ragas_judge_llm)
    FactsSimilarityMetric_result = await FactsSimilarityMetric_metric.ascore(
        actual_facts=supporting_sentences ,
        expected_facts=expected_facts
    )   
    
    # Parse tool calls - user_input expects List[AIMessage with tool_calls]
    tool_messages = []
    
    for tool_info in tool_used:
        if isinstance(tool_info, dict):
            tool_name = tool_info.get('tool', '')
            times_used = tool_info.get('times_used', 1)
            # Create ToolCall for each usage
            tool_call = []
            for _ in range(times_used):
                tool_call.append(ToolCall(name=tool_name, args={}))
        
            tool_messages.append(AIMessage(content=f"Called tool", tool_calls=tool_call))
    
    
    
    ToolCallAccuracy_metric = ToolCallAccuracy()
    ToolCallAccuracy_result = await ToolCallAccuracy_metric.ascore(
        user_input=tool_messages,
        reference_tool_calls=[ToolCall(name="Wikipedia", args={}),ToolCall(name="Wikipedia", args={} )]
    )
    
    

    # Return results for metric evaluation
    return {
        **row,  # Include original data
        "ContextPrecision":  ContextPrecision_result.value,
        "ContextRecall": ContextRecall_result.value,
        "ExactMatch": ExactMatch_result.value,
        "ToolCallAccuracy": ToolCallAccuracy_result.value,
        "AnswerAccuracy": AnswerAccuracy_result.value,
        "FactsSimilarity": FactsSimilarityMetric_result.value,
    }

# Run evaluation on the dataset
async def main():
    results = await my_experiment.arun(dataset)
    df = results.to_pandas()
    df.to_csv("my_results.csv", index=False, encoding='utf-8-sig')

    print("✅ Saved to my_results.csv")
    print(df)

if __name__ == "__main__":
    import atexit
    import multiprocessing
    
    # Force cleanup of multiprocessing resources on Windows
    def cleanup():
        try:
            multiprocessing.util._exit_function()
        except:
            pass
    
    atexit.register(cleanup)
    asyncio.run(main())