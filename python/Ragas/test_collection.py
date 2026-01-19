import asyncio
from unittest import result
from Ragas.RagasLLM import RagasLLM
from ragas import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision, AnswerAccuracy
import sys
import os
from dotenv import load_dotenv
from ragas.metrics.collections import ExactMatch
from ragas.metrics.collections import ToolCallAccuracy

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

load_dotenv()
# Import the class we defined above (assuming it's in the same file for now)
from BeeAIClient import BeeAIClient 
from InstructorRagasLLM import InstructorRagasLLM

async def main():
    print("🚀 Initializing BeeAI Client...")
    
    # 1. Create the raw client (The "Driver")
    # This object knows how to talk to BeeAI but outputs raw text.
    #raw_client = BeeAIClient(model_name="vertexai:gemini-2.0-flash-lite-001")
    raw_client= RagasLLM.from_name(model_name="vertexai:gemini-2.0-flash-lite-001")
    ragas_judge_llm = InstructorRagasLLM.from_name(model_name="vertexai:gemini-2.0-flash-lite-001")
    print("🧠 Wrapping with Ragas llm_factory...")

    # 2. Upgrade to a Structured Output LLM (The "Brain")
    # llm_factory takes our raw client and wraps it with Instructor/LiteLLM logic
    # to enable JSON parsing required by Ragas metrics.
    
    judge_llm = llm_factory(
        provider="vertexai",
        model="gemini-2.0-flash-lite-001", # Used for logging/metadata
        client=raw_client,       # <--- Ensure this is passed correctly
        adapter="litellm"       # Explicitly specify the adapter if needed
    )
    
   
    print(f"✅ LLM Ready! Type: {type(ragas_judge_llm)}")
    # Expected: <class 'ragas.llms.instructor.InstructorLLM'>

    # 3. Initialize a Metric
    # Now ContextPrecision can ask for JSON and the factory will handle parsing
    ragas_judge_llm.is_async = True
    metric = ContextPrecision(llm=ragas_judge_llm)
    new_tool_metric = ToolCallAccuracy(llm=ragas_judge_llm)
    
    print("✅ Metric initialized successfully.")

    # --- Test Run (Optional) ---
    # To run this, you would need a SingleTurnSample
    # score = await metric.single_turn_ascore(sample)
    # print(f"Score: {score}")
    # Evaluate
    result = await metric.ascore(
        user_input="Where is the Eiffel Tower located?",
        reference="The Eiffel Tower is located in Paris.",
        retrieved_contexts=[
        "The Eiffel Tower is located in Paris.",
        "The Brandenburg Gate is located in Berlin."
    ]
    )
    
    
    
    print(f"Context Precision Score: {result.value}")

if __name__ == "__main__":
    asyncio.run(main())