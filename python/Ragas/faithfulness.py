import sys
import os
from unittest import result
import warnings
from dotenv import load_dotenv
from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import (
    faithfulness,       
    context_recall,     
    context_precision
)
from Metrics import ToolUsageMetric, FactsSimilarityMetric
import asyncio
from RagasLLM import RagasLLM
from ragas.llms import llm_factory

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

load_dotenv()

from ragas.llms import llm_factory

async def ContextPrecision_with_ground_truth(llm: RagasLLM):
    # Create metric
    # Evaluate
    result = await scorer.ascore(
        user_input="Where is the Eiffel Tower located?",
        reference="The Eiffel Tower is located in Paris.",
        retrieved_contexts=[
        "The Eiffel Tower is located in Paris.",
        "The Brandenburg Gate is located in Berlin."
    ]
    )
    print(f"Context Precision Score: {result.value}")

def main():
    print("Initializing Adapter...")
    
    llm = RagasLLM.from_name(model_name="vertexai:gemini-2.0-flash-lite-001")
              
    print("Preparing Data...")
    data = {
        'question': ["When was the first super bowl?"],
        'answer': ["The first superbowl was held on Jan 15, 1967"],
        'contexts': [[
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
        ]],
        'reference': ["The first superbowl was held on Jan 15, 1967"],
        'ground_truth': ["The first superbowl was held on Jan 15, 1967"],
        'tools_called': ["wikipedia"],
        'expected_tools': ["wikipedia"]
           
    }
    dataset = Dataset.from_dict(data)

    print("Running Evaluation...")

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_recall, context_precision],
        llm=llm
    )
    df = results.to_pandas()
    print(df)
    
    print(f"\n Faithfulness Score: {results['faithfulness']}")
    print(f"Context Recall Score: {results['context_recall']}")
    print(f"Context Precision Score: {results['context_precision']}")
    
   

    
    
    
if __name__ == "__main__":
    main()