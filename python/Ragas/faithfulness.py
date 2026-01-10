import sys
import os
import warnings
from dotenv import load_dotenv
from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import (
    faithfulness,       
    context_recall,     
    context_precision,  
)

from RagasLLM import RagasLLM

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

load_dotenv()

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
        'reference': ["The first superbowl was held on Jan 15, 1967"]
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