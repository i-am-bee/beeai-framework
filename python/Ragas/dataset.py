import json
from pathlib import Path
from ragas import Dataset

json_path = Path(__file__).parent.parent / "eval/agents/requirement/evaluation_dataset_2_clean.json"
with open(json_path, encoding="utf-8") as f:
    items = json.load(f)

dataset = Dataset(name="my_evaluation", backend="local/csv", root_dir=...)
for item in items:
    dataset.append({
        "id": item["id"],
        "question": [item["question"]],
        "answer": [item["answer"]],
        "contexts": item["relevant_sentences"],
        "supporting_titles": item["supporting_titles"],
    })
dataset.save()