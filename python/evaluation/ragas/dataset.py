import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation.dataset import load_items
from ragas import Dataset

_data_dir = Path(__file__).parent / "data"
_data_dir.mkdir(exist_ok=True)

dataset = Dataset(name="my_evaluation", backend="local/csv", root_dir=str(_data_dir))
for i, item in enumerate(load_items(), start=1):
    dataset.append({
        "id": f"sample_{i}",
        "question": [item["question"]],
        "answer": [item["answer"]],
        "contexts": item["relevant_sentences"],
        "supporting_titles": item["supporting_titles"],
    })
dataset.save()
