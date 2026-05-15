"""Build the Ragas CSV dataset used by the experiment runner.

Run this once before executing ``experiment.py``::

    python -m examples.evaluation.ragas.dataset

It reads the shared ``dataset.json`` via ``load_items()`` and writes a
Ragas-native dataset to ``./data/datasets/my_evaluation.csv``.
"""

from pathlib import Path

from examples.evaluation.dataset import load_items
from ragas import Dataset


def build_dataset() -> Dataset:
    """Create and persist the Ragas dataset from the shared JSON items."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    dataset = Dataset(name="my_evaluation", backend="local/csv", root_dir=str(data_dir))
    for i, item in enumerate(load_items(), start=1):
        dataset.append({
            "id": f"sample_{i}",
            "question": item["question"],
            "answer": item["answer"],
            "contexts": item["relevant_sentences"],
            "supporting_titles": item["supporting_titles"],
        })
    dataset.save()
    return dataset


if __name__ == "__main__":
    build_dataset()
