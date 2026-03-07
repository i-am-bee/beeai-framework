from ragas import Dataset
import os

# Create data directory if it doesn't exist
data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(data_dir, exist_ok=True)

# Create a new dataset
dataset = Dataset(name="my_evaluation", backend="local/csv", root_dir=data_dir)

# Add a sample to the dataset
dataset.append({
    "id": "sample_1",
    'question': ["Which magazine was started first Arthur's Magazine or First for Women?"],
    'answer': ["Arthur's Magazine"],
    'contexts': [
        "Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.",
        "First for Women is a woman's magazine published by Bauer Media Group in the USA."
    ],
    "supporting_titles": [
        "Arthur's Magazine",
        "First for Women"
      ]
})

dataset.append({
    "id": "sample_2",
      "question": ["The Oberoi family is part of a hotel company that has a head office in what city?"],
      "answer": ["Delhi"],
      'contexts': [
        "The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.",
        "The Oberoi Group is a hotel company with its head office in Delhi."
      ],
      "supporting_titles": [
        "Oberoi family",
        "The Oberoi Group"
      ]
    }
)

# Save the dataset to disk
dataset.save()
