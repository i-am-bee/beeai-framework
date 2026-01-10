from ragas import Dataset

# Create a new dataset
dataset = Dataset(name="my_evaluation", backend="local/csv", root_dir="./data")

# Add a sample to the dataset
dataset.append({
    "id": "sample_1",
    'question': ["When was the first super bowl?"],
    'answer': ["The first superbowl was held on Jan 15, 1967"],
    'contexts': [[
        "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
    ]],
    'reference': ["The first superbowl was held on Jan 15, 1967"]
    
})