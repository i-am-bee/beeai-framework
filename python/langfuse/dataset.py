from config import langfuse

def create_dataset():
    dataset_name = "beeai_experiments_hotpotQA"
    try:
        return langfuse.get_dataset(dataset_name)
    except:
        print(f"Creating new dataset: {dataset_name}")
        return langfuse.create_dataset(
            name=dataset_name,
            description="Dataset for BeeAI agent experiments",
            metadata={"relevant_sentences": [], "supporting_titles": []}
        )



def add_dataset_item(query, expected_output, relevant_sentences, supporting_titles):
    langfuse.create_dataset_item(
        dataset_name="beeai_experiments_hotpotQA",
        input={"query": query},
        expected_output={"result": expected_output},
        metadata={"relevant_sentences": relevant_sentences, "supporting_titles": supporting_titles}
    )

def get_dataset():
    return langfuse.get_dataset("beeai_experiments_hotpotQA")