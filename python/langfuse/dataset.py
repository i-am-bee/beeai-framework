from config import langfuse

def create_dataset():
    langfuse.create_dataset(
        name="beeai_experiments",
        description="Dataset for BeeAI agent experiments"
    )

def add_dataset_item(query, expected_output):
    langfuse.create_dataset_item(
        dataset_name="beeai_experiments",
        input={"query": query},
        expected_output={"result": expected_output}
    )

def get_dataset():
    return langfuse.get_dataset("beeai_experiments")