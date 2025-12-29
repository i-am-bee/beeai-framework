import asyncio
from dataset import create_dataset, add_dataset_item
from experiment import run_experiment

async def main():
    # Create dataset (run once)
    create_dataset()
    test_cases = [
        ("What is the capital of France?", "Paris"),
        ("Who is the CEO of Tesla?", "Elon Musk"),
        ("Write a python print statement that prints 'Hello World'", "print('Hello World')")
    ]
    for query, expected in test_cases:
        add_dataset_item(query, expected)
   
if __name__ == "__main__":
    asyncio.run(main())