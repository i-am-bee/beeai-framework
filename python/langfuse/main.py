import asyncio
from dataset import create_dataset, add_dataset_item
from experiment import run_experiment

async def main():
    # Create dataset (run once)
    create_dataset()
    add_dataset_item("What is the weather in Tokyo?", "Expected weather info")
    
    # Run experiment
    await run_experiment()

if __name__ == "__main__":
    asyncio.run(main())