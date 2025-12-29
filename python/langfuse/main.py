import asyncio
from dataset import create_dataset, add_dataset_item
from experiment import run_experiment

async def main():
    await run_experiment()

if __name__ == "__main__":
    asyncio.run(main())