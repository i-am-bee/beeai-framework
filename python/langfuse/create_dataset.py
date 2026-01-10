import asyncio
from dataset import create_dataset, add_dataset_item
from experiment import run_experiment

async def main():
    # Create dataset (run once)
    create_dataset()
    add_dataset_item("The Oberoi family is part of a hotel company that has a head office in what city?", "Delhi", [
        "The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.",
        "The Oberoi Group is a hotel company with its head office in Delhi."], ["Oberoi family","The Oberoi Group"])
   
 
if __name__ == "__main__":
    asyncio.run(main())