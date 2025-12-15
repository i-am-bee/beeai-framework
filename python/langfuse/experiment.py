import asyncio
from beeai_framework.agents.types import AgentExecutionConfig
from config import langfuse
from agent import create_agent
from dataset import get_dataset

async def run_experiment():
    agent = await create_agent()
    dataset = get_dataset()
    
    for item in dataset.items:
        print(f"Running evaluation for item: {item.id} (Input: {item.input})")
        
        # Use the item.run() context manager with required arguments
        with item.run(
            run_name="beeai_experiment_v1",
            run_metadata={"model": "gpt-4o-mini", "temperature": 0.7},
            run_description="BeeAI agent experiment"
        ) as root_span:
            # Run your BeeAI agent
            response = await agent.run(
                prompt=item.input["query"],
                execution=AgentExecutionConfig(max_iterations=5),
            )
            
            # Update the trace with input and output
            root_span.update_trace(
                input=item.input,
                output={"result": response.result.text if response.result else None}
            )
            
            # Score the result against expected output
            if item.expected_output and response.result:
                root_span.score_trace(name="exact_match", value=1.0)
            else:
                root_span.score_trace(name="exact_match", value=0.0)
    
    print(f"\nFinished processing dataset for run 'beeai_experiment_v1'.")
    
    # Flush events
    langfuse.flush()
