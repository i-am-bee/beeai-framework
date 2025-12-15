import asyncio
from beeai_framework.agents.types import AgentExecutionConfig
from config import langfuse
from agent import create_agent
from dataset import get_dataset

async def run_experiment():
    agent = create_agent()  # Remove await - it's not async
    dataset = get_dataset()
    
    for item in dataset.items:
        print(f"Running evaluation for item: {item.id} (Input: {item.input})")
        
        with item.run(
            run_name="beeai_experiment_v1",
            run_metadata={"model": "gpt-4o-mini", "temperature": 0.7},
            run_description="BeeAI agent experiment"
        ) as root_span:
            # Run your BeeAI agent - use positional argument
            response = await agent.run(
                item.input["query"],  # Positional, not prompt=
                execution=AgentExecutionConfig(max_iterations=5),
            )
            
            # RequirementAgent uses .final_answer, not .result.text
            output_text = response.final_answer if hasattr(response, 'final_answer') else str(response)
            
            # Update the trace with input and output
            root_span.update_trace(
                input=item.input,
                output={"result": output_text}
            )
            
            # Score the result against expected output
            if item.expected_output and output_text:
                root_span.score_trace(name="exact_match", value=1.0)
            else:
                root_span.score_trace(name="exact_match", value=0.0)
    
    print(f"\nFinished processing dataset for run 'beeai_experiment_v1'.")
    
    # Flush events
    langfuse.flush()
    
if __name__ == "__main__":
    asyncio.run(run_experiment())