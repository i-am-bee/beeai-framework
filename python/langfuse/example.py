import asyncio
from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.types import ChatModelParameters
from beeai_framework.memory import TokenMemory
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from langfuse import Evaluation
from agent import create_agent


# Run the agent
async def main():
    agent = create_agent()
    response = await agent.run(
        "What is the capital of France?",
        execution=AgentExecutionConfig(
            max_retries_per_step=3, 
            total_max_retries=10, 
            max_iterations=5
        ),
    )
    
    # RequirementAgent returns different structure
    print("Agent Response:", response)
    print("\nFull response:")
    ##print(response.model_dump())
    print(response.last_message.text)
    
    return response
 
if __name__ == "__main__":
    asyncio.run(main())