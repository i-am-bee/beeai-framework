import asyncio
import os
import sys
import traceback

from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.agents.react import ReActAgent
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.feedo import FeedoSearchTool, FeedoToolInput


async def main() -> None:
    # 1. Initialize the Feedo tool
    # Note: Requires FEEDO_USAGE_KEY environment variable. 
    # Get a free testnet key at https://feedo.ink
    print("Initializing Feedo Memory Tool...")
    feedo_tool = FeedoSearchTool()

    # 2. You can run the tool directly
    print("Testing direct tool call...")
    tool_input = FeedoToolInput(action="add", query="The project codename is 'Apollo'.", topic="project")
    result = await feedo_tool.run(tool_input)
    print(f"Direct result: {result.get_text_content()}")

    # 3. Or pass it to an Agent so it can autonomously manage memory
    print("\nInitializing ReActAgent with Feedo Tool...")
    llm = OpenAIChatModel("gpt-4o")
    agent = ReActAgent(llm=llm, tools=[feedo_tool], memory=UnconstrainedMemory())

    print("\nAsking agent to recall the project name...")
    response = await agent.run(
        {"prompt": "What is the project codename? If you don't know, search your memory."}
    )
    print(f"Agent Response: {response.result.text}")


if __name__ == "__main__":
    try:
        # We ensure a usage key is present for the example to work
        if not os.environ.get("FEEDO_USAGE_KEY"):
            print("Please set the FEEDO_USAGE_KEY environment variable.")
            print("Example: export FEEDO_USAGE_KEY='0x123...'")
            print("You can get a free testnet key at https://feedo.ink")
            sys.exit(1)
            
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
