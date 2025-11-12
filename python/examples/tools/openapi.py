import asyncio
import json
import os
import sys
import traceback

from aiofiles import open

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.errors import FrameworkError
from beeai_framework.tools.openapi import OpenAPITool


async def main() -> None:
    # Retrieve the schema
    current_dir = os.path.dirname(__file__)
    async with open(f"{current_dir}/assets/github_openapi.json") as file:
        content = await file.read()
        open_api_schema = json.loads(content)

        # Create a tool for each operation in the schema
        tools = OpenAPITool.from_schema(open_api_schema)
        print(f"Retrieved {len(tools)} tools")
        print("\n".join([t.name for t in tools]))

        # Create an agent
        agent = RequirementAgent(llm="ollama:granite4:micro", tools=tools)

        # Run the agent
        prompt = "How many repositories are in 'i-am-bee' org?"
        print("User:", prompt)
        response = await agent.run(prompt)
        print("Agent 🤖 : ", response.last_message.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
