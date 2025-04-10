import asyncio
import json
import os
import sys
import traceback

import aiofiles

from beeai_framework.agents.react import ReActAgent
from beeai_framework.backend import ChatModel
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.openapi import OpenAPITool


async def main() -> None:
    llm = ChatModel.from_name("ollama:llama3.1")
    current_dir = os.path.dirname(__file__)
    async with aiofiles.open(os.path.join(current_dir, "assets/github_openapi.json")) as file:
        open_api_schema = json.loads(await file.read())

    agent = ReActAgent(llm=llm, tools=[OpenAPITool(open_api_schema)], memory=UnconstrainedMemory())

    response = await agent.run("How many repositories are in 'i-am-bee' org?").on(
        "update", lambda data, event: print(f"Agent ({data.update.key}) 🤖 : ", data.update.value)
    )

    print("Agent 🤖 : ", response.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
