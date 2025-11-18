import asyncio
import os
import sys
import traceback
from typing import Any

import aiofiles
import yaml
from dotenv import load_dotenv

from beeai_framework.agents.tool_calling import ToolCallingAgent
from beeai_framework.backend import ChatModel
from beeai_framework.emitter import EventMeta
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.openapi import AfterFetchEvent, BeforeFetchEvent, OpenAPITool


async def main() -> None:
    llm = ChatModel.from_name("ollama:llama3.1")
    current_dir = os.path.dirname(__file__)
    async with aiofiles.open(os.path.join(current_dir, "assets/github_openapi.json")) as file:
        open_api_schema = yaml.safe_load(await file.read())

    api_tool = OpenAPITool(open_api_schema)

    def print_fetch_event(data: Any, event: EventMeta) -> None:
        if isinstance(data, BeforeFetchEvent):
            print(f"Agent ({event.name}) 🤖 : ", data.input)
        elif isinstance(data, AfterFetchEvent):
            print(f"Agent ({event.name}) 🤖 : ", data.data)

    api_tool.emitter.on("*", print_fetch_event)

    agent = ToolCallingAgent(llm=llm, tools=[api_tool], memory=UnconstrainedMemory())

    response = await agent.run("How many repositories are in 'i-am-bee' org?")

    print("Agent 🤖 : ", response.result.text)


if __name__ == "__main__":
    try:
        load_dotenv()
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
