# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
import traceback

from beeai_framework.agents.react import ReActAgent
from beeai_framework.backend import ChatModel
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.weather import OpenMeteoTool


async def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.3:8b")
    agent = ReActAgent(llm=llm, tools=[OpenMeteoTool()], memory=UnconstrainedMemory())

    result = await agent.run("What's the current weather in London?")

    print(result.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
