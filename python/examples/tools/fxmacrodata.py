import asyncio
import sys
import traceback

from beeai_framework.agents.react import ReActAgent
from beeai_framework.backend import ChatModel
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.fxmacrodata import FXMacroDataTool


async def main() -> None:
    llm = ChatModel.from_name("ollama:llama3.1")
    # USD data works without a key. Set FXMACRODATA_API_KEY for other currencies.
    agent = ReActAgent(llm=llm, tools=[FXMacroDataTool()], memory=UnconstrainedMemory())

    result = await agent.run("What was the latest US CPI inflation print, and when is the next release?")

    print(result.last_message.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
