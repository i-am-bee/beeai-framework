import asyncio
import json
import re
import sys
import traceback

from beeai_framework import SystemMessage, ToolMessage, UserMessage
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AnyMessage, MessageToolResultContent
from beeai_framework.backend.types import ChatModelParameters
from beeai_framework.errors import FrameworkError
from beeai_framework.tools import ToolOutput
from beeai_framework.tools.search import DuckDuckGoSearchTool
from beeai_framework.tools.tool import AnyTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool


async def main() -> None:
    model = ChatModel.from_name("ollama:llama3.1", ChatModelParameters(temperature=0))
    tools: list[AnyTool] = [DuckDuckGoSearchTool(), OpenMeteoTool()]
    messages: list[AnyMessage] = [
        SystemMessage("You are a helpful assistant. Use tools to provide a correct answer."),
        UserMessage("What's the fastest marathon time?"),
    ]

    while True:
        response = await model.create(
            messages=messages,
            tools=tools,
        )

        tool_calls = response.get_tool_calls()

        tool_results: list[ToolMessage] = []

        for tool_call in tool_calls:
            print(f"-> running '{tool_call.tool_name}' tool with {tool_call.args}")
            tool: AnyTool = next(tool for tool in tools if tool.name == tool_call.tool_name)
            assert tool is not None
            res: ToolOutput = await tool.run(json.loads(tool_call.args))
            result = res.get_text_content()
            print(f"<- got response from '{tool_call.tool_name}'", re.sub(r"\s+", " ", result)[:90] + " (truncated)")
            tool_results.append(
                ToolMessage(
                    MessageToolResultContent(
                        result=result,
                        tool_name=tool_call.tool_name,
                        tool_call_id=tool_call.id,
                    )
                )
            )

        messages.extend(tool_results)

        answer = response.get_text_content()

        if answer:
            print(f"Agent: {answer}")
            break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
