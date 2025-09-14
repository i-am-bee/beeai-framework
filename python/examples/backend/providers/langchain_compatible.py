import asyncio
import json
import sys
import traceback

from beeai_framework.adapters.langchain.backend.chat import LangChainChatModel
from beeai_framework.backend import AnyMessage, MessageToolResultContent, ToolMessage, UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.tools.weather import OpenMeteoTool

# prevent import error for Ollama
cur_dir = sys.path.pop(0)
while cur_dir in sys.path:
    sys.path.remove(cur_dir)

from langchain_ollama.chat_models import ChatOllama  # noqa: E402


async def basic() -> None:
    langchain_llm = ChatOllama(model="llama3.1:8b")

    llm = LangChainChatModel(langchain_llm)
    user_message = UserMessage("Hello!")
    response = await llm.create(messages=[user_message])

    print(response.get_text_content())
    print(response.finish_reason)
    print(response.usage)
    print(response.cost)
    print(response.messages)


async def tool_calling() -> None:
    langchain_llm = ChatOllama(model="llama3.1:8b")

    llm = LangChainChatModel(langchain_llm)
    tool = OpenMeteoTool()

    messages: list[AnyMessage] = [UserMessage("What's the current weather in London? Use a tool.")]
    response = await llm.create(messages=messages, tools=[tool], stream=True)
    print(response.get_tool_calls())
    tool_call = response.get_tool_calls()[0]
    assert tool_call.tool_name == tool.name
    tool_response = await tool.run(json.loads(tool_call.args))
    messages.extend(response.messages)
    messages.append(
        ToolMessage(
            MessageToolResultContent(
                tool_name=tool.name,
                result=tool_response.get_text_content(),
                tool_call_id=tool_call.id,
            )
        )
    )
    response = await llm.create(messages=messages)
    print(response.get_text_content())
    messages.extend(response.messages)
    messages.append(UserMessage("Thank you!"))
    response = await llm.create(messages=messages)
    print(response.get_text_content())


if __name__ == "__main__":
    try:
        asyncio.run(tool_calling())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
