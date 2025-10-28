import asyncio
import json
import sys
import traceback
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from beeai_framework.adapters.langchain.backend.chat import LangChainChatModel
from beeai_framework.backend import (
    AnyMessage,
    ChatModelNewTokenEvent,
    MessageToolResultContent,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from beeai_framework.emitter import EventMeta
from beeai_framework.errors import AbortError, FrameworkError
from beeai_framework.parsers.field import ParserField
from beeai_framework.parsers.line_prefix import LinePrefixParser, LinePrefixParserNode
from beeai_framework.tools.weather import OpenMeteoTool
from beeai_framework.utils import AbortSignal

# prevent import error for langchain_ollama (only needed in this context)
cur_dir = sys.path.pop(0)
while cur_dir in sys.path:
    sys.path.remove(cur_dir)

from langchain_ollama.chat_models import ChatOllama as LangChainOllamaChat  # noqa: E402


async def langchain_ollama_from_name() -> None:
    langchain_llm = LangChainOllamaChat(model="granite4:micro")
    llm = LangChainChatModel(langchain_llm)
    user_message = UserMessage("what states are part of New England?")
    response = await llm.run([user_message])
    print(response.get_text_content())


async def langchain_ollama_granite_from_name() -> None:
    langchain_llm = LangChainOllamaChat(model="granite4:micro")
    llm = LangChainChatModel(langchain_llm)
    user_message = UserMessage("what states are part of New England?")
    response = await llm.run([user_message])
    print(response.get_text_content())


async def langchain_ollama_sync() -> None:
    langchain_llm = LangChainOllamaChat(model="granite4:micro")
    llm = LangChainChatModel(langchain_llm)
    user_message = UserMessage("what is the capital of Massachusetts?")
    response = await llm.run([user_message])
    print(response.get_text_content())


async def langchain_ollama_stream() -> None:
    langchain_llm = LangChainOllamaChat(model="granite4:micro")
    llm = LangChainChatModel(langchain_llm)
    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.run([user_message], stream=True)
    print(response.get_text_content())


async def langchain_ollama_stream_abort() -> None:
    langchain_llm = LangChainOllamaChat(model="granite4:micro")
    llm = LangChainChatModel(langchain_llm)
    user_message = UserMessage("What is the smallest of the Cape Verde islands?")

    try:
        response = await llm.run([user_message], stream=True, signal=AbortSignal.timeout(0.5))

        if response is not None:
            print(response.get_text_content())
        else:
            print("No response returned.")
    except AbortError as err:
        print(f"Aborted: {err}")


async def langchain_ollama_structure() -> None:
    class TestSchema(BaseModel):
        answer: str = Field(description="your final answer")

    langchain_llm = LangChainOllamaChat(model="granite4:micro")
    llm = LangChainChatModel(langchain_llm)
    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.run([user_message], response_format=TestSchema)
    print(response.output_structured)


async def langchain_ollama_stream_parser() -> None:
    langchain_llm = LangChainOllamaChat(model="granite4:micro")
    llm = LangChainChatModel(langchain_llm)

    parser = LinePrefixParser(
        nodes={
            "test": LinePrefixParserNode(
                prefix="Prefix: ", field=ParserField.from_type(str), is_start=True, is_end=True
            )
        }
    )

    async def on_new_token(data: ChatModelNewTokenEvent, event: EventMeta) -> None:
        await parser.add(data.value.get_text_content())

    user_message = UserMessage("Produce 3 lines each starting with 'Prefix: ' followed by a sentence and a new line.")
    await llm.run([user_message], stream=True).observe(lambda emitter: emitter.on("new_token", on_new_token))
    result = await parser.end()
    print(result)


async def langchain_ollama_tool_calling() -> None:
    langchain_llm = LangChainOllamaChat(model="granite4:micro")
    llm = LangChainChatModel(langchain_llm)
    llm.parameters.stream = True
    weather_tool = OpenMeteoTool()
    messages: list[AnyMessage] = [
        SystemMessage(
            f"""You are a helpful assistant that uses tools to provide answers.
Current date is {datetime.now(tz=UTC).date()!s}
"""
        ),
        UserMessage("What is the current weather in Berlin?"),
    ]
    response = await llm.run(messages, tools=[weather_tool], tool_choice="required")
    messages.extend(response.output)
    tool_call_msg = response.get_tool_calls()[0]
    print(tool_call_msg.model_dump())
    tool_response = await weather_tool.run(json.loads(tool_call_msg.args))
    tool_response_msg = ToolMessage(
        MessageToolResultContent(
            result=tool_response.get_text_content(), tool_name=tool_call_msg.tool_name, tool_call_id=tool_call_msg.id
        )
    )
    print(tool_response_msg.to_plain())
    final_response = await llm.run([*messages, tool_response_msg], tools=[])
    print(final_response.get_text_content())


async def main() -> None:
    print("*" * 10, "langchain_ollama_from_name")
    await langchain_ollama_from_name()
    print("*" * 10, "langchain_ollama_granite_from_name")
    await langchain_ollama_granite_from_name()
    print("*" * 10, "langchain_ollama_sync")
    await langchain_ollama_sync()
    print("*" * 10, "langchain_ollama_stream")
    await langchain_ollama_stream()
    print("*" * 10, "langchain_ollama_stream_abort")
    await langchain_ollama_stream_abort()
    print("*" * 10, "langchain_ollama_structure")
    await langchain_ollama_structure()
    print("*" * 10, "langchain_ollama_stream_parser")
    await langchain_ollama_stream_parser()
    print("*" * 10, "langchain_ollama_tool_calling")
    await langchain_ollama_tool_calling()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
