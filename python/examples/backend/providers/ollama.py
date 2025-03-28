import asyncio
import json
import sys
import traceback
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from beeai_framework import SystemMessage
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.events import ChatModelNewTokenEvent
from beeai_framework.backend.message import AnyMessage, MessageToolResultContent, ToolMessage, UserMessage
from beeai_framework.cancellation import AbortSignal
from beeai_framework.emitter import EventMeta
from beeai_framework.errors import AbortError, FrameworkError
from beeai_framework.parsers.field import ParserField
from beeai_framework.parsers.line_prefix import LinePrefixParser, LinePrefixParserNode
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool


async def ollama_from_name() -> None:
    llm = ChatModel.from_name("ollama:llama3.1")
    user_message = UserMessage("what states are part of New England?")
    response = await llm.create(messages=[user_message])
    print(response.get_text_content())


async def ollama_granite_from_name() -> None:
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")
    user_message = UserMessage("what states are part of New England?")
    response = await llm.create(messages=[user_message])
    print(response.get_text_content())


async def ollama_sync() -> None:
    llm = OllamaChatModel("llama3.1")
    user_message = UserMessage("what is the capital of Massachusetts?")
    response = await llm.create(messages=[user_message])
    print(response.get_text_content())


async def ollama_stream() -> None:
    llm = OllamaChatModel("llama3.1")
    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.create(messages=[user_message], stream=True)
    print(response.get_text_content())


async def ollama_stream_abort() -> None:
    llm = OllamaChatModel("llama3.1")
    user_message = UserMessage("What is the smallest of the Cape Verde islands?")

    try:
        response = await llm.create(messages=[user_message], stream=True, abort_signal=AbortSignal.timeout(0.5))

        if response is not None:
            print(response.get_text_content())
        else:
            print("No response returned.")
    except AbortError as err:
        print(f"Aborted: {err}")


async def ollama_structure() -> None:
    class TestSchema(BaseModel):
        answer: str = Field(description="your final answer")

    llm = OllamaChatModel("llama3.1")
    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.create_structure(schema=TestSchema, messages=[user_message])
    print(response.object)


async def ollama_stream_parser() -> None:
    llm = OllamaChatModel("llama3.1")

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
    await llm.create(messages=[user_message], stream=True).observe(
        lambda emitter: emitter.on("new_token", on_new_token)
    )
    result = await parser.end()
    print(result)


async def ollama_tool_calling() -> None:
    llm = OllamaChatModel("llama3.1")
    weather_tool = OpenMeteoTool()
    messages: list[AnyMessage] = [
        SystemMessage(
            f"""You are a helpful assistant that uses tools to provide answers.
Current date is {datetime.now(tz=UTC).date()!s}
"""
        ),
        UserMessage("What is the current weather in Berlin?"),
    ]
    response = await llm.create(messages=messages, tools=[weather_tool], tool_choice="required")
    messages.extend(response.messages)
    tool_call_msg = response.get_tool_calls()[0]
    print(tool_call_msg.model_dump())
    tool_response = await weather_tool.run(json.loads(tool_call_msg.args))
    tool_response_msg = ToolMessage(
        MessageToolResultContent(
            result=tool_response.get_text_content(), tool_name=tool_call_msg.tool_name, tool_call_id=tool_call_msg.id
        )
    )
    print(tool_response_msg.to_plain())
    final_response = await llm.create(messages=[*messages, tool_response_msg], tools=[])
    print(final_response.get_text_content())


async def main() -> None:
    print("*" * 10, "ollama_from_name")
    await ollama_from_name()
    print("*" * 10, "ollama_granite_from_name")
    await ollama_granite_from_name()
    print("*" * 10, "ollama_sync")
    await ollama_sync()
    print("*" * 10, "ollama_stream")
    await ollama_stream()
    print("*" * 10, "ollama_stream_abort")
    await ollama_stream_abort()
    print("*" * 10, "ollama_structure")
    await ollama_structure()
    print("*" * 10, "ollama_stream_parser")
    await ollama_stream_parser()
    print("*" * 10, "ollama_tool_calling")
    await ollama_tool_calling()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
