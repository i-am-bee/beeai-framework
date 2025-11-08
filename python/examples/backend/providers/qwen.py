# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from beeai_framework.adapters.qwen import QwenChatModel
from beeai_framework.backend import ChatModel, ChatModelNewTokenEvent, UserMessage
from beeai_framework.emitter import EventMeta
from beeai_framework.errors import AbortError
from beeai_framework.parsers.field import ParserField
from beeai_framework.parsers.line_prefix import LinePrefixParser, LinePrefixParserNode
from beeai_framework.utils import AbortSignal


async def qwen_from_name() -> None:
    llm = ChatModel.from_name("qwen:qwen-turbo")
    user_message = UserMessage("what states are part of New England?")
    response = await llm.run([user_message])
    print(response.get_text_content())


async def qwen_sync() -> None:
    llm = QwenChatModel("qwen-plus")
    user_message = UserMessage("what is the capital of Massachusetts?")
    response = await llm.run([user_message])
    print(response.get_text_content())


async def qwen_stream() -> None:
    llm = QwenChatModel("qwen-plus")
    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.run([user_message], stream=True)
    print(response.get_text_content())


async def qwen_stream_abort() -> None:
    llm = QwenChatModel("qwen-plus")
    user_message = UserMessage("What is the smallest of the Cape Verde islands?")

    try:
        response = await llm.run([user_message], stream=True, signal=AbortSignal.timeout(0.5))

        if response is not None:
            print(response.get_text_content())
        else:
            print("No response returned.")
    except AbortError as err:
        print(f"Aborted: {err}")


async def qwen_structure() -> None:
    class TestSchema(BaseModel):
        answer: str = Field(description="your final answer")

    llm = QwenChatModel("qwen-plus")
    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.run([user_message], response_format=TestSchema)
    print(response.output_structured)


async def qwen_stream_parser() -> None:
    llm = QwenChatModel("qwen-plus")

    parser = LinePrefixParser(
        nodes={
            "test": LinePrefixParserNode(
                prefix="Prefix: ", field=ParserField.from_type(str), is_start=True, is_end=True
            )
        }
    )

    async def on_new_token(data: ChatModelNewTokenEvent, event: EventMeta) -> None:
        await parser.add(chunk=data.value.get_text_content())

    user_message = UserMessage("Produce 3 lines each starting with 'Prefix: ' followed by a sentence and a new line.")
    await llm.run([user_message], stream=True).observe(lambda emitter: emitter.on("new_token", on_new_token))
    result = await parser.end()
    print(result)


async def main() -> None:
    print("*" * 10, "qwen_from_name")
    await qwen_from_name()
    print("*" * 10, "qwen_sync")
    await qwen_sync()
    print("*" * 10, "qwen_stream")
    await qwen_stream()
    print("*" * 10, "qwen_stream_abort")
    await qwen_stream_abort()
    print("*" * 10, "qwen_structure")
    await qwen_structure()
    print("*" * 10, "qwen_stream_parser")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
