# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from beeai_framework.adapters.runcrate import RuncrateChatModel
from beeai_framework.backend import ChatModel, UserMessage


async def runcrate_from_name() -> None:
    llm = ChatModel.from_name("runcrate:deepseek-ai/DeepSeek-V4.1-Flash")
    user_message = UserMessage("what states are part of New England?")
    response = await llm.run([user_message])
    print(response.get_text_content())


async def runcrate_sync() -> None:
    llm = RuncrateChatModel("deepseek-ai/DeepSeek-V4.1-Flash")
    user_message = UserMessage("what is the capital of Massachusetts?")
    response = await llm.run([user_message])
    print(response.get_text_content())


async def runcrate_stream() -> None:
    llm = RuncrateChatModel("deepseek-ai/DeepSeek-V4.1-Flash")
    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.run([user_message], stream=True)
    print(response.get_text_content())


async def runcrate_structure() -> None:
    class TestSchema(BaseModel):
        answer: str = Field(description="your final answer")

    llm = RuncrateChatModel("deepseek-ai/DeepSeek-V4.1-Flash")
    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.run([user_message], response_format=TestSchema)
    print(response.output_structured)


async def main() -> None:
    print("*" * 10, "runcrate_from_name")
    await runcrate_from_name()
    print("*" * 10, "runcrate_sync")
    await runcrate_sync()
    print("*" * 10, "runcrate_stream")
    await runcrate_stream()
    print("*" * 10, "runcrate_structure")
    await runcrate_structure()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
