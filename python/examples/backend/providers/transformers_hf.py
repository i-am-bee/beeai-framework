import asyncio

from pydantic import BaseModel, Field

from beeai_framework.adapters.transformers.backend.chat import TransformersChatModel
from beeai_framework.backend import UserMessage
from beeai_framework.errors import AbortError
from beeai_framework.utils import AbortSignal

llm = TransformersChatModel("ibm-granite/granite-3.3-2b-instruct")


async def transformers_sync() -> None:
    user_message = UserMessage("what is the capital of Massachusetts?")
    response = await llm.run([user_message])
    print(response.get_text_content())


async def transformers_stream() -> None:
    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.run([user_message], stream=True)
    print(response.get_text_content())


async def transformers_stream_abort() -> None:
    user_message = UserMessage("What is the smallest of the Cape Verde islands?")

    try:
        response = await llm.run([user_message], stream=True, signal=AbortSignal.timeout(0.5))

        if response is not None:
            print(response.get_text_content())
        else:
            print("No response returned.")
    except AbortError as err:
        print(f"Aborted: {err}")


async def transformers_structure() -> None:
    class TestSchema(BaseModel):
        answer: str = Field(description="your final answer")

    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.run([user_message], response_format=TestSchema, stream=True)
    print(response.last_message.text)
    print(response.output_structured)


async def main() -> None:
    print("*" * 10, "transformers_sync")
    await transformers_sync()
    print("*" * 10, "transformers_stream")
    await transformers_stream()
    print("*" * 10, "transformers_stream_abort")
    await transformers_stream_abort()
    print("*" * 10, "transformers_structure")
    await transformers_structure()


if __name__ == "__main__":
    asyncio.run(main())
