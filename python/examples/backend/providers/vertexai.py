import asyncio

from pydantic import BaseModel, Field

from beeai_framework.adapters.vertexai.backend.chat import VertexAIChatModel
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.events import ChatModelNewTokenEvent
from beeai_framework.backend.message import UserMessage
from beeai_framework.cancellation import AbortSignal
from beeai_framework.emitter import EventMeta
from beeai_framework.errors import AbortError
from beeai_framework.parsers.field import ParserField
from beeai_framework.parsers.line_prefix import LinePrefixParser, LinePrefixParserNode


async def vertexai_from_name() -> None:
    llm = ChatModel.from_name("vertexai:gemini-2.0-flash-lite-001")
    user_message = UserMessage("what states are part of New England?")
    response = await llm.create(messages=[user_message])
    print(response.get_text_content())


async def vertexai_sync() -> None:
    llm = VertexAIChatModel("gemini-2.0-flash-lite-001")
    user_message = UserMessage("what is the capital of Massachusetts?")
    response = await llm.create(messages=[user_message])
    print(response.get_text_content())


async def vertexai_stream() -> None:
    llm = VertexAIChatModel("gemini-2.0-flash-lite-001")
    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.create(messages=[user_message], stream=True)
    print(response.get_text_content())


async def vertexai_stream_abort() -> None:
    llm = VertexAIChatModel("gemini-2.0-flash-lite-001")
    user_message = UserMessage("What is the smallest of the Cape Verde islands?")

    try:
        response = await llm.create(messages=[user_message], stream=True, abort_signal=AbortSignal.timeout(0.5))

        if response is not None:
            print(response.get_text_content())
        else:
            print("No response returned.")
    except AbortError as err:
        print(f"Aborted: {err}")


async def vertexai_structure() -> None:
    class TestSchema(BaseModel):
        answer: str = Field(description="your final answer")

    llm = VertexAIChatModel("gemini-2.0-flash-lite-001")
    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.create_structure(schema=TestSchema, messages=[user_message])
    print(response.object)


async def vertexai_stream_parser() -> None:
    llm = VertexAIChatModel("gemini-2.0-flash-lite-001")

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


async def main() -> None:
    print("*" * 10, "vertexai_from_name")
    await vertexai_from_name()
    print("*" * 10, "vertexai_sync")
    await vertexai_sync()
    print("*" * 10, "vertexai_stream")
    await vertexai_stream()
    print("*" * 10, "vertexai_stream_abort")
    await vertexai_stream_abort()
    print("*" * 10, "vertexai_structure")
    await vertexai_structure()
    print("*" * 10, "vertexai_stream_parser")
    await vertexai_stream_parser()


if __name__ == "__main__":
    asyncio.run(main())
