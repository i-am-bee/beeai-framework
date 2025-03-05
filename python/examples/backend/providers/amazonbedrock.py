import asyncio

from traitlets import Callable

from beeai_framework.adapters.amazonbedrock.backend.chat import AmazonBedrockChatModel
from beeai_framework.backend.chat import ChatModel, ChatModelOutput
from beeai_framework.backend.message import UserMessage
from beeai_framework.cancellation import AbortSignal
from beeai_framework.emitter import EventMeta
from beeai_framework.errors import AbortError
from beeai_framework.parsers.field import ParserField
from beeai_framework.parsers.line_prefix import LinePrefixParser, LinePrefixParserNode


async def amazonbedrock_from_name() -> None:
    llm = ChatModel.from_name("amazonbedrock:meta.llama3-8b-instruct-v1:0")
    user_message = UserMessage("what states are part of New England?")
    response = await llm.create(messages=[user_message])
    print(response.get_text_content())


async def amazonbedrock_sync() -> None:
    llm = AmazonBedrockChatModel("meta.llama3-8b-instruct-v1:0")
    user_message = UserMessage("what is the capital of Massachusetts?")
    response = await llm.create(messages=[user_message])
    print(response.get_text_content())


async def amazonbedrock_stream() -> None:
    llm = AmazonBedrockChatModel("meta.llama3-8b-instruct-v1:0")
    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.create(messages=[user_message], stream=True)
    print(response.get_text_content())


async def amazonbedrock_stream_abort() -> None:
    llm = AmazonBedrockChatModel("meta.llama3-8b-instruct-v1:0")
    user_message = UserMessage("What is the smallest of the Cape Verde islands?")

    try:
        response = await llm.create(messages=[user_message], stream=True, abort_signal=AbortSignal.timeout(0.5))

        if response is not None:
            print(response.get_text_content())
        else:
            print("No response returned.")
    except AbortError as err:
        print(f"Aborted: {err}")


# TODO: See https://github.com/i-am-bee/beeai-framework/issues/491
# async def amazonbedrock_structure() -> None:
#     class TestSchema(BaseModel):
#         answer: str = Field(description="your final answer")

#     llm = AmazonBedrockChatModel("meta.llama3-8b-instruct-v1:0")
#     user_message = UserMessage("How many islands make up the country of Cape Verde?")
#     response = await llm.create_structure(schema=TestSchema, messages=[user_message])
#     print(response.object)


async def amazonbedrock_stream_parser() -> None:
    llm = AmazonBedrockChatModel("meta.llama3-8b-instruct-v1:0")

    parser = LinePrefixParser(
        nodes={
            "test": LinePrefixParserNode(
                prefix="Prefix: ", field=ParserField.from_type(str), is_start=True, is_end=True
            )
        }
    )

    async def on_new_token(value: tuple[ChatModelOutput, Callable], event: EventMeta) -> None:
        data, abort = value
        await parser.add(data.get_text_content())

    user_message = UserMessage("Produce 3 lines each starting with 'Prefix: ' followed by a sentence and a new line.")
    await llm.create(messages=[user_message], stream=True).observe(lambda emitter: emitter.on("newToken", on_new_token))
    result = await parser.end()
    print(result)


async def main() -> None:
    print("*" * 10, "amazonbedrock_from_name")
    await amazonbedrock_from_name()
    print("*" * 10, "amazonbedrock_sync")
    await amazonbedrock_sync()
    print("*" * 10, "amazonbedrock_stream")
    await amazonbedrock_stream()
    print("*" * 10, "amazonbedrock_stream_abort")
    await amazonbedrock_stream_abort()
    # TODO: Reinstate structured tests when working
    # print("*" * 10, "amazonbedrock_structure")
    # await amazonbedrock_structure()
    print("*" * 10, "amazonbedrock_stream_parser")
    await amazonbedrock_stream_parser()


if __name__ == "__main__":
    asyncio.run(main())
