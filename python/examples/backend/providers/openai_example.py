import asyncio
import datetime
import sys
import traceback

from pydantic import BaseModel, Field

from beeai_framework.adapters.openai import OpenAIChatModel, OpenAIEmbeddingModel
from beeai_framework.backend import (
    ChatModel,
    ChatModelNewTokenEvent,
    ChatModelParameters,
    MessageToolResultContent,
    ToolMessage,
    UserMessage,
)
from beeai_framework.emitter import EventMeta
from beeai_framework.errors import AbortError, FrameworkError
from beeai_framework.parsers.field import ParserField
from beeai_framework.parsers.line_prefix import LinePrefixParser, LinePrefixParserNode
from beeai_framework.tools.weather import OpenMeteoTool, OpenMeteoToolInput
from beeai_framework.utils import AbortSignal


async def openai_from_name() -> None:
    llm = ChatModel.from_name("openai:gpt-4.1-mini")
    user_message = UserMessage("what states are part of New England?")
    response = await llm.run([user_message])
    print(response.get_text_content())


async def openai_granite_from_name() -> None:
    llm = ChatModel.from_name("openai:gpt-4.1-mini")
    user_message = UserMessage("what states are part of New England?")
    response = await llm.run([user_message])
    print(response.get_text_content())


async def openai_sync() -> None:
    llm = OpenAIChatModel("gpt-4.1-mini")
    user_message = UserMessage("what is the capital of Massachusetts?")
    response = await llm.run([user_message])
    print(response.get_text_content())


async def openai_stream() -> None:
    llm = OpenAIChatModel("gpt-4.1-mini")
    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.run([user_message], stream=True)
    print(response.get_text_content())


async def openai_stream_abort() -> None:
    llm = OpenAIChatModel("gpt-4.1-mini")
    user_message = UserMessage("What is the smallest of the Cape Verde islands?")

    try:
        response = await llm.run([user_message], stream=True, signal=AbortSignal.timeout(0.5))

        if response is not None:
            print(response.get_text_content())
        else:
            print("No response returned.")
    except AbortError as err:
        print(f"Aborted: {err}")


async def openai_structure() -> None:
    class TestSchema(BaseModel):
        answer: str = Field(description="your final answer")

    llm = OpenAIChatModel("gpt-4.1-mini")
    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.run([user_message], response_format=TestSchema, stream=True)
    print(response.output_structured)


async def openai_stream_parser() -> None:
    llm = OpenAIChatModel("gpt-4.1-mini")

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


async def openai_tool_calling() -> None:
    llm = ChatModel.from_name("openai:gpt-4.1-mini", ChatModelParameters(stream=True, temperature=0))
    user_message = UserMessage(f"What is the current weather in Boston? Current date is {datetime.datetime.today()}.")
    weather_tool = OpenMeteoTool()
    response = await llm.run([user_message], tools=[weather_tool])
    tool_call_msg = response.get_tool_calls()[0]
    print(tool_call_msg.model_dump())
    tool_response = await weather_tool.run(OpenMeteoToolInput(location_name="Boston"))
    tool_response_msg = ToolMessage(
        MessageToolResultContent(
            result=tool_response.get_text_content(),
            tool_name=weather_tool.name,
            tool_call_id=response.get_tool_calls()[0].id,
        )
    )
    print(tool_response_msg.to_plain())
    final_response = await llm.run([user_message, *response.output, tool_response_msg], tools=[])
    print(final_response.get_text_content())


async def openai_embedding() -> None:
    embedding_llm = OpenAIEmbeddingModel()

    response = await embedding_llm.create(["Text", "to", "embed"])

    for row in response.embeddings:
        print(*row)


async def openai_file_example() -> None:
    llm = ChatModel.from_name("openai:gpt-4.1-mini")
    data_uri = "data:application/pdf;base64,JVBERi0xLjQKMSAwIG9iago8PC9UeXBlIC9DYXRhbG9nCi9QYWdlcyAyIDAgUgo+PgplbmRvYmoKMiAwIG9iago8PC9UeXBlIC9QYWdlcwovS2lkcyBbMyAwIFJdCi9Db3VudCAxCj4+CmVuZG9iagozIDAgb2JqCjw8L1R5cGUgL1BhZ2UKL1BhcmVudCAyIDAgUgovTWVkaWFCb3ggWzAgMCA1OTUgODQyXQovQ29udGVudHMgNSAwIFIKL1Jlc291cmNlcyA8PC9Qcm9jU2V0IFsvUERGIC9UZXh0XQovRm9udCA8PC9GMSA0IDAgUj4+Cj4+Cj4+CmVuZG9iago0IDAgb2JqCjw8L1R5cGUgL0ZvbnQKL1N1YnR5cGUgL1R5cGUxCi9OYW1lIC9GMQovQmFzZUZvbnQgL0hlbHZldGljYQovRW5jb2RpbmcgL01hY1JvbWFuRW5jb2RpbmcKPj4KZW5kb2JqCjUgMCBvYmoKPDwvTGVuZ3RoIDUzCj4+CnN0cmVhbQpCVAovRjEgMjAgVGYKMjIwIDQwMCBUZAooRHVtbXkgUERGKSBUagpFVAplbmRzdHJlYW0KZW5kb2JqCnhyZWYKMCA2CjAwMDAwMDAwMDAgNjU1MzUgZgowMDAwMDAwMDA5IDAwMDAwIG4KMDAwMDAwMDA2MyAwMDAwMCBuCjAwMDAwMDAxMjQgMDAwMDAgbgowMDAwMDAwMjc3IDAwMDAwIG4KMDAwMDAwMDM5MiAwMDAwMCBuCnRyYWlsZXIKPDwvU2l6ZSA2Ci9Sb290IDEgMCBSCj4+CnN0YXJ0eHJlZgo0OTUKJSVFT0YK"

    file_message = UserMessage.from_file(file_data=data_uri, format="text")
    print(file_message.to_plain())
    response = await llm.run([UserMessage("Read content of the file."), file_message])
    print(response.get_text_content())


async def main() -> None:
    await openai_file_example()
    return

    print("*" * 10, "openai_from_name")
    await openai_from_name()
    print("*" * 10, "openai_granite_from_name")
    await openai_granite_from_name()
    print("*" * 10, "openai_sync")
    await openai_sync()
    print("*" * 10, "openai_stream")
    await openai_stream()
    print("*" * 10, "openai_stream_abort")
    await openai_stream_abort()
    print("*" * 10, "openai_structure")
    await openai_structure()
    print("*" * 10, "openai_stream_parser")
    await openai_stream_parser()
    print("*" * 10, "openai_tool_calling")
    await openai_tool_calling()
    print("*" * 10, "openai_embedding")
    await openai_embedding()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
