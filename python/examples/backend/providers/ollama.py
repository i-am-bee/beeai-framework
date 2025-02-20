import asyncio

from pydantic import BaseModel, Field
from pyventus import EventLinker

from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.backend.chat import ChatModel, ChatModelOutput
from beeai_framework.backend.message import UserMessage
from beeai_framework.cancellation import AbortSignal
from beeai_framework.parsers.line_prefix import LinePrefixParser, Prefix


async def ollama_from_name() -> None:
    llm = ChatModel.from_name("ollama:llama3.1")
    user_message = UserMessage("what states are part of New England?")
    response = await llm.create({"messages": [user_message]})
    print(response.get_text_content())


async def ollama_granite_from_name() -> None:
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")
    user_message = UserMessage("what states are part of New England?")
    response = await llm.create({"messages": [user_message]})
    print(response.get_text_content())


async def ollama_sync() -> None:
    llm = OllamaChatModel("llama3.1")
    user_message = UserMessage("what is the capital of Massachusetts?")
    response = await llm.create({"messages": [user_message]})
    print(response.get_text_content())


async def ollama_stream() -> None:
    llm = OllamaChatModel("llama3.1")
    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.create({"messages": [user_message], "stream": True})
    print(response.get_text_content())


async def ollama_stream_abort() -> None:
    llm = OllamaChatModel("llama3.1")
    user_message = UserMessage("What is the smallest of the Cape Verde islands?")
    response = await llm.create({"messages": [user_message], "stream": True, "abort_signal": AbortSignal.timeout(0.5)})

    if response is not None:
        print(response.get_text_content())
    else:
        print("No response returned.")


async def ollama_structure() -> None:
    class TestSchema(BaseModel):
        answer: str = Field(description="your final answer")

    llm = OllamaChatModel("llama3.1")
    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.create_structure(
        {
            "schema": TestSchema,
            "messages": [user_message],
        }
    )
    print(response.object)


async def ollama_stream_parser() -> None:
    llm = OllamaChatModel("llama3.1")

    parser = LinePrefixParser(prefixes=[Prefix(name="test", line_prefix="Prefix: ")])

    @EventLinker.on("newToken")
    async def listener(data: dict[str, ChatModelOutput]) -> None:
        output: ChatModelOutput = data["value"]
        for result in parser.feed(output.get_text_content()):
            if result is not None:
                print(result.prefix.name, result.content)

    user_message = UserMessage("Produce 3 lines each starting with 'Prefix: ' followed by a sentence and a new line.")
    await llm.create({"messages": [user_message], "stream": True})

    # Pick up any remaining lines in parser buffer
    for result in parser.finalize():
        if result is not None:
            print(result.prefix.name, result.content)


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


if __name__ == "__main__":
    asyncio.run(main())
