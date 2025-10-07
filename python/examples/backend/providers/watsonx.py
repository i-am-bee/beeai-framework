import asyncio
import datetime
import json
import sys
import traceback

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from beeai_framework.adapters.watsonx import WatsonxChatModel
from beeai_framework.adapters.watsonx.backend.embedding import WatsonxEmbeddingModel
from beeai_framework.backend import ChatModel, MessageToolResultContent, ToolMessage, UserMessage
from beeai_framework.errors import AbortError, FrameworkError
from beeai_framework.tools.weather import OpenMeteoTool
from beeai_framework.utils import AbortSignal

# Load environment variables
load_dotenv()

# Setting can be passed here during initiation or pre-configured via environment variables
llm = WatsonxChatModel(
    "ibm/granite-3-8b-instruct",
    # settings={
    #     "project_id": "WATSONX_PROJECT_ID",
    #     "api_key": "WATSONX_API_KEY",
    #     "base_url": "WATSONX_API_URL",
    # },
)


async def watsonx_from_name() -> None:
    watsonx_llm = ChatModel.from_name(
        "watsonx:ibm/granite-3-8b-instruct",
        # {
        #     "project_id": "WATSONX_PROJECT_ID",
        #     "api_key": "WATSONX_API_KEY",
        #     "base_url": "WATSONX_API_URL",
        # },
    )
    user_message = UserMessage("what states are part of New England?")
    response = await watsonx_llm.run([user_message])
    print(response.get_text_content())


async def watsonx_sync() -> None:
    user_message = UserMessage("what is the capital of Massachusetts?")
    response = await llm.run([user_message])
    print(response.get_text_content())


async def watsonx_stream() -> None:
    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.run([user_message], stream=True)
    print(response.get_text_content())


async def watsonx_images() -> None:
    image_llm = ChatModel.from_name(
        "watsonx:meta-llama/llama-3-2-11b-vision-instruct",
    )
    response = await image_llm.run(
        [
            UserMessage("What is the dominant color in the picture?"),
            UserMessage.from_image(
                "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAHUlEQVR4nGI5Y6bFQApgIkn1qIZRDUNKAyAAAP//0ncBT3KcmKoAAAAASUVORK5CYII="
            ),
        ],
    )
    print(response.get_text_content())


async def watsonx_stream_abort() -> None:
    user_message = UserMessage("What is the smallest of the Cape Verde islands?")

    try:
        response = await llm.run([user_message], stream=True, signal=AbortSignal.timeout(0.5))

        if response is not None:
            print(response.get_text_content())
        else:
            print("No response returned.")
    except AbortError as err:
        print(f"Aborted: {err}")


async def watson_structure() -> None:
    class TestSchema(BaseModel):
        answer: str = Field(description="your final answer")

    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.run([user_message], response_format=TestSchema)
    print(response.output_structured)


async def watson_tool_calling() -> None:
    watsonx_llm = ChatModel.from_name("watsonx:ibm/granite-3-3-8b-instruct")
    user_message = UserMessage(f"What is the current weather in Boston? Current date is {datetime.datetime.today()}.")
    weather_tool = OpenMeteoTool()
    response = await watsonx_llm.run([user_message], tools=[weather_tool], stream=True)
    tool_call_msg = response.get_tool_calls()[0]
    print(tool_call_msg.model_dump())
    tool_response = await weather_tool.run(json.loads(tool_call_msg.args))
    tool_response_msg = ToolMessage(
        MessageToolResultContent(
            result=tool_response.get_text_content(), tool_name=tool_call_msg.tool_name, tool_call_id=tool_call_msg.id
        )
    )
    print(tool_response_msg.to_plain())
    final_response = await watsonx_llm.run([user_message, *response.output, tool_response_msg], tools=[])
    print(final_response.get_text_content())


async def watsonx_debug() -> None:
    # Log every request
    llm.emitter.on(
        "*",
        lambda data, event: print(
            f"Time: {event.created_at.time().isoformat()}",
            f"Event: {event.name}",
            f"Data: {str(data)[:90]}...",
        ),
    )

    response = await llm.run(
        [UserMessage("Hello world!")],
    )
    print(response.output[0].to_plain())


async def watsonx_embedding() -> None:
    embedding_llm = WatsonxEmbeddingModel()

    response = await embedding_llm.create(["Text", "to", "embed"])

    for row in response.embeddings:
        print(*row)


async def watsonx_file_example() -> None:
    """Example of sending a file as part of a user message.

    Uses the new factory method UserMessage.from_file. For demonstration we use a tiny
    base64 encoded plain text data URI; in real usage you could pass a file_id referencing
    an uploaded file or another supported format (pdf, markdown, etc.).
    """

    # Minimal inline text file ("Hello Watsonx") as data URI
    data_uri = "data:text/plain;base64,SGVsbG8gV2F0c29ueCAh"

    file_message = UserMessage.from_file(file_data=data_uri, format="txt")
    response = await llm.run([file_message])
    print(response.get_text_content())


async def main() -> None:
    print("*" * 10, "watsonx_from_name")
    await watsonx_from_name()
    print("*" * 10, "watsonx_images")
    await watsonx_images()
    print("*" * 10, "watsonx_sync")
    await watsonx_sync()
    print("*" * 10, "watsonx_stream")
    await watsonx_stream()
    print("*" * 10, "watsonx_stream_abort")
    await watsonx_stream_abort()
    print("*" * 10, "watson_structure")
    await watson_structure()
    print("*" * 10, "watson_tool_calling")
    await watson_tool_calling()
    print("*" * 10, "watsonx_debug")
    await watsonx_debug()
    print("*" * 10, "watsonx_embedding")
    await watsonx_embedding()
    print("*" * 10, "watsonx_file_example")
    await watsonx_file_example()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
