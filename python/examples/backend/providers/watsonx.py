import asyncio
import json
import sys
import traceback

from pydantic import BaseModel, Field

from beeai_framework import ToolMessage
from beeai_framework.adapters.watsonx.backend.chat import WatsonxChatModel
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import MessageToolResultContent, UserMessage
from beeai_framework.cancellation import AbortSignal
from beeai_framework.errors import AbortError, FrameworkError
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool

# Setting can be passed here during initiation or pre-configured via environment variables
llm = WatsonxChatModel(
    "ibm/granite-3-8b-instruct",
    # settings={
    #     "project_id": "WATSONX_PROJECT_ID",
    #     "api_key": "WATSONX_API_KEY",
    #     "api_base": "WATSONX_API_URL",
    # },
)


async def watsonx_from_name() -> None:
    watsonx_llm = ChatModel.from_name(
        "watsonx:ibm/granite-3-8b-instruct",
        # {
        #     "project_id": "WATSONX_PROJECT_ID",
        #     "api_key": "WATSONX_API_KEY",
        #     "api_base": "WATSONX_API_URL",
        # },
    )
    user_message = UserMessage("what states are part of New England?")
    response = await watsonx_llm.create(messages=[user_message])
    print(response.get_text_content())


async def watsonx_sync() -> None:
    user_message = UserMessage("what is the capital of Massachusetts?")
    response = await llm.create(messages=[user_message])
    print(response.get_text_content())


async def watsonx_stream() -> None:
    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.create(messages=[user_message], stream=True)
    print(response.get_text_content())


async def watsonx_images() -> None:
    image_llm = ChatModel.from_name(
        "watsonx:meta-llama/llama-3-2-11b-vision-instruct",
    )
    response = await image_llm.create(
        messages=[
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
        response = await llm.create(messages=[user_message], stream=True, abort_signal=AbortSignal.timeout(0.5))

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
    response = await llm.create_structure(schema=TestSchema, messages=[user_message])
    print(response.object)


async def watson_tool_calling() -> None:
    watsonx_llm = ChatModel.from_name(
        "watsonx:ibm/granite-3-8b-instruct",
    )
    user_message = UserMessage("What is the current weather in Boston?")
    weather_tool = OpenMeteoTool()
    response = await watsonx_llm.create(messages=[user_message], tools=[weather_tool])
    tool_call_msg = response.get_tool_calls()[0]
    print(tool_call_msg.model_dump())
    tool_response = await weather_tool.run(json.loads(tool_call_msg.args))
    tool_response_msg = ToolMessage(
        MessageToolResultContent(
            result=tool_response.get_text_content(), tool_name=tool_call_msg.tool_name, tool_call_id=tool_call_msg.id
        )
    )
    print(tool_response_msg.to_plain())
    final_response = await watsonx_llm.create(messages=[user_message, tool_response_msg], tools=[])
    print(final_response.get_text_content())


async def watsonx_debug() -> None:
    # Log every request
    llm.emitter.match(
        "*",
        lambda data, event: print(
            f"Time: {event.created_at.time().isoformat()}",
            f"Event: {event.name}",
            f"Data: {str(data)[:90]}...",
        ),
    )

    response = await llm.create(
        messages=[UserMessage("Hello world!")],
    )
    print(response.messages[0].to_plain())


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


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
