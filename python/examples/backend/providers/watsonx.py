import asyncio

from pydantic import BaseModel, Field

from beeai_framework.adapters.watsonx.backend.chat import WatsonxChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.cancellation import AbortSignal

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
    watsonx_llm = WatsonxChatModel.from_name(
        "watsonx:ibm/granite-3-8b-instruct",
        # {
        #     "project_id": "WATSONX_PROJECT_ID",
        #     "api_key": "WATSONX_API_KEY",
        #     "api_base": "WATSONX_API_URL",
        # },
    )
    user_message = UserMessage("what states are part of New England?")
    response = await watsonx_llm.create({"messages": [user_message]})
    print(response.get_text_content())


async def watsonx_sync() -> None:
    user_message = UserMessage("what is the capital of Massachusetts?")
    response = await llm.create({"messages": [user_message]})
    print(response.get_text_content())


async def watsonx_stream() -> None:
    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.create({"messages": [user_message], "stream": True})
    print(response.get_text_content())


async def watsonx_stream_abort() -> None:
    user_message = UserMessage("What is the smallest of the Cape Verde islands?")
    response = await llm.create({"messages": [user_message], "stream": True, "abort_signal": AbortSignal.timeout(0.5)})

    if response is not None:
        print(response.get_text_content())
    else:
        print("No response returned.")


async def watson_structure() -> None:
    class TestSchema(BaseModel):
        answer: str = Field(description="your final answer")

    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.create_structure(
        {
            "schema": TestSchema,
            "messages": [user_message],
        }
    )
    print(response.object)


async def main() -> None:
    print("*" * 10, "watsonx_from_name")
    await watsonx_from_name()
    print("*" * 10, "watsonx_sync")
    await watsonx_sync()
    print("*" * 10, "watsonx_stream")
    await watsonx_stream()
    print("*" * 10, "watsonx_stream_abort")
    await watsonx_stream_abort()
    print("*" * 10, "watson_structure")
    await watson_structure()


if __name__ == "__main__":
    asyncio.run(main())
