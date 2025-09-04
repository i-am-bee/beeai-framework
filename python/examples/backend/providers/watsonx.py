import asyncio
import datetime

from dotenv import load_dotenv

from beeai_framework.backend import ChatModel, UserMessage
from beeai_framework.tools.weather import OpenMeteoTool

load_dotenv()


async def main() -> None:
    watsonx_llm = ChatModel.from_name("watsonx:ibm/granite-3-3-8b-instruct")
    user_message = UserMessage(f"What is the current weather in Boston? Current date is {datetime.datetime.today()}.")
    weather_tool = OpenMeteoTool()
    response = await watsonx_llm.create(
        messages=[user_message],
        tools=[weather_tool],
        tool_choice=weather_tool,
        stream=True,
    )
    print(response.get_tool_calls()[0])


if __name__ == "__main__":
    asyncio.run(main())
