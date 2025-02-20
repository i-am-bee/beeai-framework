import asyncio
import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from pydantic import BaseModel, Field

from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.agents.types import BeeInput, BeeRunInput
from beeai_framework.backend.chat import ChatModel
from beeai_framework.emitter.emitter import Emitter, EventMeta
from beeai_framework.emitter.types import EmitterOptions
from beeai_framework.memory.token_memory import TokenMemory
from beeai_framework.tools.tool import StringToolOutput, Tool
from beeai_framework.utils.custom_logger import BeeLogger
from examples.helpers.io import ConsoleReader

# Load environment variables
load_dotenv()

# Configure logging - using DEBUG instead of trace
logger = BeeLogger("app", level=logging.DEBUG)

reader = ConsoleReader()


def get_env_var(key: str, default: str | None = None) -> str:
    """Helper function to get environment variables with defaults"""
    return os.getenv(key, default)


class LangChainWikipediaToolInput(BaseModel):
    query: str = Field(description="The topic or question to search for on Wikipedia.")


class LangChainWikipediaTool(Tool):
    """Adapter class to integrate LangChain's Wikipedia tool with our framework"""

    name = "Wikipedia"
    description = "Search factual and historical information from Wikipedia about given topics."
    input_schema = LangChainWikipediaToolInput

    def __init__(self) -> None:
        super().__init__()
        wikipedia = WikipediaAPIWrapper()
        self.wikipedia = WikipediaQueryRun(api_wrapper=wikipedia)

    def _run(self, input: LangChainWikipediaToolInput, _: Any | None = None) -> Any:
        query = input.query
        try:
            result = self.wikipedia.run(query)
            return StringToolOutput(json.dumps(result))
        except Exception as e:
            logger.error(f"Wikipedia search error: {e!s}")
            return f"Error searching Wikipedia: {e!s}"


def create_agent() -> BeeAgent:
    """Create and configure the agent with tools and LLM"""

    llm = ChatModel.from_name(
        "ollama:granite3.1-dense:8b",
        # ChatModelParameters(temperature=0, presence_penalty=1.0),
    )

    # Configure tools with LangChain's Wikipedia tool
    # tools = [LangChainWikipediaTool(), OpenMeteoTool()]
    tools = [LangChainWikipediaTool()]

    # Add code interpreter tool if URL is configured
    code_interpreter_url = get_env_var("CODE_INTERPRETER_URL")
    if code_interpreter_url:
        # Note: Python tool implementation would go here
        pass

    # Create agent with memory and tools
    agent = BeeAgent(BeeInput(llm=llm, tools=tools, memory=TokenMemory(llm)))

    return agent


async def process_agent_events(event_data: dict[str, Any], event_meta: EventMeta) -> None:
    """Process agent events and log appropriately"""

    if event_meta.name == "error":
        reader.write("Agent 🤖 : ", event_data["error"])
    elif event_meta.name == "retry":
        reader.write("Agent 🤖 : ", "retrying the action...")
    elif event_meta.name == "update":
        reader.write(f"Agent({event_data['update']['key']}) 🤖 : ", event_data["update"]["parsedValue"])
    # elif event_meta.name == "start":
    #     reader.write("Agent 🤖 : ", "starting new iteration")
    # elif event_meta.name == "success":
    #     reader.write("Agent 🤖 : ", "success")


async def observer(emitter: Emitter) -> None:
    emitter.on("*.*", process_agent_events, EmitterOptions(match_nested=True))


async def main() -> None:
    """Main application loop"""

    try:
        # Create agent
        agent = create_agent()

        # Log code interpreter status if configured
        code_interpreter_url = get_env_var("CODE_INTERPRETER_URL")
        if code_interpreter_url:
            reader.write(
                "🛠️ System: ",
                f"The code interpreter tool is enabled. Please ensure that it is running on {code_interpreter_url}",
            )

        reader.write("🛠️ System: ", "Agent initialized with LangChain Wikipedia tool.")

        # Main interaction loop with user input
        for prompt in reader:
            # Run agent with the prompt
            response = await agent.run(
                BeeRunInput(prompt=prompt),
                {
                    "execution": {
                        "max_retries_per_step": 3,
                        "total_max_retries": 10,
                        "max_iterations": 20,
                    }
                },
            ).observe(observer)

            reader.write("Agent 🤖 : ", response.result.text)

    except Exception as e:
        logger.error(f"Application error: {e!s}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
