import asyncio
import json
import sys
import traceback
from urllib.parse import quote

import requests

from beeai_framework import ReActAgent, tool
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.errors import FrameworkError
from beeai_framework.logger import Logger
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.tools import StringToolOutput

logger = Logger(__name__)


# defining a tool using the `tool` decorator
@tool
def basic_calculator(expression: str) -> StringToolOutput:
    """
    A calculator tool that performs mathematical operations.

    Args:
        expression: The mathematical expression to evaluate (e.g., "2 + 3 * 4").

    Returns:
        The result of the mathematical expression
    """
    try:
        encoded_expression = quote(expression)
        math_url = f"https://newton.vercel.app/api/v2/simplify/{encoded_expression}"

        response = requests.get(
            math_url,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()

        return StringToolOutput(json.dumps(response.json()))
    except Exception as e:
        raise RuntimeError(f"Error evaluating expression: {e!s}") from Exception


async def main() -> None:
    # using the tool in an agent

    chat_model = ChatModel.from_name("ollama:granite3.1-dense:8b")

    agent = ReActAgent(llm=chat_model, tools=[basic_calculator], memory=UnconstrainedMemory())

    result = await agent.run("What is the square root of 36?", execution=AgentExecutionConfig(total_max_retries=10))

    print(result.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
