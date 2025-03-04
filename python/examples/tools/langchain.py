import asyncio
import random
import sys
import traceback

import langchain
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from beeai_framework.adapters.langchain.tools import LangChainTool
from beeai_framework.errors import FrameworkError


class RandomNumberToolArgsSchema(BaseModel):
    min: int = Field(description="The minimum integer", ge=0)
    max: int = Field(description="The maximum integer", ge=0)


def random_number_func(min: int, max: int) -> int:
    """Generate a random integer between two given integers. The two given integers are inclusive."""
    return random.randint(min, max)


async def main() -> None:
    generate_random_number = StructuredTool.from_function(
        func=random_number_func,
        # coroutine=async_random_number_func, <- if you want to specify an async method instead
        name="GenerateRandomNumber",
        description="Generate a random number between a minimum and maximum value.",
        args_schema=RandomNumberToolArgsSchema,
        return_direct=True,
    )

    tool = LangChainTool(generate_random_number)
    response = await tool.run(
        {"min": 1, "max": 10},  # LangChain tool input
        {"timeout": 10 * 1000},  # LangChain run options
    )

    print(response)


if __name__ == "__main__":
    langchain.debug = False
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
