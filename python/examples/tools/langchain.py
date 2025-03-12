# To run this example, the optional packages:
#
#    - langchain-core
#    - langchain-community
#
# need to be installed

import asyncio
import pathlib
import random
import sys
import traceback
from typing import Any

import langchain
from langchain_community.tools.file_management.list_dir import ListDirectoryTool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from beeai_framework.adapters.langchain.tools import LangChainTool
from beeai_framework.errors import FrameworkError


async def directory_list_tool() -> None:
    list_dir_tool = ListDirectoryTool()
    tool = LangChainTool[Any](list_dir_tool)
    dir_path = str(pathlib.Path(__file__).parent.resolve())
    response = await tool.run({"dir_path": dir_path})
    print(f"Listing contents of {dir_path}:\n{response}")


async def custom_structured_tool() -> None:
    class RandomNumberToolArgsSchema(BaseModel):
        min: int = Field(description="The minimum integer", ge=0)
        max: int = Field(description="The maximum integer", ge=0)

    def random_number_func(min: int, max: int) -> int:
        """Generate a random integer between two given integers. The two given integers are inclusive."""
        return random.randint(min, max)

    generate_random_number = StructuredTool.from_function(
        func=random_number_func,
        # coroutine=async_random_number_func, <- if you want to specify an async method instead
        name="GenerateRandomNumber",
        description="Generate a random number between a minimum and maximum value.",
        args_schema=RandomNumberToolArgsSchema,
        return_direct=True,
    )

    tool = LangChainTool[Any](generate_random_number)
    response = await tool.run({"min": 1, "max": 10})

    print(f"Your random number: {response}")


async def main() -> None:
    print("*" * 10, "Using custom StructuredTool")
    await custom_structured_tool()
    print("*" * 10, "Using ListDirectoryTool")
    await directory_list_tool()


if __name__ == "__main__":
    langchain.debug = False
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
