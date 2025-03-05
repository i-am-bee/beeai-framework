import asyncio

from beeai_framework import tool
from beeai_framework.tools import ToolError


async def main() -> None:
    @tool
    def dummy() -> None:
        """
        A dummy tool.
        """
        raise ToolError("Dummy error.")

    await dummy.run({})


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ToolError as e:
        print("===CAUSE===")
        print(e.get_cause())
        print("===EXPLAIN===")
        print(e.explain())
