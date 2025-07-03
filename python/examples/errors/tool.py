# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio

from beeai_framework.tools import ToolError, tool


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
