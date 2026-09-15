import asyncio

from beeai_framework.adapters.hol_guard import HOLGuardMiddleware
from beeai_framework.tools import tool


@tool(name="Bash", description="Example shell tool protected by HOL Guard")
async def bash(command: str) -> str:
    return f"Tool would execute: {command}"


async def main() -> None:
    # HOLGuardMiddleware fails closed by default if the local hol-guard process is unavailable.
    result = await bash.run({"command": "echo hello"}).middleware(HOLGuardMiddleware())
    print(result.get_text_content())


if __name__ == "__main__":
    asyncio.run(main())
