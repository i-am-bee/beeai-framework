# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio

from beeai_framework.backend import AssistantMessage, UserMessage
from beeai_framework.memory import UnconstrainedMemory


async def main() -> None:
    memory = UnconstrainedMemory()
    await memory.add(UserMessage("What is your name?"))

    serialized = memory.serialize()
    restored = UnconstrainedMemory.from_serialized(serialized)

    await restored.add(AssistantMessage("Bee"))

    print(len(restored.messages))
    print(restored.messages[0].text)


if __name__ == "__main__":
    asyncio.run(main())
