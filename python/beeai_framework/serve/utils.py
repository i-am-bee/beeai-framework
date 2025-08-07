# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from beeai_framework.agents import AnyAgent
from beeai_framework.logger import Logger
from beeai_framework.memory import BaseMemory
from beeai_framework.serve.memory_manager import MemoryManager

logger = Logger(__name__)


async def init_agent_memory(
    agent: AnyAgent, memory_manager: MemoryManager, session_id: str | None, *, stateful: bool = True
) -> None:
    async def create_empty_memory() -> BaseMemory:
        memory = await agent.memory.clone()
        memory.reset()
        return memory

    if stateful and session_id:
        if not await memory_manager.contains(session_id):
            await memory_manager.set(session_id, await create_empty_memory())
        memory = await memory_manager.get(session_id)
    else:
        memory = await create_empty_memory()

    try:
        agent.memory = memory
    except Exception:
        logger.debug("Agent does not support setting a new memory, resetting existing one for the agent.")
        agent.memory.reset()
