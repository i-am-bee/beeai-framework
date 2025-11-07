# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from beeai_framework.adapters.agentstack.serve.server import AgentStackMemoryManager
from beeai_framework.agents import AnyAgent
from beeai_framework.serve import MemoryManager, init_agent_memory

try:
    import agentstack_sdk.server.context as agentstack_context

    from beeai_framework.adapters.a2a.agents._utils import convert_a2a_to_framework_message
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [agentstack not found.\nRun 'pip install \"beeai-framework[agentstack]\"' to install."
    ) from e


async def init_agent_stack_memory(
    agent: AnyAgent, memory_manager: MemoryManager, context: agentstack_context.RunContext
) -> None:
    if isinstance(memory_manager, AgentStackMemoryManager):
        history = [message async for message in context.load_history() if message.parts]
        agent.memory.reset()
        await agent.memory.add_many([convert_a2a_to_framework_message(message) for message in history])
    else:
        await init_agent_memory(agent, memory_manager, context.context_id)
