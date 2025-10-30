# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from typing import Annotated

from beeai_framework.adapters.agentstack.serve.server import AgentStackMemoryManager
from beeai_framework.agents import AnyAgent
from beeai_framework.backend import AssistantMessage, MessageTextContent, MessageToolCallContent, ToolMessage
from beeai_framework.backend.message import AnyMessage
from beeai_framework.serve import MemoryManager, init_agent_memory

try:
    import agentstack_sdk.a2a.extensions as agentstack_extensions
    import agentstack_sdk.a2a.types as agentstack_types
    import agentstack_sdk.server.context as agentstack_context

    from beeai_framework.adapters.a2a.agents._utils import convert_a2a_to_framework_message
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [agentstack not found.\nRun 'pip install \"beeai-framework[agentstack]\"' to install."
    ) from e


def send_message_trajectory(
    msg: AnyMessage,
    trajectory: Annotated[
        agentstack_extensions.TrajectoryExtensionServer, agentstack_extensions.TrajectoryExtensionSpec()
    ],
) -> Generator[agentstack_types.Metadata[str, agentstack_extensions.Trajectory]]:
    if isinstance(msg, AssistantMessage):
        for content in msg.content:
            if isinstance(content, MessageTextContent):
                yield trajectory.trajectory_metadata(title="assistant", content=content.text)
            elif isinstance(content, MessageToolCallContent):
                if content.tool_name == "final_answer":
                    continue
                yield trajectory.trajectory_metadata(title=f"{content.tool_name} (request)", content=content.args)
    elif isinstance(msg, ToolMessage):
        for tool_call in msg.get_tool_results():
            if tool_call.tool_name == "final_answer":
                continue

            yield trajectory.trajectory_metadata(
                title=f"{tool_call.tool_name} (response)", content=str(tool_call.result)
            )


async def init_agent_stack_memory(
    agent: AnyAgent, memory_manager: MemoryManager, context: agentstack_context.RunContext
) -> None:
    if isinstance(memory_manager, AgentStackMemoryManager):
        history = [message async for message in context.load_history() if message.parts]
        agent.memory.reset()
        await agent.memory.add_many([convert_a2a_to_framework_message(message) for message in history])
    else:
        await init_agent_memory(agent, memory_manager, context.context_id)
