# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from typing import Annotated

from beeai_framework.adapters.beeai_platform.serve.server import BeeAIPlatformMemoryManager, BeeAIPlatformServerMetadata
from beeai_framework.agents import AnyAgent
from beeai_framework.backend import AssistantMessage, MessageTextContent, MessageToolCallContent, ToolMessage
from beeai_framework.backend.message import AnyMessage
from beeai_framework.serve import MemoryManager, init_agent_memory

try:
    import beeai_sdk.a2a.extensions as beeai_extensions
    import beeai_sdk.a2a.types as beeai_types
    import beeai_sdk.server.context as beeai_context

    from beeai_framework.adapters.a2a.agents._utils import convert_a2a_to_framework_message
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [beeai-platform] not found.\nRun 'pip install \"beeai-framework[beeai-platform]\"' to install."
    ) from e


def send_message_trajectory(
    msg: AnyMessage,
    trajectory: Annotated[beeai_extensions.TrajectoryExtensionServer, beeai_extensions.TrajectoryExtensionSpec()],
) -> Generator[beeai_types.Metadata[str, beeai_extensions.Trajectory]]:
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


def _init_metadata(
    agent: AnyAgent,
    base: BeeAIPlatformServerMetadata | None = None,
) -> BeeAIPlatformServerMetadata:
    copy = (base or {}).copy()
    if not copy.get("name"):
        copy["name"] = agent.meta.name
    if not copy.get("description"):
        copy["description"] = agent.meta.description
    return copy


async def init_beeai_platform_memory(
    agent: AnyAgent, memory_manager: MemoryManager, context: beeai_context.RunContext
) -> None:
    if isinstance(memory_manager, BeeAIPlatformMemoryManager):
        history = [message async for message in context.store.load_history() if message.parts]
        agent.memory.reset()
        # last message is provided directly to the run method
        await agent.memory.add_many([convert_a2a_to_framework_message(message) for message in history[:-1]])
    else:
        await init_agent_memory(agent, memory_manager, context.context_id)
