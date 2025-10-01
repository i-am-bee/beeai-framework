# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from typing_extensions import TypeVar, override

from beeai_framework.adapters.a2a.serve.executors.base_a2a_agent_executor import BaseA2AAgentExecutor
from beeai_framework.agents.requirement.events import RequirementAgentStartEvent, RequirementAgentSuccessEvent
from beeai_framework.agents.tool_calling import ToolCallingAgentStartEvent, ToolCallingAgentSuccessEvent
from beeai_framework.backend import AssistantMessage, MessageToolCallContent, ToolMessage
from beeai_framework.emitter import Emitter, EventMeta
from beeai_framework.runnable import Runnable
from beeai_framework.utils.lists import find_index

try:
    import a2a.server.agent_execution as a2a_agent_execution
    import a2a.server.tasks as a2a_server_tasks
    import a2a.types as a2a_types
    import a2a.utils as a2a_utils
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [a2a] not found.\nRun 'pip install \"beeai-framework[a2a]\"' to install."
    ) from e

from beeai_framework.agents import AnyAgent

AnyAgentLike = TypeVar("AnyAgentLike", bound=AnyAgent, default=AnyAgent)
AnyRunnable = TypeVar("AnyRunnable", bound=Runnable[Any], default=Runnable[Any])


class ToolCallingAgentExecutor(BaseA2AAgentExecutor):
    @override
    async def _process_events(
        self,
        emitter: Emitter,
        context: a2a_agent_execution.RequestContext,
        updater: a2a_server_tasks.TaskUpdater,
    ) -> None:
        last_msg = None

        async def process_event(
            data: RequirementAgentStartEvent
            | RequirementAgentSuccessEvent
            | ToolCallingAgentStartEvent
            | ToolCallingAgentSuccessEvent,
            event: EventMeta,
        ) -> None:
            nonlocal last_msg
            messages = data.state.memory.messages
            if last_msg is None:
                last_msg = messages[-1]

            cur_index = find_index(messages, lambda msg: msg is last_msg, fallback=-1, reverse_traversal=True)
            for message in messages[cur_index + 1 :]:
                if (isinstance(message, ToolMessage) and message.content[0].tool_name == "final_answer") or (
                    isinstance(message, AssistantMessage)
                    and isinstance(message.content[0], MessageToolCallContent)
                    and message.content[0].tool_name == "final_answer"
                ):
                    continue

                await updater.start_work(
                    a2a_utils.new_agent_parts_message(
                        parts=[
                            a2a_types.Part(
                                root=a2a_types.TextPart(text=content)
                                if isinstance(content, str)
                                else a2a_types.DataPart(data=content.model_dump())
                            )
                            for content in message.content
                        ]
                    ),
                )
                last_msg = message

        emitter.on("start", process_event)
        emitter.on("success", process_event)
