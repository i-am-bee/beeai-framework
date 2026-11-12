# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from pydantic_core import to_jsonable_python
from typing_extensions import TypeVar, override

from beeai_framework.adapters.a2a.serve.executors.base_a2a_agent_executor import BaseA2AAgentExecutor
from beeai_framework.agents.react import ReActAgentUpdateEvent
from beeai_framework.agents.react.types import ReActAgentIterationResult
from beeai_framework.emitter import Emitter, EventMeta
from beeai_framework.utils.strings import to_json

try:
    import a2a.helpers as a2a_helpers
    import a2a.server.agent_execution as a2a_agent_execution
    import a2a.server.tasks as a2a_server_tasks
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [a2a] not found.\nRun 'pip install \"beeai-framework[a2a]\"' to install."
    ) from e

from beeai_framework.agents import AnyAgent
from beeai_framework.logger import Logger

AnyAgentLike = TypeVar("AnyAgentLike", bound=AnyAgent, default=AnyAgent)

logger = Logger(__name__)


class ReActAgentExecutor(BaseA2AAgentExecutor):
    @override
    async def _process_events(
        self,
        emitter: Emitter,
        context: a2a_agent_execution.RequestContext,
        updater: a2a_server_tasks.TaskUpdater,
    ) -> None:
        async def process_event(data: ReActAgentUpdateEvent, event: EventMeta) -> None:
            text = ""
            if isinstance(data.data, ReActAgentIterationResult):
                if data.data.final_answer:
                    text = data.data.final_answer
                elif data.data.tool_output:
                    text = data.data.tool_output
                elif data.data.tool_name or data.data.tool_input:
                    text = to_json({"tool_name": data.data.tool_name, "tool_input": data.data.tool_input})
                elif data.data.thought:
                    text = data.data.thought

            await updater.start_work(
                a2a_helpers.new_text_message(
                    text,
                    context_id=context.context_id,
                    task_id=context.task_id,
                )
                if isinstance(data.data, ReActAgentIterationResult)
                else a2a_helpers.new_message(parts=[a2a_helpers.new_data_part(to_jsonable_python(data.data))]),
            )

        emitter.on("update", process_event)
