# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel

from beeai_framework.adapters.watsonx_orchestrate.serve.agent import (
    WatsonxOrchestrateServerAgent,
    WatsonxOrchestrateServerAgentEmitFn,
    WatsonxOrchestrateServerAgentMessageEvent,
    WatsonxOrchestrateServerAgentThinkEvent,
    WatsonxOrchestrateServerAgentToolCallEvent,
    WatsonxOrchestrateServerAgentToolResponse,
)
from beeai_framework.agents.react import ReActAgent, ReActAgentUpdateEvent
from beeai_framework.backend import AnyMessage
from beeai_framework.emitter import EmitterOptions, EventMeta
from beeai_framework.tools import Tool, ToolStartEvent, ToolSuccessEvent
from beeai_framework.utils.cloneable import Cloneable


class WatsonxOrchestrateServerReActAgent(WatsonxOrchestrateServerAgent[ReActAgent]):
    @property
    def model_id(self) -> str:
        return self._agent._input.llm.model_id

    async def _stream(self, input: list[AnyMessage], emit: WatsonxOrchestrateServerAgentEmitFn) -> None:
        cloned_agent = await self._agent.clone() if isinstance(self._agent, Cloneable) else self._agent

        async def on_tool_start(data: ToolStartEvent, meta: EventMeta) -> None:
            assert meta.trace, "ToolStartEvent must have trace"
            assert isinstance(meta.creator, Tool)

            await emit(
                WatsonxOrchestrateServerAgentToolCallEvent(
                    id=meta.trace.run_id,
                    name=meta.creator.name,
                    args=data.input.model_dump() if isinstance(data.input, BaseModel) else data.input,
                )
            )

        async def on_tool_success(data: ToolSuccessEvent, meta: EventMeta) -> None:
            assert meta.trace, "ToolSuccessEvent must have trace"
            assert isinstance(meta.creator, Tool)

            await emit(
                WatsonxOrchestrateServerAgentToolResponse(
                    name=meta.creator.name, id=meta.trace.run_id, result=data.output.get_text_content()
                )
            )

        # Tool events are emitted by the nested tool runs, which the run's own event stream
        # (match_nested=False) does not surface, so they are subscribed to explicitly.
        run = cloned_agent.run(input).on(
            lambda event: isinstance(event.creator, Tool) and event.name == "start",
            on_tool_start,
            EmitterOptions(match_nested=True),
        )
        run = run.on(
            lambda event: isinstance(event.creator, Tool) and event.name == "success",
            on_tool_success,
            EmitterOptions(match_nested=True),
        )

        async for data, event in run:
            match (data, event.name):
                case (ReActAgentUpdateEvent(), "partial_update"):
                    update = data.update.value
                    if not isinstance(update, str):
                        update = update.get_text_content()
                    match data.update.key:
                        case "thought":
                            await emit(WatsonxOrchestrateServerAgentThinkEvent(text=update))
                        case "final_answer":
                            await emit(WatsonxOrchestrateServerAgentMessageEvent(text=update))
