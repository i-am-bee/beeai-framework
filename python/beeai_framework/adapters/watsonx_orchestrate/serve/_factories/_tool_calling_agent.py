# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from beeai_framework.adapters.watsonx_orchestrate.serve.agent import (
    WatsonxOrchestrateServerAgent,
    WatsonxOrchestrateServerAgentEmitFn,
    WatsonxOrchestrateServerAgentMessageEvent,
    WatsonxOrchestrateServerAgentToolCallEvent,
    WatsonxOrchestrateServerAgentToolResponse,
)
from beeai_framework.agents.tool_calling import ToolCallingAgent
from beeai_framework.backend import AnyMessage
from beeai_framework.emitter import EmitterOptions, EventMeta
from beeai_framework.tools import Tool, ToolStartEvent, ToolSuccessEvent
from beeai_framework.utils.cloneable import Cloneable


class WatsonxOrchestrateServerToolCallingAgent(WatsonxOrchestrateServerAgent[ToolCallingAgent]):
    @property
    def model_id(self) -> str:
        return self._agent._llm.model_id

    async def _stream(self, input: list[AnyMessage], emit: WatsonxOrchestrateServerAgentEmitFn) -> None:
        cloned_agent = await self._agent.clone() if isinstance(self._agent, Cloneable) else self._agent

        async def on_tool_success(data: ToolSuccessEvent, meta: EventMeta) -> None:
            assert meta.trace, "ToolSuccessEvent must have trace"
            assert isinstance(meta.creator, Tool)

            if meta.creator.name == "final_answer":
                return

            await emit(
                WatsonxOrchestrateServerAgentToolResponse(
                    name=meta.creator.name, id=meta.trace.run_id, result=data.output.get_text_content()
                )
            )

        async def on_tool_start(data: ToolStartEvent, meta: EventMeta) -> None:
            assert meta.trace, "ToolStartEvent must have trace"
            assert isinstance(meta.creator, Tool)

            if meta.creator.name == "final_answer":
                return

            await emit(
                WatsonxOrchestrateServerAgentToolCallEvent(
                    id=meta.trace.run_id,
                    name=meta.creator.name,
                    args=data.input.model_dump() if isinstance(data.input, BaseModel) else data.input,
                )
            )

        response = await (
            cloned_agent.run(input)
            .on(
                lambda event: isinstance(event.creator, Tool) and event.name == "start",
                on_tool_start,
                EmitterOptions(match_nested=True),
            )
            .on(
                lambda event: isinstance(event.creator, Tool) and event.name == "success",
                on_tool_success,
                EmitterOptions(match_nested=True),
            )
        )
        await emit(WatsonxOrchestrateServerAgentMessageEvent(text=response.last_message.text))
