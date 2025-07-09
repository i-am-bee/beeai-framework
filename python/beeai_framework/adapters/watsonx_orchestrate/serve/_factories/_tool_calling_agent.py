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
from beeai_framework.agents.experimental.utils._tool import FinalAnswerTool
from beeai_framework.agents.tool_calling import ToolCallingAgent
from beeai_framework.backend import AssistantMessage
from beeai_framework.emitter import EmitterOptions, EventMeta
from beeai_framework.tools import Tool, ToolStartEvent, ToolSuccessEvent


class WatsonxOrchestrateServerToolCallingAgent(WatsonxOrchestrateServerAgent[ToolCallingAgent]):
    @property
    def model_id(self) -> str:
        return self._agent._llm.model_id

    async def _run(self) -> AssistantMessage:
        response = await self._agent.run(prompt=None)
        return response.result

    async def _stream(self, emit: WatsonxOrchestrateServerAgentEmitFn) -> None:
        async def on_tool_success(data: ToolSuccessEvent, meta: EventMeta) -> None:
            assert meta.trace, "ToolSuccessEvent must have trace"
            assert isinstance(meta.creator, Tool)

            await emit(
                WatsonxOrchestrateServerAgentToolResponse(
                    name=meta.creator.name, id=meta.trace.run_id, result=data.output.get_text_content()
                )
            )

        async def on_tool_start(data: ToolStartEvent, meta: EventMeta) -> None:
            assert meta.trace, "ToolStartEvent must have trace"
            assert isinstance(meta.creator, Tool)

            if isinstance(meta.creator, FinalAnswerTool):
                return

            await emit(
                WatsonxOrchestrateServerAgentToolCallEvent(
                    id=meta.trace.run_id,
                    name=meta.creator.name,
                    args=data.input.model_dump() if isinstance(data.input, BaseModel) else data.input,
                )
            )

        response = await (
            self._agent.run(prompt=None)
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
        await emit(WatsonxOrchestrateServerAgentMessageEvent(text=response.result.text))
