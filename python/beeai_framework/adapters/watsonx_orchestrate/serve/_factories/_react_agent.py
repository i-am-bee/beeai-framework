# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from beeai_framework.adapters.watsonx_orchestrate.serve.agent import (
    WatsonxOrchestrateServerAgent,
    WatsonxOrchestrateServerAgentEmitFn,
    WatsonxOrchestrateServerAgentMessageEvent,
    WatsonxOrchestrateServerAgentThinkEvent,
)
from beeai_framework.agents.react import ReActAgent, ReActAgentUpdateEvent
from beeai_framework.backend import AssistantMessage
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware


class WatsonxOrchestrateServerReActAgent(WatsonxOrchestrateServerAgent[ReActAgent]):
    @property
    def model_id(self) -> str:
        return self._agent._input.llm.model_id

    async def _run(self) -> AssistantMessage:
        response = await self._agent.run(prompt=None).middleware(GlobalTrajectoryMiddleware())
        return response.result

    async def _stream(self, emit: WatsonxOrchestrateServerAgentEmitFn) -> None:
        async for data, event in self._agent.run():
            match (data, event.name):
                case (ReActAgentUpdateEvent(), "partial_update"):
                    update = data.update.value
                    if not isinstance(update, str):
                        update = update.get_text_content()
                    match data.update.key:
                        # TODO: ReAct agent does not use native-tool calling capabilities (ignore or simulate?)
                        case "thought":
                            await emit(WatsonxOrchestrateServerAgentThinkEvent(text=update))
                        case "final_answer":
                            await emit(WatsonxOrchestrateServerAgentMessageEvent(text=update))
