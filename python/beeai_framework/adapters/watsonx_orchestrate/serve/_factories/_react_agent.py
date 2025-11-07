# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0


from beeai_framework.adapters.watsonx_orchestrate.serve.agent import (
    WatsonxOrchestrateServerAgent,
    WatsonxOrchestrateServerAgentEmitFn,
    WatsonxOrchestrateServerAgentMessageEvent,
    WatsonxOrchestrateServerAgentThinkEvent,
)
from beeai_framework.agents.react import ReActAgent, ReActAgentUpdateEvent
from beeai_framework.backend import AnyMessage
from beeai_framework.utils.cloneable import Cloneable


class WatsonxOrchestrateServerReActAgent(WatsonxOrchestrateServerAgent[ReActAgent]):
    @property
    def model_id(self) -> str:
        return self._agent._input.llm.model_id

    async def _stream(self, input: list[AnyMessage], emit: WatsonxOrchestrateServerAgentEmitFn) -> None:
        cloned_agent = await self._agent.clone() if isinstance(self._agent, Cloneable) else self._agent
        async for data, event in cloned_agent.run(input):
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
