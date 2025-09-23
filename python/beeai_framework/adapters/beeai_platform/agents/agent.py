# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Unpack

import httpx

from beeai_framework.adapters.a2a.agents.agent import A2AAgentOptions

try:
    import a2a.types as a2a_types
    from beeai_sdk.a2a.extensions import (
        EmbeddingDemand,
        EmbeddingFulfillment,
        EmbeddingServiceExtensionClient,
        EmbeddingServiceExtensionSpec,
        LLMDemand,
        LLMFulfillment,
        LLMServiceExtensionClient,
        LLMServiceExtensionSpec,
        PlatformApiExtensionClient,
        PlatformApiExtensionSpec,
    )
    from beeai_sdk.platform import ModelProvider
    from beeai_sdk.platform.context import Context, ContextPermissions, Permissions
    from beeai_sdk.platform.model_provider import ModelCapability

    from beeai_framework.adapters.a2a.agents import A2AAgent, A2AAgentErrorEvent, A2AAgentUpdateEvent

except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [beeai-platform] not found.\nRun 'pip install \"beeai-framework[beeai-platform]\"' to install."
    ) from e

from beeai_framework.adapters.beeai_platform.agents.events import (
    BeeAIPlatformAgentErrorEvent,
    BeeAIPlatformAgentUpdateEvent,
    beeai_platform_agent_event_types,
)
from beeai_framework.adapters.beeai_platform.agents.types import (
    BeeAIPlatformAgentOutput,
)
from beeai_framework.agents import AgentError, BaseAgent
from beeai_framework.backend.message import AnyMessage
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.emitter.emitter import EventMeta
from beeai_framework.memory import BaseMemory
from beeai_framework.runnable import runnable_entry
from beeai_framework.utils.strings import to_safe_word


class BeeAIPlatformAgent(BaseAgent[BeeAIPlatformAgentOutput]):
    def __init__(
        self, *, url: str | None = None, agent_card: a2a_types.AgentCard | None = None, memory: BaseMemory
    ) -> None:
        super().__init__()
        self._agent = A2AAgent(url=url, agent_card=agent_card, memory=memory)

    @property
    def name(self) -> str:
        return self._agent.name

    @runnable_entry
    async def run(
        self, input: str | AnyMessage | list[AnyMessage] | a2a_types.Message, /, **kwargs: Unpack[A2AAgentOptions]
    ) -> BeeAIPlatformAgentOutput:
        self._agent.set_run_params(
            context_id=kwargs.get("context_id"),
            task_id=kwargs.get("task_id"),
            clear_context=kwargs.get("clear_context"),
        )

        context = RunContext.get()

        async def update_event(data: A2AAgentUpdateEvent, event: EventMeta) -> None:
            await context.emitter.emit(
                "update",
                BeeAIPlatformAgentUpdateEvent(value=data.value),
            )

        async def error_event(data: A2AAgentErrorEvent, event: EventMeta) -> None:
            await context.emitter.emit(
                "error",
                BeeAIPlatformAgentErrorEvent(message=data.message),
            )

        message = self._agent.convert_to_a2a_message(input)
        message.metadata = (message.metadata or {}) | await self._get_metadata()

        response = await self._agent.run(message, **kwargs).on("update", update_event).on("error", error_event)

        return BeeAIPlatformAgentOutput(output=response.output, event=response.event)

    async def check_agent_exists(
        self,
    ) -> None:
        try:
            await self._agent.check_agent_exists()
        except Exception as e:
            raise AgentError("Can't connect to beeai platform agent.", cause=e)

    async def _get_metadata(self) -> dict[str, Any]:
        if not self._agent.agent_card:
            await self._agent._load_agent_card()

        assert self._agent.agent_card is not None, "Agent card should not be empty after loading."

        context = await Context.create()
        context_token = await context.generate_token(
            grant_global_permissions=Permissions(llm={"*"}, embeddings={"*"}, a2a_proxy={"*"}),
            grant_context_permissions=ContextPermissions(files={"*"}, vector_stores={"*"}, context_data={"*"}),
        )
        llm_spec = LLMServiceExtensionSpec.from_agent_card(self._agent.agent_card)
        embedding_spec = EmbeddingServiceExtensionSpec.from_agent_card(self._agent.agent_card)
        platform_extension_spec = PlatformApiExtensionSpec.from_agent_card(self._agent.agent_card)

        async def get_fulfillemnt_args(
            capability: ModelCapability, demand: LLMDemand | EmbeddingDemand
        ) -> dict[str, Any]:
            matches = await ModelProvider.match(
                suggested_models=demand.suggested,
                capability=capability,
            )

            if not matches:
                raise AgentError(f"No matching model found for {capability}.")

            return {
                "api_base": "{platform_url}/api/v1/openai/",
                "api_key": context_token.token.get_secret_value(),
                "api_model": matches[0].model_id,
            }

        metadata = (
            (
                LLMServiceExtensionClient(llm_spec).fulfillment_metadata(
                    llm_fulfillments={
                        key: LLMFulfillment(**(await get_fulfillemnt_args(ModelCapability.LLM, demand)))
                        for key, demand in llm_spec.params.llm_demands.items()
                    }
                )
                if llm_spec
                else {}
            )
            | (
                EmbeddingServiceExtensionClient(embedding_spec).fulfillment_metadata(
                    embedding_fulfillments={
                        key: EmbeddingFulfillment(**(await get_fulfillemnt_args(ModelCapability.EMBEDDING, demand)))
                        for key, demand in embedding_spec.params.embedding_demands.items()
                    }
                )
                if embedding_spec
                else {}
            )
            | (
                PlatformApiExtensionClient(platform_extension_spec).api_auth_metadata(
                    auth_token=context_token.token, expires_at=context_token.expires_at
                )
                if platform_extension_spec
                else {}
            )
        )

        return metadata

    @classmethod
    async def from_platform(cls, url: str, memory: BaseMemory) -> list["BeeAIPlatformAgent"]:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{url}/api/v1/providers")

            response.raise_for_status()
            return [
                BeeAIPlatformAgent(
                    agent_card=a2a_types.AgentCard(**provider["agent_card"]), memory=await memory.clone()
                )
                for provider in response.json().get("items", [])
            ]

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["beeai_platform", "agent", to_safe_word(self.name)],
            creator=self,
            events=beeai_platform_agent_event_types,
        )

    @property
    def memory(self) -> BaseMemory:
        return self._agent.memory

    @memory.setter
    def memory(self, memory: BaseMemory) -> None:
        self._agent.memory = memory

    async def clone(self) -> "BeeAIPlatformAgent":
        cloned = BeeAIPlatformAgent(
            url=self._agent._url, agent_card=self._agent.agent_card, memory=await self._agent.memory.clone()
        )
        cloned.emitter = await self.emitter.clone()
        return cloned
