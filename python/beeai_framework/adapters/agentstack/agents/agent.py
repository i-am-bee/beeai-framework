# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Literal, Unpack

from pydantic import Secret

from beeai_framework.adapters.agentstack.context import AgentStackContext

try:
    import a2a.types as a2a_types
    from agentstack_sdk.a2a.extensions import (
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
    from agentstack_sdk.platform import ModelProvider, PlatformClient, Provider, get_platform_client
    from agentstack_sdk.platform.context import Context, ContextPermissions, Permissions
    from agentstack_sdk.platform.model_provider import ModelCapability

    from beeai_framework.adapters.a2a.agents import A2AAgent, A2AAgentErrorEvent, A2AAgentUpdateEvent

except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [agentstack] not found.\nRun 'pip install \"beeai-framework[agentstack]\"' to install."
    ) from e

from beeai_framework.adapters.agentstack.agents.events import (
    AgentStackAgentErrorEvent,
    AgentStackAgentUpdateEvent,
    agent_stack_agent_event_types,
)
from beeai_framework.adapters.agentstack.agents.types import (
    AgentStackAgentOutput,
    AgentStackAgentStatus,
)
from beeai_framework.agents import AgentError, AgentMeta, AgentOptions, BaseAgent
from beeai_framework.backend.message import AnyMessage
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.emitter.emitter import EventMeta
from beeai_framework.memory import BaseMemory, UnconstrainedMemory
from beeai_framework.runnable import runnable_entry
from beeai_framework.utils.strings import to_safe_word


class AgentStackAgentOptions(AgentOptions, total=False):
    agent_stack_context: Context | Literal["clear"]
    """
    User can specify custom context for the request. Can be used to support multiple users in one client.
    """
    client: PlatformClient
    """
    User can specify custom client to be used for the request.
    """
    streaming: bool


class AgentStackAgent(BaseAgent[AgentStackAgentOutput]):
    def __init__(
        self,
        *,
        url: str | None = None,
        agent_card: a2a_types.AgentCard | None = None,
        memory: BaseMemory,
        auth_token: str | Secret[str] | None = None,
        agent_stack_url: str | None = None,
    ) -> None:
        super().__init__()
        self._agent_stack_context: Context | None = None
        self._agent = A2AAgent(url=url, agent_card=agent_card, memory=memory)
        self._auth_token = auth_token
        self._agent_stack_url = agent_stack_url

    @property
    def name(self) -> str:
        return self._agent.name

    @runnable_entry
    async def run(
        self,
        input: str | AnyMessage | list[AnyMessage] | a2a_types.Message,
        /,
        **kwargs: Unpack[AgentStackAgentOptions],
    ) -> AgentStackAgentOutput:
        context_param = kwargs.pop("agent_stack_context", None)
        try:
            # try to extract existing agent stack context
            agent_stack_context = AgentStackContext.get()
        except LookupError:
            agent_stack_context = None

        if context_param:
            self._agent_stack_context = None if context_param == "clear" else context_param

        context = RunContext.get()

        async def update_event(data: A2AAgentUpdateEvent, event: EventMeta) -> None:
            await context.emitter.emit(
                "update",
                AgentStackAgentUpdateEvent(value=data.value),
            )

        async def error_event(data: A2AAgentErrorEvent, event: EventMeta) -> None:
            await context.emitter.emit(
                "error",
                AgentStackAgentErrorEvent(message=data.message),
            )

        client = _create_platform_client(kwargs.pop("client", self._agent_stack_url), self._auth_token)
        async with client:
            if not self._agent_stack_context and not agent_stack_context:
                self._agent_stack_context = await Context.create(client=client)

            message = self._agent.convert_to_a2a_message(
                input,
                context_id=agent_stack_context.context.context_id
                if agent_stack_context
                else self._agent_stack_context.id,  # type: ignore[union-attr]
                metadata=agent_stack_context.metadata if agent_stack_context else await self._get_metadata(client),
            )

            response = await (
                self._agent.run(message, **kwargs, httpx_client=client)  # type: ignore[misc]
                .on("update", update_event)
                .on("error", error_event)
            )

            return AgentStackAgentOutput(output=response.output, event=response.event, context=response.context)

    async def check_agent_exists(
        self,
    ) -> None:
        try:
            await self._agent.check_agent_exists()
        except Exception as e:
            raise AgentError("Can't connect to agent stack agent.", cause=e)

    async def _get_metadata(self, client: PlatformClient) -> dict[str, Any]:
        if not self._agent.agent_card:
            await self._agent._load_agent_card()

        assert self._agent.agent_card is not None, "Agent card should not be empty after loading."

        assert self._agent_stack_context is not None, "Agent stack context should not be empty."

        context_token = await self._agent_stack_context.generate_token(
            client=client,
            grant_global_permissions=Permissions(llm={"*"}, embeddings={"*"}, a2a_proxy={"*"}, contexts={"read"}),
            grant_context_permissions=ContextPermissions(files={"*"}, vector_stores={"*"}, context_data={"*"}),
        )
        llm_spec = LLMServiceExtensionSpec.from_agent_card(self._agent.agent_card)
        embedding_spec = EmbeddingServiceExtensionSpec.from_agent_card(self._agent.agent_card)
        agent_stack_extension_spec = PlatformApiExtensionSpec.from_agent_card(self._agent.agent_card)

        async def get_fulfillemnt_args(
            capability: ModelCapability, demand: LLMDemand | EmbeddingDemand
        ) -> dict[str, Any]:
            matches = await ModelProvider.match(suggested_models=demand.suggested, capability=capability, client=client)

            if not matches:
                raise AgentError(f"No matching model found for {capability}.")

            return {
                "api_base": "{platform_url}/api/v1/openai/",
                "api_key": context_token.token.get_secret_value(),
                "api_model": matches[0].model_id,
            }

        metadata = (
            (
                PlatformApiExtensionClient(agent_stack_extension_spec).api_auth_metadata(
                    auth_token=context_token.token, expires_at=context_token.expires_at
                )
                if agent_stack_extension_spec
                else {}
            )
            | (
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
        )

        return metadata

    @classmethod
    async def from_agent_stack(
        cls,
        url: str | PlatformClient | None = None,
        memory: BaseMemory | None = None,
        *,
        states: set[AgentStackAgentStatus] | None = None,
        auth_token: str | Secret[str] | None = None,
    ) -> list["AgentStackAgent"]:
        if memory is None:
            memory = UnconstrainedMemory()
        if states is None:
            states = {s for s in AgentStackAgentStatus if s != AgentStackAgentStatus.OFFLINE}

        client = _create_platform_client(url, auth_token)

        providers = await Provider.list(client=client)
        return [
            AgentStackAgent(
                agent_card=provider.agent_card,
                agent_stack_url=url if isinstance(url, str) else None,
                memory=await memory.clone(),
                auth_token=auth_token,
            )
            for provider in providers
            if provider.state in states
        ]

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["agentstack", "agent", to_safe_word(self.name)],
            creator=self,
            events=agent_stack_agent_event_types,
        )

    @property
    def meta(self) -> AgentMeta:
        return self._agent.meta

    @property
    def memory(self) -> BaseMemory:
        return self._agent.memory

    @memory.setter
    def memory(self, memory: BaseMemory) -> None:
        self._agent.memory = memory

    async def clone(self) -> "AgentStackAgent":
        cloned = AgentStackAgent(
            url=self._agent._url,
            agent_card=self._agent.agent_card,
            memory=await self._agent.memory.clone(),
            auth_token=self._auth_token,
            agent_stack_url=self._agent_stack_url,
        )
        cloned.emitter = await self.emitter.clone()
        return cloned


def _create_platform_client(
    url_or_client: str | PlatformClient | None, auth_token: str | Secret[str] | None = None
) -> PlatformClient:
    if isinstance(url_or_client, str):
        return PlatformClient(base_url=url_or_client, auth_token=auth_token)
    else:
        return url_or_client or get_platform_client()
