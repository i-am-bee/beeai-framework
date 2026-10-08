# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import contextlib
from collections.abc import Sequence
from typing import Any, Self

import uvicorn
from pydantic import BaseModel
from typing_extensions import TypedDict, TypeVar, Unpack, override

import beeai_framework.adapters.watsonx_orchestrate.serve._factories as factories
from beeai_framework.adapters.watsonx_orchestrate.serve.agent import WatsonxOrchestrateServerAgent
from beeai_framework.adapters.watsonx_orchestrate.serve.api import WatsonxOrchestrateAPI
from beeai_framework.agents import AgentExecutionConfig
from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.tool_calling import ToolCallingAgent
from beeai_framework.logger import Logger
from beeai_framework.runnable import Runnable
from beeai_framework.serve import MemoryManager
from beeai_framework.serve.errors import FactoryAlreadyRegisteredError
from beeai_framework.serve.server import Server
from beeai_framework.serve.utils import agent_execution_options, checked_execution_config
from beeai_framework.utils import ModelLike
from beeai_framework.utils.models import to_model

AnyAgentLike = TypeVar("AnyAgentLike", bound=Runnable[Any], default=Runnable[Any])
AnyWatsonxOrchestrateServerAgentLike = TypeVar(
    "AnyWatsonxOrchestrateServerAgentLike",
    bound=WatsonxOrchestrateServerAgent[Any],
    default=WatsonxOrchestrateServerAgent[Any],
)

logger = Logger(__name__)


class WatsonxOrchestrateServerConfig(BaseModel):
    """Configuration for the WatsonxOrchestrateServer."""

    host: str = "0.0.0.0"
    port: int = 9999
    api_key: str | None = None

    fast_api_kwargs: dict[str, Any] | None = None


class WatsonxOrchestrateServerMetadata(TypedDict, total=False):
    execution: AgentExecutionConfig
    """Run defaults for built-in agents. Fields set to None retain the agent's defaults."""


class WatsonxOrchestrateServer(
    Server[
        AnyAgentLike,
        AnyWatsonxOrchestrateServerAgentLike,
        WatsonxOrchestrateServerConfig,
    ],
):
    def __init__(
        self,
        *,
        config: ModelLike[WatsonxOrchestrateServerConfig] | None = None,
        api_cls: type[WatsonxOrchestrateAPI] = WatsonxOrchestrateAPI,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        super().__init__(
            config=to_model(WatsonxOrchestrateServerConfig, config or WatsonxOrchestrateServerConfig()),
            memory_manager=memory_manager,
        )
        self._api_cls = api_cls
        self._execution: AgentExecutionConfig | None = None

    def serve(self) -> None:
        if not self._members:
            raise ValueError("No agents registered to the server.")

        api = self._api_cls(
            create_agent=self._create_agent,
            api_key=self._config.api_key,
            fast_api_kwargs=self._config.fast_api_kwargs,
            memory_manager=self._memory_manager,
        )
        uvicorn.run(api.app, host=self._config.host, port=self._config.port)

    def _create_agent(self) -> AnyWatsonxOrchestrateServerAgentLike:
        member = self._members[0]
        # pyrefly: ignore [missing-attribute]
        agent = type(self)._factories[type(member)](member)
        agent.run_options = dict(agent_execution_options(self._execution))
        return agent

    @override
    def register(self, input: AnyAgentLike, **metadata: Unpack[WatsonxOrchestrateServerMetadata]) -> Self:
        if self._members:
            raise ValueError("WatsonxOrchestrateServer only supports one agent.")

        self._execution = checked_execution_config(input, metadata.get("execution"))
        return super().register(input)

    @override
    def register_many(self, input: Sequence[AnyAgentLike]) -> Self:
        raise NotImplementedError("register_many is not implemented for WatsonxOrchestrateServer")


with contextlib.suppress(FactoryAlreadyRegisteredError):
    WatsonxOrchestrateServer.register_factory(
        ReActAgent,
        # pyrefly: ignore [bad-argument-type]
        lambda agent: factories.WatsonxOrchestrateServerReActAgent(agent),
    )

with contextlib.suppress(FactoryAlreadyRegisteredError):
    WatsonxOrchestrateServer.register_factory(
        ToolCallingAgent,
        # pyrefly: ignore [bad-argument-type]
        lambda agent: factories.WatsonxOrchestrateServerToolCallingAgent(agent),
    )

with contextlib.suppress(FactoryAlreadyRegisteredError):
    WatsonxOrchestrateServer.register_factory(
        RequirementAgent,
        # pyrefly: ignore [bad-argument-type]
        lambda agent: factories.WatsonxOrchestrateServerRequirementAgent(agent),
    )

with contextlib.suppress(FactoryAlreadyRegisteredError):
    WatsonxOrchestrateServer.register_factory(Runnable, lambda agent: factories.WatsonxOrchestrateServerRunnable(agent))  # type: ignore[type-abstract]
