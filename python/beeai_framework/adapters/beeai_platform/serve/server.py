# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import os
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any, Self

import uvicorn
from pydantic import BaseModel
from typing_extensions import TypedDict, TypeVar, Unpack, override

from beeai_framework.adapters.beeai_platform.serve._dummy_context_store import (
    DummyContextStore,
)
from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.tool_calling import ToolCallingAgent
from beeai_framework.memory import BaseMemory
from beeai_framework.runnable import Runnable
from beeai_framework.serve.errors import FactoryAlreadyRegisteredError

try:
    import a2a.types as a2a_types
    import beeai_sdk.server as beeai_server
    import beeai_sdk.server.agent as beeai_agent
    import beeai_sdk.server.store.context_store as beeai_context_store
    import beeai_sdk.server.store.platform_context_store as beeai_platform_context_store
    from beeai_sdk.a2a.extensions.ui.agent_detail import AgentDetail
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [beeai-platform] not found.\nRun 'pip install \"beeai-framework[beeai-platform]\"' to install."
    ) from e

from beeai_framework.serve import MemoryManager, Server
from beeai_framework.utils.models import ModelLike, to_model

AnyAgentLike = TypeVar("AnyAgentLike", bound=Runnable[Any], default=Runnable[Any])


# this class is only placeholder to use ContextStore from the beeai-sdk
class BeeAIPlatformMemoryManager(MemoryManager):
    async def set(self, key: str, value: BaseMemory) -> None:
        raise NotImplementedError("This method is not supported by BeeAIPlatformMemoryManager.")

    async def get(self, key: str) -> BaseMemory:
        raise NotImplementedError("This method is not supported by BeeAIPlatformMemoryManager.")

    async def contains(self, key: str) -> bool:
        raise NotImplementedError("This method is not supported by BeeAIPlatformMemoryManager.")


class BeeAIPlatformServerConfig(BaseModel):
    """Configuration for the BeeAIServer."""

    host: str = "127.0.0.1"
    port: int = 9999
    configure_telemetry: bool = True

    configure_logger: bool | None = None
    self_registration: bool | None = True
    run_limit: int | None = None
    run_ttl: timedelta | None = None
    uds: str | None = None
    fd: int | None = None
    loop: uvicorn.config.LoopSetupType | None = None
    http: type[asyncio.Protocol] | uvicorn.config.HTTPProtocolType | None = None
    ws: type[asyncio.Protocol] | uvicorn.config.WSProtocolType | None = None
    ws_max_size: int | None = None
    ws_max_queue: int | None = None
    ws_ping_interval: float | None = None
    ws_ping_timeout: float | None = None
    ws_per_message_deflate: bool | None = None
    lifespan: uvicorn.config.LifespanType | None = None
    env_file: str | os.PathLike[str] | None = None
    log_config: dict[str, Any] | str | None = None
    log_level: str | int | None = None
    access_log: bool | None = None
    use_colors: bool | None = None
    interface: uvicorn.config.InterfaceType | None = None
    reload: bool | None = None
    reload_dirs: list[str] | str | None = None
    reload_delay: float | None = None
    reload_includes: list[str] | str | None = None
    reload_excludes: list[str] | str | None = None
    workers: int | None = None
    proxy_headers: bool | None = None
    server_header: bool | None = None
    date_header: bool | None = None
    forwarded_allow_ips: list[str] | str | None = None
    root_path: str | None = None
    limit_concurrency: int | None = None
    limit_max_requests: int | None = None
    backlog: int | None = None
    timeout_keep_alive: int | None = None
    timeout_notify: int | None = None
    timeout_graceful_shutdown: int | None = None
    callback_notify: Callable[..., Awaitable[None]] | None = None
    ssl_keyfile: str | os.PathLike[str] | None = None
    ssl_certfile: str | os.PathLike[str] | None = None
    ssl_keyfile_password: str | None = None
    ssl_version: int | None = None
    ssl_cert_reqs: int | None = None
    ssl_ca_certs: str | None = None
    ssl_ciphers: str | None = None
    headers: list[tuple[str, str]] | None = None
    factory: bool | None = None
    h11_max_incomplete_event_size: int | None = None


class BeeAIPlatformServerMetadata(TypedDict, total=False):
    name: str
    description: str
    additional_interfaces: list[a2a_types.AgentInterface]
    capabilities: a2a_types.AgentCapabilities
    default_input_modes: list[str]
    default_output_modes: list[str]
    detail: AgentDetail
    documentation_url: str
    icon_url: str
    preferred_transport: str
    provider: a2a_types.AgentProvider
    security: list[dict[str, list[str]]]
    security_schemes: dict[str, a2a_types.SecurityScheme]
    skills: list[a2a_types.AgentSkill]
    supports_authenticated_extended_card: bool
    version: str


class BeeAIPlatformServer(
    Server[
        AnyAgentLike,
        beeai_agent.AgentFactory,
        BeeAIPlatformServerConfig,
    ],
):
    def __init__(
        self, *, config: ModelLike[BeeAIPlatformServerConfig] | None = None, memory_manager: MemoryManager | None = None
    ) -> None:
        super().__init__(
            config=to_model(BeeAIPlatformServerConfig, config or BeeAIPlatformServerConfig()),
            memory_manager=memory_manager or BeeAIPlatformMemoryManager(),
        )
        self._metadata_by_agent: dict[AnyAgentLike, BeeAIPlatformServerMetadata] = {}
        self._server = beeai_server.Server()

    def _setup_member(self) -> beeai_context_store.ContextStore:
        if not self._members:
            raise ValueError("No agents registered to the server.")

        member = self._members[0]
        factory = type(self)._get_factory(member)
        config = self._metadata_by_agent.get(member, BeeAIPlatformServerMetadata())
        self._server._agent_factory = factory(member, metadata=config, memory_manager=self._memory_manager)  # type: ignore[call-arg]
        return (
            beeai_platform_context_store.PlatformContextStore()
            if isinstance(self._memory_manager, BeeAIPlatformMemoryManager)
            else DummyContextStore()
        )

    def serve(self) -> None:
        context_store = self._setup_member()
        with contextlib.suppress(KeyboardInterrupt):
            self._server.run(
                **self._config.model_dump(exclude_none=True, exclude={"context_store": True}),
                context_store=context_store,
            )

    async def aserve(self) -> None:
        context_store = self._setup_member()
        with contextlib.suppress(KeyboardInterrupt):
            await self._server.serve(
                **self._config.model_dump(exclude_none=True, exclude={"context_store": True}),
                context_store=context_store,
            )

    @override
    def register(self, input: AnyAgentLike, **metadata: Unpack[BeeAIPlatformServerMetadata]) -> Self:
        if len(self._members) != 0:
            raise ValueError("BeeAIPlatformServer only supports one agent.")
        else:
            super().register(input)
            metadata = metadata or BeeAIPlatformServerMetadata()
            detail = metadata.setdefault("detail", AgentDetail(interaction_mode="multi-turn"))
            detail.framework = detail.framework or "BeeAI"

            self._metadata_by_agent[input] = metadata
            return self


def register() -> None:
    from beeai_framework.adapters.beeai_platform.serve.factories import (
        _react_agent_factory,
        _requirement_agent_factory,
        _runnable_factory,
        _tool_calling_agent_factory,
    )

    with contextlib.suppress(FactoryAlreadyRegisteredError):
        BeeAIPlatformServer.register_factory(ReActAgent, _react_agent_factory)  # type: ignore[arg-type]

    with contextlib.suppress(FactoryAlreadyRegisteredError):
        BeeAIPlatformServer.register_factory(ToolCallingAgent, _tool_calling_agent_factory)  # type: ignore[arg-type]

    with contextlib.suppress(FactoryAlreadyRegisteredError):
        BeeAIPlatformServer.register_factory(RequirementAgent, _requirement_agent_factory)  # type: ignore[arg-type]

    with contextlib.suppress(FactoryAlreadyRegisteredError):
        BeeAIPlatformServer.register_factory(Runnable, _runnable_factory)  # type: ignore


register()
