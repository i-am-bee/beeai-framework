# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import signal
from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal, Self

import uvicorn
from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict, Unpack, override

from beeai_framework.adapters.a2a.serve.executors.base_a2a_executor import BaseA2AExecutor
from beeai_framework.adapters.a2a.serve.executors.react_agent_executor import ReActAgentExecutor
from beeai_framework.adapters.a2a.serve.executors.tool_calling_agent_executor import ToolCallingAgentExecutor
from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.runnable import AnyRunnable, AnyRunnableTypeVar, Runnable
from beeai_framework.serve import MemoryManager
from beeai_framework.serve.errors import FactoryAlreadyRegisteredError
from beeai_framework.serve.utils import checked_execution_config

try:
    import a2a.server.agent_execution as a2a_agent_execution
    import a2a.server.events as a2a_server_events
    import a2a.server.request_handlers as a2a_request_handlers
    import a2a.server.routes as a2a_routes
    import a2a.server.tasks as a2a_server_tasks
    import a2a.types as a2a_types
    import a2a.utils as a2a_utils
    import grpc
    from a2a.compat.v0_3 import a2a_v0_3_pb2, a2a_v0_3_pb2_grpc
    from a2a.compat.v0_3.grpc_handler import CompatGrpcHandler
    from a2a.types import a2a_pb2, a2a_pb2_grpc
    from grpc_health.v1 import health, health_pb2, health_pb2_grpc
    from grpc_reflection.v1alpha import reflection
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import PlainTextResponse, Response
    from starlette.routing import BaseRoute, Route
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [a2a] not found.\nRun 'pip install \"beeai-framework[a2a]\"' to install."
    ) from e

from beeai_framework.agents import AgentExecutionConfig, BaseAgent

# pyrefly: ignore [deprecated]
from beeai_framework.agents.tool_calling.agent import ToolCallingAgent
from beeai_framework.logger import Logger
from beeai_framework.serve.server import Server
from beeai_framework.utils import ModelLike
from beeai_framework.utils.models import to_model

logger = Logger(__name__)

_PROTOCOL_BINDINGS: dict[str, str] = {
    "jsonrpc": a2a_utils.TransportProtocol.JSONRPC,
    "http_json": a2a_utils.TransportProtocol.HTTP_JSON,
    "grpc": a2a_utils.TransportProtocol.GRPC,
}


class A2AServerConfig(BaseModel):
    """Configuration for the A2AServer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    host: str = "0.0.0.0"
    port: int = 9999
    protocol: Literal["jsonrpc", "grpc", "http_json"] = "jsonrpc"
    agent_card_port: int | None = None
    """
    Applicable only to the gRPC protocol. Specifies the port for the HTTP server to expose AgentCard.
    """
    server_credentials: grpc.ServerCredentials | None = None
    """
    Applicable only to the gRPC protocol.
    """
    enable_v0_3_compat: bool = True
    """
    Whether to also serve clients of the A2A protocol 0.3. The AgentCard then lists the 0.3 interface too.
    """


class A2AServerMetadata(TypedDict, total=False):
    """Configuration to be used within AgentCard."""

    name: str
    description: str
    url: str
    version: str
    default_input_modes: list[str]
    default_output_modes: list[str]
    capabilities: a2a_types.AgentCapabilities
    skills: list[a2a_types.AgentSkill]
    task_store: a2a_server_tasks.TaskStore | None
    queue_manager: a2a_server_events.QueueManager | None
    push_notifier: a2a_server_tasks.PushNotificationSender | None
    request_context_builder: a2a_agent_execution.RequestContextBuilder | None

    send_trajectory: bool
    """
    Whether to send trajectory data to the client.
    """

    execution: AgentExecutionConfig
    """Run defaults for built-in agents. Fields set to None retain the agent's defaults."""


class A2AServer(
    Server[
        AnyRunnableTypeVar,
        BaseA2AExecutor,
        A2AServerConfig,
    ],
):
    def __init__(
        self, *, config: ModelLike[A2AServerConfig] | None = None, memory_manager: MemoryManager | None = None
    ) -> None:
        super().__init__(config=to_model(A2AServerConfig, config or A2AServerConfig()), memory_manager=memory_manager)
        self._metadata_by_agent: dict[AnyRunnable, A2AServerMetadata] = {}
        self._ready = False

    def serve(self) -> None:
        if len(self._members) == 0:
            raise ValueError("No agents registered to the server.")

        member = self._members[0]
        factory = type(self)._get_factory(member)
        metadata = self._metadata_by_agent.get(member, {})
        executor = factory(member, metadata=metadata, memory_manager=self._memory_manager)  # type: ignore[call-arg]

        if self._config.protocol not in _PROTOCOL_BINDINGS:
            raise ValueError(f"Unsupported protocol {self._config.protocol}")
        address = f"{self._config.host}:{self._config.port}"
        self._add_interfaces(
            executor.agent_card,
            _PROTOCOL_BINDINGS[self._config.protocol],
            metadata.get("url", address if self._config.protocol == "grpc" else f"http://{address}"),
        )

        request_handler = a2a_request_handlers.DefaultRequestHandler(
            agent_executor=executor,
            task_store=metadata.get("task_store", None) or a2a_server_tasks.InMemoryTaskStore(),
            agent_card=executor.agent_card,
            queue_manager=metadata.get("queue_manager", None),
            push_sender=metadata.get("push_sender", metadata.get("push_notifier", None)),  # type: ignore
            request_context_builder=metadata.get("request_context_builder", None),
        )

        compat = self._config.enable_v0_3_compat
        if self._config.protocol == "grpc":
            asyncio.run(self._start_grpc_server(executor.agent_card, request_handler))
            return

        routes = (
            a2a_routes.create_jsonrpc_routes(request_handler, rpc_url="/", enable_v0_3_compat=compat)
            if self._config.protocol == "jsonrpc"
            else a2a_routes.create_rest_routes(request_handler, enable_v0_3_compat=compat)
        )
        app = self._create_app(executor.agent_card, request_handler, routes)
        uvicorn.run(app, host=self._config.host, port=self._config.port)

    def _add_interfaces(self, agent_card: a2a_types.AgentCard, protocol_binding: str, url: str) -> None:
        """Lists the served protocol versions in the AgentCard, the 0.3 one is read by the 0.3 clients."""
        versions = [a2a_utils.constants.PROTOCOL_VERSION_1_0]
        if self._config.enable_v0_3_compat:
            versions.append(a2a_utils.constants.PROTOCOL_VERSION_0_3)
        agent_card.supported_interfaces.extend(
            a2a_types.AgentInterface(protocol_binding=protocol_binding, protocol_version=version, url=url)
            for version in versions
        )

    def _create_app(
        self,
        agent_card: a2a_types.AgentCard,
        request_handler: a2a_request_handlers.DefaultRequestHandler,
        routes: Sequence[BaseRoute],
    ) -> Starlette:
        """Creates the Starlette app serving the AgentCard, the A2A routes and the health endpoint."""

        async def health_endpoint(request: Request) -> Response:
            content = "ok" if self._ready else "not ready"
            status = 200 if self._ready else 503
            return PlainTextResponse(content, status_code=status)

        @contextlib.asynccontextmanager
        async def lifespan(_app: Any) -> AsyncIterator[None]:
            self._ready = True
            try:
                yield
            finally:
                self._ready = False
                # stops the tasks the handler keeps alive (e.g. one waiting for input)
                await request_handler.aclose()

        return Starlette(
            routes=[
                *a2a_routes.create_agent_card_routes(agent_card),
                *routes,
                Route("/health", endpoint=health_endpoint),
            ],
            lifespan=lifespan,
        )

    @override
    def register(self, input: AnyRunnableTypeVar, **metadata: Unpack[A2AServerMetadata]) -> Self:
        if len(self._members) != 0:
            raise ValueError("A2AServer only supports one agent.")
        else:
            if "execution" in metadata:
                metadata["execution"] = checked_execution_config(input, metadata["execution"])  # type: ignore[typeddict-item]
            super().register(input)
            self._metadata_by_agent[input] = metadata
            return self

    @override
    def register_many(self, input: Sequence[AnyRunnable]) -> Self:
        raise NotImplementedError("register_many is not implemented for A2AServer")

    async def _start_grpc_server(
        self, agent_card: a2a_types.AgentCard, request_handler: a2a_request_handlers.DefaultRequestHandler
    ) -> None:
        """Creates the Starlette app for the agent card server."""

        app = Starlette(routes=a2a_routes.create_agent_card_routes(agent_card))

        # Create uvicorn server for agent card
        agent_card_port = self._config.agent_card_port or 11000
        config = uvicorn.Config(
            app,
            host=self._config.host,
            port=agent_card_port,
            log_config=None,
        )
        logger.info(f"HTTP server started at port {agent_card_port}. Serving Agent Card.")
        http_server = uvicorn.Server(config)

        """Creates the gRPC server."""
        grpc_server = grpc.aio.server()
        health_servicer = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, grpc_server)
        a2a_pb2_grpc.add_A2AServiceServicer_to_server(
            a2a_request_handlers.GrpcHandler(request_handler),
            grpc_server,
        )  # type: ignore[no-untyped-call]
        a2a_service_names = [a2a_pb2.DESCRIPTOR.services_by_name["A2AService"].full_name]
        if self._config.enable_v0_3_compat:
            # the 0.3 service has its own proto package, so both are served on the same port
            a2a_v0_3_pb2_grpc.add_A2AServiceServicer_to_server(
                CompatGrpcHandler(request_handler),
                grpc_server,
            )  # type: ignore[no-untyped-call]
            a2a_service_names.append(a2a_v0_3_pb2.DESCRIPTOR.services_by_name["A2AService"].full_name)
        service_names = (
            *a2a_service_names,
            "grpc.health.v1.Health",
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(service_names, grpc_server)
        port = f"{self._config.host}:{self._config.port}"
        grpc_server.add_secure_port(
            port, self._config.server_credentials
        ) if self._config.server_credentials else grpc_server.add_insecure_port(port)
        logger.info(f"grpc server started at {port}")

        loop = asyncio.get_running_loop()

        async def shutdown(sig: signal.Signals) -> None:
            """Gracefully shutdown the servers."""
            try:
                health_servicer.set("", health_pb2.HealthCheckResponse.NOT_SERVING)
                for service_name in a2a_service_names:
                    health_servicer.set(service_name, health_pb2.HealthCheckResponse.NOT_SERVING)
            except Exception:
                pass
            http_server.should_exit = True

            await grpc_server.stop(5)

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s)))  # type: ignore[misc]

        await grpc_server.start()
        health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
        for service_name in a2a_service_names:
            health_servicer.set(service_name, health_pb2.HealthCheckResponse.SERVING)

        try:
            await asyncio.gather(http_server.serve(), grpc_server.wait_for_termination())
        finally:
            # stops the tasks the handler keeps alive (e.g. one waiting for input)
            await request_handler.aclose()


def _react_agent_factory(
    agent: ReActAgent, *, metadata: A2AServerMetadata | None = None, memory_manager: MemoryManager
) -> ReActAgentExecutor:
    return ReActAgentExecutor(
        agent=agent,
        agent_card=_create_agent_card(metadata or {}, agent),
        memory_manager=memory_manager,
        send_trajectory=metadata.get("send_trajectory", None) if metadata is not None else None,
        execution=metadata.get("execution", None) if metadata is not None else None,
    )


with contextlib.suppress(FactoryAlreadyRegisteredError):
    A2AServer.register_factory(ReActAgent, _react_agent_factory)  # type: ignore[arg-type]


def _tool_calling_agent_factory(
    agent: ToolCallingAgent, *, metadata: A2AServerMetadata | None = None, memory_manager: MemoryManager
) -> ToolCallingAgentExecutor:
    return ToolCallingAgentExecutor(
        agent=agent,
        agent_card=_create_agent_card(metadata or {}, agent),
        memory_manager=memory_manager,
        send_trajectory=metadata.get("send_trajectory", None) if metadata is not None else None,
        execution=metadata.get("execution", None) if metadata is not None else None,
    )


with contextlib.suppress(FactoryAlreadyRegisteredError):
    A2AServer.register_factory(ToolCallingAgent, _tool_calling_agent_factory)  # type: ignore[arg-type]


def _requirement_agent_factory(
    agent: RequirementAgent, *, metadata: A2AServerMetadata | None = None, memory_manager: MemoryManager
) -> ToolCallingAgentExecutor:
    return ToolCallingAgentExecutor(
        agent=agent,
        agent_card=_create_agent_card(metadata or {}, agent),
        memory_manager=memory_manager,
        send_trajectory=metadata.get("send_trajectory", None) if metadata is not None else None,
        execution=metadata.get("execution", None) if metadata is not None else None,
    )


with contextlib.suppress(FactoryAlreadyRegisteredError):
    A2AServer.register_factory(RequirementAgent, _requirement_agent_factory)  # type: ignore[arg-type]


def _runnable_factory(
    runnable: Runnable[Any], *, metadata: A2AServerMetadata | None = None, memory_manager: MemoryManager
) -> BaseA2AExecutor[Runnable[Any]]:
    return BaseA2AExecutor(
        runnable=runnable,
        agent_card=_create_agent_card(metadata or {}, runnable),
        memory_manager=memory_manager,
    )


with contextlib.suppress(FactoryAlreadyRegisteredError):
    A2AServer.register_factory(Runnable, _runnable_factory)  # type: ignore


def _create_agent_card(metadata: A2AServerMetadata, runnable: Runnable[Any]) -> a2a_types.AgentCard:
    name = metadata.get("name", runnable.meta.name if isinstance(runnable, BaseAgent) else runnable.__class__.__name__)
    description = metadata.get(
        "description",
        runnable.meta.description if isinstance(runnable, BaseAgent) else runnable.__class__.__doc__ or "",
    )

    return a2a_types.AgentCard(
        name=name,
        description=description,
        version=metadata.get("version", "1.0.0"),
        default_input_modes=metadata.get("default_input_modes", ["text"]),
        default_output_modes=metadata.get("default_output_modes", ["text"]),
        capabilities=metadata.get("capabilities", a2a_types.AgentCapabilities(streaming=True)),
        skills=metadata.get(
            "skills",
            [
                a2a_types.AgentSkill(
                    id=name,
                    description=description,
                    name=name,
                    tags=[],
                )
            ],
        ),
    )
