# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import signal
from collections.abc import Sequence
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

try:
    import a2a.server.agent_execution as a2a_agent_execution
    import a2a.server.apps as a2a_apps
    import a2a.server.events as a2a_server_events
    import a2a.server.request_handlers as a2a_request_handlers
    import a2a.server.tasks as a2a_server_tasks
    import a2a.types as a2a_types
    import a2a.utils as a2a_utils
    import grpc
    from a2a.grpc import a2a_pb2, a2a_pb2_grpc
    from grpc_health.v1 import health, health_pb2, health_pb2_grpc
    from grpc_reflection.v1alpha import reflection
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, PlainTextResponse, Response
    from starlette.routing import Route
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [a2a] not found.\nRun 'pip install \"beeai-framework[a2a]\"' to install."
    ) from e

from beeai_framework.agents import BaseAgent

# pyrefly: ignore [deprecated]
from beeai_framework.agents.tool_calling.agent import ToolCallingAgent
from beeai_framework.logger import Logger
from beeai_framework.serve.server import Server
from beeai_framework.utils import ModelLike
from beeai_framework.utils.models import to_model

logger = Logger(__name__)


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

        request_handler = a2a_request_handlers.DefaultRequestHandler(
            agent_executor=executor,
            task_store=metadata.get("task_store", None) or a2a_server_tasks.InMemoryTaskStore(),
            queue_manager=metadata.get("queue_manager", None),
            push_sender=metadata.get("push_sender", metadata.get("push_notifier", None)),  # type: ignore
            request_context_builder=metadata.get("request_context_builder", None),
        )

        server: a2a_apps.A2ARESTFastAPIApplication | a2a_apps.A2AStarletteApplication
        if self._config.protocol == "jsonrpc":
            executor.agent_card.url = metadata.get("url", f"http://{self._config.host}:{self._config.port}")
            executor.agent_card.preferred_transport = a2a_types.TransportProtocol.jsonrpc
            server = a2a_apps.A2AStarletteApplication(agent_card=executor.agent_card, http_handler=request_handler)
            app = server.build()
            self._add_health_endpoint(app)
            uvicorn.run(app, host=self._config.host, port=self._config.port)
        elif self._config.protocol == "http_json":
            executor.agent_card.url = metadata.get("url", f"http://{self._config.host}:{self._config.port}")
            executor.agent_card.preferred_transport = a2a_types.TransportProtocol.http_json
            server = a2a_apps.A2ARESTFastAPIApplication(agent_card=executor.agent_card, http_handler=request_handler)
            app = server.build()
            self._add_health_endpoint(app)
            uvicorn.run(app, host=self._config.host, port=self._config.port)
        elif self._config.protocol == "grpc":
            executor.agent_card.url = metadata.get("url", f"{self._config.host}:{self._config.port}")
            executor.agent_card.preferred_transport = a2a_types.TransportProtocol.grpc
            asyncio.run(self._start_grpc_server(executor.agent_card, request_handler))
        else:
            raise ValueError(f"Unsupported protocol {self._config.protocol}")

    def _add_health_endpoint(self, app: Starlette) -> None:
        """Add health endpoint to the Starlette app."""

        async def on_startup() -> None:
            self._ready = True

        async def on_shutdown() -> None:
            self._ready = False

        async def health_endpoint(request: Request) -> Response:
            content = "ok" if self._ready else "not ready"
            status = 200 if self._ready else 503
            return PlainTextResponse(content, status_code=status)

        app.add_event_handler("startup", on_startup)
        app.add_event_handler("shutdown", on_shutdown)
        app.routes.append(Route("/health", endpoint=health_endpoint))

    @override
    def register(self, input: AnyRunnableTypeVar, **metadata: Unpack[A2AServerMetadata]) -> Self:
        if len(self._members) != 0:
            raise ValueError("A2AServer only supports one agent.")
        else:
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

        def get_agent_card_http(request: Request) -> Response:
            return JSONResponse(agent_card.model_dump(mode="json", exclude_none=True))

        routes = [Route(a2a_utils.constants.AGENT_CARD_WELL_KNOWN_PATH, endpoint=get_agent_card_http)]
        app = Starlette(routes=routes)

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
            a2a_request_handlers.GrpcHandler(agent_card, request_handler),
            grpc_server,
        )  # type: ignore[no-untyped-call]
        service_names = (
            a2a_pb2.DESCRIPTOR.services_by_name["A2AService"].full_name,
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
                health_servicer.set(
                    a2a_pb2.DESCRIPTOR.services_by_name["A2AService"].full_name,
                    health_pb2.HealthCheckResponse.NOT_SERVING,
                )
            except Exception:
                pass
            http_server.should_exit = True

            await grpc_server.stop(5)

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s)))  # type: ignore[misc]

        await grpc_server.start()
        health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
        health_servicer.set(
            a2a_pb2.DESCRIPTOR.services_by_name["A2AService"].full_name,
            health_pb2.HealthCheckResponse.SERVING,
        )

        await asyncio.gather(http_server.serve(), grpc_server.wait_for_termination())


def _react_agent_factory(
    agent: ReActAgent, *, metadata: A2AServerMetadata | None = None, memory_manager: MemoryManager
) -> ReActAgentExecutor:
    return ReActAgentExecutor(
        agent=agent,
        agent_card=_create_agent_card(metadata or {}, agent),
        memory_manager=memory_manager,
        send_trajectory=metadata.get("send_trajectory", None) if metadata is not None else None,
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
        url=metadata.get("url", "dummy"),
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
