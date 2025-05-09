# Copyright 2025 © BeeAI a Series of LF Projects, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import timedelta
from typing import Any

import acp_sdk.models as acp_models
import acp_sdk.server.context as acp_context
import acp_sdk.server.types as acp_types
import uvicorn
from acp_sdk.server.agent import Agent as AcpBaseAgent
from pydantic import BaseModel


class AcpAgent(AcpBaseAgent):
    """A wrapper for a BeeAI agent to be used with the ACP server."""

    def __init__(
        self,
        fn: Callable[
            [list[acp_models.Message], acp_context.Context],
            AsyncGenerator[acp_types.RunYield, acp_types.RunYieldResume],
        ],
        name: str,
        description: str | None = None,
        metadata: acp_models.Metadata | None = None,
    ) -> None:
        super().__init__()
        self.fn = fn
        self._name = name
        self._description = description
        self._metadata = metadata

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description or ""

    @property
    def metadata(self) -> acp_models.Metadata:
        return self._metadata or acp_models.Metadata()

    async def run(
        self, input: list[acp_models.Message], context: acp_context.Context
    ) -> AsyncGenerator[acp_types.RunYield, acp_types.RunYieldResume]:
        try:
            gen: AsyncGenerator[acp_types.RunYield, acp_types.RunYieldResume] = self.fn(input, context)
            value = None
            while True:
                value = yield await gen.asend(value)
        except StopAsyncIteration:
            pass


class AcpServerConfig(BaseModel):
    """Configuration for the AcpServer."""

    configure_logger: bool | None = None
    configure_telemetry: bool | None = None
    self_registration: bool | None = None
    run_limit: int | None = None
    run_ttl: timedelta | None = None
    host: str | None = None
    port: int | None = None
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
