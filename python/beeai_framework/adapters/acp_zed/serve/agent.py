# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

try:
    from acp import (
        PROTOCOL_VERSION,
        InitializeResponse,
        NewSessionResponse,
        PromptResponse,
    )
    from acp import (
        Agent as ACPBaseAgent,
    )
    from acp.interfaces import Client
    from acp.schema import (
        AgentCapabilities,
        ClientCapabilities,
        CloseSessionResponse,
        HttpMcpServer,
        Implementation,
        McpServerStdio,
        SseMcpServer,
    )
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [acp_zed] not found.\nRun 'pip install \"beeai-framework[acp-zed]\"' to install."
    ) from e

from beeai_framework.adapters.acp_zed.serve._utils import PromptBlock
from beeai_framework.agents import AnyAgent

_active_session: ContextVar[str | None] = ContextVar("acp_zed_active_session", default=None)

SessionFactory = Callable[[str], Awaitable[AnyAgent]]
RunTurn = Callable[[str, list[PromptBlock], Client, AnyAgent], Coroutine[Any, Any, str]]


class FsBridge:
    """Agent-side handle to the ACP client's filesystem methods.

    Holds the connection + client-advertised capabilities so BeeAI tools can call
    `fs/read_text_file` and `fs/write_text_file` without threading the session id
    through tool arguments — it's read from a ContextVar that the wrapper sets for
    the duration of each `session/prompt`.
    """

    def __init__(self) -> None:
        self._conn: Client | None = None
        self._capabilities: ClientCapabilities | None = None

    def bind(self, conn: Client) -> None:
        self._conn = conn

    def set_capabilities(self, caps: ClientCapabilities | None) -> None:
        self._capabilities = caps

    @property
    def can_read(self) -> bool:
        fs = getattr(self._capabilities, "fs", None)
        return bool(getattr(fs, "read_text_file", False))

    @property
    def can_write(self) -> bool:
        fs = getattr(self._capabilities, "fs", None)
        return bool(getattr(fs, "write_text_file", False))

    def _active_session_id(self) -> str:
        sid = _active_session.get()
        if sid is None:
            raise RuntimeError("File-system tools must be invoked from inside an ACP prompt.")
        return sid

    def _require_conn(self) -> Client:
        if self._conn is None:
            raise RuntimeError("ACP connection has not been established yet.")
        return self._conn

    async def read_text_file(self, path: str, *, line: int | None = None, limit: int | None = None) -> str:
        response = await self._require_conn().read_text_file(
            path=path, session_id=self._active_session_id(), line=line, limit=limit
        )
        return response.content or ""

    async def write_text_file(self, path: str, content: str) -> None:
        await self._require_conn().write_text_file(content=content, path=path, session_id=self._active_session_id())


class ACPZedServerAgent(ACPBaseAgent):
    """Wrapper that exposes a BeeAI agent over the Zed Agent Client Protocol (stdio).

    One instance per process. Sessions are cloned from the template agent on demand
    and keyed by `session_id` so memory persists across multiple `session/prompt` calls
    within the same Zed chat.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        bridge: FsBridge,
        session_factory: SessionFactory,
        run_turn: RunTurn,
    ) -> None:
        super().__init__()
        self._name = name
        self._description = description
        self._bridge = bridge
        self._session_factory = session_factory
        self._run_turn = run_turn
        self._conn: Client | None = None
        self._sessions: dict[str, AnyAgent] = {}
        self._active_tasks: dict[str, asyncio.Task[str]] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def on_connect(self, conn: Client) -> None:
        self._conn = conn
        self._bridge.bind(conn)

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        self._bridge.set_capabilities(client_capabilities)
        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=AgentCapabilities(),
            agent_info=Implementation(name=self._name, title=self._name, version="0.1.0"),
        )

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        session_id = uuid4().hex
        self._sessions[session_id] = await self._session_factory(session_id)
        return NewSessionResponse(session_id=session_id, modes=None)

    async def prompt(
        self,
        prompt: list[PromptBlock],
        session_id: str,
        message_id: str | None = None,
        **kwargs: Any,
    ) -> PromptResponse:
        if session_id not in self._sessions:
            self._sessions[session_id] = await self._session_factory(session_id)
        if self._conn is None:
            raise RuntimeError("prompt() called before on_connect()")

        token = _active_session.set(session_id)
        task = asyncio.create_task(self._run_turn(session_id, prompt, self._conn, self._sessions[session_id]))
        self._active_tasks[session_id] = task
        try:
            stop_reason = await task
        except asyncio.CancelledError:
            stop_reason = "cancelled"
        finally:
            _active_session.reset(token)
            self._active_tasks.pop(session_id, None)
        return PromptResponse(stop_reason=stop_reason)

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        task = self._active_tasks.get(session_id)
        if task and not task.done():
            task.cancel()

    async def close_session(self, session_id: str, **kwargs: Any) -> CloseSessionResponse | None:
        task = self._active_tasks.pop(session_id, None)
        if task and not task.done():
            task.cancel()
        self._sessions.pop(session_id, None)
        return CloseSessionResponse()
