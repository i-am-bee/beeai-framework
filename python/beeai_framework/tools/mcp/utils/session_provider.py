# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextvars
import threading
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, ClassVar, cast

from mcp.client.session import ClientSession

from beeai_framework.logger import Logger

logger = Logger(__name__)

# An MCP client transport (e.g. ``stdio_client(...)`` or ``streamable_http_client(...)``).
# It yields ``(read_stream, write_stream)``; streamable HTTP on mcp 1.x also yields
# a ``get_session_id`` callback.
MCPClient = AbstractAsyncContextManager[tuple[Any, Any] | tuple[Any, Any, Any]]
MCPClientFactory = Callable[[], MCPClient]
MCPClientSource = MCPClient | MCPClientFactory

_active_tool_call: contextvars.ContextVar["MCPToolCallAttempt | None"] = contextvars.ContextVar(
    "beeai_mcp_tool_call_attempt", default=None
)


@dataclass
class MCPToolCallAttempt:
    """Tracks whether the current tools/call request reached the transport."""

    request_sent: bool = False


class _RequestTrackingWriteStream:
    """Mark a tools/call as sent only after the underlying stream accepts it."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    async def send(self, item: Any) -> None:
        await self._stream.send(item)
        attempt = _active_tool_call.get()
        if attempt is None:
            return
        message = getattr(item, "message", item)
        message = getattr(message, "root", message)
        if getattr(message, "method", None) == "tools/call":
            attempt.request_sent = True

    async def __aenter__(self) -> "_RequestTrackingWriteStream":
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, *exc_info: Any) -> Any:
        return await self._stream.__aexit__(*exc_info)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def is_recoverable_session_error(error: BaseException, *, request_sent: bool) -> bool:
    """Return whether retrying the call is known not to repeat a tool side effect."""
    # Streamable HTTP uses these JSON-RPC errors for a 404 caused by an expired
    # Mcp-Session-Id. MCP 1.x says "Session terminated"; MCP 2.x says "Session
    # not found". In either case the server rejects the request before dispatch.
    mcp_error = getattr(error, "error", None)
    error_code = getattr(mcp_error, "code", getattr(error, "code", None))
    error_message = getattr(mcp_error, "message", getattr(error, "message", None))
    if error_code in (-32600, 32600) and error_message in {"Session terminated", "Session not found"}:
        return True

    if request_sent:
        return False

    # MCP 2.x's dispatcher reports a closed client session as a JSON-RPC error
    # before it attempts to write the request. The request tracker above keeps
    # this safe from retrying a call that was already sent.
    if error_message == "Connection closed":
        return True

    # These errors can be raised while writing to a transport that is already
    # closed. Once send() succeeds, the marker above prevents an unsafe retry.
    closed_transport_errors = {"BrokenResourceError", "ClosedResourceError", "EndOfStream"}
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, BaseExceptionGroup):
            return bool(current.exceptions) and all(
                is_recoverable_session_error(item, request_sent=False) for item in current.exceptions
            )
        if type(current).__name__ in closed_transport_errors:
            return True
        current = current.__cause__ or current.__context__
    return False


class MCPSessionProvider:
    """Owns a reconnectable MCP transport and shares one session among its tools."""

    _lock: ClassVar[threading.RLock] = threading.RLock()

    def __init__(self, client: MCPClientSource) -> None:
        if callable(client) and not hasattr(client, "__aenter__"):
            self._client: MCPClient | None = None
            self._client_factory: MCPClientFactory | None = client
            self._recreate_cm: Callable[[], MCPClient] | None = None
        else:
            self._client = client  # type: ignore[assignment]
            self._client_factory = None
            recreate_cm = getattr(client, "_recreate_cm", None)
            self._recreate_cm = cast(Callable[[], MCPClient], recreate_cm) if callable(recreate_cm) else None

        self._client_used = False
        self._session: ClientSession | None = None
        self._session_initialized = asyncio.Event()
        self._session_stopping = asyncio.Event()
        self._session_task: asyncio.Task[None] | None = None
        self._session_error: Exception | None = None
        self._state_lock = asyncio.Lock()
        self._reconnect_lock = asyncio.Lock()
        self._tool_list_lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._generation = 0
        self._tools_listed_generation = 0
        self.refs = 0
        self._closed = False

    def acquire(self, count: int = 1) -> None:
        """Keep the provider alive while one or more MCP tools refer to it."""
        with type(self)._lock:
            self.refs += count

    def release(self) -> None:
        """Release a tool reference and stop the transport after the last tool."""
        with type(self)._lock:
            self.refs = max(0, self.refs - 1)
            should_stop = self.refs == 0
        if should_stop:
            self._stop()

    def destroy(self) -> None:
        """Stop this provider when tool discovery fails or returns no tools."""
        with type(self)._lock:
            self.refs = 0
        self._stop()

    def _stop(self) -> None:
        self._closed = True
        event = self._session_stopping
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(event.set)
        elif loop is None:
            event.set()

    def _make_client(self) -> MCPClient:
        if self._client_factory is not None:
            return self._client_factory()
        if not self._client_used:
            self._client_used = True
            assert self._client is not None
            return self._client
        if self._recreate_cm is not None:
            return self._recreate_cm()
        raise RuntimeError(
            "This MCP transport cannot be restarted. Pass a callable that creates a new transport context manager."
        )

    async def session(self) -> ClientSession:
        """Return the active session, starting a fresh transport when necessary."""
        async with self._state_lock:
            if self._closed:
                raise RuntimeError("MCP Client Session provider has been destroyed.")
            if self._session is not None:
                return self._session
            if self._session_task is None or self._session_task.done():
                self._loop = asyncio.get_running_loop()
                self._session_initialized = asyncio.Event()
                self._session_stopping = asyncio.Event()
                self._session_error = None
                self._session_task = asyncio.create_task(self._create())
            initialized = self._session_initialized

        await initialized.wait()
        async with self._state_lock:
            if self._session is not None:
                return self._session
            error = self._session_error
        if error is not None:
            raise RuntimeError(f"Failed to initialize MCP client session: {error}") from error
        raise RuntimeError("MCP Client Session has been destroyed.")

    async def reconnect(self, session: ClientSession) -> ClientSession:
        """Replace an unusable session once; concurrent callers share this reconnect."""
        async with self._reconnect_lock:
            async with self._state_lock:
                if self._session is not session:
                    current = self._session
                    task = self._session_task
                else:
                    current = None
                    task = self._session_task
                    self._session_stopping.set()

            if current is not None:
                return current
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            return await self.session()

    def mark_tools_listed(self, session: ClientSession) -> None:
        """Record discovery already performed by ``MCPTool.from_client``."""
        if session is self._session:
            self._tools_listed_generation = self._generation

    async def ensure_tools_listed(self, session: ClientSession) -> ClientSession:
        """Refresh the SDK's per-session tool cache after reconnecting."""
        async with self._tool_list_lock:
            for retry in range(2):
                async with self._state_lock:
                    if self._session is not session:
                        if self._session is None:
                            session = None  # type: ignore[assignment]
                            generation = -1
                        else:
                            session = self._session
                            generation = self._generation
                    else:
                        generation = self._generation
                        if self._tools_listed_generation == generation:
                            return session

                if session is None:
                    session = await self.session()
                    continue
                try:
                    await session.list_tools()
                except Exception as error:
                    if retry or not is_recoverable_session_error(error, request_sent=False):
                        raise
                    session = await self.reconnect(session)
                    continue

                async with self._state_lock:
                    if self._session is session and self._generation == generation:
                        self._tools_listed_generation = generation
                        return session
            return session

    async def _create(self) -> None:
        session: ClientSession | None = None
        try:
            async with (
                self._make_client() as (read, write, *_),
                ClientSession(read, cast(Any, _RequestTrackingWriteStream(write))) as session,
            ):
                await session.initialize()
                async with self._state_lock:
                    self._session = session
                    self._generation += 1
                    self._session_initialized.set()
                    stopping = self._session_stopping
                await stopping.wait()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._session_error = error
            logger.error(f"MCP client session ended: {error}")
        finally:
            async with self._state_lock:
                if self._session is session:
                    self._session = None
                if self._session_task is asyncio.current_task():
                    self._session_task = None
                self._session_initialized.set()
