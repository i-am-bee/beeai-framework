# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any, ClassVar
from weakref import WeakKeyDictionary

from mcp.client.session import ClientSession

from beeai_framework.logger import Logger

logger = Logger(__name__)

# An MCP client transport (e.g. `stdio_client(...)`, `sse_client(...)`, `streamable_http_client(...)`).
# It yields `(read_stream, write_stream)`; streamable HTTP on mcp 1.x also yields a `get_session_id` callback.
# The stream classes differ between mcp 1.x (anyio memory streams) and 2.x (stream protocols).
MCPClient = AbstractAsyncContextManager[tuple[Any, Any] | tuple[Any, Any, Any]]

CleanupFn = Callable[[], None]


class MCPSessionProvider:
    _instances: ClassVar[WeakKeyDictionary[ClientSession, "MCPSessionProvider"]] = WeakKeyDictionary()
    _lock: ClassVar[threading.RLock] = threading.RLock()

    def __init__(self, client: MCPClient) -> None:
        self._client = client
        self._session: ClientSession | None = None
        self._session_initialized = asyncio.Event()
        self._session_stopping = asyncio.Event()
        self.refs = 0
        self._started = False

    @classmethod
    def _acquire_by_session(cls, session: ClientSession) -> bool:
        with cls._lock:
            entry = cls._instances.get(session)
            if entry is None:
                return False
            entry.refs += 1
            return True

    @classmethod
    def destroy_by_session(cls, session: ClientSession) -> None:
        with cls._lock:
            entry = cls._instances.get(session)
            if entry is None or entry._session is None:
                return
            entry.refs = max(0, entry.refs - 1)
            if entry.refs != 0:
                return
            cls._instances.pop(session, None)
            stopping = entry._session_stopping
        stopping.set()

    def destroy(self) -> None:
        if self._session is None:
            return
        MCPSessionProvider.destroy_by_session(self._session)

    async def session(self) -> ClientSession:
        if self._started:
            return await self._get()

        self._started = True

        async def create() -> None:
            try:
                async with self._client as (read, write, *_), ClientSession(read, write) as _session:
                    self._session = _session
                    with type(self)._lock:
                        type(self)._instances[_session] = self
                    await _session.initialize()
                    self._session_initialized.set()
                    await self._session_stopping.wait()
            except Exception as e:
                logger.error(f"Failed to initialize MCP session: {e}")
                self._session_initialized.set()
                if isinstance(e, asyncio.CancelledError):
                    raise
            finally:
                self._session = None

        task = asyncio.create_task(create())
        task.add_done_callback(lambda *args, **kwargs: self.destroy())
        return await self._get()

    async def _get(self) -> ClientSession:
        await self._session_initialized.wait()
        if self._session is None:
            raise RuntimeError("MCP Client Session has been destroyed.")
        return self._session
