# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
from collections.abc import Callable
from typing import ClassVar
from weakref import WeakKeyDictionary

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.client.session import ClientSession
from mcp.shared.message import SessionMessage

MCPClient = contextlib._AsyncGeneratorContextManager[
    tuple[MemoryObjectReceiveStream[SessionMessage | Exception], MemoryObjectSendStream[SessionMessage]], None
]

CleanupFn = Callable[[], None]


class MCPSessionProvider:
    _instances: ClassVar[WeakKeyDictionary[ClientSession, "MCPSessionProvider"]] = WeakKeyDictionary()

    def __init__(self, client: MCPClient) -> None:
        self._client = client
        self._session: ClientSession | None = None
        self._session_initialized = asyncio.Event()
        self._session_stopping = asyncio.Event()
        self.refs = 0
        self._started = False

    @classmethod
    def destroy_by_session(cls, session: ClientSession) -> None:
        entry = cls._instances.get(session)
        if entry is not None:
            entry.destroy()

    def destroy(self) -> None:
        if self._session is None:
            return

        self.refs = max(0, self.refs - 1)
        if self.refs == 0:
            self._session_stopping.set()
            type(self)._instances.pop(self._session, None)

    async def session(self) -> ClientSession:
        if self._started:
            return await self._get()

        self._started = True

        async def create() -> None:
            try:
                async with self._client as (read, write, *_), ClientSession(read, write) as _session:
                    self._session = _session
                    type(self)._instances[_session] = self
                    await _session.initialize()
                    self._session_initialized.set()
                    await self._session_stopping.wait()
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
