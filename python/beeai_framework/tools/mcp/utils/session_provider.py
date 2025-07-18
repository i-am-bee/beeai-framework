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
    _instances: ClassVar[WeakKeyDictionary[ClientSession, CleanupFn]] = WeakKeyDictionary()

    def __init__(self, client: MCPClient) -> None:
        self._client = client

    @classmethod
    def destroy(cls, session: ClientSession) -> None:
        cleanup = cls._instances.get(session)
        if cleanup is not None:
            cleanup()

    async def session(self) -> ClientSession:
        _session: ClientSession | None = None
        _session_initialized = asyncio.Event()
        _session_stopping = asyncio.Event()

        async def create() -> None:
            nonlocal _session

            async with self._client as (read, write, *_), ClientSession(read, write) as _session:
                await _session.initialize()
                _session_initialized.set()
                await _session_stopping.wait()

        def cleanup() -> None:
            if _session is not None:
                type(self)._instances.pop(_session, None)

        task = asyncio.create_task(create())
        task.add_done_callback(lambda *args, **kwargs: cleanup())

        await _session_initialized.wait()
        assert _session is not None

        type(self)._instances[_session] = lambda: _session_stopping.set()
        return _session
