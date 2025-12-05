# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import contextlib
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, Self

try:
    import a2a.server as a2a_server
    import a2a.server.agent_execution as a2a_agent_execution
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [a2a] not found.\nRun 'pip install \"beeai-framework[a2a]\"' to install."
    ) from e

__all__ = ["A2AContext"]

# pyrefly: ignore [not-a-type]
_storage = ContextVar["A2AContext"]("a2a_server_storage")


class A2AContext:
    def __init__(
        self,
        *,
        context: a2a_agent_execution.RequestContext,
        # pyrefly: ignore [implicit-import]
        event_queue: a2a_server.events.EventQueue,
    ) -> None:
        self._cleanups: list[Callable[[], None]] = []
        self._context = context
        self._event_queue = event_queue

    @property
    def context(self) -> a2a_agent_execution.RequestContext:
        return self._context

    @property
    # pyrefly: ignore [implicit-import]
    def event_queue(self) -> a2a_server.events.EventQueue:
        return self._event_queue

    def __enter__(self) -> Self:
        ctx_key = _storage.set(self)
        self._cleanups.append(lambda: _storage.reset(ctx_key))
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        while self._cleanups:
            cleanup = self._cleanups.pop(0)
            with contextlib.suppress(Exception):
                cleanup()
