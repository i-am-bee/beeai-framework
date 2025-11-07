# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass

from beeai_framework.utils.asynchronous import ensure_async

__all__ = ["IOHandlers", "io_read", "setup_io_context"]

ReadHandler = Callable[[str], Awaitable[str]]


@dataclass
class IOHandlers:
    read: ReadHandler


_storage: ContextVar[IOHandlers] = ContextVar("io_storage")
_storage.set(IOHandlers(read=ensure_async(input)))


async def io_read(prompt: str) -> str:
    store = _storage.get()
    return await store.read(prompt)


def setup_io_context(*, read: ReadHandler) -> Callable[[], None]:
    handlers = IOHandlers(read=read)
    token = _storage.set(handlers)
    return lambda: _storage.reset(token)
