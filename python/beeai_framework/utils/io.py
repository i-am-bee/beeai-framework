# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass

from beeai_framework.utils.asynchronous import ensure_async

__all__ = ["IOHandlers", "io_read", "setup_io_context"]

ReadHandler = Callable[[str], Awaitable[str]]
IOConfirmHandler = Callable[[str], Awaitable[bool]]


@dataclass
class IOHandlers:
    read: ReadHandler
    io_confirm: IOConfirmHandler


_default_read = ensure_async(input)


async def _default_ask(question: str) -> bool:
    return input(question).startswith("yes")


_storage: ContextVar[IOHandlers] = ContextVar("io_storage")
_storage.set(IOHandlers(read=_default_read, io_confirm=_default_ask))


async def io_read(prompt: str) -> str:
    store = _storage.get()
    return await store.read(prompt)


async def io_ask(prompt: str) -> bool:
    store = _storage.get()
    return await store.io_confirm(prompt)


def setup_io_context(*, read: ReadHandler, io_confirm: IOConfirmHandler) -> Callable[[], None]:
    handlers = IOHandlers(read=read, io_confirm=io_confirm)
    token = _storage.set(handlers)
    return lambda: _storage.reset(token)
