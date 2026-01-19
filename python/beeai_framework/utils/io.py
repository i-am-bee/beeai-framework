# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable, Coroutine
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from typing_extensions import TypedDict, Unpack

from beeai_framework.utils.asynchronous import ensure_async

__all__ = ["IOHandlers", "io_read", "setup_io_context"]


class IOConfirmKwargs(TypedDict, total=False):
    title: str
    description: str
    submit_label: str
    cancel_label: str
    data: dict[str, Any]


ReadHandler = Callable[[str], Awaitable[str]]
# pyrefly: ignore [invalid-annotation]
IOConfirmHandler = Callable[[str, Unpack[IOConfirmKwargs]], Coroutine[Any, Any, bool]]


@dataclass
class IOHandlers:
    read: ReadHandler
    confirm: IOConfirmHandler


_default_read = ensure_async(input)


async def _default_confirm(prompt: str, **kwargs: Unpack[IOConfirmKwargs]) -> bool:
    return input(prompt).lower().startswith("yes")


_storage: ContextVar[IOHandlers] = ContextVar("io_storage")
# pyrefly: ignore [bad-argument-type]
_storage.set(IOHandlers(read=_default_read, confirm=_default_confirm))


async def io_read(prompt: str) -> str:
    store = _storage.get()
    return await store.read(prompt)


async def io_confirm(prompt: str, **kwargs: Any) -> bool:
    store = _storage.get()
    # pyrefly: ignore [bad-argument-count]
    return await store.confirm(prompt, **kwargs)


def setup_io_context(*, read: ReadHandler, confirm: IOConfirmHandler) -> Callable[[], None]:
    handlers = IOHandlers(read=read, confirm=confirm)
    token = _storage.set(handlers)
    return lambda: _storage.reset(token)
