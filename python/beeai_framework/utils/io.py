# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable, Coroutine
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from typing_extensions import TypedDict, Unpack

from beeai_framework.utils.asynchronous import ensure_async
from beeai_framework.utils.strings import to_json

__all__ = ["IOHandlers", "format_confirm_prompt", "io_read", "setup_io_context"]


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


def format_confirm_prompt(prompt: str, data: Any) -> str:
    """Append the tool input to a confirmation prompt so the user can review it.

    Handlers that cannot render structured data (the default console prompt and the
    ACP server's free-text confirmation) use this to keep the tool input visible.
    """
    if not data:
        return prompt
    return f"{prompt}\n{to_json(data, indent=2)}"


async def _default_confirm(prompt: str, **kwargs: Unpack[IOConfirmKwargs]) -> bool:
    return (await _default_read(format_confirm_prompt(prompt, kwargs.get("data")))).lower().startswith("yes")


_storage: ContextVar[IOHandlers] = ContextVar("io_storage")


async def io_read(prompt: str) -> str:
    # pyrefly: ignore [bad-argument-type]
    store = _storage.get(IOHandlers(read=_default_read, confirm=_default_confirm))
    return await store.read(prompt)


async def io_confirm(prompt: str, **kwargs: Any) -> bool:
    # pyrefly: ignore [bad-argument-type]
    store = _storage.get(IOHandlers(read=_default_read, confirm=_default_confirm))
    # pyrefly: ignore [bad-argument-count]
    return await store.confirm(prompt, **kwargs)


def setup_io_context(*, read: ReadHandler, confirm: IOConfirmHandler) -> Callable[[], None]:
    handlers = IOHandlers(read=read, confirm=confirm)
    token = _storage.set(handlers)
    return lambda: _storage.reset(token)
