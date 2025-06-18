# Copyright 2025 © BeeAI a Series of LF Projects, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
