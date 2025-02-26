# Copyright 2025 IBM Corp.
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


import contextlib
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from beeai_framework.errors import AbortError
from beeai_framework.utils.custom_logger import BeeLogger

logger = BeeLogger(__name__)


class AbortSignal(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._aborted = False
        self._reason: str | None = None
        self._listeners: list[Callable] = []

    @property
    def aborted(self) -> bool:
        return self._aborted

    @property
    def reason(self) -> str:
        return self._reason

    def add_event_listener(self, callback: Callable[[], None]) -> None:
        self._listeners.append(callback)

    def remove_event_listener(self, callback: Callable[[], None]) -> None:
        with contextlib.suppress(ValueError):
            self._listeners.remove(callback)

    def _abort(self, reason: str | None = None) -> None:
        self._aborted = True
        self._reason = reason
        for callback in self._listeners:
            if callback:
                callback()

    @classmethod
    def timeout(cls, duration: int) -> "AbortSignal":
        signal = cls()

        def _callback() -> None:
            signal._timer.cancel()
            signal._abort(f"Operation timed out after {duration} ms")

        signal._timer = threading.Timer(duration * 1.0, _callback)
        signal._timer.start()

        return signal

    def throw_if_aborted(self) -> None:
        if self._aborted:
            raise AbortError(self._reason)


class AbortController:
    def __init__(self) -> None:
        self._signal = AbortSignal()

    @property
    def signal(self) -> AbortSignal:
        return self._signal

    def abort(self, reason: str | None = None) -> None:
        self._signal._abort(reason)


def register_signals(controller: AbortController, signals: list[AbortSignal]) -> None:
    def trigger_abort(reason: str | None = None) -> None:
        controller.abort(reason)

    for signal in filter(lambda x: x is not None, signals):
        if signal.aborted:
            trigger_abort(signal.reason)
        signal.add_event_listener(trigger_abort)


async def abort_signal_handler(
    fn: Callable[[], Awaitable[Any]], signal: AbortSignal | None = None, on_abort: Callable[[], None] | None = None
) -> Awaitable[Any]:
    def abort_handler() -> None:
        if on_abort:
            on_abort()
        if signal:
            signal.cancel()

    if signal:
        if signal.aborted:
            raise AbortError(signal.reason)
        else:
            signal.add_event_listener(abort_handler)

    try:
        return await fn()
    finally:
        if signal:
            signal.remove_event_listener(abort_handler)
