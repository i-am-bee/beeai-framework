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

import asyncio
import functools
import inspect
import types
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

StateType = TypeVar("StateType", bound=BaseModel)
# TODO: Dont know how to type self
F = TypeVar("F", bound=Callable[[Any], Coroutine[Any, Any, None]])
# TODO: Dont know how to type self
F_FIN = TypeVar("F_FIN", bound=Callable[[Any, list[Any]], Coroutine[Any, Any, None]])
# TODO: Dont know how to type self
F_ROUTER = TypeVar("F_ROUTER", bound=Callable[[Any], Coroutine[Any, Any, str]])


def start(func: F) -> F:
    func._is_start = True  # type: ignore[attr-defined]
    return func


def listen(*func_names: str) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        if func.__name__ in func_names:
            raise ValueError(f"Method '{func.__name__}' cannot listen to itself.")
        func._listens_to = set(func_names)  # type: ignore[attr-defined]
        return func

    return decorator


def fan_in(func_name: str) -> Callable[[F_FIN], F_FIN]:
    def decorator(func: F_FIN) -> F_FIN:
        if func.__name__ == func_name:
            raise ValueError(f"Method '{func.__name__}' cannot listen to itself.")
        func._listens_to = {func_name}  # type: ignore[attr-defined]
        return func

    return decorator


def router(*func_names: str) -> Callable[[F_ROUTER], F_ROUTER]:
    def decorator(func: F_ROUTER) -> F_ROUTER:
        if func.__name__ in func_names:
            raise ValueError(f"Method '{func.__name__}' cannot listen to itself.")
        func._listens_to = set(func_names)  # type: ignore[attr-defined]
        func._is_router = True  # type: ignore[attr-defined]
        return func

    return decorator


class Workflow(Generic[StateType]):
    def __init__(self, max_concurrency: int = 20) -> None:
        self._listeners: dict[str, set[str]] = {}
        self._fired_sources_map: dict[str, set[str]] = defaultdict(set)
        self._start_method: str | None = None
        subclass = self.__class__  # Get the subclass

        # initialize static listeners from static decorator metadata
        for name, method in subclass.__dict__.items():
            if inspect.iscoroutinefunction(method):
                if hasattr(method, "_listens_to"):
                    for dependency in method._listens_to:
                        self._listeners.setdefault(dependency, set()).add(name)
                if hasattr(method, "_is_start"):
                    if self._start_method is not None:
                        raise RuntimeError("Only one @start method is allowed per class.")
                    self._start_method = name

        # Wrap static methods
        self._wrap_all_listeners()
        self._max_concurrency = max_concurrency

    def _wrap_all_listeners(self) -> None:
        for method_name in self._listeners:
            if hasattr(self, method_name):  # Only wrap if method actually exists
                self._wrap_method(method_name)

    def _wrap_method(self, method_name: str) -> None:
        orig = getattr(self, method_name)

        @functools.wraps(orig)
        async def wrapped(self: "Workflow[StateType]", *args: tuple[Any, ...], **kwargs: dict[str, Any]) -> Any:
            result = await orig(*args, **kwargs)
            await self._run_listeners(method_name)
            return result

        setattr(self, method_name, types.MethodType(wrapped, self))

    def _ready_to_run(self, listener_name: str, triggered_method: str) -> bool:
        listener_func = getattr(self, listener_name).__func__
        sources = self._fired_sources_map[listener_name]
        sources.add(triggered_method)

        if sources == listener_func._listens_to:
            self._fired_sources_map[listener_name].clear()
            return True
        return False

    async def _run_listeners(self, triggered_method: str) -> None:
        listeners = [
            getattr(self, listener_name)()
            for listener_name in self._listeners.get(triggered_method, [])
            if self._ready_to_run(listener_name, triggered_method)
        ]

        results = await asyncio.gather(*listeners)
        routes = [r for r in results if r is not None]

        for r in routes:
            await self._run_listeners(r)

    async def fan_out(self, method_name: str, args_list: list[Any], fan_in: str) -> None:
        sem = asyncio.Semaphore(self._max_concurrency)

        async def run_one(args: Any) -> Any:
            async with sem:
                method = getattr(self, method_name)
                if not isinstance(args, list | tuple):
                    args = [args]
                return await method(*args)

        results: list[Any] = await asyncio.gather(*[run_one(args) for args in args_list], return_exceptions=True)

        # Call fan_in directly, bypass static listeners
        listeners = [
            getattr(self, listener_name)(results)
            for listener_name in self._listeners.get(fan_in, [])
            if self._ready_to_run(listener_name, fan_in)
        ]

        await asyncio.gather(*listeners)

    async def run(self, state: StateType) -> None:
        if self._start_method is None:
            raise RuntimeError("No @start method defined.")

        self.state = state

        await getattr(self, self._start_method)()
