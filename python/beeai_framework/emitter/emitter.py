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

import asyncio
import copy
import functools
import inspect
import re
import uuid
from asyncio import Task
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import Any, TypeAlias

from pydantic import BaseModel, ConfigDict, InstanceOf

from beeai_framework.emitter.errors import EmitterError
from beeai_framework.emitter.types import EmitterOptions, EventTrace
from beeai_framework.emitter.utils import (
    assert_valid_name,
    assert_valid_namespace,
)
from beeai_framework.utils.types import MaybeAsync

MatcherFn: TypeAlias = Callable[["EventMeta"], bool]
Matcher: TypeAlias = str | re.Pattern[str] | MatcherFn
Callback: TypeAlias = MaybeAsync[[Any, "EventMeta"], None]
CleanupFn: TypeAlias = Callable[[], None]


class Listener(BaseModel):
    match: MatcherFn
    raw: Matcher
    callback: Callback
    options: InstanceOf[EmitterOptions] | None = None

    model_config = ConfigDict(frozen=True)


class EventMeta(BaseModel):
    id: str
    name: str
    path: str
    created_at: datetime
    source: InstanceOf["Emitter"]
    creator: object
    context: object
    group_id: str | None = None
    trace: InstanceOf[EventTrace] | None = None
    data_type: type


class Emitter:
    def __init__(
        self,
        group_id: str | None = None,
        namespace: list[str] | None = None,
        creator: object | None = None,
        context: dict[Any, Any] | None = None,
        trace: EventTrace | None = None,
        events: dict[str, type] | None = None,
    ) -> None:
        super().__init__()

        self.listeners: set[Listener] = set()
        self.group_id: str | None = group_id
        self.namespace: list[str] = namespace or []
        self.creator: object | None = creator
        self.context: dict[Any, Any] = context or {}
        self.trace: EventTrace | None = trace
        self.cleanups: list[CleanupFn] = []
        self._events: dict[str, type] = events or {}

        assert_valid_namespace(self.namespace)

    @property
    def events(self) -> dict[str, type]:
        return self._events.copy()

    @events.setter
    def events(self, new_events: dict[str, type]) -> None:
        self._events.update(new_events)

    @staticmethod
    @functools.cache
    def root() -> "Emitter":
        return Emitter(creator=object())

    def child(
        self,
        group_id: str | None = None,
        namespace: list[str] | None = None,
        creator: object | None = None,
        context: dict[Any, Any] | None = None,
        trace: EventTrace | None = None,
        events: dict[str, type] | None = None,
    ) -> "Emitter":
        child_emitter = Emitter(
            trace=trace or self.trace,
            group_id=group_id or self.group_id,
            context={**self.context, **(context or {})},
            creator=creator or self.creator,
            namespace=namespace + self.namespace if namespace else self.namespace[:],
            events=events or self.events,
        )

        cleanup = child_emitter.pipe(self)
        self.cleanups.append(cleanup)

        return child_emitter

    def pipe(self, target: "Emitter") -> CleanupFn:
        return self.on(
            "*.*",
            target._invoke,
            EmitterOptions(
                is_blocking=True,
                once=False,
                persistent=True,
            ),
        )

    def destroy(self) -> None:
        self.listeners.clear()
        for cleanup in self.cleanups:
            cleanup()
        self.cleanups.clear()

    def on(self, event: str, callback: Callback, options: EmitterOptions | None = None) -> CleanupFn:
        return self.match(event, callback, options)

    def match(self, matcher: Matcher, callback: Callback, options: EmitterOptions | None = None) -> CleanupFn:
        def create_matcher() -> MatcherFn:
            matchers: list[MatcherFn] = []
            match_nested = options.match_nested if options else None

            if matcher == "*":
                match_nested = False if match_nested is None else match_nested
                matchers.append(lambda event: event.path == ".".join([*self.namespace, event.name]))
            elif matcher == "*.*":
                match_nested = True if match_nested is None else match_nested
                matchers.append(lambda _: True)
            elif isinstance(matcher, re.Pattern):
                match_nested = True if match_nested is None else match_nested
                matchers.append(lambda event: matcher.match(event.path) is not None)
            elif callable(matcher):
                match_nested = False if match_nested is None else match_nested
                matchers.append(matcher)
            elif isinstance(matcher, str):
                if "." in matcher:
                    match_nested = True if match_nested is None else match_nested
                    matchers.append(lambda event: event.path == matcher)
                else:
                    match_nested = False if match_nested is None else match_nested
                    matchers.append(
                        lambda event: event.name == matcher and event.path == ".".join([*self.namespace, event.name])
                    )
            else:
                raise EmitterError("Invalid matcher provided!")

            if not match_nested:

                def match_same_run(event: EventMeta) -> bool:
                    return self.trace is None or (
                        self.trace.run_id == event.trace.run_id if event.trace is not None else False
                    )

                matchers.insert(0, match_same_run)

            return lambda event: all(match_fn(event) for match_fn in matchers)

        listener = Listener(match=create_matcher(), raw=matcher, callback=callback, options=options)
        self.listeners.add(listener)

        return lambda: self.listeners.remove(listener)

    async def emit(self, name: str, value: Any) -> None:
        try:
            assert_valid_name(name)
            event = self.create_event(name)
            await self._invoke(value, event)
        except Exception as e:
            raise EmitterError.ensure(e)

    async def _invoke(self, data: Any, event: EventMeta) -> None:
        executions: list[Coroutine[Any, Any, Any] | Task[Any]] = []
        for listener in self.listeners:
            if not listener.match(event):
                continue

            if listener.options and listener.options.once:
                self.listeners.remove(listener)

            async def run(ln: Listener = listener) -> Any:
                try:
                    if inspect.iscoroutinefunction(ln.callback):
                        return await ln.callback(data, event)
                    else:
                        return ln.callback(data, event)
                except Exception as e:
                    raise EmitterError.ensure(
                        e, message="One of the provided Emitter callbacks has failed.", event=event
                    )

            if listener.options and listener.options.is_blocking:
                executions.append(run())
            else:
                executions.append(asyncio.create_task(run()))

        await asyncio.gather(*executions)

    def create_event(self, name: str) -> EventMeta:
        return EventMeta(
            id=str(uuid.uuid4()),
            group_id=self.group_id,
            name=name,
            path=".".join([*self.namespace, name]),
            created_at=datetime.now(tz=UTC),
            source=self,
            creator=self.creator,
            context={**self.context},
            trace=copy.copy(self.trace),
            data_type=self.events.get(name) or type(Any),
        )
