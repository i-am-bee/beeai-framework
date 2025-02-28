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
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Generic, ParamSpec, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, InstanceOf

from beeai_framework.emitter.errors import EmitterError
from beeai_framework.emitter.types import EmitterOptions, EventTrace
from beeai_framework.emitter.utils import (
    assert_valid_name,
    assert_valid_namespace,
)
from beeai_framework.utils.types import MaybeAsync

T = TypeVar("T", bound=BaseModel)
P = ParamSpec("P")

MatcherFn: TypeAlias = Callable[["EventMeta"], bool]
Matcher: TypeAlias = str | MatcherFn
Callback: TypeAlias = MaybeAsync[[P, "EventMeta"], None]
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


class Emitter(Generic[T]):
    def __init__(
        self,
        group_id: str | None = None,
        namespace: list[str] | None = None,
        creator: object | None = None,
        context: object | None = None,
        trace: EventTrace | None = None,
    ) -> None:
        super().__init__()

        self.listeners: set[Listener] = set()
        self.group_id: str | None = group_id
        self.namespace: list[str] = namespace or []
        self.creator: object | None = creator
        self.context: object = context or {}
        self.trace: EventTrace | None = trace
        self.cleanups: list[CleanupFn] = []

        assert_valid_namespace(self.namespace)

    @staticmethod
    @functools.cache
    def root() -> "Emitter":
        return Emitter(creator=object())

    def child(
        self,
        group_id: str | None = None,
        namespace: list[str] | None = None,
        creator: object | None = None,
        context: object | None = None,
        trace: EventTrace | None = None,
    ) -> "Emitter":
        child_emitter = Emitter(
            trace=trace or self.trace,
            group_id=group_id or self.group_id,
            context={**self.context, **(context or {})},
            creator=creator or self.creator,
            namespace=namespace + self.namespace if namespace else self.namespace[:],
        )

        cleanup = child_emitter.pipe(self)
        self.cleanups.append(cleanup)

        return child_emitter

    def pipe(self, target: "Emitter") -> CleanupFn:
        return self.on(
            "*.*",
            target.invoke,
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
            match_nested = options.match_nested if options else False

            if matcher == "*":
                match_nested = False if match_nested is None else match_nested
                matchers.append(lambda event: event.path == ".".join([*self.namespace, event.name]))
            elif matcher == "*.*":
                match_nested = True if match_nested is None else match_nested
                matchers.append(lambda _: True)
            # TODO is_valid_regex matches on all strings, not just regex patterns
            # elif is_valid_regex(matcher):
            #     match_nested = True if match_nested is None else match_nested
            #     matchers.append(lambda event: bool(re.search(matcher, event.path)))
            elif isinstance(matcher, Callable):
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
                    return self.trace is None or self.trace.run_id == event.trace.run_id

                matchers.insert(0, match_same_run)

            return lambda event: all(match_fn(event) for match_fn in matchers)

        listener = Listener(match=create_matcher(), raw=matcher, callback=callback, options=options)
        self.listeners.add(listener)

        return lambda: self.listeners.remove(listener)

    async def emit(self, name: str, value: Any) -> None:
        assert_valid_name(name)

        event = self.create_event(name)
        await self.invoke(value, event)

    async def invoke(self, data: Any, event: EventMeta) -> None:
        executions = []
        for listener in self.listeners:
            if not listener.match(event):
                continue

            if listener.options and listener.options.once:
                self.listeners.remove(listener)

            async def run(ln: Listener = listener) -> Any:
                if inspect.iscoroutinefunction(ln.callback):
                    return await ln.callback(data, event)
                else:
                    return ln.callback(data, event)

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
        )
