# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import functools
import re
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, TypeAlias

from pydantic import BaseModel, ConfigDict, InstanceOf

from beeai_framework.emitter.errors import EmitterError
from beeai_framework.emitter.types import EmitterOptions, EventTrace
from beeai_framework.emitter.utils import (
    assert_valid_name,
    assert_valid_namespace,
)
from beeai_framework.utils.asynchronous import ensure_async
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
    context: dict[str, Any]
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

        self._listeners: set[Listener] = set()
        self._group_id: str | None = group_id
        self.namespace: list[str] = namespace or []
        self.creator: object | None = creator
        self.context: dict[Any, Any] = context or {}
        self.trace: EventTrace | None = trace
        self._cleanups: list[CleanupFn] = []
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
            group_id=group_id or self._group_id,
            context={**self.context, **(context or {})},
            creator=creator or self.creator,
            namespace=namespace + self.namespace if namespace else self.namespace[:],
            events=events or self.events,
        )

        cleanup = child_emitter.pipe(self)
        self._cleanups.append(cleanup)

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
        self._listeners.clear()
        for cleanup in self._cleanups:
            cleanup()
        self._cleanups.clear()

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
        self._listeners.add(listener)

        return lambda: self._listeners.remove(listener) if listener in self._listeners else None

    async def emit(self, name: str, value: Any) -> None:
        try:
            assert_valid_name(name)
            event = self._create_event(name)
            await self._invoke(value, event)
        except Exception as e:
            raise EmitterError.ensure(e)

    async def _invoke(self, data: Any, event: EventMeta) -> None:
        async def run(ln: Listener) -> Any:
            try:
                ln_async = ensure_async(ln.callback)
                return await ln_async(data, event)
            except Exception as e:
                raise EmitterError.ensure(
                    e,
                    message=f"One of the provided emitter callbacks has failed. Event: {event.path}",
                    event=event,
                )

        async with asyncio.TaskGroup() as tg:
            for listener in self._listeners:
                if not listener.match(event):
                    continue

                if listener.options and listener.options.once:
                    self._listeners.remove(listener)

                task = tg.create_task(run(listener))
                if listener.options and listener.options.is_blocking:
                    _ = await task

    def _create_event(self, name: str) -> EventMeta:
        return EventMeta(
            id=str(uuid.uuid4()),
            group_id=self._group_id,
            name=name,
            path=".".join([*self.namespace, name]),
            created_at=datetime.now(tz=UTC),
            source=self,
            creator=self.creator,
            context={**self.context},
            trace=copy.copy(self.trace),
            data_type=self.events.get(name) or type(Any),
        )

    async def clone(self) -> "Emitter":
        cloned = Emitter(
            str(self._group_id),
            self.namespace.copy(),
            self.creator if self.creator else None,
            self.context.copy(),
            self.trace.model_copy() if self.trace else None,
            self._events.copy(),
        )
        cloned._cleanups = self._cleanups
        cloned._listeners = {listener.model_copy() for listener in self._listeners}
        return cloned
