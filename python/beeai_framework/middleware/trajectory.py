# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
import sys
from collections.abc import Callable
from functools import cached_property
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from beeai_framework.agents import BaseAgent
from beeai_framework.agents.requirement.requirements import Requirement
from beeai_framework.backend import AnyMessage, ChatModel
from beeai_framework.context import (
    RunContext,
    RunContextFinishEvent,
    RunContextStartEvent,
    RunContextSuccessEvent,
    RunMiddlewareProtocol,
)
from beeai_framework.emitter import Emitter, EmitterOptions, EventMeta
from beeai_framework.emitter.utils import create_internal_event_matcher
from beeai_framework.errors import FrameworkError
from beeai_framework.logger import Logger
from beeai_framework.tools import Tool
from beeai_framework.utils.strings import to_json


@runtime_checkable
class Writeable(Protocol):
    def write(self, s: str) -> int: ...


class TraceLevel(BaseModel):
    """Information about how deep the given entity is in the execution tree."""

    relative: int = Field(0, ge=0, description="Relative depth to the included (observed) elements.")
    absolute: int = Field(0, ge=0, description="Absolute depth from the root.")


class GlobalTrajectoryMiddleware(RunMiddlewareProtocol):
    def __init__(
        self,
        *,
        target: Writeable | Logger | bool | None = None,
        included: list[type] | None = None,
        excluded: list[type] | None = None,
        pretty: bool = False,
        prefix_by_type: dict[type, str] | None = None,
        exclude_none: bool = True,
        enabled: bool = True,
        match_nested: bool = True,
        emitter_priority: int | None = None,
        formatter: Callable[["GlobalTrajectoryMiddlewareFormatterInput"], str] | None = None,
    ) -> None:
        """
        Args:
            target: Specify a file or stream to write the trajectory to.
            included: List of classes to include in the trajectory.
            excluded: List of classes to exclude from the trajectory.
            pretty: Use pretty formatting for the trajectory.
            prefix_by_type: Customize how instances of individual classes should be printed.
            exclude_none: Exclude None values from the printing.
            enabled: Enable/Disable the logging.
            match_nested: Whether to observe trajectories of nested run contexts.
            emitter_priority: Defines a priority for registered events.
                Setting higher priority may result in capturing events without any modifications from other middlewares.
        """
        super().__init__()
        self.enabled = enabled
        self._included = included or []
        self._excluded = excluded or []
        self._cleanups: list[Callable[[], None]] = []
        self._target = _create_target(target)
        self._ctx: RunContext | None = None
        self._pretty = pretty
        self._last_message: AnyMessage | None = None
        self._trace_level: dict[str, TraceLevel] = {}
        self._prefix_by_type = {BaseAgent: "🤖 ", ChatModel: "💬 ", Tool: "🛠️ ", Requirement: "🔎 "} | (
            prefix_by_type or {}
        )
        self._exclude_none = exclude_none
        self._match_nested = match_nested
        self._emitter_priority = emitter_priority if emitter_priority is not None else -1  # run later
        self._formatter = formatter or (
            lambda x: f"{x.prefix}{x.class_name}[{x.instance_name or x.class_name}][{x.event_name}]"
        )

    @cached_property
    def emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["middleware", "global_trajectory"], events=global_trajectory_middleware_events
        )

    def _bind_emitter(self, emitter: Emitter) -> None:
        self._cleanups.append(
            emitter.on("*.*", lambda _, event: self._log_trace_id(event), EmitterOptions(match_nested=True))
        )

        @emitter.on(
            create_internal_event_matcher(["start", "success", "error", "finish"]),
            options=EmitterOptions(match_nested=False, is_blocking=True, priority=self._emitter_priority),
        )
        async def handle_top_level_event(data: Any, meta: EventMeta) -> None:
            assert isinstance(meta.creator, RunContext)
            assert meta.trace

            self._log_trace_id(meta)
            if not self._is_allowed(meta):
                return

            if not self.enabled:
                return

            internal_handler = getattr(self, f"on_internal_{meta.name}")
            await internal_handler(data, meta)

        self._cleanups.append(lambda: emitter.off(callback=handle_top_level_event))

        if self._match_nested:

            @emitter.on(
                create_internal_event_matcher("start"),
                options=EmitterOptions(match_nested=True, is_blocking=True, priority=self._emitter_priority),
            )
            async def handle_nested_event(data: Any, meta: EventMeta) -> None:
                assert isinstance(meta.creator, RunContext)
                if meta.creator.emitter is not emitter:
                    await handle_top_level_event(data, meta)  # type: ignore
                    self._bind_emitter(meta.creator.emitter)

            self._cleanups.append(lambda: emitter.off(callback=handle_nested_event))

    def bind(self, ctx: RunContext) -> None:
        while self._cleanups:
            self._cleanups.pop(0)()

        self._trace_level.clear()
        self._trace_level[ctx.run_id] = TraceLevel()
        self._ctx = ctx

        self._bind_emitter(ctx.emitter)

    def _log_trace_id(self, meta: EventMeta) -> None:
        if not meta.trace or not meta.trace.run_id:
            return

        if meta.trace.run_id in self._trace_level:
            return

        if meta.trace.parent_run_id is meta.trace.run_id:
            return

        if meta.trace.parent_run_id:
            allowed = self._is_allowed(meta)
            parent_trace = self._trace_level.get(meta.trace.parent_run_id, TraceLevel())
            self._trace_level[meta.trace.run_id] = TraceLevel(
                relative=parent_trace.relative + (1 if allowed else 0), absolute=parent_trace.absolute + 1
            )

    def _is_allowed(self, meta: EventMeta) -> bool:
        target: object = meta.creator
        if isinstance(target, RunContext):
            target = target.instance

        for excluded in self._excluded:
            if isinstance(target, excluded):
                return False

        if not self._included:
            return True

        return any(isinstance(target, included) for included in self._included)

    def _extract_name(self, meta: EventMeta) -> str:
        target: object = meta.creator
        if isinstance(target, RunContext):
            target = target.instance

        class_name = type(target).__name__
        target_name = (
            target.meta.name if isinstance(target, BaseAgent) else target.name if hasattr(target, "name") else None
        )
        prefix = next((v for k, v in self._prefix_by_type.items() if isinstance(target, k)), "")

        input = GlobalTrajectoryMiddlewareFormatterInput(
            prefix=prefix,
            class_name=class_name,
            instance_name=target_name,
            event_name=meta.name,
        )
        return self._formatter(input)

    def _format_prefix(self, meta: EventMeta) -> str:
        assert meta.trace
        indent = self._get_trace_level(meta).relative
        indent_parent = self._get_trace_level(meta, type="parent").relative
        indent_diff = indent - indent_parent

        prefix = ""
        prefix += "  " * (indent_parent * 2)

        if meta.name != "start" and indent:
            prefix += "<"

        prefix += "--" * indent_diff

        if meta.name == "start" and prefix and indent:
            prefix += ">"

        if prefix:
            prefix = f"{prefix} "

        name = self._extract_name(meta)
        return f"{prefix}{name}: "

    def _get_trace_level(self, meta: EventMeta, *, type: Literal["self", "parent"] = "self") -> TraceLevel:
        assert meta.trace
        run_id = meta.trace.parent_run_id or "" if type == "parent" else meta.trace.run_id
        return self._trace_level.get(run_id, TraceLevel())

    def _format_payload(self, value: Any) -> str:
        if isinstance(value, str | int | bool | float | None):
            return str(value)

        if isinstance(value, FrameworkError):
            return value.explain()

        return to_json(value, indent=2 if self._pretty else None, sort_keys=False, exclude_none=self._exclude_none)

    async def on_internal_start(self, payload: RunContextStartEvent, meta: EventMeta) -> None:
        prefix = self._format_prefix(meta)
        message = f"{prefix}{self._format_payload(payload)}"

        await self.emitter.emit(
            "start",
            GlobalTrajectoryMiddlewareStartEvent(
                message=message, level=self._get_trace_level(meta), origin=(payload, meta)
            ),
        )
        self._target.write(f"{message}\n")

    async def on_internal_success(self, payload: RunContextSuccessEvent, meta: EventMeta) -> None:
        prefix = self._format_prefix(meta)
        message = f"{prefix}{self._format_payload(payload.output)}"

        await self.emitter.emit(
            "success",
            GlobalTrajectoryMiddlewareSuccessEvent(
                message=message, level=self._get_trace_level(meta), origin=(payload, meta)
            ),
        )
        self._target.write(f"{message}\n")

    async def on_internal_error(self, payload: FrameworkError, meta: EventMeta) -> None:
        prefix = self._format_prefix(meta)
        message = f"{prefix}{self._format_payload(payload)}"

        await self.emitter.emit(
            "error",
            GlobalTrajectoryMiddlewareErrorEvent(
                message=message, level=self._get_trace_level(meta), origin=(payload, meta)
            ),
        )
        self._target.write(f"{message}\n")

    async def on_internal_finish(self, payload: RunContextFinishEvent, meta: EventMeta) -> None:
        prefix = self._format_prefix(meta)
        message = f"{prefix}{self._format_payload(payload.error or payload.output)}"

        await self.emitter.emit(
            "finish",
            GlobalTrajectoryMiddlewareFinishEvent(
                message=message, level=self._get_trace_level(meta), origin=(payload, meta)
            ),
        )


class GlobalTrajectoryMiddlewareEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    message: str
    level: TraceLevel
    origin: tuple[Any, EventMeta]


class GlobalTrajectoryMiddlewareFormatterInput(BaseModel):
    prefix: str
    class_name: str
    event_name: str
    instance_name: str | None


class GlobalTrajectoryMiddlewareStartEvent(GlobalTrajectoryMiddlewareEvent):
    origin: tuple[RunContextStartEvent, EventMeta]


class GlobalTrajectoryMiddlewareSuccessEvent(GlobalTrajectoryMiddlewareEvent):
    origin: tuple[RunContextSuccessEvent, EventMeta]


class GlobalTrajectoryMiddlewareErrorEvent(GlobalTrajectoryMiddlewareEvent):
    origin: tuple[FrameworkError, EventMeta]


class GlobalTrajectoryMiddlewareFinishEvent(GlobalTrajectoryMiddlewareEvent):
    origin: tuple[RunContextFinishEvent, EventMeta]


global_trajectory_middleware_events: dict[str, type] = {
    "start": GlobalTrajectoryMiddlewareStartEvent,
    "success": GlobalTrajectoryMiddlewareSuccessEvent,
    "error": GlobalTrajectoryMiddlewareErrorEvent,
    "finish": GlobalTrajectoryMiddlewareFinishEvent,
}


def _logger_to_writeable(logger: Logger) -> Writeable:
    class CustomWriteable(Writeable):
        def write(self, s: str) -> int:
            msg = s.removesuffix("\n")
            logger.log(msg=msg, level=logger.level)
            return len(msg)

    return CustomWriteable()


def _create_target(input: Writeable | Logger | bool | None) -> Writeable:
    if input is None or input is True:
        # pyrefly: ignore [bad-return]
        return sys.stdout
    elif input is False:

        class NullWriteable(Writeable):
            def write(self, s: str) -> int:
                return len(s)

        return NullWriteable()
    elif isinstance(input, Logger):
        return _logger_to_writeable(input)
    else:
        return input
