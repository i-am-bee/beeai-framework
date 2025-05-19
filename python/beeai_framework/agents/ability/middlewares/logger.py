import sys
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from beeai_framework.agents import BaseAgent
from beeai_framework.agents.ability.abilities.ability import Ability
from beeai_framework.agents.ability.agent import AbilityAgent
from beeai_framework.agents.ability.events import AbilityAgentStartEvent, AbilityAgentSuccessEvent
from beeai_framework.backend import AnyMessage, ChatModelStartEvent, ChatModelSuccessEvent
from beeai_framework.context import RunContext, RunContextFinishEvent, RunContextStartEvent, RunMiddleware
from beeai_framework.emitter import EmitterOptions, EventMeta
from beeai_framework.utils.lists import find_index
from beeai_framework.utils.strings import to_json


@runtime_checkable
class Writeable(Protocol):
    def write(self, s: str) -> int: ...
    def flush(self) -> None: ...


class AbilityAgentLoggerMiddleware(RunMiddleware):
    def __init__(
        self,
        *,
        messages: bool = False,
        ability_calls: bool = True,
        llm_calls: bool = False,
        target: Writeable | None = None,
        pretty: bool = False,
    ) -> None:
        super().__init__()
        self._cleanups: list[Callable[[], None]] = []
        self._target: Writeable = target if target is not None else sys.stdout
        self._ctx: RunContext | None = None
        self._log_messages = messages
        self._ability_calls = ability_calls
        self._llm_calls = llm_calls
        self._pretty = pretty
        self._last_message: AnyMessage | None = None
        self._trace_level: dict[str, int] = {}

    def bind(self, ctx: RunContext) -> None:
        instance = ctx.instance
        if not isinstance(instance, AbilityAgent):
            raise ValueError("Middleware can only be used with AbilityAgent")

        while self._cleanups:
            fn = self._cleanups.pop(0)
            fn()

        self._trace_level.clear()
        self._trace_level[ctx.run_id] = 0

        self._ctx = ctx

        # must be last to be executed as first
        self._cleanups.append(
            ctx.emitter.match("*.*", lambda _, event: self._log_trace_id(event), EmitterOptions(match_nested=True))
        )

        self._cleanups.append(ctx.emitter.on("start", self.on_start))
        self._cleanups.append(ctx.emitter.on("success", self.on_success))

        # if self._llm_calls:
        #    llm = instance._llm
        self._cleanups.append(
            ctx.emitter.match(
                lambda event: event.name == "start" and bool(event.context.get("internal")),
                self.on_internal_start,
                EmitterOptions(match_nested=True),
            )
        )
        self._cleanups.append(
            ctx.emitter.match(
                lambda event: event.name == "finish" and bool(event.context.get("internal")),
                self.on_internal_finish,
                EmitterOptions(match_nested=True),
            )
        )
        # self._cleanups.append(
        #    ctx.emitter.match(
        #        lambda event: event.name == "start" and isinstance(event.creator, ChatModel),
        #        self.on_llm_start,
        #        EmitterOptions(match_nested=True),
        #    )
        # )
        # self._cleanups.append(
        #    ctx.emitter.match(
        #        lambda event: event.name == "success" and isinstance(event.creator, ChatModel),
        #        self.on_llm_success,
        #        EmitterOptions(match_nested=True),
        #    ),
        # )

        # self._cleanups.append(
        #    ctx.emitter.match(
        #        create_event_matcher("start", llm), self.on_llm_start, EmitterOptions(match_nested=True)
        #    )
        # )
        # self._cleanups.append(
        #    ctx.emitter.match(
        #        create_event_matcher("success", llm), self.on_llm_success, EmitterOptions(match_nested=True)
        #    )
        # )

        # if self._ability_calls:
        #    for ability in instance._abilities:
        #        for name, method in [
        #            ("start", self.on_ability_start),
        #            ("finish", self.on_ability_finish),
        #        ]:
        #            ctx.emitter.match(
        #                create_internal_event_matcher(name, ability, parent_run_id=ctx.run_id),
        #                method,
        #                EmitterOptions(match_nested=True),
        #            )

    def _log_trace_id(self, meta: EventMeta) -> None:
        if not meta.trace or not meta.trace.run_id:
            return

        if meta.trace.run_id in self._trace_level:
            return

        if meta.trace.parent_run_id:
            parent_level = self._trace_level.get(meta.trace.parent_run_id, 0)
            self._trace_level[meta.trace.run_id] = parent_level + 1

    def _extract_name(self, meta: EventMeta) -> str:
        target: object = meta.creator
        if isinstance(target, RunContext):
            target = target.instance

        class_name = type(target).__qualname__

        if isinstance(target, BaseAgent):
            return f"{class_name}[{target.meta.name}][{meta.name}]"
        elif isinstance(target, BaseTool) or isinstance(target, Ability):
            return f"{class_name}[{target.name}][{meta.name}]"

        return f"{class_name}[{meta.name}]"

    def _get_trace_level(self, meta: EventMeta) -> tuple[int, int]:
        assert meta.trace
        indent = self._trace_level[meta.trace.run_id]
        parent_indent = self._trace_level.get(meta.trace.parent_run_id, 0)
        return indent, parent_indent

    def _write(self, text: str, meta: EventMeta) -> None:
        assert meta.trace

        self._log_trace_id(meta)
        indent, indent_parent = self._get_trace_level(meta)
        indent_diff = indent - indent_parent

        prefix = ""
        prefix += "  " * indent_parent
        if indent_parent > 0:
            prefix += "  " * indent_parent

        if meta.name == "finish":
            prefix += "<"

        prefix += "--" * indent_diff

        # if indent_diff >= 0:
        if meta.name == "start" and prefix:
            prefix += ">"
        # else:
        #    prefix = f"<-{prefix}"

        if prefix:
            prefix = f"{prefix} "

        name = self._extract_name(meta)
        self._target.write(f"{prefix}{name}: {text}\n")

    def _format_json(self, value: Any) -> str:
        if isinstance(value, BaseModel):
            return value.model_dump_json(
                indent=2 if self._pretty else None,
                fallback=lambda value: vars(value) if isinstance(value, dict) else str(value),  # TODO: improve
            )
        else:
            return to_json(value, indent=2 if self._pretty else None, sort_keys=False)

    # def on_ability_start(self, data: RunContextStartEvent, meta: EventMeta) -> None:
    #    self._write(f"calling with {data}", meta)
    #
    # def on_ability_finish(self, data: RunContextFinishEvent, meta: EventMeta) -> None:
    #    self._write(f"finish with {data}", meta)
    #    if data.error is not None:
    #        self._write(data.error.explain(), meta)

    def on_llm_start(self, data: ChatModelStartEvent, meta: EventMeta) -> None:
        # self._write("calling", meta)
        self._write(f"calling with {self._format_json(data.input)}", meta)

    def on_llm_success(self, data: ChatModelSuccessEvent, meta: EventMeta) -> None:
        self._write(f"{self._format_json(data)}", meta)

    def on_internal_start(self, data: RunContextStartEvent, meta: EventMeta) -> None:
        self._write(f"{self._format_json(data)}", meta)

    def on_internal_finish(self, data: RunContextFinishEvent, meta: EventMeta) -> None:
        if data.error is None:
            self._write(f"{self._format_json(data.output)}", meta)
        else:
            self._write("error has occurred", meta)
            self._write(data.error.explain(), meta)

    def on_start(self, data: AbilityAgentStartEvent, meta: EventMeta) -> None:
        self._process_new_messages(data.state.memory.messages, meta)

    def on_success(self, data: AbilityAgentSuccessEvent, meta: EventMeta) -> None:
        self._process_new_messages(data.state.memory.messages, meta)

    def _process_new_messages(self, messages: list[AnyMessage], meta: EventMeta) -> None:
        if not messages:
            return

        cur_index = (
            find_index(messages, lambda msg: msg is self._last_message, fallback=-1, reverse_traversal=True)
            if self._last_message is not None
            else -1
        )
        new_messages = messages[cur_index + 1 :]
        for message in new_messages:
            if self._log_messages:
                self._write(f"new message ({message.role})", meta)
                for chunk in message.content:
                    self._write(f"{self._format_json(chunk)}", meta)

            self._last_message = message
