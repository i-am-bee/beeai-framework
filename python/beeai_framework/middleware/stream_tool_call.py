# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from functools import cached_property
from typing import Any

from pydantic import BaseModel

from beeai_framework.backend import (
    ChatModel,
    ChatModelNewTokenEvent,
    ChatModelOutput,
    ChatModelStartEvent,
    ChatModelSuccessEvent,
)
from beeai_framework.backend.utils import parse_broken_json
from beeai_framework.context import RunContext, RunMiddlewareProtocol
from beeai_framework.emitter import Emitter, EmitterOptions, EventMeta
from beeai_framework.tools import AnyTool


class StreamToolCallMiddleware(RunMiddlewareProtocol):
    """
    Middleware for handling streaming tool calls in a ChatModel.

    This middleware observes, listens to Chat Model stream updates and parses the tool calls on demand so that
    they can be consumed as soon as possible.
    """

    def __init__(self, target: AnyTool, key: str, *, match_nested: bool = False, force_streaming: bool = False) -> None:
        """
        Args:
            target: The tool that we are waiting for to be called.
            key: Refers to the name of the attribute in the tool's schema that we want to stream
            match_nested: Whether the middleware should be applied only to the top level.
            force_streaming: Set's the stream flag on the ChatModel.
        """

        self._target = target
        self._key = key
        self._output = ChatModelOutput(output=[])
        self._buffer = ""
        self._delta = ""
        self._match_nested = match_nested
        self._force_streaming = force_streaming
        self._cleanups: list[Callable[[], None]] = []

    def bind(self, ctx: "RunContext") -> None:
        self._output = ChatModelOutput(output=[])
        self._buffer = ""
        self._delta = ""

        self._cleanups.append(
            ctx.instance.emitter.on(
                lambda meta: isinstance(meta.creator, ChatModel) and meta.name == "start",
                callback=self._handle_start,
                options=self._create_emitter_options(),
            )
        )
        self._cleanups.append(
            ctx.instance.emitter.on(
                lambda meta: isinstance(meta.creator, ChatModel) and meta.name == "new_token",
                callback=self._handle_new_token,
                options=self._create_emitter_options(),
            )
        )
        self._cleanups.append(
            ctx.instance.emitter.on(
                lambda meta: isinstance(meta.creator, ChatModel) and meta.name == "success",
                callback=self._handle_success,
                options=self._create_emitter_options(),
            )
        )

    async def add(self, chunk: ChatModelOutput) -> None:
        await self._handle_new_token(
            ChatModelNewTokenEvent(value=chunk, abort=lambda: None), EventMeta.model_construct()
        )

    def unbind(self) -> None:
        while self._cleanups:
            fn = self._cleanups.pop(0)
            fn()

    def _create_emitter_options(self) -> EmitterOptions:
        return EmitterOptions(match_nested=self._match_nested, is_blocking=True)

    @cached_property
    def emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["middleware", "stream_tool_call"])

    async def _process(self, tool_name: str, args: Any) -> None:
        if tool_name != self._target.name:
            return

        parsed_args = parse_broken_json(args, fallback={}, stream_stable=True) if isinstance(args, str) else args

        try:
            output_structured = self._target.input_schema.model_validate(parsed_args)
        except Exception:
            return

        if output_structured and hasattr(output_structured, self._key):  # assumption, could be parametrized
            output = getattr(output_structured, self._key) or ""
            self._delta = output[len(self._buffer) :]
            self._buffer = output
            if not self._delta:
                return
        else:
            output = ""

        await self.emitter.emit(
            "update",
            StreamToolCallMiddlewareUpdateEvent(output_structured=output_structured, delta=self._delta, output=output),
        )

    async def _handle_start(self, data: ChatModelStartEvent, meta: EventMeta) -> None:
        if self._force_streaming:
            data.input.stream = True
            data.input.stream_partial_tool_calls = True

    async def _handle_success(self, data: ChatModelSuccessEvent, meta: EventMeta) -> None:
        if self._output.is_empty():
            await self._handle_new_token(ChatModelNewTokenEvent(value=data.value, abort=lambda: None), meta)

    async def _handle_new_token(self, data: ChatModelNewTokenEvent, meta: EventMeta) -> None:
        self._output.merge(data.value)

        tool_calls = self._output.get_tool_calls()
        for tool_call in tool_calls:
            await self._process(tool_call.tool_name, tool_call.args)
        else:
            tool_call = parse_broken_json(self._output.get_text_content(), fallback={}, stream_stable=True)
            if not isinstance(tool_call, dict):
                return

            tool_call = tool_call.get("item", tool_call)  # WrappedRootModel was used
            if not isinstance(tool_call, dict):
                return

            await self._process(tool_call.get("name", ""), tool_call.get("parameters"))


class StreamToolCallMiddlewareUpdateEvent(BaseModel):
    output_structured: BaseModel | Any
    output: str
    delta: str
