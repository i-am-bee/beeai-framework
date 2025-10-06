# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from functools import cached_property
from typing import Any

from pydantic import BaseModel

from beeai_framework.backend import ChatModel, ChatModelNewTokenEvent, ChatModelOutput
from beeai_framework.backend.utils import parse_broken_json
from beeai_framework.context import RunContext, RunMiddlewareProtocol
from beeai_framework.emitter import Emitter, EmitterOptions, EventMeta
from beeai_framework.tools import AnyTool


class StreamToolCallMiddleware(RunMiddlewareProtocol):
    def __init__(self, target: AnyTool, key: str, *, emitter_options: EmitterOptions | None = None) -> None:
        self._target = target
        self._key = key
        self._output = ChatModelOutput(output=[])
        self._buffer = ""
        self._delta = ""
        self._emitter_options = emitter_options

    def bind(self, ctx: "RunContext") -> None:
        self._output = ChatModelOutput(output=[])
        self._buffer = ""
        self._delta = ""

        # could be more general
        if not isinstance(ctx.instance, ChatModel):
            raise ValueError("Middleware is intended to be used with a ChatModel")

        ctx.emitter.on("new_token", self.handler, self._emitter_options)

    @cached_property
    def emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["middleware", "stream"])

    async def _process(self, tool_name: str, args: Any) -> None:
        if tool_name != self._target.name:
            return

        parsed_args = parse_broken_json(args, fallback={}) if isinstance(args, str) else args

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
            UpdateEvent(output_structured=output_structured, delta=self._delta, output=output),
        )

    async def handler(self, data: ChatModelNewTokenEvent, meta: EventMeta) -> None:
        self._output.merge(data.value)

        tool_calls = self._output.get_tool_calls()
        for tool_call in tool_calls:
            await self._process(tool_call.tool_name, tool_call.args)
        else:
            tool_call = parse_broken_json(self._output.get_text_content(), fallback={})
            if not isinstance(tool_call, dict):
                return

            await self._process(tool_call.get("name", ""), tool_call.get("parameters"))


class UpdateEvent(BaseModel):
    output_structured: BaseModel | Any
    output: str
    delta: str
