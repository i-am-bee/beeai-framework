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
from inspect import isfunction
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from beeai_framework.agents.base import BaseAgent
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.experimental.autoflow.utils import get_input_schema, get_output_schema, get_output_schema_for_value
from beeai_framework.tools import StringToolOutput, ToolOutput
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import ToolRunOptions
from beeai_framework.utils.models import to_model
from beeai_framework.utils.strings import to_safe_word

TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)


@runtime_checkable
class Runnable(Protocol[TInput, TOutput]):
    name: str
    description: str
    input_schema: type[TInput]
    output_schema: type[TOutput]

    async def run(self, /, input: TInput) -> TOutput: ...


RunnableInput = (
    Callable[..., Awaitable[TOutput] | TOutput] | BaseAgent[TOutput] | Tool[Any, Any, Any] | Runnable[Any, TOutput]
)


def runnable(fn: RunnableInput[TOutput]) -> Runnable[Any, TOutput]:
    if isinstance(fn, Runnable):
        return fn

    class DynamicRunnable(Runnable[Any, Any]):
        def __init__(self) -> None:
            if isinstance(fn, BaseAgent):
                self.name = fn.meta.name
                self.description = fn.meta.description
                self.input_schema = get_input_schema(fn.run)
                self.output_schema = get_output_schema(fn.run)
                self.handler = lambda input: fn(**input.model_dump())  # type: ignore
            elif isinstance(fn, Tool):
                self.name = fn.name
                self.description = fn.description
                self.input_schema = fn.input_schema
                self.output_schema = get_output_schema_for_value(f"{fn.name}Output", str)
                self.handler = fn.run
            elif isfunction(fn):
                self.name = fn.__name__
                self.description = fn.__doc__ or ""
                self.input_schema = get_input_schema(fn)
                self.output_schema = get_output_schema(fn)
                self.handler = lambda input: fn(**input.model_dump())

            if not self.description:
                raise ValueError("Description is required!")

        async def run(self, /, *args: None, **kwargs: Any) -> TOutput:
            input = self.input_schema.model_validate(kwargs)
            response: TOutput = await self.handler(input)  # type: ignore
            return response

    return DynamicRunnable()


def runnable_custom(
    *,
    input_schema: type[TInput],
    output_schema: type[TOutput],
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[[TInput], Awaitable[TOutput]]], Runnable[TInput, TOutput]]:
    def wrapper(fn: Callable[[TInput], Awaitable[TOutput]]) -> Runnable[TInput, TOutput]:
        if isinstance(fn, Runnable):
            # TODO: ?
            return fn

        class DynamicRunnable(Runnable[TInput, TOutput]):
            def __init__(self) -> None:
                self.name = name or fn.__name__
                self.description = description or fn.__doc__ or ""
                self.input_schema = input_schema
                self.output_schema = output_schema

            async def run(self, *args: Any, **kwargs: Any) -> TOutput:
                return await fn(to_model(self.input_schema, kwargs))

        return DynamicRunnable()

    return wrapper


def runnable_to_tool(runnable: Runnable[Any, Any]) -> Tool[Any, Any, Any]:
    class FunctionTool(Tool[Any, Any, Any]):
        name = runnable.name
        description = runnable.description
        input_schema = runnable.input_schema

        def __init__(self, options: dict[str, Any] | None = None) -> None:
            super().__init__(options)

        def _create_emitter(self) -> Emitter:
            return Emitter.root().child(
                namespace=["tool", "custom", to_safe_word(runnable.name)],
                creator=self,
            )

        async def _run(self, input: Any, options: ToolRunOptions | None, context: RunContext) -> ToolOutput:
            tool_input_dict = input.model_dump()
            result = await runnable.run(**tool_input_dict)

            if isinstance(result, ToolOutput):
                return result
            else:
                return StringToolOutput(result=str(result))

    return FunctionTool()
