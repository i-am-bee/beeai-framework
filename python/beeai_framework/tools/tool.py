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


import inspect
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from typing import Any, Generic, TypeAlias

from pydantic import BaseModel, ConfigDict, ValidationError, create_model
from typing_extensions import TypeVar

from beeai_framework.cache.base import BaseCache
from beeai_framework.cache.null_cache import NullCache
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.logger import Logger
from beeai_framework.retryable import Retryable, RetryableConfig, RetryableContext, RetryableInput
from beeai_framework.tools.errors import ToolError, ToolInputValidationError
from beeai_framework.tools.events import (
    ToolErrorEvent,
    ToolRetryEvent,
    ToolStartEvent,
    ToolSuccessEvent,
    tool_event_types,
)
from beeai_framework.tools.types import StringToolOutput, ToolOutput, ToolRunOptions
from beeai_framework.utils.strings import to_safe_word

logger = Logger(__name__)

TInput = TypeVar("TInput", bound=BaseModel)
TRunOptions = TypeVar("TRunOptions", bound=ToolRunOptions, default=ToolRunOptions)
TOutput = TypeVar("TOutput", bound=ToolOutput, default=ToolOutput)


class Tool(Generic[TInput, TRunOptions, TOutput], ABC):
    def __init__(self, options: dict[str, Any] | None = None) -> None:
        self.options: dict[str, Any] | None = options or None
        self.cache = self.options.get("cache", NullCache[TOutput]()) if self.options else NullCache[TOutput]()

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def input_schema(self) -> type[TInput]:
        pass

    @cached_property
    def emitter(self) -> Emitter:
        emitter = self._create_emitter()
        emitter.events = tool_event_types
        return emitter

    @abstractmethod
    def _create_emitter(self) -> Emitter:
        pass

    @abstractmethod
    async def _run(self, input: TInput, options: TRunOptions | None, context: RunContext) -> TOutput:
        pass

    def _generate_key(self, input: TInput | dict[str, Any], options: TRunOptions | None = None) -> str:
        options_dict = options.model_dump(exclude_none=True) if options else {}
        options_dict.pop("signal", None)
        options_dict.pop("retry_options", None)
        return BaseCache.generate_key(input, options_dict)

    async def clear_cache(self) -> None:
        await self.cache.clear()

    def validate_input(self, input: TInput | dict[str, Any]) -> TInput:
        try:
            return self.input_schema.model_validate(input)
        except ValidationError as e:
            raise ToolInputValidationError("Tool input validation error", cause=e)

    def run(self, input: TInput | dict[str, Any], options: TRunOptions | None = None) -> Run[TOutput]:
        async def handler(context: RunContext) -> TOutput:
            error_propagated = False

            try:
                validated_input = self.validate_input(input)

                async def executor(_: RetryableContext) -> TOutput:
                    nonlocal error_propagated
                    error_propagated = False
                    await context.emitter.emit("start", ToolStartEvent(input=validated_input, options=options))

                    if self.cache.enabled:
                        cache_key = self._generate_key(input, options)
                        result = await self.cache.get(cache_key)
                        if result:
                            return result

                    result = await self._run(validated_input, options, context)
                    if self.cache.enabled:
                        await self.cache.set(cache_key, result)

                    return result

                async def on_error(error: Exception, _: RetryableContext) -> None:
                    nonlocal error_propagated
                    error_propagated = True
                    err = ToolError.ensure(error)
                    await context.emitter.emit(
                        "error", ToolErrorEvent(error=err, input=validated_input, options=options)
                    )
                    if FrameworkError.is_fatal(err) is True:
                        raise err

                async def on_retry(ctx: RetryableContext, last_error: Exception) -> None:
                    err = ToolError.ensure(last_error)
                    await context.emitter.emit(
                        "retry", ToolRetryEvent(error=err, input=validated_input, options=options)
                    )

                output = await Retryable(
                    RetryableInput(
                        executor=executor,
                        on_error=on_error,
                        on_retry=on_retry,
                        config=RetryableConfig(
                            max_retries=(
                                (options.retry_options.max_retries or 0) if options and options.retry_options else 0
                            ),
                            factor=((options.retry_options.factor or 1) if options and options.retry_options else 1),
                            signal=context.signal,
                        ),
                    )
                ).get()

                await context.emitter.emit(
                    "success", ToolSuccessEvent(output=output, input=validated_input, options=options)
                )
                return output
            except Exception as e:
                err = ToolError.ensure(e, tool=self)
                if not error_propagated:
                    await context.emitter.emit("error", ToolErrorEvent(error=err, input=input, options=options))
                raise err
            finally:
                await context.emitter.emit("finish", None)

        return RunContext.enter(
            self,
            handler,
            signal=options.signal if options else None,
            run_params={"input": input, "options": options},
        )


# this method was inspired by the discussion that was had in this issue:
# https://github.com/pydantic/pydantic/issues/1391
@typing.no_type_check
def get_input_schema(tool_function: Callable) -> type[BaseModel]:
    input_model_name = tool_function.__name__

    args, _, _, defaults, kwonlyargs, kwonlydefaults, annotations = inspect.getfullargspec(tool_function)
    defaults = defaults or []
    args = args or []

    non_default_args = len(args) - len(defaults)
    try:
        defaults = (...,) * non_default_args + defaults
    except TypeError:
        defaults = [
            ...,
        ] * non_default_args + defaults

    keyword_only_params = {param: kwonlydefaults.get(param, Any) for param in kwonlyargs}
    params = {param: (annotations.get(param, Any), default) for param, default in zip(args, defaults, strict=False)}

    input_model = create_model(
        input_model_name,
        **params,
        **keyword_only_params,
        __config__=ConfigDict(extra="allow", arbitrary_types_allowed=True),
    )

    return input_model


def tool(tool_function: Callable[..., Any]) -> "AnyTool":
    tool_name = tool_function.__name__
    tool_description = inspect.getdoc(tool_function)
    tool_input = get_input_schema(tool_function)

    if tool_description is None:
        raise ValueError("No tool description provided.")

    class FunctionTool(Tool[Any, ToolRunOptions, ToolOutput]):
        name = tool_name
        description = tool_description or ""
        input_schema = tool_input

        def __init__(self, options: dict[str, Any] | None = None) -> None:
            super().__init__(options)

        def _create_emitter(self) -> Emitter:
            return Emitter.root().child(
                namespace=["tool", "custom", to_safe_word(self.name)],
                creator=self,
            )

        async def _run(self, input: Any, options: ToolRunOptions | None, context: RunContext) -> ToolOutput:
            tool_input_dict = input.model_dump()
            if inspect.iscoroutinefunction(tool_function):
                result = await tool_function(**tool_input_dict)
            else:
                result = tool_function(**tool_input_dict)

            if isinstance(result, ToolOutput):
                return result
            else:
                return StringToolOutput(result=str(result))

    f_tool = FunctionTool()
    return f_tool


AnyTool: TypeAlias = Tool[Any, Any, Any]
