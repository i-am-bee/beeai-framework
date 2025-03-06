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
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from typing import Any, Generic, TypeAlias

from pydantic import BaseModel, ConfigDict, ValidationError, create_model
from typing_extensions import TypeVar

from beeai_framework.cancellation import AbortSignal
from beeai_framework.context import Run, RunContext, RunContextInput, RunInstance
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.logger import Logger
from beeai_framework.retryable import Retryable, RetryableConfig, RetryableContext, RetryableInput
from beeai_framework.tools.errors import ToolError, ToolInputValidationError
from beeai_framework.utils.strings import to_json, to_safe_word

logger = Logger(__name__)

IN = TypeVar("IN", bound=BaseModel)


class RetryOptions(BaseModel):
    max_retries: int | None = None
    factor: int | None = None


class ToolRunOptions(BaseModel):
    retry_options: RetryOptions | None = None
    signal: AbortSignal | None = None


OPT = TypeVar("OPT", bound=ToolRunOptions, default=ToolRunOptions)


class ToolOutput(ABC):
    @abstractmethod
    def get_text_content(self) -> str:
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass

    def __str__(self) -> str:
        return self.get_text_content()


OUT = TypeVar("OUT", bound=ToolOutput, default=ToolOutput)


class StringToolOutput(ToolOutput):
    def __init__(self, result: str = "") -> None:
        super().__init__()
        self.result = result

    def is_empty(self) -> bool:
        return len(self.result) == 0

    def get_text_content(self) -> str:
        return self.result


class JSONToolOutput(ToolOutput):
    def __init__(self, result: Any) -> None:
        self.result = result

    def get_text_content(self) -> str:
        return to_json(self.result)

    def is_empty(self) -> bool:
        return not self.result


class Tool(Generic[IN, OPT, OUT], ABC):
    def __init__(self, options: dict[str, Any] | None = None) -> None:
        self.options: dict[str, Any] | None = options or None

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
    def input_schema(self) -> type[IN]:
        pass

    @cached_property
    def emitter(self) -> Emitter:
        return self._create_emitter()

    @abstractmethod
    def _create_emitter(self) -> Emitter:
        pass

    @abstractmethod
    async def _run(self, input: IN, options: OPT | None, context: RunContext) -> OUT:
        pass

    def validate_input(self, input: IN | dict[str, Any]) -> IN:
        try:
            return self.input_schema.model_validate(input)
        except ValidationError as e:
            raise ToolInputValidationError("Tool input validation error", cause=e)

    def run(self, input: IN | dict[str, Any], options: OPT | None = None) -> Run[OUT]:
        async def run_tool(context: RunContext) -> OUT:
            error_propagated = False

            try:
                validated_input = self.validate_input(input)

                meta = {"input": validated_input, "options": options}

                async def executor(_: RetryableContext) -> Any:
                    nonlocal error_propagated
                    error_propagated = False
                    await context.emitter.emit("start", meta)
                    return await self._run(validated_input, options, context)

                async def on_error(error: Exception, _: RetryableContext) -> None:
                    nonlocal error_propagated
                    error_propagated = True
                    err = ToolError.ensure(error)
                    await context.emitter.emit("error", {"error": err, **meta})
                    if err.is_fatal is True:
                        raise err from None

                async def on_retry(ctx: RetryableContext, last_error: Exception) -> None:
                    err = ToolError.ensure(last_error)
                    await context.emitter.emit("retry", {"error": err, **meta})

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

                await context.emitter.emit("success", {"output": output, **meta})
                return output
            except Exception as e:
                err = ToolError.ensure(e)
                if not error_propagated:
                    await context.emitter.emit("error", {"error": err, "input": input, "options": options})
                raise err
            finally:
                await context.emitter.emit("finish", None)

        return RunContext.enter(
            RunInstance(emitter=self.emitter),
            RunContextInput(params=[input, options], signal=options.signal if options else None),
            run_tool,
        )


# this method was inspired by the discussion that was had in this issue:
# https://github.com/pydantic/pydantic/issues/1391
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


def tool(tool_function: Callable) -> Tool:
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


AnyTool: TypeAlias = Tool[BaseModel, ToolRunOptions, ToolOutput]
