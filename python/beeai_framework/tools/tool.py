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
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from beeai_framework.context import Run, RunContext, RunContextInput, RunInstance
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.retryable import Retryable, RetryableConfig, RetryableContext, RetryableInput
from beeai_framework.tools.errors import ToolError, ToolInputValidationError
from beeai_framework.utils import BeeLogger
from beeai_framework.utils.strings import to_safe_word

logger = BeeLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ToolOutput(ABC):
    @abstractmethod
    def get_text_content(self) -> str:
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass

    def to_string(self) -> str:
        return self.get_text_content()


class StringToolOutput(ToolOutput):
    def __init__(self, result: str = "") -> None:
        super().__init__()
        self.result = result

    def is_empty(self) -> bool:
        return len(self.result) == 0

    def get_text_content(self) -> str:
        return self.result


class Tool(Generic[T], ABC):
    options: dict[str, Any]

    emitter: Emitter

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        if options is None:
            options = {}
        self.options = options

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
    def input_schema(self) -> type[T]:
        pass

    @abstractmethod
    async def _run(self, input: Any, options: dict[str, Any] | None = None) -> Any:
        pass

    def validate_input(self, input: T | dict[str, Any]) -> T:
        try:
            return self.input_schema.model_validate(input)
        except ValidationError as e:
            raise ToolInputValidationError("Tool input validation error") from e

    def prompt_data(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": str(self.input_schema.model_json_schema(mode="serialization")),
        }

    def run(self, input: T | dict[str, Any], options: dict[str, Any] | None = None) -> Run[Any]:
        async def run_tool(context: RunContext) -> Any:
            error_propagated = False

            try:
                validated_input = self.validate_input(input)

                meta = {"input": validated_input, "options": options}

                async def executor(_: RetryableContext) -> Any:
                    nonlocal error_propagated
                    error_propagated = False
                    await context.emitter.emit("start", meta)
                    return await self._run(validated_input, options)

                async def on_error(error: Exception, _: RetryableContext) -> None:
                    nonlocal error_propagated
                    error_propagated = True
                    err = FrameworkError.ensure(error)
                    await context.emitter.emit("error", {"error": err, **meta})
                    if err.is_fatal:
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
                            max_retries=options.get("max_retries") if options else 1, signal=context.signal
                        ),
                    )
                ).get()

                await context.emitter.emit("success", {"output": output, **meta})
                return output
            except Exception as e:
                err = ToolError.ensure(e)
                if not error_propagated:
                    await context.emitter.emit("error", {"error": err, "input": input, "options": options})
                raise err from None
            finally:
                await context.emitter.emit("finish", None)

        return RunContext.enter(
            RunInstance(emitter=self.emitter),
            RunContextInput(params=[input, options], signal=options.signal if options else None),
            run_tool,
        )


# this method was inspired by the discussion that was had in this issue:
# https://github.com/pydantic/pydantic/issues/1391
def get_input_schema(tool_function: Callable) -> BaseModel:
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

    class FunctionTool(Tool):
        name = tool_name
        description = tool_description
        input_schema = tool_input

        def __init__(self, options: dict[str, Any] | None = None) -> None:
            super().__init__(options)
            self.emitter = Emitter.root().child(
                namespace=["tool", "custom", to_safe_word(self.name)],
                creator=self,
            )

        async def _run(self, tool_in: Any, _: dict[str, Any] | None = None) -> None:
            tool_input_dict = tool_in.model_dump()
            if inspect.iscoroutinefunction(tool_function):
                return await tool_function(**tool_input_dict)
            else:
                return tool_function(**tool_input_dict)

    f_tool = FunctionTool()
    return f_tool
