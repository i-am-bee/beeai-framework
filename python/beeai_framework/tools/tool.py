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

from beeai_framework.tools.errors import ToolInputValidationError
from beeai_framework.utils import BeeLogger

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
    def _run(self, input: Any, options: dict[str, Any] | None = None) -> Any:
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

    def run(self, input: T | dict[str, Any], options: dict[str, Any] | None = None) -> Any:
        return self._run(self.validate_input(input), options)


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

        def _run(self, tool_in: Any, _: dict[str, Any] | None = None) -> None:
            tool_input_dict = tool_in.model_dump()
            return tool_function(**tool_input_dict)

    f_tool = FunctionTool()
    return f_tool
