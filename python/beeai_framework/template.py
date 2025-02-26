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


from collections.abc import Callable
from typing import Generic, TypedDict, TypeVar

import chevron
from pydantic import BaseModel, Field

from beeai_framework.errors import FrameworkError
from beeai_framework.utils.models import ModelLike, to_model


class Prompt(TypedDict):
    prompt: str | None


T = TypeVar("T", bound=BaseModel)


class PromptTemplateInput(BaseModel, Generic[T]):
    input_schema: type[T] = Field(..., alias="schema")
    template: str
    functions: dict[str, Callable[[], str]] | None = None
    defaults: dict[str, str] | None = {}


class PromptTemplate(Generic[T]):
    def __init__(self, config: PromptTemplateInput) -> None:
        self._config = config

    def render(self, input: ModelLike[T]) -> str:
        data = to_model(self._config.input_schema, input).model_dump()

        if self._config.defaults:
            for key, value in self._config.defaults.items():
                if data.get(key) is None:
                    data.update({key: value})

        # Apply function derived data
        if self._config.functions:
            for key in self._config.functions:
                if key in data:
                    raise PromptTemplateError(f"Function named '{key}' clashes with input data field!")
                data[key] = self._config.functions[key]()

        return chevron.render(template=self._config.template, data=data)

    def fork(self, customizer: Callable[[PromptTemplateInput], "PromptTemplate"] | None) -> "PromptTemplate":
        new_config = customizer(self._config) if customizer else self._config
        if not isinstance(new_config, PromptTemplateInput):
            raise ValueError("Return type from customizer must be a PromptTemplateInput or nothing.")
        return PromptTemplate(new_config)


class PromptTemplateError(FrameworkError):
    """Raised for errors caused by PromptTemplate."""

    def __init__(self, message: str = "PromptTemplate error", *, cause: Exception | None = None) -> None:
        super().__init__(message, is_fatal=True, is_retryable=False, cause=cause)
