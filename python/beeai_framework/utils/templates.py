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
from pydantic import BaseModel

from beeai_framework.utils.errors import PromptTemplateError
from beeai_framework.utils.models import ModelLike, to_model


class Prompt(TypedDict):
    prompt: str | None


T = TypeVar("T", bound=BaseModel)


class PromptTemplate(Generic[T]):
    def __init__(self, schema: type[T], template: str, functions: dict[str, Callable[[], str]] | None = None) -> None:
        self._schema: type[T] = schema
        self._template: str = template
        self._functions: dict[str, Callable[[], str]] | None = functions

    def render(self, input: ModelLike[T]) -> str:
        data = to_model(self._schema, input).model_dump()

        # Apply function derived data
        if self._functions:
            for key in self._functions:
                if key in data:
                    raise PromptTemplateError(f"Function named '{key}' clashes with input data field!")
                data[key] = self._functions[key]()

        return chevron.render(template=self._template, data=data)
