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

from abc import ABC, abstractmethod
from typing import Any, Generic

from pydantic import BaseModel
from typing_extensions import TypeVar

from beeai_framework.utils import AbortSignal
from beeai_framework.utils.strings import to_json


class RetryOptions(BaseModel):
    max_retries: int | None = None
    factor: int | None = None


class ToolRunOptions(BaseModel):
    retry_options: RetryOptions | None = None
    signal: AbortSignal | None = None


T = TypeVar("T", default=Any)


class ToolOutput(ABC, BaseModel, Generic[T]):
    result: T

    def __init__(self, result: T, **kwargs: Any) -> None:
        super().__init__(result=result, **kwargs)

    @abstractmethod
    def get_text_content(self) -> str:
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass

    def __str__(self) -> str:
        return self.get_text_content()


class StringToolOutput(ToolOutput[str]):
    def __init__(self, result: str = "") -> None:
        super().__init__(result)

    def is_empty(self) -> bool:
        return len(self.result) == 0

    def get_text_content(self) -> str:
        return self.result


class JSONToolOutput(ToolOutput[Any]):
    def get_text_content(self) -> str:
        return to_json(self.result)

    def is_empty(self) -> bool:
        return not self.result
