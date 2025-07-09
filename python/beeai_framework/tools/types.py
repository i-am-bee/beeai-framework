# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

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


class ToolOutput(ABC):
    @abstractmethod
    def get_text_content(self) -> str:
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass

    def __str__(self) -> str:
        return self.get_text_content()


class StringToolOutput(ToolOutput):
    def __init__(self, result: str = "") -> None:
        super().__init__()
        self.result = result

    def is_empty(self) -> bool:
        return len(self.result) == 0

    def get_text_content(self) -> str:
        return self.result


class JSONToolOutput(ToolOutput, Generic[T]):
    def __init__(self, result: T) -> None:
        self.result = result

    def to_json_safe(self) -> Any:
        return self.result

    def get_text_content(self) -> str:
        return to_json(self.result, sort_keys=False)

    def is_empty(self) -> bool:
        return not self.result
