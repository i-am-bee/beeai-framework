# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import BaseModel

from beeai_framework.tools import ToolOutput
from beeai_framework.utils.strings import to_json


class SearchToolResult(BaseModel):
    title: str
    description: str
    url: str


class SearchToolOutput(ToolOutput):
    def __init__(self, results: list[SearchToolResult]) -> None:
        super().__init__()
        self.results = results

    def get_text_content(self) -> str:
        return to_json(self.results)

    def is_empty(self) -> bool:
        return len(self.results) == 0

    def sources(self) -> list[str]:
        return [result.url for result in self.results]

    def to_json_safe(self) -> list[Any]:
        return [r.model_dump() for r in self.results]
