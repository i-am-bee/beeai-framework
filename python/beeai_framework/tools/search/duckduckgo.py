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


from typing import Self

from duckduckgo_search import DDGS
from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.logger import Logger
from beeai_framework.tools import ToolError
from beeai_framework.tools.search import SearchToolOutput, SearchToolResult
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import ToolRunOptions

logger = Logger(__name__)


class DuckDuckGoSearchType:
    STRICT = "STRICT"
    MODERATE = "MODERATE"
    OFF = "OFF"


class DuckDuckGoSearchToolInput(BaseModel):
    query: str = Field(description="The search query.")


class DuckDuckGoSearchToolResult(SearchToolResult):
    pass


class DuckDuckGoSearchToolOutput(SearchToolOutput):
    pass


class DuckDuckGoSearchTool(Tool[DuckDuckGoSearchToolInput, ToolRunOptions, DuckDuckGoSearchToolOutput]):
    name = "DuckDuckGo"
    description = "Search for online trends, news, current events, real-time information, or research topics."
    input_schema = DuckDuckGoSearchToolInput

    def __init__(self, max_results: int = 10, safe_search: str = DuckDuckGoSearchType.STRICT) -> None:
        super().__init__()
        self.max_results = max_results
        self.safe_search = safe_search

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "search", "duckduckgo"],
            creator=self,
        )

    async def _run(
        self, input: DuckDuckGoSearchToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> DuckDuckGoSearchToolOutput:
        try:
            results = DDGS().text(input.query, max_results=self.max_results, safesearch=self.safe_search)
            search_results: list[SearchToolResult] = [
                DuckDuckGoSearchToolResult(
                    title=result.get("title") or "", description=result.get("body") or "", url=result.get("href") or ""
                )
                for result in results
            ]
            return DuckDuckGoSearchToolOutput(search_results)

        except Exception as e:
            raise ToolError("Error performing search:") from e

    async def clone(self) -> Self:
        cloned = await super().clone()
        cloned.max_results = self.max_results
        cloned.safe_search = self.safe_search
        return cloned
