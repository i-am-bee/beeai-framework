# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from asyncio import to_thread
from typing import Any, Self

import requests
from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools import ToolError
from beeai_framework.tools.search import SearchToolOutput, SearchToolResult
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import ToolRunOptions


class FirecrawlSearchToolInput(BaseModel):
    query: str = Field(description="The web search query.")


class FirecrawlSearchToolOutput(SearchToolOutput):
    pass


class FirecrawlSearchTool(Tool[FirecrawlSearchToolInput, ToolRunOptions, FirecrawlSearchToolOutput]):
    name = "Firecrawl"
    description = (
        "Search the web for current information, news, and research topics. "
        "Returns result titles, URLs, and query-relevant highlights."
    )
    input_schema = FirecrawlSearchToolInput

    def __init__(
        self,
        api_key: str | None = None,
        *,
        max_results: int = 10,
        base_url: str = "https://api.firecrawl.dev",
        timeout: float = 30,
        location: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(options)
        self.api_key = api_key or os.environ.get("FIRECRAWL_API_KEY")
        self.max_results = max_results
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.location = location

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "search", "firecrawl"],
            creator=self,
        )

    async def clone(self) -> Self:
        tool = self.__class__(
            api_key=self.api_key,
            max_results=self.max_results,
            base_url=self.base_url,
            timeout=self.timeout,
            location=self.location,
            options=self.options,
        )
        tool.name = self.name
        tool.description = self.description
        tool.middlewares.extend(self.middlewares)
        tool._cache = await self.cache.clone()
        return tool

    async def _run(
        self, input: FirecrawlSearchToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> FirecrawlSearchToolOutput:
        if not self.api_key:
            raise ToolError("FIRECRAWL_API_KEY is required to use Firecrawl search.")

        body: dict[str, Any] = {
            "query": input.query,
            "limit": self.max_results,
            "sources": ["web"],
            "origin": "beeai-framework",
        }
        if self.location:
            body["location"] = self.location

        try:
            response = await to_thread(
                requests.post,
                f"{self.base_url}/v2/search",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=body,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as error:
            raise ToolError("Error performing Firecrawl search.") from error

        return FirecrawlSearchToolOutput(
            [
                SearchToolResult(
                    title=item.get("title") or "",
                    description=item.get("description") or "",
                    url=item["url"],
                )
                for item in (payload.get("data") or {}).get("web") or []
                if item.get("url")
            ]
        )
