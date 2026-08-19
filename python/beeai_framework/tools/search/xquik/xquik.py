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


class XquikSearchToolInput(BaseModel):
    query: str = Field(description="The X/Twitter search query.")


class XquikSearchToolOutput(SearchToolOutput):
    pass


class XquikSearchTool(Tool[XquikSearchToolInput, ToolRunOptions, XquikSearchToolOutput]):
    name = "Xquik"
    description = "Search public X/Twitter posts through the Xquik REST API."
    input_schema = XquikSearchToolInput

    def __init__(
        self,
        api_key: str | None = None,
        *,
        max_results: int = 10,
        base_url: str = "https://xquik.com",
        timeout: float = 30,
        options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(options)
        self.api_key = api_key or os.environ.get("XQUIK_API_KEY")
        self.max_results = max_results
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "search", "xquik"],
            creator=self,
        )

    async def clone(self) -> Self:
        tool = self.__class__(
            api_key=self.api_key,
            max_results=self.max_results,
            base_url=self.base_url,
            timeout=self.timeout,
            options=self.options,
        )
        tool.name = self.name
        tool.description = self.description
        tool.middlewares.extend(self.middlewares)
        tool._cache = await self.cache.clone()
        return tool

    async def _run(
        self, input: XquikSearchToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> XquikSearchToolOutput:
        if not self.api_key:
            raise ToolError("XQUIK_API_KEY is required to use Xquik search.")

        try:
            response = await to_thread(
                requests.get,
                f"{self.base_url}/api/v1/x/tweets/search",
                headers={"x-api-key": self.api_key},
                params={"q": input.query, "queryType": "Latest", "limit": self.max_results},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as error:
            raise ToolError("Error performing Xquik search.") from error

        return XquikSearchToolOutput([self._tweet_to_result(tweet) for tweet in payload.get("tweets", [])])

    @staticmethod
    def _tweet_to_result(tweet: dict[str, Any]) -> SearchToolResult:
        author = tweet.get("author") or {}
        username = author.get("username") or ""
        tweet_id = tweet.get("id") or ""
        title = f"X post by @{username}" if username else "X post"
        url = tweet.get("url") or (f"https://x.com/{username}/status/{tweet_id}" if username and tweet_id else "")
        return SearchToolResult(
            title=title,
            description=tweet.get("text") or "",
            url=url,
        )
