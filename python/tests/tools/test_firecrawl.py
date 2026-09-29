# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import requests

from beeai_framework.tools import ToolError, ToolInputValidationError
from beeai_framework.tools.search.firecrawl import (
    FirecrawlSearchTool,
    FirecrawlSearchToolInput,
    FirecrawlSearchToolOutput,
)


class MockResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error")

    def json(self) -> dict:
        return self.payload


@pytest.fixture
def tool() -> FirecrawlSearchTool:
    return FirecrawlSearchTool(api_key="test-key")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_invalid_input_type(tool: FirecrawlSearchTool) -> None:
    with pytest.raises(ToolInputValidationError):
        await tool.run(input={"search": "BeeAI"})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_output(tool: FirecrawlSearchTool, monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_post(url: str, **kwargs: dict) -> MockResponse:
        assert url == "https://api.firecrawl.dev/v2/search"
        assert kwargs["headers"] == {"Authorization": "Bearer test-key", "Content-Type": "application/json"}
        assert kwargs["json"] == {"query": "BeeAI", "limit": 10, "sources": ["web"], "origin": "beeai-framework"}
        return MockResponse(
            {
                "success": True,
                "data": {
                    "web": [
                        {
                            "url": "https://framework.beeai.dev",
                            "title": "BeeAI Framework",
                            "description": "Build production-ready multi-agent systems",
                            "position": 1,
                        },
                        {"title": "Result without URL"},
                    ]
                },
            }
        )

    monkeypatch.setattr("requests.post", mock_post)

    result = await tool.run(input=FirecrawlSearchToolInput(query="BeeAI"))

    assert type(result) is FirecrawlSearchToolOutput
    assert result.sources() == ["https://framework.beeai.dev"]
    assert "Build production-ready multi-agent systems" in result.get_text_content()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    with pytest.raises(ToolError, match="FIRECRAWL_API_KEY"):
        await FirecrawlSearchTool(api_key="").run(input=FirecrawlSearchToolInput(query="BeeAI"))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_location(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_post(url: str, **kwargs: dict) -> MockResponse:
        assert kwargs["json"]["location"] == "Germany"
        return MockResponse({"success": True, "data": {"web": []}})

    monkeypatch.setattr("requests.post", mock_post)

    result = await FirecrawlSearchTool(api_key="test-key", location="Germany").run(
        input=FirecrawlSearchToolInput(query="BeeAI")
    )

    assert result.is_empty()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_http_error(tool: FirecrawlSearchTool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("requests.post", lambda url, **kwargs: MockResponse({}, status_code=401))

    with pytest.raises(ToolError, match="Error performing Firecrawl search"):
        await tool.run(input=FirecrawlSearchToolInput(query="BeeAI"))
