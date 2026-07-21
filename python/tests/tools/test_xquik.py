# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from beeai_framework.tools import ToolError, ToolInputValidationError
from beeai_framework.tools.search.xquik import XquikSearchTool, XquikSearchToolInput, XquikSearchToolOutput


class MockResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


@pytest.fixture
def tool() -> XquikSearchTool:
    return XquikSearchTool(api_key="test-key")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_invalid_input_type(tool: XquikSearchTool) -> None:
    with pytest.raises(ToolInputValidationError):
        await tool.run(input={"search": "BeeAI"})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_output(tool: XquikSearchTool, monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_get(url: str, **kwargs: dict) -> MockResponse:
        assert url == "https://xquik.com/api/v1/x/tweets/search"
        assert kwargs["headers"] == {"x-api-key": "test-key"}
        assert kwargs["params"] == {"q": "BeeAI", "queryType": "Latest", "limit": 10}
        return MockResponse(
            {
                "tweets": [
                    {
                        "id": "123",
                        "text": "BeeAI framework update",
                        "author": {"username": "beeai"},
                    }
                ]
            }
        )

    monkeypatch.setattr("requests.get", mock_get)

    result = await tool.run(input=XquikSearchToolInput(query="BeeAI"))

    assert type(result) is XquikSearchToolOutput
    assert result.sources() == ["https://x.com/beeai/status/123"]
    assert "BeeAI framework update" in result.get_text_content()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_api_key() -> None:
    with pytest.raises(ToolError, match="XQUIK_API_KEY"):
        await XquikSearchTool(api_key="").run(input=XquikSearchToolInput(query="BeeAI"))
