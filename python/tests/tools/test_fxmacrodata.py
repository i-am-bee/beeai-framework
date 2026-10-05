# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import httpx
import pytest

from beeai_framework.tools import JSONToolOutput, ToolError, ToolInputValidationError
from beeai_framework.tools.fxmacrodata import FXMacroDataTool, FXMacroDataToolInput

"""
Utility functions and classes
"""

FREEMIUM_DELAY = {
    "applied": True,
    "delay_seconds": 900,
    "delay_minutes": 15,
    "cutoff": 1791111300,
    "cutoff_iso": "2026-10-04T10:55:00Z",
    "withheld_count": 0,
}


@pytest.fixture
def tool(monkeypatch: pytest.MonkeyPatch) -> FXMacroDataTool:
    monkeypatch.delenv("FXMACRODATA_API_KEY", raising=False)
    return FXMacroDataTool()


@pytest.fixture
def requests(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """Route the tool's HTTP client to an in-memory transport and record each request."""
    actual_client = httpx.AsyncClient
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/v1/announcements/eur/inflation" and "X-API-Key" not in request.headers:
            return httpx.Response(
                401, json={"error": "api_key_required", "detail": "This endpoint requires an API key."}
            )
        if request.url.path.startswith("/v1/announcements/"):
            return httpx.Response(
                200,
                json={
                    "currency": "USD",
                    "indicator": "inflation",
                    "freemium_delay": FREEMIUM_DELAY,
                    "data": [{"date": "2026-08-31", "val": 3.4, "announcement_datetime": 1789475400}],
                },
            )
        if request.url.path.startswith("/v1/data_catalogue/"):
            return httpx.Response(200, json={"inflation": {"name": "Inflation (CPI)", "unit": "%"}})
        return httpx.Response(200, json={"currency": "USD", "data": [{"release": "inflation"}]})

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(respond)
        return actual_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return seen


"""
Unit Tests
"""


@pytest.mark.unit
def test_input_normalizes_currency_and_indicator() -> None:
    input = FXMacroDataToolInput(operation="indicator_history", currency=" usd ", indicator="Policy_Rate")
    assert input.currency == "USD"
    assert input.indicator == "policy_rate"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input",
    [
        {"operation": "indicator_history", "currency": "USD"},
        {"operation": "indicator_history", "currency": "US", "indicator": "inflation"},
        {"operation": "indicator_history", "currency": "USD", "indicator": "../forex"},
        {"operation": "indicator_history", "currency": "USD", "indicator": "inflation", "limit": 101},
        {"operation": "forex", "currency": "USD"},
    ],
)
async def test_call_invalid_input(tool: FXMacroDataTool, input: dict[str, Any]) -> None:
    with pytest.raises(ToolInputValidationError):
        await tool.run(input=input)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_indicator_history_without_key(tool: FXMacroDataTool, requests: list[httpx.Request]) -> None:
    result = await tool.run(
        input={
            "operation": "indicator_history",
            "currency": "usd",
            "indicator": "inflation",
            "start_date": "2026-07-01",
            "end_date": "2026-09-30",
            "limit": 5,
        }
    )

    request = requests[0]
    assert request.url.scheme == "https"
    assert request.url.host == "api.fxmacrodata.com"
    assert request.url.path == "/v1/announcements/usd/inflation"
    assert dict(request.url.params) == {"limit": "5", "start_date": "2026-07-01", "end_date": "2026-09-30"}
    assert "X-API-Key" not in request.headers

    assert isinstance(result, JSONToolOutput)
    assert result.result["freemium_delay"] == FREEMIUM_DELAY
    assert result.result["data"][0]["val"] == 3.4


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_key_is_sent_as_header(monkeypatch: pytest.MonkeyPatch, requests: list[httpx.Request]) -> None:
    monkeypatch.setenv("FXMACRODATA_API_KEY", "env-key")
    await FXMacroDataTool().run(input={"operation": "indicator_history", "currency": "EUR", "indicator": "inflation"})
    await FXMacroDataTool(api_key="explicit-key").run(input={"operation": "data_catalogue", "currency": "EUR"})

    assert [request.headers["X-API-Key"] for request in requests] == ["env-key", "explicit-key"]
    assert all("api_key" not in request.url.params for request in requests)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_data_catalogue(tool: FXMacroDataTool, requests: list[httpx.Request]) -> None:
    result = await tool.run(input={"operation": "data_catalogue", "currency": "USD", "start_date": "2026-01-01"})

    assert requests[0].url.path == "/v1/data_catalogue/usd"
    assert not requests[0].url.params
    assert "inflation" in result.result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_release_calendar(tool: FXMacroDataTool, requests: list[httpx.Request]) -> None:
    await tool.run(
        input={
            "operation": "release_calendar",
            "currency": "USD",
            "indicator": "inflation",
            "start_date": "2026-10-01",
            "end_date": "2026-10-31",
            "limit": 10,
        }
    )

    assert requests[0].url.path == "/v1/calendar/usd"
    assert dict(requests[0].url.params) == {
        "indicator": "inflation",
        "start_date": "2026-10-01",
        "end_date": "2026-10-31",
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_error_includes_api_detail(tool: FXMacroDataTool, requests: list[httpx.Request]) -> None:
    with pytest.raises(ToolError, match="401: This endpoint requires an API key"):
        await tool.run(input={"operation": "indicator_history", "currency": "EUR", "indicator": "inflation"})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_malformed_key_is_rejected_without_echo(requests: list[httpx.Request]) -> None:
    with pytest.raises(ToolError) as raised:
        await FXMacroDataTool(api_key="test-key\nX-Other: 1").run(
            input={"operation": "data_catalogue", "currency": "USD"}
        )

    assert "test-key" not in str(raised.value)
    assert not requests


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"detail": "Unknown indicator"}),
        httpx.Response(200, json=["unexpected"]),
        httpx.Response(200, text="<html></html>"),
        httpx.Response(302, headers={"Location": "https://elsewhere.example/"}),
    ],
)
async def test_unexpected_response_raises_tool_error(
    monkeypatch: pytest.MonkeyPatch, response: httpx.Response
) -> None:
    actual_client = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(lambda request: response)
        return actual_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    with pytest.raises(ToolError):
        await FXMacroDataTool(api_key="test-key").run(input={"operation": "data_catalogue", "currency": "USD"})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clone_keeps_configuration() -> None:
    tool = FXMacroDataTool(api_key="test-key", timeout=5)
    cloned = await tool.clone()
    assert cloned.api_key == "test-key"
    assert cloned.timeout == 5


"""
E2E Tests
"""


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_usd_indicator_history(tool: FXMacroDataTool) -> None:
    result = await tool.run(
        input=FXMacroDataToolInput(operation="indicator_history", currency="USD", indicator="inflation", limit=3)
    )
    assert isinstance(result, JSONToolOutput)
    assert result.result["data"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_usd_data_catalogue(tool: FXMacroDataTool) -> None:
    result = await tool.run(input={"operation": "data_catalogue", "currency": "USD"})
    assert "inflation" in result.get_text_content()
