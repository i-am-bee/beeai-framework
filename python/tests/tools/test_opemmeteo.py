# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import httpx
import pytest

from beeai_framework.tools import JSONToolOutput, ToolInputValidationError
from beeai_framework.tools.weather import OpenMeteoTool, OpenMeteoToolInput

"""
Utility functions and classes
"""


@pytest.fixture
def tool() -> OpenMeteoTool:
    return OpenMeteoTool()


"""
E2E Tests
"""


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_call_model(tool: OpenMeteoTool) -> None:
    await tool.run(
        input=OpenMeteoToolInput(
            location_name="Cambridge",
            country="US",
            temperature_unit="fahrenheit",
        )
    )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_call_dict(tool: OpenMeteoTool) -> None:
    await tool.run(input={"location_name": "White Plains"})


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_call_invalid_missing_field(tool: OpenMeteoTool) -> None:
    with pytest.raises(ToolInputValidationError):
        await tool.run(input={})


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_call_invalid_bad_type(tool: OpenMeteoTool) -> None:
    with pytest.raises(ToolInputValidationError):
        await tool.run(input={"location_name": 1})


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_output(tool: OpenMeteoTool) -> None:
    result = await tool.run(input={"location_name": "White Plains"})
    assert isinstance(result, JSONToolOutput)
    assert "current" in result.get_text_content()


"""
Unit Tests
"""


@pytest.mark.unit
def test_temperature_unit_accepts_any_case() -> None:
    """The validator that lowercases this field only registers when
    field_validator is the outer decorator, and a model is as likely to send
    "Celsius" as "celsius"."""
    assert OpenMeteoToolInput(location_name="Berlin", temperature_unit="CELSIUS").temperature_unit == "celsius"
    assert OpenMeteoToolInput(location_name="Berlin", temperature_unit="Fahrenheit").temperature_unit == "fahrenheit"


class _GeocodeResponse:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results = results

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return {"results": self._results}


_PARIS_RESULTS = [
    {"name": "Paris", "country": "France", "country_code": "FR", "admin1": "Île-de-France Region", "latitude": 48.85},
    {"name": "Paris", "country": "United States", "country_code": "US", "admin1": "Texas", "latitude": 33.66},
]


@pytest.fixture
def geocode_requests(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    urls: list[str] = []

    async def get(self: httpx.AsyncClient, url: str, **kwargs: Any) -> _GeocodeResponse:
        urls.append(url)
        return _GeocodeResponse(_PARIS_RESULTS)

    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    return urls


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_input",
    [
        {"location_name": "Paris", "country": "United States"},
        {"location_name": "Paris", "country": "us"},
        {"location_name": "Paris, Texas"},
        {"location_name": "Paris, Texas, United States"},
    ],
)
async def test_geocode_picks_the_place_in_the_requested_country(
    tool: OpenMeteoTool, geocode_requests: list[str], tool_input: dict[str, str]
) -> None:
    geocode = await tool._geocode(OpenMeteoToolInput(**tool_input))

    assert geocode["country_code"] == "US"
    assert "country=" not in geocode_requests[0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_geocode_without_country_takes_the_first_place(tool: OpenMeteoTool, geocode_requests: list[str]) -> None:
    geocode = await tool._geocode(OpenMeteoToolInput(location_name="Paris"))

    assert geocode["country_code"] == "FR"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_input",
    [
        {"location_name": "Paris", "country": "Canada"},
        {"location_name": "Paris, TX"},
    ],
)
async def test_geocode_with_unmatched_qualifier_takes_the_first_place(
    tool: OpenMeteoTool, geocode_requests: list[str], tool_input: dict[str, str]
) -> None:
    geocode = await tool._geocode(OpenMeteoToolInput(**tool_input))

    assert geocode["country_code"] == "FR"
