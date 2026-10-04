# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import date
from typing import Any, Literal, Self

import httpx
from pydantic import BaseModel, Field, field_validator, model_validator

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.logger import Logger
from beeai_framework.tools import JSONToolOutput, ToolError
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import ToolRunOptions

logger = Logger(__name__)

FXMACRODATA_BASE_URL = "https://api.fxmacrodata.com/v1"


class FXMacroDataToolInput(BaseModel):
    operation: Literal["indicator_history", "data_catalogue", "release_calendar"] = Field(
        description=(
            "indicator_history returns the published values of one indicator, "
            "data_catalogue lists the indicators available for a currency, "
            "release_calendar returns upcoming release dates."
        )
    )
    currency: str = Field(description="Three-letter currency code, for example USD, EUR or JPY.")
    indicator: str | None = Field(
        description=(
            "Indicator slug such as inflation, gdp or policy_rate. Required for indicator_history, "
            "optional filter for release_calendar. Use data_catalogue to find valid slugs."
        ),
        default=None,
    )
    start_date: date | None = Field(description="Start date in the format YYYY-MM-DD.", default=None)
    end_date: date | None = Field(description="End date in the format YYYY-MM-DD.", default=None)
    limit: int | None = Field(
        description="Maximum number of rows for indicator_history, most recent first (1-100).",
        default=None,
        ge=1,
        le=100,
    )

    @field_validator("currency", mode="before")
    @classmethod
    def _normalize_currency(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip().upper()
            if len(value) != 3 or not (value.isascii() and value.isalpha()):
                raise ValueError("currency must be a three-letter code")
        return value

    @field_validator("indicator", mode="before")
    @classmethod
    def _normalize_indicator(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip().lower() or None
            if value is not None and not (value.isascii() and value.replace("_", "").isalnum()):
                raise ValueError("indicator must be a slug such as inflation or policy_rate")
        return value

    @model_validator(mode="after")
    def _require_indicator(self) -> Self:
        if self.operation == "indicator_history" and not self.indicator:
            raise ValueError("indicator is required for indicator_history")
        return self


class FXMacroDataTool(Tool[FXMacroDataToolInput, ToolRunOptions, JSONToolOutput[dict[str, Any]]]):
    name = "FXMacroData"
    description = (
        "Retrieve macroeconomic indicator history (inflation, GDP, policy rates, employment and more), "
        "the list of available indicators, and upcoming release dates for a currency."
    )
    input_schema = FXMacroDataToolInput

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = 30,
        options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(options)
        # USD works without a key; other currencies need one.
        self.api_key = api_key or os.environ.get("FXMACRODATA_API_KEY")
        self.timeout = timeout

    async def clone(self) -> Self:
        tool = self.__class__(api_key=self.api_key, timeout=self.timeout, options=self.options)
        tool.name = self.name
        tool.description = self.description
        tool.input_schema = self.input_schema
        tool.middlewares.extend(self.middlewares)
        tool._cache = await self.cache.clone()
        return tool

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "fxmacrodata"],
            creator=self,
        )

    def get_request(self, input: FXMacroDataToolInput) -> tuple[str, dict[str, Any]]:
        currency = input.currency.lower()
        params: dict[str, Any] = {}

        if input.operation == "indicator_history":
            path = f"/announcements/{currency}/{input.indicator}"
            if input.limit is not None:
                params["limit"] = input.limit
        elif input.operation == "data_catalogue":
            path = f"/data_catalogue/{currency}"
        else:
            path = f"/calendar/{currency}"
            if input.indicator:
                params["indicator"] = input.indicator

        if input.operation != "data_catalogue":
            if input.start_date:
                params["start_date"] = input.start_date.isoformat()
            if input.end_date:
                params["end_date"] = input.end_date.isoformat()

        return f"{FXMACRODATA_BASE_URL}{path}", params

    async def _run(
        self, input: FXMacroDataToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> JSONToolOutput[dict[str, Any]]:
        url, params = self.get_request(input)
        logger.debug(f"Using FXMacroData URL: {url} with params {params}")

        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, params=params, headers=headers)

        if response.is_error:
            try:
                detail = response.json().get("detail") or response.text
            except (ValueError, AttributeError):
                detail = response.text
            raise ToolError(f"FXMacroData request failed with status {response.status_code}: {detail}")

        # Returned as-is. Keyless USD responses include a freemium_delay object describing the delay window.
        return JSONToolOutput(response.json())
