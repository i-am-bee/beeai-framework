# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Self

import httpx
from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.logger import Logger
from beeai_framework.tools import JSONToolOutput
from beeai_framework.tools.errors import ToolError
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import ToolRunOptions

logger = Logger(__name__)


class PlivoMakeCallToolInput(BaseModel):
    to: str = Field(description="Destination phone number to call in E.164 format, e.g. +14150000001.")
    answer_url: str = Field(
        description="Publicly reachable URL that returns Plivo answer XML (e.g. a <Speak> element) when the call is answered."
    )


class PlivoMakeCallTool(Tool[PlivoMakeCallToolInput, ToolRunOptions, JSONToolOutput[dict[str, Any]]]):
    name = "PlivoMakeCall"
    description = "Place an outbound phone call using Plivo that runs the answer XML at the given URL when answered."
    input_schema = PlivoMakeCallToolInput

    async def clone(self) -> Self:
        tool = self.__class__(options=self.options)
        tool.name = self.name
        tool.description = self.description
        tool.input_schema = self.input_schema
        tool.middlewares.extend(self.middlewares)
        tool._cache = await self.cache.clone()
        return tool

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "plivo", "call"],
            creator=self,
        )

    async def _run(
        self, input: PlivoMakeCallToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> JSONToolOutput[dict[str, Any]]:
        auth_id = os.environ.get("PLIVO_AUTH_ID")
        auth_token = os.environ.get("PLIVO_AUTH_TOKEN")
        src = os.environ.get("PLIVO_SRC")
        if not auth_id or not auth_token or not src:
            raise ToolError("PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, and PLIVO_SRC must be set.")

        logger.debug(f"Placing Plivo call to {input.to}")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.plivo.com/v1/Account/{auth_id}/Call/",
                auth=(auth_id, auth_token),
                json={"from": src, "to": input.to, "answer_url": input.answer_url},
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            response.raise_for_status()
            return JSONToolOutput(response.json())
