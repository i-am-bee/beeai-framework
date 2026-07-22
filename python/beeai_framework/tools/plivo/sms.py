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


class PlivoSendMessageToolInput(BaseModel):
    to: str = Field(description="Recipient phone number in E.164 format, e.g. +14150000001.")
    text: str = Field(description="The text body of the SMS message to send.")


class PlivoSendMessageTool(Tool[PlivoSendMessageToolInput, ToolRunOptions, JSONToolOutput[dict[str, Any]]]):
    name = "PlivoSendMessage"
    description = "Send an SMS text message to a phone number using Plivo."
    input_schema = PlivoSendMessageToolInput

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
            namespace=["tool", "plivo", "sms"],
            creator=self,
        )

    async def _run(
        self, input: PlivoSendMessageToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> JSONToolOutput[dict[str, Any]]:
        auth_id = os.environ.get("PLIVO_AUTH_ID")
        auth_token = os.environ.get("PLIVO_AUTH_TOKEN")
        src = os.environ.get("PLIVO_SRC")
        if not auth_id or not auth_token or not src:
            raise ToolError("PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, and PLIVO_SRC must be set.")

        logger.debug(f"Sending Plivo SMS to {input.to}")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.plivo.com/v1/Account/{auth_id}/Message/",
                auth=(auth_id, auth_token),
                json={"src": src, "dst": input.to, "text": input.text, "type": "sms"},
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            response.raise_for_status()
            return JSONToolOutput(response.json())
