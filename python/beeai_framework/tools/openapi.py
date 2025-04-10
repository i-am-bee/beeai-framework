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


from typing import Any
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import httpx
from pydantic import BaseModel, InstanceOf

from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.tools import StringToolOutput, Tool, ToolError, ToolRunOptions
from beeai_framework.utils.strings import to_safe_word


class OpenAPIToolInput(BaseModel):
    path: str
    parameters: dict[str, Any] | None = None
    body: dict[str, Any] | None = None
    method: str


class OpenAPIToolOutput(StringToolOutput):
    def __init__(self, status: int, result: str = "") -> None:
        super().__init__()
        self.status = status
        self.result = result or ""


class BeforeFetchEvent(BaseModel):
    input: dict[str, Any]
    url: str


class AfterFetchEvent(BaseModel):
    data: InstanceOf[OpenAPIToolOutput]
    url: str


class OpenAPITool(Tool[OpenAPIToolInput, ToolRunOptions, OpenAPIToolOutput]):
    def __init__(
        self,
        open_api_schema: dict[str, Any],
        name: str | None = None,
        description: str | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.open_api_schema = open_api_schema
        self.headers = headers or {}

        self.url = url or next(
            server.get("url") for server in self.open_api_schema.get("servers", []) if server.get("url") is not None
        )
        if self.url is None:
            raise ToolError("OpenAPI schema hasn't any server with url specified. Pass it manually.")

        self._name = name or self.open_api_schema.get("info", {}).get("title", "").strip()
        if self._name is None:
            raise ToolError("OpenAPI schema hasn't 'name' specified. Pass it manually.")

        self._description = (
            description
            or self.open_api_schema.get("info", {}).get("description", None)
            or (
                "Performs REST API requests to the servers and retrieves the response. "
                "The server API interfaces are defined in OpenAPI schema. \n"
                "Only use the OpenAPI tool if you need to communicate to external servers."
            )
        )

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "web", "openAPI", to_safe_word(self._name)],
            creator=self,
            events={"before_fetch": BeforeFetchEvent, "after_fetch": AfterFetchEvent},
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def input_schema(self) -> type[OpenAPIToolInput]:
        return OpenAPIToolInput

    async def _run(
        self, tool_input: OpenAPIToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> OpenAPIToolOutput:
        parsed_url = urlparse(urljoin(self.url, tool_input.path or ""))
        search_params = parse_qs(parsed_url.query)
        search_params.update(tool_input.parameters or {})
        new_params = urlencode(search_params, doseq=True)
        url = urlunparse(parsed_url._replace(query=new_params))

        await self.emitter.emit("before_fetch", {"input": tool_input.model_dump()})
        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=tool_input.method,
                    url=str(url),
                    headers={"Accept": "application/json"}.update(self.headers),
                    data=tool_input.body,
                )
                output = OpenAPIToolOutput(response.status_code, response.text)
                await self.emitter.emit("after_fetch", {"url": url, "data": output})
                return output
        except httpx.HTTPError as err:
            raise ToolError(f"Request to {url} has failed.", cause=err)
