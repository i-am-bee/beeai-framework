# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Union
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import httpx
from pydantic import BaseModel, Field, InstanceOf, RootModel

from beeai_framework.backend.utils import inline_schema_refs
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.tools import StringToolOutput, Tool, ToolError, ToolRunOptions
from beeai_framework.utils import JSONSchemaModel
from beeai_framework.utils.strings import to_safe_word


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


class OpenAPITool(Tool[BaseModel, ToolRunOptions, OpenAPIToolOutput]):
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
    def input_schema(self) -> type[BaseModel]:
        def get_referenced_object(json: dict[str, Any], ref_path: str) -> dict[str, Any]:
            path_segments = ref_path.split("/")
            current_object = json
            for segment in path_segments:
                if segment == "#":
                    continue
                current_object = current_object[segment]

            if current_object is None:
                raise ValueError(f"Reference {ref_path} not found in OpenAPI schema.")

            return current_object

        schemas: list[dict[str, Any]] = []

        def resolve_schema_refs(input: Any) -> Any:
            if not input:
                return input

            input = input.copy()
            input["components"] = self.open_api_schema["components"].copy()
            return inline_schema_refs(input, force=True)

        for path, path_spec in self.open_api_schema.get("paths", {}).items():
            for method, method_spec in path_spec.items():
                properties = {
                    "path": {
                        "const": path,
                        "description": (
                            "Do not replace variables in path, instead of, put them to the parameters object."
                        ),
                    },
                    "method": {
                        "const": method,
                        "description": method_spec.get("summary", method_spec.get("description")),
                    },
                }

                if method_spec.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema"):
                    request_body = method_spec["requestBody"]["content"]["application/json"]["schema"]
                    if request_body.get("$ref"):
                        request_body = get_referenced_object(self.open_api_schema, request_body["$ref"])
                    properties["body"] = resolve_schema_refs(request_body)

                if method_spec.get("parameters"):
                    parameters = {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [p["name"] for p in method_spec["parameters"] if p.get("required")],
                        "properties": {},
                    }

                    for p in method_spec["parameters"]:
                        if "$ref" not in p:
                            parameters["properties"][p["name"]] = {**p.get("schema", {}), "description": p["name"]}  # type: ignore
                        else:
                            ref_obj = get_referenced_object(self.open_api_schema, p["$ref"])
                            if ref_obj and "name" in ref_obj and "schema" in ref_obj:
                                parameters["properties"][ref_obj["name"]] = {  # type: ignore
                                    **ref_obj["schema"],
                                    "description": ref_obj["name"],
                                }

                    properties["parameters"] = resolve_schema_refs(parameters)

                schemas.append(
                    {
                        "type": "object",
                        "required": ["path", "method"],
                        "additionalProperties": False,
                        "properties": properties,
                    }
                )

        schema_models = [
            JSONSchemaModel.create(
                f"OpenAPIToolInput{to_safe_word(schema['properties']['method']['const'])}{to_safe_word(schema['properties']['path']['const'])}",
                schema,
            )
            for schema in schemas
        ]

        if len(schema_models) == 1:
            return schema_models[0]

        class OpenAPIToolInput(RootModel[Union[*schema_models]]):  # type: ignore
            root: Union[*schema_models] = Field(description="Union of valid input schemas")  # type: ignore

        return OpenAPIToolInput

    async def _run(
        self, tool_input: BaseModel, options: ToolRunOptions | None, context: RunContext
    ) -> OpenAPIToolOutput:
        input_dict = tool_input.model_dump()
        parsed_url = urlparse(urljoin(self.url, input_dict.get("path", "")))
        search_params = parse_qs(parsed_url.query)
        search_params.update(input_dict.get("parameters", {}))
        new_params = urlencode(search_params, doseq=True)
        url = urlunparse(parsed_url._replace(query=new_params))

        await self.emitter.emit("before_fetch", BeforeFetchEvent(url=str(url), input=input_dict))
        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=input_dict["method"],
                    url=str(url),
                    headers={"Accept": "application/json"}.update(self.headers),
                    data=input_dict.get("body"),
                )
                output = OpenAPIToolOutput(response.status_code, response.text)
                await self.emitter.emit("after_fetch", AfterFetchEvent(url=str(url), data=output))
                return output
        except httpx.HTTPError as err:
            raise ToolError(f"Request to {url} has failed.", cause=err)
