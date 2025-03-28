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


from importlib import import_module
from typing import Any, Literal, TypeVar, Union

import json_repair
import jsonref  # type: ignore
from pydantic import ConfigDict, Field, RootModel, create_model

from beeai_framework.backend.constants import (
    BackendProviders,
    ProviderDef,
    ProviderModelDef,
    ProviderName,
)
from beeai_framework.backend.errors import BackendError
from beeai_framework.backend.types import ChatModelToolChoice
from beeai_framework.tools.tool import AnyTool, Tool

T = TypeVar("T")

# TODO: `${ProviderName}:${string}`
FullModelName: str


def find_provider_def(value: str) -> ProviderDef | None:
    for provider in BackendProviders.values():
        if value == provider.name or value == provider.module or value in provider.aliases:
            return provider
    return None


def parse_model(name: str) -> ProviderModelDef:
    if not name:
        raise BackendError("Neither 'provider' nor 'provider:model' was specified.")

    # provider_id:model_id
    # e.g., ollama:llama3.1
    # keep remainder of string intact (maxsplit=1) because model name can also have colons
    name_parts = name.split(":", maxsplit=1)
    provider_def = find_provider_def(name_parts[0])

    if not provider_def:
        raise BackendError("Model does not contain provider name!")

    return ProviderModelDef(
        provider_id=name_parts[0],
        model_id=name_parts[1] if len(name_parts) > 1 else None,
        provider_def=provider_def,
    )


def load_model(name: ProviderName | str, model_type: Literal["embedding", "chat"] = "chat") -> type[T]:
    parsed = parse_model(name)
    provider_def = parsed.provider_def

    module_path = f"beeai_framework.adapters.{provider_def.module}.backend.{model_type}"
    module = import_module(module_path)

    class_name = f"{provider_def.name}{model_type.capitalize()}Model"
    return getattr(module, class_name)  # type: ignore


def parse_broken_json(input: str) -> Any:
    return json_repair.loads(input)


def inline_schema_refs(schema: dict[str, Any]) -> dict[str, Any]:
    if schema.get("$defs") is not None:
        schema = jsonref.replace_refs(
            schema, base_uri="", load_on_repr=True, merge_props=True, proxies=False, lazy_load=False
        )
        schema.pop("$defs", None)

    return schema


def generate_tool_union_schema(tools: list[AnyTool]) -> dict[str, Any]:
    if not tools:
        raise ValueError("No tools provided!")

    tool_schemas = [
        create_model(  # type: ignore
            tool.name,
            __module__="fn",
            __config__=ConfigDict(extra="forbid", populate_by_name=True, title=tool.name),
            **{
                "name": (Literal[tool.name], Field(description="Tool Name")),
                "parameters": (tool.input_schema, Field(description="Tool Parameters")),
            },
        )
        for tool in tools
    ]

    if len(tool_schemas) == 1:
        schema = tool_schemas[0].model_json_schema()
    else:

        class AvailableTools(RootModel[Union[*tool_schemas]]):  # type: ignore
            pass

        schema = AvailableTools.model_json_schema()

    return inline_schema_refs(schema)


def filter_tools_by_tool_choice(tools: list[AnyTool], value: ChatModelToolChoice | None) -> list[AnyTool]:
    if value == "none":
        return []

    if value == "required" or value == "auto" or value is None:
        return tools

    if isinstance(value, Tool):
        tool = [tool for tool in tools if tool is value]
        if not tool:
            raise ValueError("Invalid tool choice provided! Tool was not found.")

        return tool

    raise RuntimeError(f"Unknown tool choice: {value}")
