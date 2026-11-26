# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
import copy
from collections.abc import Iterator
from importlib import import_module
from typing import Any, Literal, TypeVar, Union

import json_repair
import jsonref  # type: ignore
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel, ConfigDict, Field, RootModel, create_model

from beeai_framework.backend.constants import (
    BackendProviders,
    ModelTypes,
    ModuleTypes,
    ProviderDef,
    ProviderModelDef,
    ProviderModuleDef,
    ProviderName,
)
from beeai_framework.backend.errors import BackendError
from beeai_framework.backend.types import ChatModelToolChoice
from beeai_framework.tools.tool import AnyTool, Tool
from beeai_framework.utils.dicts import exclude_keys_inplace
from beeai_framework.utils.models import WrappedRootModel

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


def parse_module(name: str) -> ProviderModuleDef:
    if not name:
        raise BackendError("Neither 'provider' nor 'provider:model' was specified.")

    # provider_id:model_id
    # e.g., ollama:llama3.1
    # keep remainder of string intact (maxsplit=1) because model name can also have colons
    name_parts = name.split(":", maxsplit=1)
    provider_def = find_provider_def(name_parts[0])

    if not provider_def:
        raise BackendError("Model does not contain provider name!")

    return ProviderModuleDef(
        provider_id=name_parts[0],
        entity_id=name_parts[1] if len(name_parts) > 1 else None,
        provider_def=provider_def,
    )


def load_model(name: ProviderName | str, model_type: ModelTypes = "chat") -> type[T]:
    parsed = parse_model(name)
    provider_def = parsed.provider_def

    module_path = f"beeai_framework.adapters.{provider_def.module}.backend.{model_type}"
    module = import_module(module_path)

    class_name = f"{provider_def.name}{model_type.capitalize()}Model"
    return getattr(module, class_name)  # type: ignore


def load_module(name: ProviderName | str, module_type: ModuleTypes = "vector_store") -> type[T]:
    def get_class_suffix(module_type: str) -> str:
        words = module_type.split("_")
        return "".join(word.capitalize() for word in words)

    parsed = parse_module(name)
    provider_def = parsed.provider_def

    module_path = f"beeai_framework.adapters.{provider_def.module}.backend.{module_type.lower()}"
    module = import_module(module_path)

    class_name = f"{provider_def.name}{get_class_suffix(module_type)}"
    return getattr(module, class_name)  # type: ignore


def parse_broken_json(input: str, fallback: Any | None = None, *, stream_stable: bool = False) -> Any:
    try:
        return json_repair.loads(input, stream_stable=stream_stable)
    except Exception:
        if fallback is not None:
            return fallback
        raise


_LOCAL_DEFS_PREFIX = "#/$defs/"
_RECURSIVE_REF_KEY = "__beeai_recursive_ref__"


def _local_def_name(node: Any) -> str | None:
    ref = node.get("$ref") if isinstance(node, dict) else None
    if isinstance(ref, str) and ref.startswith(_LOCAL_DEFS_PREFIX):
        return ref.removeprefix(_LOCAL_DEFS_PREFIX)
    return None


def _walk(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _find_cyclic_refs(defs: dict[str, Any]) -> dict[str, set[str]]:
    """For each definition that refers to itself (directly or through other definitions),
    return the names of the definitions it refers to that lead back to it."""
    edges: dict[str, set[str]] = {}
    for name, body in defs.items():
        refs = (_local_def_name(node) for node in _walk(body))
        edges[name] = {ref for ref in refs if ref is not None and ref in defs}

    def reachable(start: str) -> set[str]:
        seen: set[str] = set()
        stack = list(edges[start])
        while stack:
            name = stack.pop()
            if name not in seen:
                seen.add(name)
                stack.extend(edges[name])
        return seen

    reach = {name: reachable(name) for name in defs}
    return {name: {target for target in edges[name] if name in reach[target]} for name in defs if name in reach[name]}


def inline_schema_refs(schema: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    if schema.get("$defs") is not None or force is True:
        defs = schema.get("$defs")
        cyclic = _find_cyclic_refs(defs) if isinstance(defs, dict) else {}
        if cyclic:
            # A self-referencing schema cannot be fully inlined. Keep the references that close a cycle
            # (and the definitions they point to) and inline everything else.
            schema = copy.deepcopy(schema)
            for name, targets in cyclic.items():
                for node in _walk(schema["$defs"][name]):
                    if _local_def_name(node) in targets:
                        node[_RECURSIVE_REF_KEY] = node.pop("$ref")

        # pyrefly: ignore [bad-assignment]
        schema = jsonref.replace_refs(
            schema, base_uri="", load_on_repr=True, merge_props=True, proxies=False, lazy_load=False
        )
        defs = schema.pop("$defs", None)

        if cyclic:
            # A reference at the root (e.g. Pydantic's schema for a recursive model) leaves a JsonRef wrapper.
            schema = dict(schema)
            for node in _walk(schema):
                if _RECURSIVE_REF_KEY in node:
                    node["$ref"] = node.pop(_RECURSIVE_REF_KEY)
            schema["$defs"] = {name: defs[name] for name in cyclic}
            for node in _walk(schema["$defs"]):
                if _RECURSIVE_REF_KEY in node:
                    node["$ref"] = node.pop(_RECURSIVE_REF_KEY)

    return schema


def generate_tool_union_schema(
    tools: list[AnyTool],
    *,
    strict: bool,
    allow_parallel_tool_calls: bool,
    allow_top_level_union: bool,
    fallback_tool: AnyTool | None = None,
    use_enum_for_tool_names: bool = False,
) -> tuple[dict[str, Any], type[BaseModel]]:
    if not tools:
        raise ValueError("No tools provided!")

    if use_enum_for_tool_names and any(not tool.name for tool in tools):
        raise ValueError("Gemini guided tool names must not be empty.")

    # Gemini needs a singleton enum on the wire; Literal still validates the call locally.
    def tool_name_schema(schema: dict[str, Any]) -> None:
        schema["enum"] = [schema.pop("const")]

    tool_schemas = [
        create_model(  # type: ignore
            tool.name,
            __module__="fn",
            __config__=ConfigDict(extra="forbid", populate_by_name=True, title=tool.name),
            **{
                "name": (
                    # pyrefly: ignore [invalid-literal]
                    Literal[tool.name],
                    Field(
                        description="Tool Name", json_schema_extra=tool_name_schema if use_enum_for_tool_names else None
                    ),
                ),
                "parameters": (tool.input_schema, Field(description="Tool Parameters")),
            },
        )
        for tool in tools
    ]

    if len(tool_schemas) == 1:
        schema = tool_schemas[0]
    else:
        root_model_type = Union[*tool_schemas]  # type: ignore
        BaseClass, SchemaType = (  # noqa: N806
            RootModel if allow_top_level_union else WrappedRootModel,
            list[root_model_type] if allow_parallel_tool_calls else root_model_type,
        )

        class AvailableTools(BaseClass[SchemaType]):  # type: ignore
            @classmethod
            def model_validate(
                cls,
                obj: Any,
                **kwargs: Any,
            ) -> Any:
                if not fallback_tool or obj is None:
                    return super().model_validate(obj, **kwargs)

                if not isinstance(obj, dict):
                    obj = {"name": fallback_tool.name, "parameters": obj}
                    assert isinstance(obj, dict)

                final = obj
                if BaseClass is WrappedRootModel:
                    if obj.get("item") is None:
                        final = {"item": obj}
                    else:
                        obj = obj["item"]
                        assert isinstance(obj, dict)

                # Tool Schema
                if obj.get("name") is None:
                    obj["name"] = fallback_tool.name

                if "parameters" not in obj:
                    obj["parameters"] = exclude_keys_inplace(obj, set(obj.keys()) - {"name"})

                return super().model_validate(final, **kwargs)

        schema = AvailableTools

    return (
        {
            "type": "json_schema",
            "json_schema": {
                "name": "ToolCall",
                "schema": inline_schema_refs(to_strict_json_schema(schema) if strict else schema.model_json_schema()),
            },
        },
        schema,
    )


def filter_tools_by_tool_choice(tools: list[AnyTool], value: ChatModelToolChoice | None) -> list[AnyTool]:
    if value == "none":
        return []

    if value == "required" or value == "auto" or value is None:
        return tools

    if isinstance(value, Tool):
        tool = [tool for tool in tools if tool is value]
        if not tool:
            raise ValueError(f"Invalid tool choice provided! Tool '{value}' was not found.")

        return tool

    raise RuntimeError(f"Unknown tool choice: {value}")
