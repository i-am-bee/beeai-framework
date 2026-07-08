# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import BaseModel, Field

from beeai_framework.utils.dicts import exclude_none
from beeai_framework.utils.lists import remove_falsy

Schema = dict[str, Any]

__all__ = ["SimplifyJsonSchemaConfig", "simplify_json_schema"]


class SimplifyJsonSchemaConfig(BaseModel):
    group_types: bool = Field(
        False,
        description="Group entries of anyOf/oneOf into the 'type' field. Might not be supported by some LLM providers.",
    )
    excluded_properties_by_type: dict[str, set[str]] = Field(
        default_factory=lambda: {},
        description="Exclude certain properties for a given type. Use when the given provider does not support them.",
    )


def _simplify(schema: Schema, path: list[str], config: SimplifyJsonSchemaConfig) -> Any:
    if not isinstance(schema, dict) or not schema:
        return schema

    for key in ("not",):
        if schema.get(key) == {}:
            del schema[key]

    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        excluded_keys = config.excluded_properties_by_type.get(schema_type) or set[str]()
        for key in excluded_keys:
            schema.pop(key, None)

    if schema_type == "object":
        properties = {k: _simplify(v, [*path, k], config) for k, v in schema.get("properties", {}).items()}
        schema["properties"] = exclude_none(properties)

    if schema_type == "array":
        items = _simplify(schema.get("items", {}), [*path, "items"], config)
        schema["items"] = exclude_none(items)

    for key in ("anyOf", "oneOf"):
        values = schema.get(key)
        if values and isinstance(values, list):
            values = remove_falsy([_simplify(v, [*path, key, f"{[idx]}"], config) for idx, v in enumerate(values)])

            if len(values) == 1:
                return values[0]

            if config.group_types:  # noqa: SIM102
                if values and all(v.keys() == {"type"} for v in values):
                    return {"type": [v["type"] for v in values]}

            schema[key] = values

    return schema


def simplify_json_schema(schema: Schema, config: SimplifyJsonSchemaConfig | None = None) -> None:
    _simplify(schema, ["."], config or SimplifyJsonSchemaConfig())
