# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from beeai_framework.logger import Logger
from beeai_framework.utils.dicts import exclude_none
from beeai_framework.utils.lists import remove_falsy

Schema = dict[str, Any]

logger = Logger(__name__)

__all__ = ["simplify_json_schema"]


def _simplify(schema: Schema, path: list[str]) -> Any:
    logger.debug("Visiting:", ".".join(path))
    if not isinstance(schema, dict) or not schema:
        return schema

    for key in ("not",):
        if schema.get(key) == {}:
            del schema[key]

    if schema.get("type") == "object":
        properties = {k: _simplify(v, [*path, k]) for k, v in schema.get("properties", {}).items()}
        schema["properties"] = exclude_none(properties)

    if schema.get("type") == "array":
        items = _simplify(schema.get("items", {}), [*path, "items"])
        schema["items"] = exclude_none(items)

    for key in ("anyOf", "oneOf"):
        values = schema.get(key)
        if values and isinstance(values, list):
            values = remove_falsy([_simplify(v, [*path, key, f"{[idx]}"]) for idx, v in enumerate(values)])

            if len(values) == 1:
                logger.debug("<-", values[0])
                return values[0]

            # Not supported by certain providers
            # if values and all([v.keys() == {"type"} for v in values]):
            #     print("<-", "collapse types")
            #     return {"type": [v["type"] for v in values]}

            schema[key] = values

    logger.debug("<-", schema)
    return schema


def simplify_json_schema(schema: Schema) -> None:
    _simplify(schema, ["."])
