# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any


def exclude_keys(input: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {k: input[k] for k in input.keys() - keys}


def include_keys(input: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    valid_keys = [k for k in input if k in keys]
    return {k: input[k] for k in valid_keys}


def exclude_none(input: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in input.items() if v is not None}


def exclude_non_annotated(input: Mapping[str, Any], cls: type[Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(input, dict):
        raise ValueError("input must be a TypedDict")

    excluded: dict[str, Any] = {}

    valid_keys = cls.__annotations__.keys()
    for k, v in list(input.items()):
        if k not in valid_keys:
            excluded[k] = v
            input.pop(k)

    return excluded


def remap_key(obj: dict[str, Any], *, source: str, target: str, fallback: Any | None = None) -> dict[str, Any]:
    clone = {**obj}
    clone[target] = clone.pop(source, fallback)
    return clone


def set_attr_if_none(obj: dict[str, Any], attrs: list[str], value: Any) -> None:
    for attr, next_attr in zip(attrs, attrs[1:] + [None], strict=False):
        if not isinstance(obj, dict):
            raise ValueError(f"obj must be a dict, got {type(obj)}")

        if obj.get(attr) is not None:
            obj = obj[attr]
        elif next_attr is None:
            obj[attr] = value
        else:
            obj[attr] = {}
            obj = obj[attr]
