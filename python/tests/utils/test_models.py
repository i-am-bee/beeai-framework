# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from beeai_framework.utils.models import JSONSchemaModel, to_any_model


class Dog(BaseModel):
    name: str


class Cat(BaseModel):
    name: str


def test_to_any_model_raises_on_invalid() -> None:
    with pytest.raises(ValueError, match="Failed to create a model instance from the passed object!\nDog\nCat"):
        to_any_model([Dog, Cat], {"age": 5})


@pytest.mark.unit
def test_json_schema_model_recursive_schema() -> None:
    # e.g. an MCP tool whose input is a tree
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"root": {"$ref": "#/$defs/Node"}},
        "required": ["root"],
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
                },
                "required": ["name"],
            }
        },
    }

    model = JSONSchemaModel.create("tree", schema)

    value = {"root": {"name": "a", "children": [{"name": "b", "children": []}]}}
    assert model.model_validate(value).model_dump(exclude_unset=True) == value
    with pytest.raises(ValidationError):
        model.model_validate({"root": {"children": []}})

    json_schema = json.loads(json.dumps(model.model_json_schema()))
    assert "$defs" not in json_schema
    items = json_schema["properties"]["root"]["properties"]["children"]["items"]
    assert items["type"] == "object"
    assert items["additionalProperties"] is True
