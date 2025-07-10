# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from types import NoneType
from typing import Any, Literal, Optional, get_args

import pytest
from pydantic import ValidationError

from beeai_framework.utils import JSONSchemaModel

"""
Utility functions and classes
"""


@pytest.fixture
def test_json_schema() -> dict[str, list[str] | str | Any]:
    return {
        "title": "User",
        "type": "object",
        "properties": {
            "object": {"const": "user"},
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "is_active": {"type": "boolean", "default": True},
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                    "zipcode": {"type": "integer"},
                },
                "required": ["city"],
            },
            "roles": {"type": "array", "items": {"type": "string", "enum": ["admin", "user", "guest"]}},
            "hobby": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None, "title": "An Arg"},
            "contact": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
        },
        "required": ["name", "age", "contact"],
    }


@pytest.fixture
def schema_with_additional_properties() -> dict[str, list[str] | str | Any]:
    return {
        "properties": {
            "input": {
                "additionalProperties": True,
                "title": "Input",
                "type": "object",
            },
            "config": {
                "type": "object",
                "properties": {
                    "max_retries": {"type": "integer", "default": 3, "title": "Max Retries"},
                },
            },
        },
        "required": ["input"],
        "title": "runArguments",
        "type": "object",
    }


"""
Unit Tests
"""


@pytest.mark.unit
def test_schema_with_additional_properties(schema_with_additional_properties: dict[str, list[str] | str | Any]) -> None:
    model = JSONSchemaModel.create("test_schema", schema_with_additional_properties)
    assert model.model_json_schema()

    assert model.model_fields["input"].annotation.model_fields == {}  # type: ignore
    assert get_args(model.model_fields["config"].annotation)[0].model_fields["max_retries"].annotation == int
    assert model.model_validate({"input": {"query": "test query"}})


@pytest.mark.unit
def test_json_schema_model(test_json_schema: dict[str, list[str] | str | Any]) -> None:
    model = JSONSchemaModel.create("test_schema", test_json_schema)
    assert model.model_json_schema()

    with pytest.raises(ValidationError):
        model.model_validate({"name": "aaa"})

    with pytest.raises(ValidationError):
        model.model_validate({"name": "aaa", "age": []})

    with pytest.raises(ValidationError):
        model.model_validate({"name": "aaa", "age": 123, "hobby": 123})

    # should not fail if optional fields are not included
    assert model.model_validate({"name": "aaa", "age": 25, "contact": 123456789})
    assert model.model_validate({"name": "aaa", "age": 25, "hobby": "cycling", "contact": "name@email.com"})
    assert model.model_validate({"name": "aaa", "age": 25, "hobby": None, "contact": "name@email.com"})
    assert model.model_validate(
        {"name": "aaa", "age": 25, "hobby": "cycling", "contact": "name@email.com"}
    ).model_dump() == {
        "object": "user",
        "address": None,
        "age": 25,
        "hobby": "cycling",
        "is_active": True,
        "name": "aaa",
        "roles": None,
        "contact": "name@email.com",
    }

    assert model.model_fields["object"].annotation is Literal["user"], "Expected annotation to be `Literal['user']`"  # type: ignore
    assert model.model_fields["name"].annotation is str, "Expected annotation to be `str`"
    assert model.model_fields["age"].annotation is int, "Expected annotation to be `int`"
    assert model.model_fields["is_active"].annotation is bool, "Expected annotation to be `bool`"
    # ruff: noqa: UP007, E501
    assert model.model_fields["roles"].annotation is Optional[list[Literal[("admin", "user", "guest")]]], (  # type: ignore
        "Expected correct type"
    )
    assert get_args(model.model_fields["address"].annotation)[0].model_fields["city"].annotation is str
    assert get_args(model.model_fields["address"].annotation)[0].model_fields["street"].annotation == str | NoneType
    assert get_args(model.model_fields["address"].annotation)[0].model_fields["zipcode"].annotation == int | NoneType
    assert model.model_fields["hobby"].annotation == str | NoneType, "Expected annotation to be `Optional[str]`"
    assert model.model_fields["contact"].annotation == str | int, "Expected annotation to be `Union[str, int]`"


@pytest.mark.unit
def test_preserve_default_type_not_optional() -> None:
    """
    Regression test for incorrect annotation of defaulted fields.

    Ensures that fields with a default value (but no explicit nullability)
    are treated as non-optional, with the correct type and preserved metadata.
    """
    json_schema = {
        "title": "great_toolArguments",
        "type": "object",
        "properties": {
            "an_arg": {
                "type": "string",
                "default": "default string",
                "description": "great description",
                "title": "An Arg",
            }
        },
        "required": [],
    }

    model = JSONSchemaModel.create("great_toolArguments", json_schema)
    field_info = model.model_fields["an_arg"]

    # should not wrap type in Optional
    assert field_info.annotation is str, "Expected annotation to be `str`, not `Optional[str]`"

    # should preserve the default value
    assert field_info.default == "default string", "Expected default value to be 'default string'"

    # should preserve the description
    assert field_info.description == "great description", "Expected description to be 'great description'"

    # should not mark the field as required
    assert not field_info.is_required(), "Expected field to be optional due to default value"
