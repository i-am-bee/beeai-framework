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
            "headers": {"type": "object", "description": "Optional headers to include in the request"},
        },
        "required": ["input"],
        "title": "runArguments",
        "type": "object",
    }


@pytest.fixture
def basic() -> dict[str, list[str] | str | Any]:
    return {
        "$id": "https://example.com/person.schema.json",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Person",
        "type": "object",
        "properties": {
            "firstName": {"type": "string", "description": "The person's first name."},
            "lastName": {"type": "string", "description": "The person's last name."},
            "age": {
                "description": "Age in years which must be equal to or greater than zero.",
                "type": "integer",
                "minimum": 0,
            },
        },
    }


@pytest.fixture
def arrays_of_things() -> dict[str, list[str] | str | Any]:
    return {
        "$id": "https://example.com/arrays.schema.json",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "description": "Arrays of strings and objects",
        "title": "Arrays",
        "type": "object",
        "properties": {
            "fruits": {"type": "array", "items": {"type": "string"}},
            "vegetables": {"type": "array", "items": {"$ref": "#/$defs/veggie"}},
        },
        "$defs": {
            "veggie": {
                "type": "object",
                "required": ["veggieName", "veggieLike"],
                "properties": {
                    "veggieName": {"type": "string", "description": "The name of the vegetable."},
                    "veggieLike": {"type": "boolean", "description": "Do I like this vegetable?"},
                },
            }
        },
    }


@pytest.fixture
def enumareted_values() -> dict[str, list[str] | str | Any]:
    return {
        "$id": "https://example.com/enumerated-values.schema.json",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Enumerated Values",
        "type": "object",
        "properties": {"data": {"enum": [42, True, "hello", None, (1, 2, 3)]}},
    }


@pytest.fixture
def regular_expression() -> dict[str, list[str] | str | Any]:
    return {
        "$id": "https://example.com/regex-pattern.schema.json",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Regular Expression Pattern",
        "type": "object",
        "properties": {"code": {"type": "string", "pattern": "^[A-Z]{3}-\\d{3}$"}},
    }


@pytest.fixture
def complex_object_with_nested_properties() -> dict[str, list[str] | str | Any]:
    return {
        "$id": "https://example.com/complex-object.schema.json",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Complex Object",
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                    "state": {"type": "string"},
                    "postalCode": {"type": "string", "pattern": "\\d{5}"},
                },
                "required": ["street", "city", "state", "postalCode"],
            },
            "hobbies": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "age"],
    }


@pytest.fixture
def conditional_validation_with_if_else() -> dict[str, list[str] | str | Any]:
    return {
        "$id": "https://example.com/conditional-validation-if-else.schema.json",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Conditional Validation with If-Else",
        "type": "object",
        "properties": {"isMember": {"type": "boolean"}, "membershipNumber": {"type": "string"}},
        "required": ["isMember"],
        "if": {"properties": {"isMember": {"const": True}}},
        "then": {"properties": {"membershipNumber": {"type": "string", "minLength": 10, "maxLength": 10}}},
        "else": {"properties": {"membershipNumber": {"type": "string", "minLength": 15}}},
    }


"""
Unit Tests
"""


@pytest.mark.unit
def test_schema_with_additional_properties(schema_with_additional_properties: dict[str, list[str] | str | Any]) -> None:
    model = JSONSchemaModel.create("schema_with_additional_properties", schema_with_additional_properties)
    assert model.model_json_schema()

    with pytest.raises(ValidationError):
        model.model_validate({"config": {"test": "test"}})

    assert model.model_validate({"input": {"query": "test"}})
    assert model.model_validate({"input": {"query": "test"}}).model_dump()["input"] == {"query": "test"}

    assert model.model_fields["input"].annotation is dict
    assert get_args(model.model_fields["config"].annotation)[0].model_fields["max_retries"].annotation is int
    assert model.model_validate({"input": {"query": "test query"}})


@pytest.mark.unit
@pytest.mark.parametrize("additional_properties", [True, False])
def test_empty_object_schema(additional_properties: bool) -> None:
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "additionalProperties": additional_properties,
        "properties": {},
        "type": "object",
    }

    model = JSONSchemaModel.create("test_empty_object", schema)
    assert model.model_json_schema()


@pytest.mark.unit
def test_json_schema_model(test_json_schema: dict[str, list[str] | str | Any]) -> None:
    model = JSONSchemaModel.create("test_json_schema", test_json_schema)
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
def test_basic_schema(basic: dict[str, list[str] | str | Any]) -> None:
    model = JSONSchemaModel.create("basic", basic)
    assert model.model_json_schema()

    assert model.model_fields["firstName"].annotation == str | NoneType
    assert model.model_fields["lastName"].annotation == str | NoneType
    assert model.model_fields["age"].annotation == int | NoneType


@pytest.mark.unit
def test_arrays_of_things_schema(arrays_of_things: dict[str, list[str] | str | Any]) -> None:
    model = JSONSchemaModel.create("arrays_of_things", arrays_of_things)
    assert model.model_json_schema()

    assert model.model_fields["fruits"].annotation == Optional[list[Optional[str]]]  # type: ignore[comparison-overlap]
    vegetables = get_args(model.model_fields["vegetables"].annotation)[0]
    vegetable = get_args(get_args(vegetables)[0])[0]
    assert vegetable.model_fields["veggieName"].annotation is str
    assert vegetable.model_fields["veggieLike"].annotation is bool


@pytest.mark.unit
def test_enumareted_values(enumareted_values: dict[str, list[str] | str | Any]) -> None:
    model = JSONSchemaModel.create("enumareted_values", enumareted_values)
    assert model.model_json_schema()

    assert model.model_fields["data"].annotation == Literal[42, True, "hello", None, (1, 2, 3)]  # type: ignore[comparison-overlap]


@pytest.mark.unit
def test_regular_expression(regular_expression: dict[str, list[str] | str | Any]) -> None:
    model = JSONSchemaModel.create("regular_expression", regular_expression)
    assert model.model_json_schema()

    assert model.model_fields["code"].metadata[0].pattern == "^[A-Z]{3}-\\d{3}$"
    assert model.model_fields["code"].annotation == str | NoneType


@pytest.mark.unit
def test_complex_object_with_nested_properties(
    complex_object_with_nested_properties: dict[str, list[str] | str | Any],
) -> None:
    model = JSONSchemaModel.create("complex_object_with_nested_properties", complex_object_with_nested_properties)
    assert model.model_json_schema()

    assert model.model_fields["age"].metadata[0].ge == 0
    assert model.model_fields["age"].annotation is int


@pytest.mark.unit
def test_conditional_validation_with_if_else(
    conditional_validation_with_if_else: dict[str, list[str] | str | Any],
) -> None:
    model = JSONSchemaModel.create("conditional_validation_with_if_else", conditional_validation_with_if_else)
    assert model.model_json_schema()

    assert model.model_fields["isMember"].annotation is bool
    assert model.model_fields["membershipNumber"].annotation == str | NoneType


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
