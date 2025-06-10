# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0


from typing import Any

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
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "is_active": {"type": "boolean"},
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                    "zipcode": {"type": "integer"},
                },
            },
            "roles": {"type": "array", "items": {"type": "string", "enum": ["admin", "user", "guest"]}},
        },
        "required": ["name", "age"],
    }


"""
Unit Tests
"""


@pytest.mark.unit
def test_json_schema_model(test_json_schema: dict[str, list[str] | str | Any]) -> None:
    model = JSONSchemaModel.create("test_schema", test_json_schema)
    assert model.model_json_schema()

    # should throw exception if required fields are missing
    with pytest.raises(ValidationError):  # No description provided
        model.model_validate({"name": "aaa"})

    # should not fail if optional fields are not included
    assert model.model_validate({"name": "aaa", "age": 25})
