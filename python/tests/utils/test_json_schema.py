# Copyright 2025 © BeeAI a Series of LF Projects, LLC
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
            "hobby": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None, "title": "An Arg"},
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

    with pytest.raises(ValidationError):
        model.model_validate({"name": "aaa"})

    with pytest.raises(ValidationError):
        model.model_validate({"name": "aaa", "age": []})

    with pytest.raises(ValidationError):
        model.model_validate({"name": "aaa", "age": 123, "hobby": 123})

    # should not fail if optional fields are not included
    assert model.model_validate({"name": "aaa", "age": 25})
    assert model.model_validate({"name": "aaa", "age": 25, "hobby": "cycling"})
    assert model.model_validate({"name": "aaa", "age": 25, "hobby": None})
    assert model.model_validate({"name": "aaa", "age": 25, "hobby": "cycling"}).model_dump() == {
        "address": None,
        "age": 25,
        "hobby": "cycling",
        "is_active": None,
        "name": "aaa",
        "roles": None,
    }
