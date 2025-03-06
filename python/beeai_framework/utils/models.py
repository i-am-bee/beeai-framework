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

from collections.abc import Sequence
from contextlib import suppress
from typing import Any, TypeVar, Union

from pydantic import BaseModel, Field, create_model
from pydantic_core import SchemaValidator

T = TypeVar("T", bound=BaseModel)
ModelLike = Union[T, dict]  # noqa: UP007


def to_model(cls: type[T], obj: ModelLike[T]) -> T:
    return obj if isinstance(obj, cls) else cls.model_validate(obj, strict=False, from_attributes=True)


def to_any_model(classes: Sequence[type[BaseModel]], obj: ModelLike[T]) -> Any:
    if len(classes) == 1:
        return to_model(classes[0], obj)

    for cls in classes:
        with suppress(Exception):
            return to_model(cls, obj)

    return ValueError(
        "Failed to create a model instance from the passed object!" + "\n".join(cls.__name__ for cls in classes),
    )


def to_model_optional(cls: type[T], obj: ModelLike[T] | None) -> T | None:
    return None if obj is None else to_model(cls, obj)


def check_model(model: T) -> None:
    schema_validator = SchemaValidator(schema=model.__pydantic_core_schema__)
    schema_validator.validate_python(model.__dict__)


type_mapping = {"string": str, "integer": int, "number": float, "boolean": bool}


def json_to_model(model_name: str, schema: dict) -> type[BaseModel]:
    fields = {}
    for param_name, param in schema.get("properties").items():
        fields[param_name] = (
            type_mapping.get(param.get("type")),
            Field(description=param.get("description"), default=None),
        )
    model = create_model(model_name, **fields)
    return model
