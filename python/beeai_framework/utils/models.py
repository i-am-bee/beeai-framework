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

from abc import ABC
from collections.abc import Generator, Sequence
from contextlib import suppress
from logging import Logger
from typing import Any, Literal, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, GetJsonSchemaHandler, RootModel, create_model
from pydantic.fields import FieldInfo
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, SchemaValidator

logger = Logger(__name__)

T = TypeVar("T", bound=BaseModel)
ModelLike = Union[T, dict[str, Any]]  # noqa: UP007


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


class JSONSchemaModel(ABC, BaseModel):
    _custom_json_schema: JsonSchemaValue

    model_config = ConfigDict(
        arbitrary_types_allowed=False, validate_default=True, json_schema_mode_override="validation"
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args and not kwargs and type(self).model_fields.keys() == {"root"}:
            kwargs["root"] = args[0]

        super().__init__(**kwargs)

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: CoreSchema,
        handler: GetJsonSchemaHandler,
        /,
    ) -> JsonSchemaValue:
        return cls._custom_json_schema.copy()

    @classmethod
    def create(cls, schema_name: str, schema: dict[str, Any]) -> type["JSONSchemaModel"]:
        type_mapping: dict[str, Any] = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "object": dict,
            "array": list,
            "null": None,
        }

        fields: dict[str, tuple[type, FieldInfo]] = {}
        required = set(schema.get("required", []))

        def create_field(param_name: str, param: dict[str, Any]) -> tuple[type, Any]:
            raw_type = param.get("type")
            any_of = param.get("anyOf")
            const = param.get("const")
            default = param.get("default", ...)
            enum = param.get("enum")
            target_type: Any

            # Determine base type(s)
            if const is not None:
                target_type = Literal[const]
                default = const

            elif any_of:
                sub_types = []
                for variant in any_of:
                    variant_type = type_mapping.get(variant.get("type"), Any)
                    if "enum" in variant:
                        variant_type = Literal[*variant["enum"]]
                    sub_types.append(variant_type)
                if len(sub_types) == 1:
                    target_type = sub_types[0]
                else:
                    target_type = sub_types[0]
                    for t in sub_types[1:]:
                        target_type |= t

            elif enum is not None and isinstance(enum, list):
                target_type = Literal[tuple(enum)]

            elif isinstance(raw_type, list):
                sub_types = [type_mapping.get(t, Any) for t in raw_type]
                if len(sub_types) == 1:
                    target_type = sub_types[0]
                else:
                    target_type = sub_types[0]
                    for t in sub_types[1:]:
                        target_type |= t
            else:
                target_type = type_mapping.get(str(raw_type), Any)

            is_required = param_name in required
            explicitly_nullable = (
                raw_type == "null"
                or (isinstance(raw_type, list) and "null" in raw_type)
                or (any_of and any(t.get("type") == "null" for t in any_of))
            )

            if not is_required and default is ...:
                default = None
                target_type |= type(None)

            elif explicitly_nullable:
                target_type |= type(None)

            field = Field(default=default, description=param.get("description"))

            return target_type, field

        properties = schema.get("properties", {})
        if not properties:
            properties["root"] = schema

        for param_name, param in properties.items():
            fields[param_name] = create_field(param_name, param)

        model: type[JSONSchemaModel] = create_model(  # type: ignore
            schema_name, __base__=cls, **fields
        )

        model._custom_json_schema = schema

        return model


def update_model(target: T, *, sources: list[T | None | bool], exclude_unset: bool = True) -> None:
    for source in sources:
        if not isinstance(source, BaseModel):
            continue

        for k, v in source.model_dump(exclude_unset=exclude_unset, exclude_defaults=True).items():
            setattr(target, k, v)


class ListModel(RootModel[list[T]]):
    root: list[T]

    def __iter__(self) -> Generator[tuple[str, T], None, None]:
        for i, item in enumerate(self.root):
            yield str(i), item

    def __getitem__(self, item: int) -> T:
        return self.root[item]


def to_list_model(target: type[T], field: FieldInfo | None = None) -> type[ListModel[T]]:
    field = field or Field(...)

    class CustomListModel(ListModel[target]):  # type: ignore
        root: list[target] = field  # type: ignore

    return CustomListModel
