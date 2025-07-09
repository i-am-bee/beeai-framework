# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from collections.abc import Generator, Sequence
from contextlib import suppress
from logging import Logger
from typing import Any, Literal, Optional, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, GetJsonSchemaHandler, RootModel, create_model
from pydantic.fields import FieldInfo
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, SchemaValidator

from beeai_framework.utils.dicts import remap_key

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
            any_of = param.get("anyOf")
            one_of = param.get("oneOf")
            default = param.get("default")
            const = param.get("const")

            target_field = Field(
                description=param.get("description"),
                default=default if default else const if const else None,
            )

            if one_of:
                logger.debug(
                    f"{JSONSchemaModel.__name__}: does not support 'oneOf' modifier found in {param_name} attribute."
                    f" Will use 'anyOf' instead."
                )
                return create_field(param_name, remap_key(param, source="oneOf", target="anyOf"))

            if any_of:
                target_types: list[type] = [create_field(param_name, t)[0] for t in param["anyOf"]]
                if len(target_types) == 1:
                    return create_field(param_name, remap_key(param, source="anyOf", target="type"))
                else:
                    return Union[*target_types], target_field  # type: ignore

            else:
                raw_type = param.get("type")
                enum = param.get("enum")

                target_type: type | Any = type_mapping.get(raw_type)  # type: ignore[arg-type]

                if target_type is dict:
                    target_type = cls.create(param_name, param)

                if target_type is list:
                    target_type = list[create_field(param_name, param.get("items"))[0]]  # type: ignore

                is_required = param_name in required
                explicitly_nullable = (
                    raw_type == "null"
                    or (isinstance(raw_type, list) and "null" in raw_type)
                    or (any_of and any(t.get("type") == "null" for t in any_of))
                    or (one_of and any(t.get("type") == "null" for t in one_of))
                )
                if (not is_required and not default) or explicitly_nullable:
                    target_type = Optional[target_type] if target_type else type(None)  # noqa: UP007

                if enum is not None and isinstance(enum, list):
                    target_type = Literal[tuple(enum)]
                if isinstance(const, str):
                    target_type = Literal[const]
                if not target_type:
                    logger.debug(
                        f"{JSONSchemaModel.__name__}: Can't resolve a correct type for '{param_name}' attribute."
                        f" Using 'Any' as a fallback."
                    )
                    target_type = type

            return (
                target_type,
                target_field,
            )

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
