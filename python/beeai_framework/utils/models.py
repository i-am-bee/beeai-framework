# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import contextlib
import copy
from abc import ABC
from collections.abc import Generator, Sequence
from contextlib import suppress
from typing import Any, Generic, Literal, Optional, Self, TypeGuard, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, GetJsonSchemaHandler, RootModel, ValidationError, create_model
from pydantic.fields import FieldInfo
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, SchemaValidator

from beeai_framework.utils.dicts import remap_key
from beeai_framework.utils.schema import simplify_json_schema

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
        from beeai_framework.backend.utils import inline_schema_refs

        schema = inline_schema_refs(copy.deepcopy(schema))
        simplify_json_schema(schema)

        type_mapping: dict[str, Any] = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "object": dict,
            "array": list,
            "null": type(None),
        }

        fields: dict[str, tuple[type, FieldInfo]] = {}

        def create_field(param_name: str, param: dict[str, Any], required: set[str]) -> tuple[type, Any]:
            any_of = param.get("anyOf")
            one_of = param.get("oneOf")
            default = param.get("default")
            const = param.get("const")

            target_field = Field(
                description=param.get("description"),
                default=default if default else const if const else None,
                pattern=param.get("pattern"),
                ge=param.get("minimum"),
                le=param.get("maximum"),
            )

            if one_of:
                return create_field(param_name, remap_key(param, source="oneOf", target="anyOf"), required)

            if any_of:
                target_types: list[type] = []
                for idx, t in enumerate(param["anyOf"]):
                    tmp_name = f"{param_name}_{idx}"
                    field, _ = create_field(tmp_name, t, {tmp_name})
                    target_types.append(field)

                if len(target_types) == 1:
                    return create_field(param_name, remap_key(param, source="anyOf", target="type"), required)
                else:
                    return Union[*target_types], target_field  # type: ignore

            else:
                enum = param.get("enum")
                raw_type = param.get("type")
                target_type: type | Any
                if isinstance(raw_type, list):
                    target_type = list[*[type_mapping.get(v) for v in raw_type]]  # type: ignore
                else:
                    target_type = type_mapping.get(raw_type)  # type: ignore[arg-type]

                    if target_type is dict and param.get("properties") is not None:
                        target_type = cls.create(param_name, param)
                    elif target_type is list and param.get("items"):
                        tmp_name = f"{param_name}_tmp"
                        given_field, given_field_info = create_field(tmp_name, param.get("items"), {tmp_name})  # type: ignore
                        target_type = list[given_field]  # type: ignore

                is_required = param_name in required
                explicitly_nullable = (
                    raw_type == "null"
                    or (isinstance(raw_type, list) and "null" in raw_type)
                    or (any_of and any(t.get("type") == "null" for t in any_of))
                    or (one_of and any(t.get("type") == "null" for t in one_of))
                )

                if enum is not None and isinstance(enum, list):
                    # pyrefly: ignore [invalid-literal]
                    target_type = Literal[tuple(enum)]
                if isinstance(const, str):
                    # pyrefly: ignore [not-a-type]
                    target_type = Literal[const]
                if not target_type:
                    target_type = type

                if (not is_required and not default) or explicitly_nullable:
                    target_type = Optional[target_type] if target_type else type(None)  # noqa: UP007

            return (  # type: ignore
                target_type,
                target_field,
            )

        properties = schema.get("properties", {})
        # pyrefly: ignore [bad-keyword-argument, no-matching-overload]
        updated_config = ConfigDict(**cls.model_config, title=schema.get("title", None))
        updated_config["extra"] = "allow" if schema.get("additionalProperties") else "forbid"
        updated_config["arbitrary_types_allowed"] = True

        if not properties and schema.get("type") != "object":
            properties["root"] = schema

        for param_name, param in properties.items():
            fields[param_name] = create_field(param_name, param, set(schema.get("required", [])))

        model: type[JSONSchemaModel] = create_model(  # type: ignore
            schema_name, __base__=cls, **fields, __config__=updated_config
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


class WrappedRootModel(BaseModel, Generic[T]):
    item: RootModel[T] = Field(..., title="Item")

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
        **kwargs: Any,
    ) -> Self:
        try:
            return super().model_validate(
                obj, strict=strict, from_attributes=from_attributes, context=context, **kwargs
            )
        except ValidationError as e:
            with contextlib.suppress(ValidationError):
                return cls(item=obj)
            raise e


def is_pydantic_model(obj: Any) -> TypeGuard[type[BaseModel]]:
    return isinstance(obj, type) and issubclass(obj, BaseModel)


class AnyModel(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
