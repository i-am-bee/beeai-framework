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

from collections.abc import Callable
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, RootModel, field_validator

from beeai_framework.backend.utils import parse_broken_json

T = TypeVar("T", bound=BaseModel)
TP = TypeVar("TP")


class ParserField(Generic[T]):
    def __init__(self, schema: type[T], default: T | None = None, *, as_json: bool = True) -> None:
        self.schema = schema
        self.raw = ""
        self.default = default
        self._as_json = as_json

    def write(self, chunk: str) -> None:
        self.raw += chunk

    def get(self) -> T:
        if self.default is not None and not self.raw:
            return self.default

        if self._as_json:
            input = parse_broken_json(self.raw)
            return self.schema.model_validate(input, strict=False)
        else:
            return self.schema.model_validate_strings(self.raw, strict=False)

    def get_partial(self) -> str:
        return self.raw

    def end(self) -> None:
        pass

    @staticmethod
    def from_type(
        target_type: type[TP],
        handler: Callable[[TP], TP | None] | None = None,
        /,
        default: TP | None = None,
        trim: bool = False,
    ) -> "ParserField[RootModel[TP]]":
        class CustomModel(RootModel[target_type]):  # type: ignore
            root: target_type  # type: ignore

            @field_validator("root", mode="before")
            @classmethod
            def validate_root_pre(cls, v: Any) -> Any:
                if trim and type(v) is str:
                    return v.strip()
                else:
                    return v

            @field_validator("root", mode="after")
            @classmethod
            def validate_root(cls, v: TP) -> TP:
                if not handler:
                    return v

                new_value = handler(v)
                return new_value if new_value is not None else v

        as_json = target_type in (list, dict)
        return ParserField(
            CustomModel, as_json=as_json, default=CustomModel.model_validate(default) if default is not None else None
        )
