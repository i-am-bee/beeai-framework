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

import inspect
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, TypeAdapter
from pydantic.json_schema import GenerateJsonSchema
from pydantic_ai._pydantic import FunctionSchema, function_schema

from beeai_framework.utils.models import JSONSchemaModel


def get_input_schema(f: Callable[..., Any]) -> type[BaseModel]:
    schema: FunctionSchema = function_schema(
        f,
        takes_ctx=False,
        docstring_format="auto",
        require_parameter_descriptions=False,
        schema_generator=GenerateJsonSchema,
    )
    return JSONSchemaModel.create(f"${f.__name__}Input", schema["json_schema"])


def get_output_schema(f: Callable[..., Any]) -> type[BaseModel]:
    sig = inspect.signature(f)
    return_type = sig.return_annotation
    if not return_type:
        raise ValueError("No return type!")

    return get_output_schema_for_value(f"${f.__name__}Output", return_type)


def get_output_schema_for_value(name: str, value: Any) -> type[BaseModel]:
    output_schema = TypeAdapter(value).json_schema(mode="serialization")
    return JSONSchemaModel.create(name, output_schema)
