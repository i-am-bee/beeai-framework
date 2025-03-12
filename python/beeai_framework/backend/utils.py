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


from importlib import import_module
from typing import Any, Literal, TypeVar

import json_repair

from beeai_framework.backend.constants import (
    BackendProviders,
    ProviderDef,
    ProviderModelDef,
    ProviderName,
)
from beeai_framework.backend.errors import BackendError

T = TypeVar("T")

# TODO: `${ProviderName}:${string}`
FullModelName: str


def find_provider_def(value: str) -> ProviderDef | None:
    for provider in BackendProviders.values():
        if value == provider.name or value == provider.module or value in provider.aliases:
            return provider
    return None


def parse_model(name: str) -> ProviderModelDef:
    if not name:
        raise BackendError("Neither 'provider' nor 'provider:model' was specified.")

    # provider_id:model_id
    # e.g., ollama:llama3.1
    # keep remainder of string intact (maxsplit=1) because model name can also have colons
    name_parts = name.split(":", maxsplit=1)
    provider_def = find_provider_def(name_parts[0])

    if not provider_def:
        raise BackendError("Model does not contain provider name!")

    return ProviderModelDef(
        provider_id=name_parts[0],
        model_id=name_parts[1] if len(name_parts) > 1 else None,
        provider_def=provider_def,
    )


def load_model(name: ProviderName | str, model_type: Literal["embedding", "chat"] = "chat") -> type[T]:
    parsed = parse_model(name)
    provider_def = parsed.provider_def

    module_path = f"beeai_framework.adapters.{provider_def.module}.backend.{model_type}"
    module = import_module(module_path)

    class_name = f"{provider_def.name}{model_type.capitalize()}Model"
    return getattr(module, class_name)  # type: ignore


def parse_broken_json(input: str) -> Any:
    return json_repair.loads(input)
