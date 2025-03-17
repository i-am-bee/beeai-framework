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


import os
from typing import Any

from pydantic import BaseModel

from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)


class OllamaChatModel(LiteLLMChatModel):
    @property
    def provider_id(self) -> ProviderName:
        return "ollama"

    def __init__(self, model_id: str | None = None, settings: dict[str, Any] | None = None) -> None:
        settings = settings.copy() if settings is not None else {}

        api_key = settings.get("api_key", os.getenv("OLLAMA_API_KEY") or "ollama")
        base_url = settings.get("base_url", os.getenv("OLLAMA_API_BASE")) or "http://localhost:11434"
        if not base_url.endswith("/v1"):
            base_url += "/v1"

        super().__init__(
            model_id if model_id else os.getenv("OLLAMA_CHAT_MODEL", "llama3.1"),
            provider_id="openai",
            settings=settings | {"api_key": api_key, "base_url": base_url},
        )

    def _format_response_model(self, model: type[BaseModel] | dict[str, Any]) -> dict[str, Any]:
        if isinstance(model, dict):
            return model

        return {
            "type": "json_schema",
            "json_schema": {"schema": model.model_json_schema(), "name": model.__name__, "strict": True},
        }
