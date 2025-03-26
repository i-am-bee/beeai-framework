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

from beeai_framework.adapters.litellm import utils
from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)


class VertexAIChatModel(LiteLLMChatModel):
    @property
    def provider_id(self) -> ProviderName:
        return "vertexai"

    def __init__(self, model_id: str | None = None, settings: dict[str, Any] | None = None) -> None:
        _settings = settings.copy() if settings is not None else {}

        vertexai_project = _settings.get("vertexai_project", os.getenv("GOOGLE_VERTEX_PROJECT"))
        if not vertexai_project:
            raise ValueError(
                "Project ID is required for Vertex AI model. Specify *vertexai_project* "
                + "or set GOOGLE_VERTEX_PROJECT environment variable"
            )

        vertexai_location = _settings.get("vertexai_location", os.getenv("GOOGLE_VERTEX_LOCATION"))

        # Ensure standard google auth credentials are available
        # Set GOOGLE_APPLICATION_CREDENTIALS / GOOGLE_CREDENTIALS / GOOGLE_APPLICATION_CREDENTIALS_JSON

        super().__init__(
            model_id if model_id else os.getenv("GOOGLE_VERTEX_CHAT_MODEL", "gemini-2.0-flash-lite-001"),
            provider_id="vertex_ai",
            settings=_settings
            | {
                "vertex_project": vertexai_project,
                "vertex_location": vertexai_location,
            },
        )
        self._settings["extra_headers"] = utils.parse_extra_headers(
            self._settings.get("extra_headers"), os.getenv("GOOGLE_VERTEX_API_HEADERS")
        )
