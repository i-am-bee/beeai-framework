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


import json
import os

from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)


class OpenAIChatModel(LiteLLMChatModel):
    @property
    def provider_id(self) -> ProviderName:
        return "openai"

    def __init__(self, model_id: str | None = None, settings: dict | None = None) -> None:
        _settings = settings.copy() if settings is not None else {}

        # Extra headers -- ignored if invalid
        extra_headers = None
        extra_headers_json = _settings.get("extra_headers", os.getenv("OPENAI_EXTRA_HEADERS"))

        if extra_headers_json:
            try:
                parsed_headers = json.loads(extra_headers_json)
                if isinstance(parsed_headers, dict):
                    extra_headers = parsed_headers
            except json.JSONDecodeError:
                pass

        if extra_headers:
            _settings["headers"] = extra_headers

        super().__init__(
            model_id if model_id else os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"),
            provider_id="openai",
            settings=_settings,
        )
