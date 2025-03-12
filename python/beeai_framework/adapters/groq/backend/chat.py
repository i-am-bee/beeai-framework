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

from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)


class GroqChatModel(LiteLLMChatModel):
    @property
    def provider_id(self) -> ProviderName:
        return "groq"

    def __init__(self, model_id: str | None = None, settings: dict[str, Any] | None = None) -> None:
        _settings = settings.copy() if settings is not None else {}

        api_key = _settings.get("api_key", os.getenv("GROQ_API_KEY"))

        super().__init__(
            model_id if model_id else os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant"),
            provider_id="groq",
            settings=_settings | {"api_key": api_key},
        )
