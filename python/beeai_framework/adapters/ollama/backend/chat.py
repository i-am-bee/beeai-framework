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

from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.constants import ProviderName
from beeai_framework.utils.custom_logger import BeeLogger

logger = BeeLogger(__name__)


class OllamaChatModel(LiteLLMChatModel):
    @property
    def provider_id(self) -> ProviderName:
        return "ollama"

    def __init__(self, model_id: str | None = None, settings: dict | None = None) -> None:
        _settings = settings.copy() if settings is not None else {}

        ollama_api_base_from_env = os.getenv("OLLAMA_API_BASE")

        if ollama_api_base_from_env and "base_url" not in _settings:
            _settings["base_url"] = ollama_api_base_from_env

        super().__init__(
            model_id if model_id else os.getenv("OLLAMA_CHAT_MODEL", "llama3.1"),
            settings={"base_url": "http://localhost:11434"} | _settings,
        )
