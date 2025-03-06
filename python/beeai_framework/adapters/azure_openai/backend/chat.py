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

from dotenv import load_dotenv

from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)
load_dotenv()


class AzureOpenAIChatModel(LiteLLMChatModel):
    @property
    def provider_id(self) -> ProviderName:
        return "azure_openai"

    def __init__(self, model_id: str | None = None, settings: dict | None = None) -> None:
        _settings = settings.copy() if settings is not None else {}

        api_key = _settings.get("api_key", os.getenv("AZURE_API_KEY"))
        if not api_key:
            raise ValueError(
                "Access key is required for Azure OpenAI. Specify *api_key* "
                + "or set AZURE_API_KEY  environment variable"
            )
        api_base = _settings.get("api_base", os.getenv("AZURE_API_BASE"))
        if not api_base:
            raise ValueError(
                "Base URL is required for Azure OpenAI. Specify *api_base* "
                + "or set AZURE_API_BASE  environment variable"
            )
        api_version = _settings.get("api_version", os.getenv("AZURE_API_VERSION"))
        if not api_version:
            raise ValueError(
                "API Version is required for Azure OpenAI. Specify *api_version* "
                + "or set AZURE_API_VERSION  environment variable"
            )
        super().__init__(
            (model_id if model_id else os.getenv("AZURE_CHAT_MODEL", "gpt-4o-mini")),
            provider_id="azure",
            settings=_settings
            | {
                "api_key": api_key,
                "api_base": api_base,
                "api_version": api_version,
            },
        )
