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

from dotenv import load_dotenv

from beeai_framework.adapters.litellm import utils
from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)
load_dotenv()


class AnthropicChatModel(LiteLLMChatModel):
    @property
    def provider_id(self) -> ProviderName:
        return "anthropic"

    def __init__(self, model_id: str | None = None, settings: dict[str, Any] | None = None) -> None:
        _settings = settings.copy() if settings is not None else {}

        api_key = _settings.get("api_key", os.getenv("ANTHROPIC_API_KEY"))
        if not api_key:
            raise ValueError(
                "API Key is required for Anthropic. Specify *api_key* "
                + "or set ANTHROPIC_API_KEY  environment variable"
            )

        super().__init__(
            (model_id if model_id else os.getenv("ANTHROPIC_CHAT_MODEL", "claude-3-haiku-20240307")),
            provider_id="anthropic",
            settings=_settings | {"api_key": api_key},
        )
        self._settings["extra_headers"] = utils.parse_extra_headers(
            self._settings.get("extra_headers"), os.getenv("ANTHROPIC_API_HEADERS")
        )
