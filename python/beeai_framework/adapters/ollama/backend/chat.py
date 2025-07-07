# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
from typing import ClassVar

from typing_extensions import Unpack

from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.adapters.litellm.utils import parse_extra_headers
from beeai_framework.backend.chat import ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)


class OllamaChatModel(LiteLLMChatModel):
    tool_choice_support: ClassVar[set[str]] = set()

    @property
    def provider_id(self) -> ProviderName:
        return "ollama"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        text_completion: bool | None = False,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        super().__init__(
            model_id if model_id else os.getenv("OLLAMA_CHAT_MODEL", "llama3.1"),
            provider_id="text-completion-openai" if text_completion else "openai",
            **kwargs,
        )

        self._assert_setting_value("api_key", api_key, envs=["OLLAMA_API_KEY"], fallback="ollama")
        self._assert_setting_value(
            "base_url", base_url, envs=["OLLAMA_API_BASE"], fallback="http://localhost:11434", aliases=["api_base"]
        )
        if not self._settings["base_url"].endswith("/v1"):
            self._settings["base_url"] += "/v1"

        self._settings["extra_headers"] = parse_extra_headers(
            self._settings.get("extra_headers"), os.getenv("OLLAMA_API_HEADERS")
        )
