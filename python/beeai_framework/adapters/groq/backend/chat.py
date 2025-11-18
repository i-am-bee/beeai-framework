# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os

from typing_extensions import Unpack

from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.chat import ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)


class GroqChatModel(LiteLLMChatModel):
    @property
    def provider_id(self) -> ProviderName:
        return "groq"

    def __init__(
        self, model_id: str | None = None, api_key: str | None = None, **kwargs: Unpack[ChatModelKwargs]
    ) -> None:
        super().__init__(
            model_id if model_id else os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant"),
            provider_id="groq",
            **kwargs,
        )
        self._assert_setting_value("api_key", api_key, envs=["GROQ_API_KEY"])
