# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os

from dotenv import load_dotenv
from typing_extensions import Unpack

from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.adapters.litellm.utils import parse_extra_headers
from beeai_framework.backend.chat import ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)
load_dotenv()


class AzureOpenAIChatModel(LiteLLMChatModel):
    @property
    def provider_id(self) -> ProviderName:
        return "azure_openai"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        super().__init__(
            model_id=(model_id if model_id is not None else os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-4o-mini")),
            provider_id="azure",
            **kwargs,
        )

        self._assert_setting_value("api_key", api_key, envs=["AZURE_OPENAI_API_KEY", "AZURE_API_KEY"])
        self._assert_setting_value(
            "base_url", base_url, envs=["AZURE_OPENAI_API_BASE", "AZURE_API_BASE"], aliases=["api_base"]
        )
        self._assert_setting_value("api_version", api_version, envs=["AZURE_OPENAI_API_VERSION", "AZURE_API_VERSION"])
        self._settings["extra_headers"] = parse_extra_headers(
            self._settings.get("extra_headers"), os.getenv("AZURE_OPENAI_API_HEADERS")
        )
