# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Unpack

from beeai_framework.adapters.litellm import LiteLLMChatModel, utils
from beeai_framework.backend.chat import ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)

RUNCRATE_API_BASE = "https://api.runcrate.ai/v1"


class RuncrateChatModel(LiteLLMChatModel):
    """
    A chat model implementation for the Runcrate provider, leveraging LiteLLM.

    Runcrate provides an OpenAI-compatible API for open models. This adapter routes requests
    through LiteLLM's OpenAI provider with the Runcrate base URL. Model IDs are the ones listed
    at https://api.runcrate.ai/v1/models (for tool calling, pick one that lists `tools` in its
    `supported_features`).
    """

    @property
    def provider_id(self) -> ProviderName:
        return "runcrate"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        """
        Initializes the RuncrateChatModel.

        Args:
            model_id: The ID of the Runcrate model to use. If not provided, it falls back to the
                RUNCRATE_CHAT_MODEL environment variable, and then defaults to
                'deepseek-ai/DeepSeek-V4.1-Flash'.
            api_key: The Runcrate API key. Falls back to the RUNCRATE_API_KEY env var.
            base_url: The Runcrate API base URL. Falls back to the RUNCRATE_API_BASE env var, and
                then to 'https://api.runcrate.ai/v1'.
            **kwargs: Additional settings to configure the provider.
        """
        super().__init__(
            model_id if model_id else os.getenv("RUNCRATE_CHAT_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash"),
            provider_id="openai",  # LiteLLM serves OpenAI-compatible endpoints through its openai provider
            **kwargs,
        )

        self._assert_setting_value("api_key", api_key, envs=["RUNCRATE_API_KEY"])
        self._assert_setting_value(
            "base_url", base_url, envs=["RUNCRATE_API_BASE"], aliases=["api_base"], fallback=RUNCRATE_API_BASE
        )
        self._settings["extra_headers"] = utils.parse_extra_headers(
            self._settings.get("extra_headers"), os.getenv("RUNCRATE_API_HEADERS")
        )
