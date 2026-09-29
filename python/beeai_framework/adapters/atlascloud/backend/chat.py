# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os

from typing_extensions import Unpack

from beeai_framework.adapters.litellm import LiteLLMChatModel, utils
from beeai_framework.backend.chat import ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)

ATLASCLOUD_API_BASE = "https://api.atlascloud.ai/v1"


class AtlasCloudChatModel(LiteLLMChatModel):
    """
    A chat model implementation for the Atlas Cloud provider, leveraging LiteLLM.

    Atlas Cloud provides an OpenAI-compatible API. This adapter routes requests
    through LiteLLM's OpenAI provider with the Atlas Cloud base URL.

    Model ids are vendor-qualified, e.g. "openai/gpt-4.1-mini" or
    "deepseek-ai/DeepSeek-V3.1-Terminus".
    """

    @property
    def provider_id(self) -> ProviderName:
        """The provider ID for Atlas Cloud."""
        return "atlascloud"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        """
        Initializes the AtlasCloudChatModel.

        Args:
            model_id: The ID of the Atlas Cloud model to use. If not provided,
                it falls back to the ATLASCLOUD_CHAT_MODEL environment variable,
                and then defaults to 'openai/gpt-4.1-mini'.
            api_key: The Atlas Cloud API key. Falls back to ATLASCLOUD_API_KEY env var.
            base_url: The Atlas Cloud API base URL. Falls back to ATLASCLOUD_API_BASE
                env var, then defaults to 'https://api.atlascloud.ai/v1'.
            **kwargs: Additional settings to configure the provider.
        """
        super().__init__(
            model_id if model_id else os.getenv("ATLASCLOUD_CHAT_MODEL", "openai/gpt-4.1-mini"),
            provider_id="openai",
            **kwargs,
        )

        self._assert_setting_value("api_key", api_key, envs=["ATLASCLOUD_API_KEY"])
        self._assert_setting_value(
            "base_url",
            base_url,
            envs=["ATLASCLOUD_API_BASE"],
            aliases=["api_base"],
            allow_empty=True,
            fallback=ATLASCLOUD_API_BASE,
        )
        self._settings["extra_headers"] = utils.parse_extra_headers(
            self._settings.get("extra_headers"), os.getenv("ATLASCLOUD_API_HEADERS")
        )
