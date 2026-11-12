# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os

from typing_extensions import Unpack

from beeai_framework.adapters.litellm import LiteLLMChatModel, utils
from beeai_framework.backend.chat import ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)

NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"


class NVIDIAChatModel(LiteLLMChatModel):
    """
    A chat model implementation for the NVIDIA provider, leveraging LiteLLM.

    NVIDIA NIM provides an OpenAI-compatible API. This adapter routes requests
    through LiteLLM's OpenAI provider with the NVIDIA base URL.

    The default endpoint is the NVIDIA API catalog
    (https://integrate.api.nvidia.com/v1). Point it at a self-hosted NIM with
    an explicit ``base_url`` / NVIDIA_API_BASE.

    The default model is nvidia/nemotron-3.5-super-vl-120b-a12b.
    Explicit model IDs and NVIDIA_CHAT_MODEL overrides are preserved.
    """

    @property
    def provider_id(self) -> ProviderName:
        """The provider ID for NVIDIA."""
        return "nvidia"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        """
        Initializes the NVIDIAChatModel.

        Args:
            model_id: The ID of the NVIDIA model to use. If not provided,
                it falls back to the NVIDIA_CHAT_MODEL environment variable,
                and then defaults to 'nvidia/nemotron-3.5-super-vl-120b-a12b'.
            api_key: The NVIDIA API key. Falls back to NVIDIA_API_KEY env var.
            base_url: The NVIDIA API base URL. Falls back to NVIDIA_API_BASE
                env var, and finally to 'https://integrate.api.nvidia.com/v1'.
            **kwargs: Additional settings to configure the provider.
        """
        super().__init__(
            model_id if model_id else os.getenv("NVIDIA_CHAT_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b"),
            provider_id="openai",
            **kwargs,
        )

        self._assert_setting_value("api_key", api_key, envs=["NVIDIA_API_KEY"])
        self._assert_setting_value(
            "base_url",
            base_url,
            envs=["NVIDIA_API_BASE"],
            aliases=["api_base"],
            allow_empty=True,
            fallback=NVIDIA_API_BASE,
        )
        self._settings["extra_headers"] = utils.parse_extra_headers(
            self._settings.get("extra_headers"), os.getenv("NVIDIA_API_HEADERS")
        )
