# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os

from typing_extensions import Unpack

from beeai_framework.adapters.litellm import LiteLLMChatModel, utils
from beeai_framework.backend.chat import ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)

MINIMAX_API_BASE = "https://api.minimax.io/v1"
MINIMAX_API_BASE_CN = "https://api.minimaxi.com/v1"

# Region identifiers accepted by ``region`` / the MINIMAX_API_REGION env var,
# mapped to the matching OpenAI-compatible base URL. MiniMax serves the same API
# from two regional gateways; this mapping lets callers select one by name
# instead of discovering and typing the base URL by hand.
MINIMAX_REGION_BASE_URLS = {
    "global": MINIMAX_API_BASE,
    "global_en": MINIMAX_API_BASE,
    "cn": MINIMAX_API_BASE_CN,
    "cn_zh": MINIMAX_API_BASE_CN,
}
DEFAULT_MINIMAX_REGION = "global"


def resolve_minimax_base_url(region: str | None) -> str:
    """
    Resolve a MiniMax region identifier to its OpenAI-compatible base URL.

    An empty or missing region resolves to the global endpoint. Unknown
    identifiers raise ``ValueError`` listing the supported regions.
    """
    normalized = (region or "").strip().lower()
    if not normalized:
        return MINIMAX_API_BASE
    try:
        return MINIMAX_REGION_BASE_URLS[normalized]
    except KeyError:
        raise ValueError(
            f"Unknown MiniMax region '{region}'. Supported regions: {', '.join(sorted(MINIMAX_REGION_BASE_URLS))}."
        ) from None


class MiniMaxChatModel(LiteLLMChatModel):
    """
    A chat model implementation for the MiniMax provider, leveraging LiteLLM.

    MiniMax provides an OpenAI-compatible API. This adapter routes requests
    through LiteLLM's OpenAI provider with the MiniMax base URL.

    MiniMax exposes the same API from two regional gateways: the global
    endpoint (https://api.minimax.io/v1) and the CN endpoint
    (https://api.minimaxi.com/v1). Select one with the ``region`` argument or
    the MINIMAX_API_REGION environment variable, or override it entirely with
    an explicit ``base_url`` / MINIMAX_API_BASE.

    Available models include MiniMax-M3 (default), MiniMax-M2.7,
    and MiniMax-M2.7-highspeed.
    """

    @property
    def provider_id(self) -> ProviderName:
        """The provider ID for MiniMax."""
        return "minimax"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        region: str | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        """
        Initializes the MinimaxChatModel.

        Args:
            model_id: The ID of the MiniMax model to use. If not provided,
                it falls back to the MINIMAX_CHAT_MODEL environment variable,
                and then defaults to 'MiniMax-M3'.
            api_key: The MiniMax API key. Falls back to MINIMAX_API_KEY env var.
            base_url: The MiniMax API base URL. Falls back to MINIMAX_API_BASE
                env var, then to the endpoint selected by ``region``, and finally
                to the global endpoint 'https://api.minimax.io/v1'.
            region: The MiniMax region whose endpoint is used when neither
                ``base_url`` nor MINIMAX_API_BASE is set. Accepts 'global'
                (alias 'global_en', https://api.minimax.io/v1) or 'cn'
                (alias 'cn_zh', https://api.minimaxi.com/v1). Falls back to the
                MINIMAX_API_REGION env var, then to the global endpoint.
            **kwargs: Additional settings to configure the provider.
        """
        super().__init__(
            model_id if model_id else os.getenv("MINIMAX_CHAT_MODEL", "MiniMax-M3"),
            provider_id="openai",
            **kwargs,
        )

        region = region if region is not None else os.getenv("MINIMAX_API_REGION")
        self._assert_setting_value("api_key", api_key, envs=["MINIMAX_API_KEY"])
        self._assert_setting_value(
            "base_url",
            base_url,
            envs=["MINIMAX_API_BASE"],
            aliases=["api_base"],
            allow_empty=True,
            fallback=resolve_minimax_base_url(region),
        )
        self._settings["extra_headers"] = utils.parse_extra_headers(
            self._settings.get("extra_headers"), os.getenv("MINIMAX_API_HEADERS")
        )
