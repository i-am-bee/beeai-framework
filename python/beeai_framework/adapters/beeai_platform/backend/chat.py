# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import ClassVar

from dotenv import load_dotenv
from typing_extensions import Unpack

from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.chat import ChatModelKwargs, ToolChoiceType
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)
load_dotenv()


class BeeAIPlatformChatModel(LiteLLMChatModel):
    tool_choice_support: ClassVar[set[ToolChoiceType]] = {"none"}

    @property
    def provider_id(self) -> ProviderName:
        return "beeai"

    def __init__(
        self,
        model_ids: list[str],
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        self.model_ids = model_ids
        super().__init__(
            model_ids[0],
            provider_id="openai",
            **kwargs,
        )

        self._assert_setting_value(
            "base_url", base_url, envs=["BEEAI_PLATFORM_API_BASE"], aliases=["api_base"], allow_empty=True
        )
        self._assert_setting_value("api_key", api_key, envs=["BEEAI_PLATFORM_API_KEY"], allow_empty=True)
