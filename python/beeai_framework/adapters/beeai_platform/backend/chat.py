# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from dotenv import load_dotenv
from typing_extensions import Unpack

from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.chat import ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)
load_dotenv()


class BeeAIPlatformChatModel(LiteLLMChatModel):
    @property
    def provider_id(self) -> ProviderName:
        return "beeai"

    def __init__(
        self,
        model_ids: list[str] | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        self.model_ids = model_ids or []
        super().__init__(
            "dummyModel",
            provider_id="beeai",
            **kwargs,
        )
