# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os

from typing_extensions import Unpack

from beeai_framework.adapters.litellm import utils
from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.chat import ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)


class VertexAIChatModel(LiteLLMChatModel):
    @property
    def provider_id(self) -> ProviderName:
        return "vertexai"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        project: str | None = None,
        location: str | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        super().__init__(
            model_id if model_id else os.getenv("GOOGLE_VERTEX_CHAT_MODEL", "gemini-2.0-flash-lite-001"),
            provider_id="vertex_ai",
            **kwargs,
        )

        self._assert_setting_value("vertexai_project", project, display_name="project", envs=["GOOGLE_VERTEX_PROJECT"])
        self._assert_setting_value(
            "vertexai_location", location, display_name="location", envs=["GOOGLE_VERTEX_LOCATION"]
        )
        self._settings["extra_headers"] = utils.parse_extra_headers(
            self._settings.get("extra_headers"), os.getenv("GOOGLE_VERTEX_API_HEADERS")
        )
