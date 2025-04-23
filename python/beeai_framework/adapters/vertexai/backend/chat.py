# Copyright 2025 © BeeAI a Series of LF Projects, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
