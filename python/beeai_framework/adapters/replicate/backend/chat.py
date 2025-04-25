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
from typing import Any

from dotenv import load_dotenv
from typing_extensions import Unpack

from beeai_framework.adapters.litellm import utils
from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.chat import ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.types import ChatModelInput
from beeai_framework.logger import Logger

logger = Logger(__name__)
load_dotenv()


class ReplicateChatModel(LiteLLMChatModel):
    @property
    def provider_id(self) -> ProviderName:
        return "replicate"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        api_key: str | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        super().__init__(
            (model_id if model_id else os.getenv("REPLICATE_CHAT_MODEL", "meta/meta-llama-3-8b-instruct")),
            provider_id="replicate",
            **kwargs,
        )

        self._assert_setting_value("api_key", api_key, envs=["REPLICATE_API_KEY"])
        self._settings["extra_headers"] = utils.parse_extra_headers(
            self._settings.get("extra_headers"), os.getenv("REPLICATE_API_HEADERS")
        )

    def _transform_input(self, input: ChatModelInput) -> dict[str, Any]:
        result = super()._transform_input(input)
        for message in result["messages"]:
            content = message["content"]
            if isinstance(content, str):
                continue

            new_content = ""
            if isinstance(content, dict):
                content = [content]

            if isinstance(content, list):
                for part in content:
                    if part["type"] == "text":
                        new_content += part["text"]

            # TODO: not finishesd

            message["content"] = new_content

        return result
