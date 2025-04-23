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


class AmazonBedrockChatModel(LiteLLMChatModel):
    @property
    def provider_id(self) -> ProviderName:
        return "amazon_bedrock"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        region: str | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        super().__init__(
            (model_id if model_id else os.getenv("AWS_CHAT_MODEL", "llama-3.1-8b-instant")),
            provider_id="bedrock",
            **kwargs,
        )

        self._assert_setting_value(
            "aws_access_key_id", access_key_id, display_name="access_key_id", envs=["AWS_ACCESS_KEY_ID"]
        )
        self._assert_setting_value(
            "aws_secret_access_key",
            secret_access_key,
            display_name="secret_access_key",
            envs=["AWS_SECRET_ACCESS_KEY"],
        )
        self._assert_setting_value(
            "aws_region_name", region, envs=["AWS_REGION", "AWS_REGION_NAME"], display_name="region"
        )

        self._settings["extra_headers"] = utils.parse_extra_headers(
            self._settings.get("extra_headers"), os.getenv("AWS_API_HEADERS")
        )
