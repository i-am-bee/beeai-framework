# Copyright 2025 IBM Corp.
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

from beeai_framework.adapters.litellm import utils
from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)


class AmazonBedrockChatModel(LiteLLMChatModel):
    @property
    def provider_id(self) -> ProviderName:
        return "amazon_bedrock"

    def __init__(self, model_id: str | None = None, settings: dict[str, Any] | None = None) -> None:
        _settings = settings.copy() if settings is not None else {}

        aws_access_key_id = _settings.get("aws_access_key_id", os.getenv("AWS_ACCESS_KEY_ID"))
        if not aws_access_key_id:
            raise ValueError(
                "Access key is required for Amazon Bedrock model. Specify *aws_access_key_id* "
                + "or set AWS_ACCESS_KEY_ID environment variable"
            )

        aws_secret_access_key = _settings.get("aws_secret_access_key", os.getenv("AWS_SECRET_ACCESS_KEY"))
        if not aws_secret_access_key:
            raise ValueError(
                "Secret key is required for Amazon Bedrock model. Specify *aws_secret_access_key* "
                + "or set AWS_SECRET_ACCESS_KEY environment variable"
            )

        # Determine region name with the specified precedence
        aws_region_name = _settings.get("aws_region_name")
        if not aws_region_name:
            aws_region_name = os.getenv("AWS_REGION")
        if not aws_region_name:
            aws_region_name = os.getenv("AWS_REGION_NAME")

        if not aws_region_name:
            raise ValueError(
                "Region is required for Amazon Bedrock model. Specify *aws_region_name* in settings, "
                + "set the AWS_REGION environment variable, or set the AWS_REGION_NAME environment variable."
            )

        super().__init__(
            (model_id if model_id else os.getenv("AWS_CHAT_MODEL", "llama-3.1-8b-instant")),
            provider_id="bedrock",
            settings=_settings
            | {
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key,
                "aws_region_name": aws_region_name,
            },
        )
        self._settings["extra_headers"] = utils.parse_extra_headers(
            self._settings.get("extra_headers"), os.getenv("AWS_API_HEADERS")
        )
