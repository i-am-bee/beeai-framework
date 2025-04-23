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

from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.chat import ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)


class XAIChatModel(LiteLLMChatModel):
    @property
    def provider_id(self) -> ProviderName:
        return "xai"

    def __init__(
        self, model_id: str | None = None, *, api_key: str | None = None, **kwargs: Unpack[ChatModelKwargs]
    ) -> None:
        super().__init__(model_id if model_id else os.getenv("XAI_CHAT_MODEL", "grok-2"), provider_id="xai", **kwargs)
        self._assert_setting_value("api_key", api_key, envs=["XAI_API_KEY"])
