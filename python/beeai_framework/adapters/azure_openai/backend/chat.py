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

from dotenv import load_dotenv

from beeai_framework.adapters.litellm import utils
from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)
load_dotenv()


class AzureOpenAIChatModel(LiteLLMChatModel):
    """
    Chat model for Azure OpenAI service, extending the LiteLLMChatModel.

    This class provides an interface to interact with Azure OpenAI's chat models,
    handling authentication and configuration through a combination of settings,
    environment variables, and the LiteLLM library.

    Configuration Precedence:
        1.  `settings` (dictionary passed to the constructor):
            Values in this dictionary take the highest priority.
            **Note:** Settings in this dictionary will override all values.
            Valid keys include `api_key`, `api_base`, `api_version` and any
            other settings supported by `LiteLLMChatModel`.
        2. Environment Variables (in order of precedence):
            -   `AZURE_OPENAI_API_KEY`: API key for Azure OpenAI. This takes precedence over `AZURE_API_KEY`
            -   `AZURE_API_KEY`: Alias for `AZURE_OPENAI_API_KEY`.
            -   `AZURE_OPENAI_API_BASE`: Base URL for the Azure OpenAI service. This takes precedence over
                `AZURE_API_BASE`
            -   `AZURE_API_BASE`: Alias for `AZURE_OPENAI_API_BASE`.
            -   `AZURE_OPENAI_API_VERSION`: API version to use.  This takes precedence over `AZURE_API_VERSION`.
            -   `AZURE_API_VERSION`: Alias for `AZURE_OPENAI_API_VERSION`.
            -   `AZURE_OPENAI_CHAT_MODEL`: Default chat model to use. Defaults to "gpt-4o-mini"
        3. Defaults
            - `"gpt-4o-mini"`: used as the default value for the modelId if not supplied.
        4 `ChatModelParameters`:
            Values in the `ChatModelParameters` passed into the underlying `LiteLLMChatModel` are overridden
            by the above.

    Additional environment variables may be supported by the underlying LiteLLM library.
    These are not used/validated directly by this class:
    - `AZURE_AD_TOKEN`
    - `AZURE_API_TYPE`

    Raises:
        ValueError: If required configurations (API key, base URL, API version) are missing.

    See Also: LiteLLMChatModel
    """

    @property
    def provider_id(self) -> ProviderName:
        return "azure_openai"

    def __init__(self, model_id: str | None = None, settings: dict[str, Any] | None = None) -> None:
        """
        Inherits from LiteLLMChatModel, and takes all of the same parameters.

        Initializes the AzureOpenAIChatModel.

        Args:
            model_id (str, optional): The specific model ID to use.
                If not provided, it defaults to the `AZURE_OPENAI_CHAT_MODEL`
                environment variable, or "gpt-4o-mini".
                The parent class has further settings available.
            settings (dict, optional): A dictionary of settings to override
                default behaviors.
                Available keys: `api_key`, `api_base`, `api_version` and any
                other settings supported by `LiteLLMChatModel`.

        Raises:
            ValueError: If any of the required configurations are not found.

        """

        # Default settings (can be overriden by `settings` or environment variables)
        config = {
            "api_key": os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY"),
            "api_base": os.getenv("AZURE_OPENAI_API_BASE") or os.getenv("AZURE_API_BASE"),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("AZURE_API_VERSION"),
        }

        # Update default settings with those provided in `settings` if provided
        if settings:
            config.update(settings)

        # Error checking
        if not config["api_key"]:
            raise ValueError(
                "Access key is required for Azure OpenAI. Specify *api_key* in settings "
                "or set the AZURE_OPENAI_API_KEY or AZURE_API_KEY environment variable."
            )
        if not config["api_base"]:
            raise ValueError(
                "Base URL is required for Azure OpenAI. Specify *api_base* in settings "
                "or set the AZURE_OPENAI_API_BASE or AZURE_API_BASE environment variable."
            )
        if not config["api_version"]:
            raise ValueError(
                "API Version is required for Azure OpenAI. Specify *api_version* in settings "
                "or set the AZURE_OPENAI_API_VERSION or AZURE_API_VERSION environment variable."
            )

        super().__init__(
            model_id=(model_id if model_id is not None else os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-4o-mini")),
            provider_id="azure",  # LiteLLM uses 'azure' for Azure OpenAI
            settings=config,
        )
        self._settings["extra_headers"] = utils.parse_extra_headers(
            self._settings.get("extra_headers"), os.getenv("AZURE_OPENAI_API_HEADERS")
        )
