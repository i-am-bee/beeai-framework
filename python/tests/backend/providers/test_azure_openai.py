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


import pytest

from beeai_framework.adapters.azure_openai.backend.chat import AzureOpenAIChatModel


class TestAzureOpenAIChatModel:
    """Unit tests for the AzureOpenAIChatModel class."""

    @pytest.mark.unit
    def test_init_with_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization with settings."""
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_BASE", raising=False)
        monkeypatch.delenv("AZURE_API_BASE", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
        monkeypatch.delenv("AZURE_API_VERSION", raising=False)
        settings = {
            "api_key": "test_api_key",
            "api_base": "test_api_base",
            "api_version": "test_api_version",
        }
        model = AzureOpenAIChatModel(model_id="gpt-4o", settings=settings)

        assert model._settings["api_key"] == "test_api_key"
        assert model._settings["api_base"] == "test_api_base"
        assert model._settings["api_version"] == "test_api_version"
        assert model.model_id == "gpt-4o"
        assert model._litellm_provider_id == "azure"

    @pytest.mark.unit
    def test_init_with_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization with environment variables."""
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_BASE", raising=False)
        monkeypatch.delenv("AZURE_API_BASE", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
        monkeypatch.delenv("AZURE_API_VERSION", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env_api_key")
        monkeypatch.setenv("AZURE_OPENAI_API_BASE", "env_api_base")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "env_api_version")

        model = AzureOpenAIChatModel(model_id="gpt-4o")

        assert model._settings["api_key"] == "env_api_key"
        assert model._settings["api_base"] == "env_api_base"
        assert model._settings["api_version"] == "env_api_version"
        assert model.model_id == "gpt-4o"

    @pytest.mark.unit
    def test_init_with_alias_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization with alias environment variables."""
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_BASE", raising=False)
        monkeypatch.delenv("AZURE_API_BASE", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
        monkeypatch.delenv("AZURE_API_VERSION", raising=False)
        monkeypatch.setenv("AZURE_API_KEY", "env_api_key")
        monkeypatch.setenv("AZURE_API_BASE", "env_api_base")
        monkeypatch.setenv("AZURE_API_VERSION", "env_api_version")

        model = AzureOpenAIChatModel(model_id="gpt-4o")

        assert model._settings["api_key"] == "env_api_key"
        assert model._settings["api_base"] == "env_api_base"
        assert model._settings["api_version"] == "env_api_version"

    @pytest.mark.unit
    def test_init_with_mixed_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization with mixed environment variables."""
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_BASE", raising=False)
        monkeypatch.delenv("AZURE_API_BASE", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
        monkeypatch.delenv("AZURE_API_VERSION", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env_openai_api_key")
        monkeypatch.setenv("AZURE_API_BASE", "env_api_base")
        monkeypatch.setenv("AZURE_API_VERSION", "env_api_version")

        model = AzureOpenAIChatModel(model_id="gpt-4o")

        assert model._settings["api_key"] == "env_openai_api_key"
        assert model._settings["api_base"] == "env_api_base"
        assert model._settings["api_version"] == "env_api_version"

    @pytest.mark.unit
    def test_init_with_no_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization with no configuration."""
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_BASE", raising=False)
        monkeypatch.delenv("AZURE_API_BASE", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
        monkeypatch.delenv("AZURE_API_VERSION", raising=False)

        with pytest.raises(ValueError, match="Access key is required for Azure OpenAI"):
            AzureOpenAIChatModel(model_id="gpt-4o")
        monkeypatch.setenv("AZURE_API_KEY", "env_api_key")

        with pytest.raises(ValueError, match="Base URL is required for Azure OpenAI"):
            AzureOpenAIChatModel(model_id="gpt-4o")

        monkeypatch.setenv("AZURE_API_BASE", "env_api_base")
        with pytest.raises(ValueError, match="API Version is required for Azure OpenAI"):
            AzureOpenAIChatModel(model_id="gpt-4o")

    @pytest.mark.unit
    def test_settings_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test precedence order: settings > env vars > alias env vars."""
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_BASE", raising=False)
        monkeypatch.delenv("AZURE_API_BASE", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
        monkeypatch.delenv("AZURE_API_VERSION", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env_openai_api_key")
        monkeypatch.setenv("AZURE_API_KEY", "env_api_key")
        monkeypatch.setenv("AZURE_OPENAI_API_BASE", "env_openai_api_base")
        monkeypatch.setenv("AZURE_API_BASE", "env_api_base")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "env_openai_api_version")
        monkeypatch.setenv("AZURE_API_VERSION", "env_api_version")
        settings = {
            "api_key": "settings_api_key",
            "api_base": "settings_api_base",
            "api_version": "settings_api_version",
        }
        model = AzureOpenAIChatModel(model_id="gpt-4o", settings=settings)

        assert model._settings["api_key"] == "settings_api_key"
        assert model._settings["api_base"] == "settings_api_base"
        assert model._settings["api_version"] == "settings_api_version"

    @pytest.mark.unit
    def test_env_var_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test precedence order for environment variables: AZURE_OPENAI > AZURE."""
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_BASE", raising=False)
        monkeypatch.delenv("AZURE_API_BASE", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
        monkeypatch.delenv("AZURE_API_VERSION", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env_openai_api_key")
        monkeypatch.setenv("AZURE_API_KEY", "env_api_key")
        monkeypatch.setenv("AZURE_OPENAI_API_BASE", "env_openai_api_base")
        monkeypatch.setenv("AZURE_API_BASE", "env_api_base")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "env_openai_api_version")
        monkeypatch.setenv("AZURE_API_VERSION", "env_api_version")

        model = AzureOpenAIChatModel(model_id="gpt-4o")

        assert model._settings["api_key"] == "env_openai_api_key"
        assert model._settings["api_base"] == "env_openai_api_base"
        assert model._settings["api_version"] == "env_openai_api_version"

    @pytest.mark.unit
    def test_default_model_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test default model ID when not provided."""
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_BASE", raising=False)
        monkeypatch.delenv("AZURE_API_BASE", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
        monkeypatch.delenv("AZURE_API_VERSION", raising=False)
        monkeypatch.setenv("AZURE_API_KEY", "env_api_key")
        monkeypatch.setenv("AZURE_API_BASE", "env_api_base")
        monkeypatch.setenv("AZURE_API_VERSION", "env_api_version")

        model = AzureOpenAIChatModel()

        assert model.model_id == "gpt-4o-mini"

        monkeypatch.setenv("AZURE_OPENAI_CHAT_MODEL", "new_model")
        model = AzureOpenAIChatModel()
        assert model.model_id == "new_model"
