# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import patch

import pytest

from beeai_framework.adapters.atlascloud.backend.chat import ATLASCLOUD_API_BASE, AtlasCloudChatModel
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.constants import BackendProviders


class TestAtlasCloudProviderRegistration:
    """Test that Atlas Cloud is properly registered as a provider."""

    def test_atlascloud_in_backend_providers(self) -> None:
        assert "AtlasCloud" in BackendProviders
        provider = BackendProviders["AtlasCloud"]
        assert provider.name == "AtlasCloud"
        assert provider.module == "atlascloud"
        assert "atlascloud" in provider.aliases

    def test_provider_def_has_correct_structure(self) -> None:
        provider = BackendProviders["AtlasCloud"]
        assert hasattr(provider, "name")
        assert hasattr(provider, "module")
        assert hasattr(provider, "aliases")


class TestAtlasCloudChatModelInit:
    """Test AtlasCloudChatModel initialization."""

    @patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "test-key-123"})
    def test_default_model_id(self) -> None:
        model = AtlasCloudChatModel()
        assert model.model_id == "openai/gpt-4.1-mini"

    @patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "test-key-123"})
    def test_custom_model_id(self) -> None:
        model = AtlasCloudChatModel("deepseek-ai/DeepSeek-V3.1-Terminus")
        assert model.model_id == "deepseek-ai/DeepSeek-V3.1-Terminus"

    @patch.dict(
        os.environ,
        {"ATLASCLOUD_API_KEY": "test-key-123", "ATLASCLOUD_CHAT_MODEL": "openai/gpt-5.6-terra"},
    )
    def test_model_from_env(self) -> None:
        model = AtlasCloudChatModel()
        assert model.model_id == "openai/gpt-5.6-terra"

    @patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "test-key-123"})
    def test_provider_id(self) -> None:
        model = AtlasCloudChatModel()
        assert model.provider_id == "atlascloud"

    @patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "test-key-123"})
    def test_default_base_url(self) -> None:
        model = AtlasCloudChatModel()
        assert model._settings.get("base_url") == ATLASCLOUD_API_BASE

    @patch.dict(
        os.environ,
        {"ATLASCLOUD_API_KEY": "test-key-123", "ATLASCLOUD_API_BASE": "https://custom.atlascloud.ai/v1"},
    )
    def test_custom_base_url_from_env(self) -> None:
        model = AtlasCloudChatModel()
        assert model._settings.get("base_url") == "https://custom.atlascloud.ai/v1"

    @patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "test-key-123"})
    def test_custom_base_url_param(self) -> None:
        model = AtlasCloudChatModel(base_url="https://proxy.example.com/v1")
        assert model._settings.get("base_url") == "https://proxy.example.com/v1"

    @patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "test-key-123"})
    def test_api_key_stored(self) -> None:
        model = AtlasCloudChatModel()
        assert model._settings.get("api_key") == "test-key-123"

    def test_missing_api_key_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ATLASCLOUD_API_KEY", None)
            with pytest.raises(ValueError, match=r"api_key.*required"):
                AtlasCloudChatModel()

    @patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "test-key-123"})
    def test_explicit_api_key(self) -> None:
        model = AtlasCloudChatModel(api_key="explicit-key")
        assert model._settings.get("api_key") == "explicit-key"

    @patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "test-key-123"})
    def test_load_from_name(self) -> None:
        model = ChatModel.from_name("atlascloud:openai/gpt-4.1-mini")
        assert isinstance(model, AtlasCloudChatModel)
        assert model.model_id == "openai/gpt-4.1-mini"
