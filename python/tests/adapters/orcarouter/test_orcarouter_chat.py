# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import patch

import pytest

from beeai_framework.adapters.orcarouter.backend.chat import ORCAROUTER_API_BASE, OrcaRouterChatModel
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.constants import BackendProviders


class TestOrcaRouterProviderRegistration:
    """Test that OrcaRouter is properly registered as a provider."""

    def test_orcarouter_in_backend_providers(self) -> None:
        assert "OrcaRouter" in BackendProviders
        provider = BackendProviders["OrcaRouter"]
        assert provider.name == "OrcaRouter"
        assert provider.module == "orcarouter"
        assert "orcarouter" in provider.aliases

    def test_provider_def_has_correct_structure(self) -> None:
        provider = BackendProviders["OrcaRouter"]
        assert hasattr(provider, "name")
        assert hasattr(provider, "module")
        assert hasattr(provider, "aliases")


class TestOrcaRouterChatModelInit:
    """Test OrcaRouterChatModel initialization."""

    @patch.dict(os.environ, {"ORCAROUTER_API_KEY": "test-key-123"})
    def test_default_model_id(self) -> None:
        model = OrcaRouterChatModel()
        assert model.model_id == "orcarouter/auto"

    @patch.dict(os.environ, {"ORCAROUTER_API_KEY": "test-key-123"})
    def test_custom_model_id(self) -> None:
        model = OrcaRouterChatModel("deepseek/deepseek-v4-flash")
        assert model.model_id == "deepseek/deepseek-v4-flash"

    @patch.dict(
        os.environ,
        {"ORCAROUTER_API_KEY": "test-key-123", "ORCAROUTER_CHAT_MODEL": "anthropic/claude-opus-4.7"},
    )
    def test_model_from_env(self) -> None:
        model = OrcaRouterChatModel()
        assert model.model_id == "anthropic/claude-opus-4.7"

    @patch.dict(os.environ, {"ORCAROUTER_API_KEY": "test-key-123"})
    def test_provider_id(self) -> None:
        model = OrcaRouterChatModel()
        assert model.provider_id == "orcarouter"

    @patch.dict(os.environ, {"ORCAROUTER_API_KEY": "test-key-123"})
    def test_default_base_url(self) -> None:
        model = OrcaRouterChatModel()
        assert model._settings.get("base_url") == ORCAROUTER_API_BASE

    @patch.dict(
        os.environ,
        {"ORCAROUTER_API_KEY": "test-key-123", "ORCAROUTER_API_BASE": "https://custom.orcarouter.ai/v1"},
    )
    def test_custom_base_url_from_env(self) -> None:
        model = OrcaRouterChatModel()
        assert model._settings.get("base_url") == "https://custom.orcarouter.ai/v1"

    @patch.dict(os.environ, {"ORCAROUTER_API_KEY": "test-key-123"})
    def test_custom_base_url_param(self) -> None:
        model = OrcaRouterChatModel(base_url="https://proxy.example.com/v1")
        assert model._settings.get("base_url") == "https://proxy.example.com/v1"

    @patch.dict(os.environ, {"ORCAROUTER_API_KEY": "test-key-123"})
    def test_api_key_stored(self) -> None:
        model = OrcaRouterChatModel()
        assert model._settings.get("api_key") == "test-key-123"

    def test_missing_api_key_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ORCAROUTER_API_KEY", None)
            with pytest.raises(ValueError, match=r"api_key.*required"):
                OrcaRouterChatModel()

    @patch.dict(os.environ, {"ORCAROUTER_API_KEY": "test-key-123"})
    def test_explicit_api_key(self) -> None:
        model = OrcaRouterChatModel(api_key="explicit-key")
        assert model._settings.get("api_key") == "explicit-key"


class TestOrcaRouterModelLoading:
    """Test that OrcaRouter models can be loaded via the factory method."""

    @patch.dict(os.environ, {"ORCAROUTER_API_KEY": "test-key-123"})
    def test_load_from_name(self) -> None:
        model = ChatModel.from_name("orcarouter:orcarouter/auto")
        assert isinstance(model, OrcaRouterChatModel)
        assert model.model_id == "orcarouter/auto"

    @patch.dict(os.environ, {"ORCAROUTER_API_KEY": "test-key-123"})
    def test_load_from_alias(self) -> None:
        model = ChatModel.from_name("orcarouter:deepseek/deepseek-v4-flash")
        assert isinstance(model, OrcaRouterChatModel)
        assert model.model_id == "deepseek/deepseek-v4-flash"
