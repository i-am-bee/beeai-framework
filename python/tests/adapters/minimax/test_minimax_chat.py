# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import patch

import pytest

from beeai_framework.adapters.minimax.backend.chat import (
    MINIMAX_API_BASE,
    MINIMAX_API_BASE_CN,
    MiniMaxChatModel,
    resolve_minimax_base_url,
)
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.constants import BackendProviders


class TestMiniMaxProviderRegistration:
    """Test that MiniMax is properly registered as a provider."""

    def test_minimax_in_backend_providers(self) -> None:
        assert "MiniMax" in BackendProviders
        provider = BackendProviders["MiniMax"]
        assert provider.name == "MiniMax"
        assert provider.module == "minimax"
        assert "minimax" in provider.aliases

    def test_provider_def_has_correct_structure(self) -> None:
        provider = BackendProviders["MiniMax"]
        assert hasattr(provider, "name")
        assert hasattr(provider, "module")
        assert hasattr(provider, "aliases")


class TestMiniMaxChatModelInit:
    """Test MiniMaxChatModel initialization."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    def test_default_model_id(self) -> None:
        model = MiniMaxChatModel()
        assert model.model_id == "MiniMax-M3"

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    def test_custom_model_id(self) -> None:
        model = MiniMaxChatModel("MiniMax-M2.7")
        assert model.model_id == "MiniMax-M2.7"

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    def test_highspeed_model_id(self) -> None:
        model = MiniMaxChatModel("MiniMax-M2.7-highspeed")
        assert model.model_id == "MiniMax-M2.7-highspeed"

    @patch.dict(
        os.environ,
        {"MINIMAX_API_KEY": "test-key-123", "MINIMAX_CHAT_MODEL": "MiniMax-M2.7-highspeed"},
    )
    def test_model_from_env(self) -> None:
        model = MiniMaxChatModel()
        assert model.model_id == "MiniMax-M2.7-highspeed"

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    def test_provider_id(self) -> None:
        model = MiniMaxChatModel()
        assert model.provider_id == "minimax"

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    def test_default_base_url(self) -> None:
        model = MiniMaxChatModel()
        assert model._settings.get("base_url") == MINIMAX_API_BASE

    @patch.dict(
        os.environ,
        {"MINIMAX_API_KEY": "test-key-123", "MINIMAX_API_BASE": "https://custom.minimax.io/v1"},
    )
    def test_custom_base_url_from_env(self) -> None:
        model = MiniMaxChatModel()
        assert model._settings.get("base_url") == "https://custom.minimax.io/v1"

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    def test_custom_base_url_param(self) -> None:
        model = MiniMaxChatModel(base_url="https://proxy.example.com/v1")
        assert model._settings.get("base_url") == "https://proxy.example.com/v1"

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    def test_api_key_stored(self) -> None:
        model = MiniMaxChatModel()
        assert model._settings.get("api_key") == "test-key-123"

    def test_missing_api_key_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing MINIMAX_API_KEY
            os.environ.pop("MINIMAX_API_KEY", None)
            with pytest.raises(ValueError, match=r"api_key.*required"):
                MiniMaxChatModel()

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    def test_explicit_api_key(self) -> None:
        model = MiniMaxChatModel(api_key="explicit-key")
        assert model._settings.get("api_key") == "explicit-key"


class TestMiniMaxModelLoading:
    """Test that MiniMax models can be loaded via the factory method."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    def test_load_from_name(self) -> None:
        model = ChatModel.from_name("minimax:MiniMax-M3")
        assert isinstance(model, MiniMaxChatModel)
        assert model.model_id == "MiniMax-M3"

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    def test_load_from_alias(self) -> None:
        model = ChatModel.from_name("minimax:MiniMax-M2.7")
        assert isinstance(model, MiniMaxChatModel)
        assert model.model_id == "MiniMax-M2.7"


class TestMiniMaxRegionSelection:
    """Test regional (global / CN) endpoint selection."""

    def test_resolve_global_variants(self) -> None:
        assert resolve_minimax_base_url("global") == MINIMAX_API_BASE
        assert resolve_minimax_base_url("global_en") == MINIMAX_API_BASE

    def test_resolve_cn_variants(self) -> None:
        assert resolve_minimax_base_url("cn") == MINIMAX_API_BASE_CN
        assert resolve_minimax_base_url("cn_zh") == MINIMAX_API_BASE_CN

    def test_resolve_is_case_insensitive_and_trimmed(self) -> None:
        assert resolve_minimax_base_url("  CN  ") == MINIMAX_API_BASE_CN

    def test_resolve_empty_defaults_to_global(self) -> None:
        assert resolve_minimax_base_url(None) == MINIMAX_API_BASE
        assert resolve_minimax_base_url("") == MINIMAX_API_BASE

    def test_resolve_unknown_region_raises(self) -> None:
        with pytest.raises(ValueError, match=r"Unknown MiniMax region"):
            resolve_minimax_base_url("mars")

    def test_endpoints_differ(self) -> None:
        assert MINIMAX_API_BASE == "https://api.minimax.io/v1"
        assert MINIMAX_API_BASE_CN == "https://api.minimaxi.com/v1"
        assert MINIMAX_API_BASE != MINIMAX_API_BASE_CN

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    def test_region_param_selects_cn(self) -> None:
        model = MiniMaxChatModel(region="cn")
        assert model._settings.get("base_url") == MINIMAX_API_BASE_CN

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    def test_region_param_selects_global(self) -> None:
        model = MiniMaxChatModel(region="global")
        assert model._settings.get("base_url") == MINIMAX_API_BASE

    @patch.dict(
        os.environ,
        {"MINIMAX_API_KEY": "test-key-123", "MINIMAX_API_REGION": "cn_zh"},
    )
    def test_region_from_env_selects_cn(self) -> None:
        model = MiniMaxChatModel()
        assert model._settings.get("base_url") == MINIMAX_API_BASE_CN

    @patch.dict(
        os.environ,
        {"MINIMAX_API_KEY": "test-key-123", "MINIMAX_API_REGION": "cn"},
    )
    def test_explicit_base_url_overrides_region(self) -> None:
        model = MiniMaxChatModel(base_url="https://proxy.example.com/v1", region="cn")
        assert model._settings.get("base_url") == "https://proxy.example.com/v1"

    @patch.dict(
        os.environ,
        {
            "MINIMAX_API_KEY": "test-key-123",
            "MINIMAX_API_BASE": "https://api.minimax.io/v1",
            "MINIMAX_API_REGION": "cn",
        },
    )
    def test_api_base_env_overrides_region(self) -> None:
        model = MiniMaxChatModel()
        assert model._settings.get("base_url") == MINIMAX_API_BASE

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    def test_default_without_region_is_global(self) -> None:
        model = MiniMaxChatModel()
        assert model._settings.get("base_url") == MINIMAX_API_BASE
