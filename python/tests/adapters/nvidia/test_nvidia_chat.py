# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import patch

import pytest

from beeai_framework.adapters.nvidia.backend.chat import (
    NVIDIA_API_BASE,
    NVIDIAChatModel,
)
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.constants import BackendProviders


class TestNVIDIAProviderRegistration:
    """Test that NVIDIA is properly registered as a provider."""

    def test_nvidia_in_backend_providers(self) -> None:
        assert "NVIDIA" in BackendProviders
        provider = BackendProviders["NVIDIA"]
        assert provider.name == "NVIDIA"
        assert provider.module == "nvidia"
        assert "nvidia" in provider.aliases

    def test_provider_def_has_correct_structure(self) -> None:
        provider = BackendProviders["NVIDIA"]
        assert hasattr(provider, "name")
        assert hasattr(provider, "module")
        assert hasattr(provider, "aliases")


class TestNVIDIAChatModelInit:
    """Test NVIDIAChatModel initialization."""

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-123"})
    def test_default_model_id(self) -> None:
        model = NVIDIAChatModel()
        assert model.model_id == "nvidia/nemotron-3.5-super-vl-120b-a12b"

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-123"})
    def test_custom_model_id(self) -> None:
        model = NVIDIAChatModel("nvidia/nemotron-3.5-lightning-30b-a3b")
        assert model.model_id == "nvidia/nemotron-3.5-lightning-30b-a3b"

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-123"})
    def test_small_model_id(self) -> None:
        model = NVIDIAChatModel("nvidia/nemotron-3.5-bolt-e4b")
        assert model.model_id == "nvidia/nemotron-3.5-bolt-e4b"

    @patch.dict(
        os.environ,
        {"NVIDIA_API_KEY": "test-key-123", "NVIDIA_CHAT_MODEL": "nvidia/nemotron-3.5-bolt-e4b"},
    )
    def test_model_from_env(self) -> None:
        model = NVIDIAChatModel()
        assert model.model_id == "nvidia/nemotron-3.5-bolt-e4b"

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-123"})
    def test_provider_id(self) -> None:
        model = NVIDIAChatModel()
        assert model.provider_id == "nvidia"

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-123"})
    def test_default_base_url(self) -> None:
        model = NVIDIAChatModel()
        assert model._settings.get("base_url") == NVIDIA_API_BASE

    @patch.dict(
        os.environ,
        {"NVIDIA_API_KEY": "test-key-123", "NVIDIA_API_BASE": "https://custom.nim.example.com/v1"},
    )
    def test_custom_base_url_from_env(self) -> None:
        model = NVIDIAChatModel()
        assert model._settings.get("base_url") == "https://custom.nim.example.com/v1"

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-123"})
    def test_custom_base_url_param(self) -> None:
        model = NVIDIAChatModel(base_url="https://proxy.example.com/v1")
        assert model._settings.get("base_url") == "https://proxy.example.com/v1"

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-123"})
    def test_custom_api_base_setting(self) -> None:
        model = NVIDIAChatModel(settings={"api_base": "http://localhost:8000/v1"})
        assert model._settings.get("base_url") == "http://localhost:8000/v1"

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-123"})
    def test_api_key_stored(self) -> None:
        model = NVIDIAChatModel()
        assert model._settings.get("api_key") == "test-key-123"

    def test_missing_api_key_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing NVIDIA_API_KEY
            os.environ.pop("NVIDIA_API_KEY", None)
            with pytest.raises(ValueError, match=r"api_key.*required"):
                NVIDIAChatModel()

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-123"})
    def test_explicit_api_key(self) -> None:
        model = NVIDIAChatModel(api_key="explicit-key")
        assert model._settings.get("api_key") == "explicit-key"


class TestNVIDIAModelLoading:
    """Test that NVIDIA models can be loaded via the factory method."""

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-123"})
    def test_load_from_name(self) -> None:
        model = ChatModel.from_name("nvidia:nvidia/nemotron-3.5-super-vl-120b-a12b")
        assert isinstance(model, NVIDIAChatModel)
        assert model.model_id == "nvidia/nemotron-3.5-super-vl-120b-a12b"

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-123"})
    def test_load_from_alias(self) -> None:
        model = ChatModel.from_name("nvidia:nvidia/nemotron-3.5-lightning-30b-a3b")
        assert isinstance(model, NVIDIAChatModel)
        assert model.model_id == "nvidia/nemotron-3.5-lightning-30b-a3b"


class TestNVIDIAEndpoint:
    """Test the default NVIDIA endpoint."""

    def test_default_endpoint(self) -> None:
        assert NVIDIA_API_BASE == "https://integrate.api.nvidia.com/v1"

    @patch.dict(
        os.environ,
        {"NVIDIA_API_KEY": "test-key-123", "NVIDIA_API_BASE": "https://saved.example.com/v1"},
    )
    def test_explicit_base_url_overrides_env(self) -> None:
        model = NVIDIAChatModel(base_url="https://proxy.example.com/v1")
        assert model._settings.get("base_url") == "https://proxy.example.com/v1"
