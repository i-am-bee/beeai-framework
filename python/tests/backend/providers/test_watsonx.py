# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.adapters.watsonx import WatsonxChatModel, WatsonxEmbeddingModel
from beeai_framework.backend.constants import BackendProviders

WATSONX_ENV_VARS = [
    "WATSONX_CHAT_MODEL",
    "WATSONX_EMBEDDING_MODEL",
    "WATSONX_SPACE_ID",
    "WATSONX_DEPLOYMENT_SPACE_ID",
    "WATSONX_PROJECT_ID",
    "WATSONX_REGION",
    "WATSONX_URL",
    "WATSONX_API_KEY",
    "WATSONX_APIKEY",
    "WATSONX_ZENAPIKEY",
]


@pytest.fixture(autouse=True)
def _clear_watsonx_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate every test from any ambient WATSONX_* configuration."""
    for var in WATSONX_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.mark.unit
def test_watsonx_provider_is_registered() -> None:
    provider = BackendProviders["watsonx"]
    assert provider.name == "Watsonx"
    assert provider.module == "watsonx"
    assert "watsonx" in provider.aliases
    assert "ibm" in provider.aliases


class TestWatsonxChatModel:
    @pytest.mark.unit
    def test_provider_id(self) -> None:
        model = WatsonxChatModel(project_id="test-project")
        assert model.provider_id == "watsonx"

    @pytest.mark.unit
    def test_default_model_id(self) -> None:
        model = WatsonxChatModel(project_id="test-project")
        assert model.model_id == "ibm/granite-3-3-8b-instruct"

    @pytest.mark.unit
    def test_model_id_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WATSONX_CHAT_MODEL", "ibm/granite-13b-chat")
        model = WatsonxChatModel(project_id="test-project")
        assert model.model_id == "ibm/granite-13b-chat"

    @pytest.mark.unit
    def test_explicit_model_id_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WATSONX_CHAT_MODEL", "ibm/granite-13b-chat")
        model = WatsonxChatModel("ibm/granite-3-3-8b-instruct", project_id="test-project")
        assert model.model_id == "ibm/granite-3-3-8b-instruct"

    @pytest.mark.unit
    def test_requires_project_id_when_space_id_missing(self) -> None:
        with pytest.raises(ValueError, match="project_id"):
            WatsonxChatModel()

    @pytest.mark.unit
    def test_space_id_makes_project_id_optional(self) -> None:
        model = WatsonxChatModel(space_id="test-space")
        assert model._settings["space_id"] == "test-space"
        assert model._settings.get("project_id") is None

    @pytest.mark.unit
    def test_project_id_from_env_satisfies_requirement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WATSONX_PROJECT_ID", "env-project")
        model = WatsonxChatModel()
        assert model._settings["project_id"] == "env-project"

    @pytest.mark.unit
    def test_region_defaults_and_builds_base_url(self) -> None:
        model = WatsonxChatModel(project_id="test-project")
        assert model._settings["region"] == "us-south"
        assert model._settings["base_url"] == "https://us-south.ml.cloud.ibm.com"

    @pytest.mark.unit
    def test_custom_region_builds_base_url(self) -> None:
        model = WatsonxChatModel(project_id="test-project", region="eu-de")
        assert model._settings["region"] == "eu-de"
        assert model._settings["base_url"] == "https://eu-de.ml.cloud.ibm.com"

    @pytest.mark.unit
    def test_explicit_base_url_overrides_region(self) -> None:
        model = WatsonxChatModel(project_id="test-project", region="eu-de", base_url="https://custom.example.com")
        assert model._settings["base_url"] == "https://custom.example.com"

    @pytest.mark.unit
    @pytest.mark.parametrize("env_var", ["WATSONX_API_KEY", "WATSONX_APIKEY", "WATSONX_ZENAPIKEY"])
    def test_api_key_resolved_from_each_env_variant(self, monkeypatch: pytest.MonkeyPatch, env_var: str) -> None:
        monkeypatch.setenv(env_var, "secret-key")
        model = WatsonxChatModel(project_id="test-project")
        assert model._settings["api_key"] == "secret-key"

    @pytest.mark.unit
    def test_settings_passed_via_kwarg(self) -> None:
        model = WatsonxChatModel(settings={"project_id": "kwarg-project"})
        assert model._settings["project_id"] == "kwarg-project"


class TestWatsonxEmbeddingModel:
    @pytest.mark.unit
    def test_provider_id(self) -> None:
        model = WatsonxEmbeddingModel(project_id="test-project")
        assert model.provider_id == "watsonx"

    @pytest.mark.unit
    def test_default_model_id(self) -> None:
        model = WatsonxEmbeddingModel(project_id="test-project")
        assert model.model_id == "sentence-transformers/all-minilm-l6-v2"

    @pytest.mark.unit
    def test_model_id_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WATSONX_EMBEDDING_MODEL", "ibm/slate-30m-english-rtrvr")
        model = WatsonxEmbeddingModel(project_id="test-project")
        assert model.model_id == "ibm/slate-30m-english-rtrvr"

    @pytest.mark.unit
    def test_requires_project_id_when_space_id_missing(self) -> None:
        with pytest.raises(ValueError, match="project_id"):
            WatsonxEmbeddingModel()

    @pytest.mark.unit
    def test_region_defaults_and_builds_base_url(self) -> None:
        model = WatsonxEmbeddingModel(project_id="test-project")
        assert model._settings["region"] == "us-south"
        assert model._settings["base_url"] == "https://us-south.ml.cloud.ibm.com"
