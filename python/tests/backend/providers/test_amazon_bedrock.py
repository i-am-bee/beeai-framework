# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.adapters.amazon_bedrock import AmazonBedrockChatModel


class TestAmazonBedrockChatModel:
    """Unit tests for the AmazonBedrockChatModel class."""

    @pytest.mark.unit
    def test_default_model_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The default model ID must be a valid Amazon Bedrock model ID, not a Groq alias."""
        monkeypatch.delenv("AWS_CHAT_MODEL", raising=False)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test_access_key_id")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test_secret_access_key")
        monkeypatch.setenv("AWS_REGION", "us-east-1")

        model = AmazonBedrockChatModel()

        assert model.model_id == "meta.llama3-70b-instruct-v1:0"

    @pytest.mark.unit
    def test_default_model_id_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The default model ID can be overridden via the AWS_CHAT_MODEL environment variable."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test_access_key_id")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test_secret_access_key")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.setenv("AWS_CHAT_MODEL", "amazon.titan-text-lite-v1")

        model = AmazonBedrockChatModel()

        assert model.model_id == "amazon.titan-text-lite-v1"
