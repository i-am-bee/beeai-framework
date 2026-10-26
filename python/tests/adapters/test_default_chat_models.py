# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.adapters.anthropic import AnthropicChatModel
from beeai_framework.adapters.gemini import GeminiChatModel
from beeai_framework.adapters.vertexai import VertexAIChatModel

# Models that the providers have retired: a default pointing at one fails on the first request
RETIRED_MODELS = {
    "claude-3-haiku-20240307",  # retired 2026-04-20
    "gemini-2.0-flash-lite-001",
    "gemini-2.5-flash",  # no longer served to new Gemini API keys
}


@pytest.fixture(autouse=True)
def provider_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ANTHROPIC_CHAT_MODEL", "GEMINI_CHAT_MODEL", "GOOGLE_VERTEX_CHAT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_VERTEX_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_VERTEX_LOCATION", "us-central1")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_class", "expected"),
    [
        (AnthropicChatModel, "claude-haiku-4-5"),
        (GeminiChatModel, "gemini-3.8-flash"),
        (VertexAIChatModel, "gemini-3.8-flash"),
    ],
)
def test_default_chat_model_is_not_retired(model_class: type, expected: str) -> None:
    model = model_class()

    assert model.model_id not in RETIRED_MODELS
    assert model.model_id == expected
