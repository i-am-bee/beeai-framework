# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.adapters.ollama import OllamaChatModel, OllamaEmbeddingModel


@pytest.mark.unit
@pytest.mark.parametrize(
    "base_url",
    ["http://host:11434", "http://host:11434/", "http://host:11434/v1", "http://host:11434/v1/"],
)
def test_ollama_chat_model_normalizes_base_url(base_url: str) -> None:
    model = OllamaChatModel("llama3.1", base_url=base_url)
    assert model._settings["base_url"] == "http://host:11434/v1"


@pytest.mark.unit
@pytest.mark.parametrize(
    "base_url",
    ["http://host:11434", "http://host:11434/", "http://host:11434/v1", "http://host:11434/v1/"],
)
def test_ollama_embedding_model_normalizes_base_url(base_url: str) -> None:
    model = OllamaEmbeddingModel("nomic-embed-text", base_url=base_url)
    assert model._settings["api_base"] == "http://host:11434/v1"
