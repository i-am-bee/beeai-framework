# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from beeai_framework.adapters.openai import OpenAIEmbeddingModel


@pytest.mark.unit
def test_embedding_model_does_not_share_the_settings_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two models built from one settings template must not share (or mutate) that dict."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    template: dict[str, Any] = {"dimensions": 512}
    first = OpenAIEmbeddingModel("text-embedding-3-small", settings=template, api_key="key-one")
    second = OpenAIEmbeddingModel("text-embedding-3-small", settings=template, api_key="key-two")

    assert first._settings is not second._settings
    assert first._settings["api_key"] == "key-one"
    assert second._settings["api_key"] == "key-two"
    assert template == {"dimensions": 512}
