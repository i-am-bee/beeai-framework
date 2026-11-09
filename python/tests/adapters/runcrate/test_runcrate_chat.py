# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import json
import os
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

from beeai_framework.adapters.runcrate.backend.chat import RUNCRATE_API_BASE, RuncrateChatModel
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.constants import BackendProviders
from beeai_framework.backend.message import UserMessage

pytestmark = pytest.mark.unit

ENV_NAMES = ("RUNCRATE_CHAT_MODEL", "RUNCRATE_API_KEY", "RUNCRATE_API_BASE", "RUNCRATE_API_HEADERS")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RUNCRATE_API_KEY", "test-key")


def test_provider_registration() -> None:
    provider = BackendProviders["Runcrate"]
    assert provider.name == "Runcrate"
    assert provider.module == "runcrate"
    assert "runcrate" in provider.aliases


def test_defaults() -> None:
    model = RuncrateChatModel()
    assert model.provider_id == "runcrate"
    assert model.model_id == "deepseek-ai/DeepSeek-V4.1-Flash"
    assert model._settings.get("base_url") == RUNCRATE_API_BASE == "https://api.runcrate.ai/v1"
    assert model._settings.get("api_key") == "test-key"


def test_explicit_settings_win_over_env() -> None:
    with patch.dict(
        os.environ, {"RUNCRATE_CHAT_MODEL": "Qwen/Qwen3.5-27B", "RUNCRATE_API_BASE": "https://env.example/v1"}
    ):
        assert RuncrateChatModel().model_id == "Qwen/Qwen3.5-27B"
        assert RuncrateChatModel()._settings.get("base_url") == "https://env.example/v1"
        model = RuncrateChatModel("moonshotai/Kimi-K3", api_key="explicit-key", base_url="https://proxy.example/v1")
    assert model.model_id == "moonshotai/Kimi-K3"
    assert model._settings.get("api_key") == "explicit-key"
    assert model._settings.get("base_url") == "https://proxy.example/v1"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNCRATE_API_KEY")
    with pytest.raises(ValueError, match=r"api_key.*required"):
        RuncrateChatModel()


def test_load_from_name() -> None:
    model = ChatModel.from_name("runcrate:Qwen/Qwen3.5-27B")
    assert isinstance(model, RuncrateChatModel)
    assert model.model_id == "Qwen/Qwen3.5-27B"


@pytest_asyncio.fixture
async def requests(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[list[httpx.Request]]:
    captured: list[httpx.Request] = []

    async def send(self: httpx.AsyncClient, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        captured.append(request)
        body = json.loads(request.content)
        response: dict[str, Any] = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        return httpx.Response(200, request=request, json=response)

    # Keep the real LiteLLM and OpenAI SDK serialization; replace only HTTP I/O.
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    yield captured
    await GLOBAL_LOGGING_WORKER.stop()


@pytest.mark.asyncio
async def test_request_goes_to_runcrate_with_bearer_key(requests: list[httpx.Request]) -> None:
    result = await RuncrateChatModel("deepseek-ai/DeepSeek-V4.1-Flash").run([UserMessage("Hello")])

    assert result.get_text_content() == "Hello"
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.runcrate.ai/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer test-key"
    body = json.loads(requests[0].content)
    assert body["model"] == "deepseek-ai/DeepSeek-V4.1-Flash"
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]


@pytest.mark.asyncio
async def test_request_uses_custom_base_url(requests: list[httpx.Request], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNCRATE_API_BASE", "https://proxy.example.com/v1")
    await ChatModel.from_name("runcrate:Qwen/Qwen3.5-27B").run([UserMessage("Hello")])

    assert requests[0].url.host == "proxy.example.com"
    assert json.loads(requests[0].content)["model"] == "Qwen/Qwen3.5-27B"
