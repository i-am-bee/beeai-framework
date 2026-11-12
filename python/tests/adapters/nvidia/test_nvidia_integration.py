# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for NVIDIA chat model.

The live tests require a valid NVIDIA_API_KEY environment variable.
Run with: pytest tests/adapters/nvidia/test_nvidia_integration.py -v

The offline tests run the same conversations end to end through LiteLLM and
the OpenAI SDK, replacing only the HTTP transport, so they run without a key.
"""

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

from beeai_framework.adapters.nvidia.backend.chat import NVIDIAChatModel
from beeai_framework.backend.message import UserMessage

MODEL_IDS = ["nvidia/nemotron-3.5-lightning-30b-a3b", "nvidia/nemotron-3.5-super-vl-120b-a12b"]

requires_api_key = pytest.mark.skipif(
    not os.getenv("NVIDIA_API_KEY"),
    reason="NVIDIA_API_KEY not set",
)


@pytest.fixture(params=MODEL_IDS)
def chat_model(request: pytest.FixtureRequest) -> NVIDIAChatModel:
    return NVIDIAChatModel(request.param)


@requires_api_key
class TestNVIDIAIntegration:
    """Integration tests that call the real NVIDIA API."""

    @pytest.mark.asyncio
    async def test_simple_chat(self, chat_model: NVIDIAChatModel) -> None:
        output = await chat_model.run(
            [UserMessage("What is 2 + 2? Reply with just the number.")],
        )
        text = output.get_text_content()
        assert "4" in text

    @pytest.mark.asyncio
    async def test_streaming(self, chat_model: NVIDIAChatModel) -> None:
        output = await chat_model.run(
            [UserMessage("Count from 1 to 3.")],
            stream=True,
        )
        text = output.get_text_content()
        assert "1" in text


@pytest_asyncio.fixture
async def offline_requests(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[list[httpx.Request]]:
    captured: list[httpx.Request] = []
    for name in ("NVIDIA_CHAT_MODEL", "NVIDIA_API_BASE", "NVIDIA_API_HEADERS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")

    async def send(self: httpx.AsyncClient, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        captured.append(request)
        body = json.loads(request.content)
        answer = "1, 2, 3" if "Count" in json.dumps(body["messages"]) else "4"
        response: dict[str, Any] = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        if body.get("stream"):
            response["object"] = "chat.completion.chunk"
            response["choices"] = [
                {"index": 0, "delta": {"role": "assistant", "content": answer}, "finish_reason": "stop"}
            ]
            return httpx.Response(
                200,
                request=request,
                headers={"content-type": "text/event-stream"},
                content=f"data: {json.dumps(response)}\n\ndata: [DONE]\n\n",
            )
        return httpx.Response(200, request=request, json=response)

    # Keep the real LiteLLM and OpenAI SDK serialization; replace only HTTP I/O.
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    yield captured
    await GLOBAL_LOGGING_WORKER.stop()


@pytest.mark.unit
class TestNVIDIAOfflineIntegration:
    """The integration conversations against a mocked NVIDIA API catalog endpoint."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_id", MODEL_IDS)
    async def test_simple_chat(self, offline_requests: list[httpx.Request], model_id: str) -> None:
        output = await NVIDIAChatModel(model_id).run(
            [UserMessage("What is 2 + 2? Reply with just the number.")],
        )
        assert "4" in output.get_text_content()
        assert len(offline_requests) == 1
        request = offline_requests[0]
        assert request.url.host == "integrate.api.nvidia.com"
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["model"] == model_id
        assert body.get("stream", False) is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_id", MODEL_IDS)
    async def test_streaming(self, offline_requests: list[httpx.Request], model_id: str) -> None:
        output = await NVIDIAChatModel(model_id).run(
            [UserMessage("Count from 1 to 3.")],
            stream=True,
        )
        assert "1" in output.get_text_content()
        assert len(offline_requests) == 1
        request = offline_requests[0]
        assert request.url.host == "integrate.api.nvidia.com"
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == model_id
        assert body["stream"] is True
