# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

from beeai_framework.adapters.nvidia.backend.chat import NVIDIAChatModel
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import MessageImageContent, MessageTextContent, UserMessage, UserMessageContent

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def requests(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[list[httpx.Request]]:
    captured: list[httpx.Request] = []
    for name in ("NVIDIA_CHAT_MODEL", "NVIDIA_API_BASE", "NVIDIA_API_HEADERS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")

    async def send(self: httpx.AsyncClient, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
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
        if body.get("stream"):
            response["object"] = "chat.completion.chunk"
            response["choices"] = [
                {"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
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


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "model_id",
    [None, "nvidia/nemotron-3.5-super-vl-120b-a12b", "nvidia/nemotron-3.5-lightning-30b-a3b", "custom-model"],
)
async def test_model_request(requests: list[httpx.Request], stream: bool, model_id: str | None) -> None:
    model = NVIDIAChatModel(model_id)
    result = await model.run([UserMessage("Hello")], stream=stream)
    assert result.get_text_content() == "Hello"
    assert len(requests) == 1
    assert requests[0].url.host == "integrate.api.nvidia.com"
    assert requests[0].headers["authorization"] == "Bearer test-key"
    body = json.loads(requests[0].content)
    assert body["model"] == (model_id or "nvidia/nemotron-3.5-super-vl-120b-a12b")
    assert body.get("stream", False) is stream
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    assert "reasoning_effort" not in body
    assert "service_tier" not in body


@pytest.mark.parametrize(
    "model_id",
    ["nvidia/nemotron-3.5-super-vl-120b-a12b", "nvidia/nemotron-3.5-lightning-30b-a3b", "custom-model"],
)
async def test_saved_model_and_custom_endpoint(
    requests: list[httpx.Request], monkeypatch: pytest.MonkeyPatch, model_id: str
) -> None:
    monkeypatch.setenv("NVIDIA_CHAT_MODEL", model_id)
    monkeypatch.setenv("NVIDIA_API_BASE", "https://proxy.example.com/v1")
    model = ChatModel.from_name("nvidia")
    await model.run([UserMessage("Hello")])
    assert requests[0].url.host == "proxy.example.com"
    assert json.loads(requests[0].content)["model"] == model_id


async def test_factory_default(requests: list[httpx.Request]) -> None:
    await ChatModel.from_name("nvidia").run([UserMessage("Hello")])
    assert requests[0].url.host == "integrate.api.nvidia.com"
    assert json.loads(requests[0].content)["model"] == "nvidia/nemotron-3.5-super-vl-120b-a12b"


async def test_image_content_is_preserved(requests: list[httpx.Request]) -> None:
    content: list[UserMessageContent] = [
        MessageTextContent(text="Describe this image"),
        MessageImageContent(image_url={"url": "https://example.com/image.png", "detail": "high"}),
    ]
    await NVIDIAChatModel().run([UserMessage(content)])
    body = json.loads(requests[0].content)
    assert body["messages"][0]["content"] == [part.model_dump() for part in content]


async def test_explicit_overrides_saved_settings(
    requests: list[httpx.Request], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NVIDIA_CHAT_MODEL", "saved-model")
    monkeypatch.setenv("NVIDIA_API_BASE", "https://saved.example.com/v1")
    model = NVIDIAChatModel("custom-model", base_url="https://custom.example.com/v1")
    await model.run([UserMessage("Hello")])
    assert requests[0].url.host == "custom.example.com"
    assert json.loads(requests[0].content)["model"] == "custom-model"


async def test_self_hosted_nim_trailing_slash(requests: list[httpx.Request]) -> None:
    model = NVIDIAChatModel(base_url="https://nim.example.com/v1/")
    await model.run([UserMessage("Hello")])
    assert requests[0].url.host == "nim.example.com"
