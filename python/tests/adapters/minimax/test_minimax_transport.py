# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

from beeai_framework.adapters.minimax.backend.chat import MiniMaxChatModel
from beeai_framework.backend.chat import ChatModel, ChatModelParameters
from beeai_framework.backend.message import MessageImageContent, MessageTextContent, UserMessage, UserMessageContent

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def requests(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[list[httpx.Request]]:
    captured: list[httpx.Request] = []
    for name in ("MINIMAX_CHAT_MODEL", "MINIMAX_API_BASE", "MINIMAX_API_REGION", "MINIMAX_API_HEADERS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")

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


@pytest.mark.parametrize("region,host", [("global", "api.minimax.io"), ("cn", "api.minimaxi.com")])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("model_id", [None, "MiniMax-M3", "MiniMax-M2.7", "custom-model"])
async def test_model_request(
    requests: list[httpx.Request], region: str, host: str, stream: bool, model_id: str | None
) -> None:
    model = MiniMaxChatModel(model_id, region=region)
    result = await model.run([UserMessage("Hello")], stream=stream)
    assert result.get_text_content() == "Hello"
    assert len(requests) == 1
    assert requests[0].url.host == host
    body = json.loads(requests[0].content)
    assert body["model"] == (model_id or "MiniMax-M3.1-Flash")
    assert body.get("stream", False) is stream
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    assert "thinking" not in body
    assert "reasoning_effort" not in body
    assert "service_tier" not in body
    assert "reasoning_split" not in body


@pytest.mark.parametrize("model_id", ["MiniMax-M3.1-Flash", "MiniMax-M3", "MiniMax-M2.7-highspeed", "custom-model"])
async def test_saved_model_and_custom_endpoint(
    requests: list[httpx.Request], monkeypatch: pytest.MonkeyPatch, model_id: str
) -> None:
    monkeypatch.setenv("MINIMAX_CHAT_MODEL", model_id)
    monkeypatch.setenv("MINIMAX_API_REGION", "cn")
    monkeypatch.setenv("MINIMAX_API_BASE", "https://proxy.example.com/v1")
    model = ChatModel.from_name("minimax")
    await model.run([UserMessage("Hello")])
    assert requests[0].url.host == "proxy.example.com"
    assert json.loads(requests[0].content)["model"] == model_id


async def test_factory_default(requests: list[httpx.Request]) -> None:
    await ChatModel.from_name("minimax").run([UserMessage("Hello")])
    assert json.loads(requests[0].content)["model"] == "MiniMax-M3.1-Flash"


@pytest.mark.parametrize(
    "model_id,extra_body",
    [
        ("MiniMax-M3.1-Flash", {"thinking": {"type": "adaptive"}, "reasoning_effort": "high"}),
        ("MiniMax-M3", {"thinking": {"type": "disabled"}}),
    ],
)
async def test_explicit_thinking_settings(
    requests: list[httpx.Request], model_id: str, extra_body: dict[str, Any]
) -> None:
    model = MiniMaxChatModel(model_id, settings={"extra_body": extra_body})
    await model.run([UserMessage("Hello")])
    body = json.loads(requests[0].content)
    for key, value in extra_body.items():
        assert body[key] == value


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
@pytest.mark.parametrize("stream", [False, True])
async def test_reasoning_effort_parameter(requests: list[httpx.Request], effort: str, stream: bool) -> None:
    await MiniMaxChatModel().run([UserMessage("Hello")], reasoning_effort=effort, stream=stream)
    assert json.loads(requests[0].content)["reasoning_effort"] == effort


@pytest.mark.parametrize("source", ["parameters", "settings"])
async def test_saved_reasoning_effort(requests: list[httpx.Request], source: str) -> None:
    model = (
        MiniMaxChatModel(parameters=ChatModelParameters(reasoning_effort="medium"))
        if source == "parameters"
        else MiniMaxChatModel(settings={"reasoning_effort": "medium"})
    )
    await model.run([UserMessage("Hello")])
    assert json.loads(requests[0].content)["reasoning_effort"] == "medium"


async def test_image_content_is_preserved(requests: list[httpx.Request]) -> None:
    content: list[UserMessageContent] = [
        MessageTextContent(text="Describe this image"),
        MessageImageContent(image_url={"url": "https://example.com/image.png", "detail": "high"}),
    ]
    await MiniMaxChatModel().run([UserMessage(content)])
    body = json.loads(requests[0].content)
    assert body["messages"][0]["content"] == [part.model_dump() for part in content]


async def test_explicit_overrides_saved_settings(
    requests: list[httpx.Request], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MINIMAX_CHAT_MODEL", "saved-model")
    monkeypatch.setenv("MINIMAX_API_BASE", "https://saved.example.com/v1")
    model = MiniMaxChatModel("custom-model", base_url="https://custom.example.com/v1", region="cn")
    await model.run([UserMessage("Hello")])
    assert requests[0].url.host == "custom.example.com"
    assert json.loads(requests[0].content)["model"] == "custom-model"
