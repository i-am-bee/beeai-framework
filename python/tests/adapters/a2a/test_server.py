# Copyright 2026 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import a2a.client as a2a_client
import a2a.types as a2a_types
import httpx
import pytest
from a2a.server.request_handlers import DefaultRequestHandler
from starlette.applications import Starlette

import beeai_framework.adapters.a2a.serve.server as a2a_server
from beeai_framework.adapters.a2a.agents import A2AAgent
from beeai_framework.adapters.a2a.serve.server import A2AServer, A2AServerConfig, A2AServerMetadata
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import AnyMessage
from beeai_framework.memory import UnconstrainedMemory
from tests.agents._scripted import ScriptedChatModel, final_answer_message, tool_call_message, weather_tool

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

BASE_URL = "http://testserver"


def make_agent() -> RequirementAgent:
    responses: list[list[AnyMessage]] = [
        [tool_call_message("weather_tool", {"city": "Prague"})],
        [final_answer_message("sunny in Prague")],
    ]
    return RequirementAgent(llm=ScriptedChatModel(responses * 2), tools=[weather_tool], memory=UnconstrainedMemory())


def build_app(monkeypatch: pytest.MonkeyPatch, config: A2AServerConfig, metadata: A2AServerMetadata) -> Starlette:
    """Returns the app that A2AServer.serve hands to uvicorn."""
    apps: list[Starlette] = []
    monkeypatch.setattr(a2a_server.uvicorn, "run", lambda app, **_: apps.append(app))
    A2AServer(config=config).register(make_agent(), **metadata).serve()
    return apps[0]


@asynccontextmanager
async def serve(app: Starlette) -> AsyncIterator[httpx.AsyncClient]:
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE_URL) as client,
    ):
        yield client


def send_v0_3_message(text: str) -> dict[str, Any]:
    message = {"kind": "message", "messageId": "message-0.3", "role": "user", "parts": [{"kind": "text", "text": text}]}
    return {"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {"message": message}}


@pytest.mark.parametrize("protocol", ["jsonrpc", "http_json"])
async def test_app_serves_a2a_clients(protocol: str, monkeypatch: pytest.MonkeyPatch) -> None:
    app = build_app(monkeypatch, A2AServerConfig(protocol=protocol), {"url": BASE_URL})  # type: ignore[arg-type]

    async with serve(app) as client:
        card = await a2a_client.A2ACardResolver(client, BASE_URL).get_agent_card()
        agent = A2AAgent(agent_card=card, memory=UnconstrainedMemory())
        response = await agent.run("Weather in Prague?", httpx_client=client)

    assert response.last_message.text == "sunny in Prague"
    assert response.event.status_update.status.state == a2a_types.TaskState.TASK_STATE_COMPLETED


async def test_app_serves_v0_3_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    app = build_app(monkeypatch, A2AServerConfig(protocol="jsonrpc"), {"url": BASE_URL})

    async with serve(app) as client:
        card = (await client.get("/.well-known/agent-card.json")).json()
        result = (await client.post("/", json=send_v0_3_message("Weather in Prague?"))).json()["result"]

    assert (card["url"], card["preferredTransport"], card["protocolVersion"]) == (BASE_URL, "JSONRPC", "0.3")
    assert [(item["protocolBinding"], item["protocolVersion"]) for item in card["supportedInterfaces"]] == [
        ("JSONRPC", "1.0"),
        ("JSONRPC", "0.3"),
    ]
    assert (result["kind"], result["status"]["state"]) == ("task", "completed")
    assert result["status"]["message"]["parts"] == [{"kind": "text", "text": "sunny in Prague"}]


async def test_app_without_v0_3_compat_rejects_v0_3_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    app = build_app(monkeypatch, A2AServerConfig(protocol="jsonrpc", enable_v0_3_compat=False), {"url": BASE_URL})

    async with serve(app) as client:
        card = (await client.get("/.well-known/agent-card.json")).json()
        response = (await client.post("/", json=send_v0_3_message("Weather in Prague?"))).json()

    assert [item["protocolVersion"] for item in card["supportedInterfaces"]] == ["1.0"]
    assert "url" not in card
    assert "error" in response


async def test_app_reports_health_and_closes_the_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[bool] = []

    async def aclose(self: DefaultRequestHandler) -> None:
        closed.append(True)

    monkeypatch.setattr(DefaultRequestHandler, "aclose", aclose)
    app = build_app(monkeypatch, A2AServerConfig(protocol="jsonrpc"), {})

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE_URL) as client:
        assert (await client.get("/health")).status_code == 503
        async with app.router.lifespan_context(app):
            assert (await client.get("/health")).text == "ok"
        assert (await client.get("/health")).status_code == 503

    assert closed == [True]
