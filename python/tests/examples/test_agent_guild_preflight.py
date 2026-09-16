# Copyright 2026 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
"""Offline fixtures exercise actual RequirementAgent and its blocking tool boundary."""

import json
import socket
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements import Requirement
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from beeai_framework.backend import AssistantMessage
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.message import MessageToolCallContent
from beeai_framework.backend.types import ChatModelInput, ChatModelOutput
from beeai_framework.context import RunContext
from beeai_framework.tools import tool
from examples.agents.requirement.agent_guild_preflight import (
    PREFLIGHT_URL,
    PreflightRequirement,
    denial,
    validate_endpoint,
)

ENDPOINT = "https://selected.example/mcp"
MCP_UNKNOWNS = frozenset({"agent_card_resolves", "agent_card_signed", "payment_claim_holds", "independent_evidence"})
pytestmark = pytest.mark.unit


class ScriptedModel(ChatModel):
    def __init__(self) -> None:
        super().__init__(
            allow_parallel_tool_calls=True,
            retry_on_empty_response=False,
            fix_invalid_tool_calls=False,
            tool_call_fallback_via_response_format=False,
        )
        self.calls = 0
        self.offered: list[list[str]] = []

    @property
    def model_id(self) -> str:
        return "synthetic-local-script"

    @property
    def provider_id(self) -> ProviderName:
        return "openai"  # A namespace only; no provider adapter is constructed.

    async def _create(self, input: ChatModelInput, run: RunContext) -> ChatModelOutput:
        self.calls += 1
        self.offered.append([t.name for t in input.tools or []])
        name, args = ("selected", {"text": "fixture"}) if self.calls == 1 else ("final_answer", {"response": "done"})
        return ChatModelOutput(
            output=[
                AssistantMessage(
                    MessageToolCallContent(
                        id=str(self.calls),
                        tool_name=name,
                        args=json.dumps(args),
                    )
                )
            ]
        )

    async def _create_stream(self, input: ChatModelInput, run: RunContext) -> AsyncGenerator[ChatModelOutput]:
        yield await self._create(input, run)


def document(*, unknown: str | None = None, failed: str | None = None) -> dict[str, Any]:
    # The six-check MCP contract after the September 2026 optional-card change:
    # reachability/handshake are observed, while card/payment/history stay unknown.
    statuses = {
        "endpoint_reachable": "proven",
        "protocol_handshake": "proven",
        "agent_card_resolves": "unknown",
        "agent_card_signed": "unknown",
        "payment_claim_holds": "unknown",
        "independent_evidence": "unknown",
    }
    if unknown:
        statuses[unknown] = "unknown"
    if failed:
        statuses[failed] = "failed"
    return {
        "target": ENDPOINT,
        "verdict": "do_not_delegate"
        if failed in ("endpoint_reachable", "protocol_handshake")
        else "delegate_with_caution"
        if failed
        else "no_failed_checks",
        "checks": [{"check": name, "status": status} for name, status in statuses.items()],
        "failed": [name for name, status in statuses.items() if status == "failed"],
        "unknowns": [name for name, status in statuses.items() if status == "unknown"],
        "scored": [name for name, status in statuses.items() if status != "unknown"],
    }


def omit_checks(doc: dict[str, Any], names: set[str] | frozenset[str]) -> None:
    doc["checks"] = [row for row in doc["checks"] if row["check"] not in names]
    for summary in ("failed", "unknowns", "scored"):
        doc[summary] = [name for name in doc[summary] if name not in names]


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Tests must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario,executed",
    [
        ("allow", 1),
        ("failed", 0),
        ("caution", 0),
        ("unknown", 0),
        ("required_unknown", 0),
        ("omitted_unknown", 0),
        ("sparse", 0),
        ("mismatch", 0),
        ("unavailable", 0),
        ("redirect", 0),
        ("oversized", 0),
        ("changed_at_start", 0),
        ("other_requirement", 0),
    ],
)
async def test_native_execution_boundary(scenario: str, executed: int) -> None:
    calls: list[str] = []
    requests: list[httpx.Request] = []

    @tool(name="selected", description="Synthetic protected operation")
    async def selected(text: str) -> str:
        calls.append(text)
        return "synthetic result"

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert str(request.url).split("?")[0] == PREFLIGHT_URL
        assert request.method == "GET" and request.url.params["url"] == ENDPOINT
        assert "authorization" not in request.headers and "cookie" not in request.headers
        body = document()
        if scenario == "failed" or (scenario == "changed_at_start" and len(requests) >= 2):
            body = document(failed="protocol_handshake")
        elif scenario == "caution":
            body = document(failed="independent_evidence")
        elif scenario == "required_unknown":
            body = document(unknown="protocol_handshake")
        elif scenario == "omitted_unknown":
            omit_checks(body, {"independent_evidence"})
        elif scenario == "sparse":
            omit_checks(body, MCP_UNKNOWNS)
        elif scenario == "mismatch":
            body["target"] = "https://different.example/mcp"
        elif scenario == "unavailable":
            raise httpx.ConnectError("Synthetic failure")
        elif scenario == "redirect":
            return httpx.Response(302, headers={"location": "https://elsewhere.example"})
        elif scenario == "oversized":
            return httpx.Response(200, content=b" " * 65_537, headers={"content-type": "application/json"})
        return httpx.Response(200, json=body)

    guard = PreflightRequirement(
        selected,
        ENDPOINT,
        disclose_endpoint=True,
        tolerated_unknowns=MCP_UNKNOWNS | {"protocol_handshake"}
        if scenario == "required_unknown"
        else MCP_UNKNOWNS - {"independent_evidence"}
        if scenario == "omitted_unknown"
        else frozenset()
        if scenario in ("unknown", "sparse")
        else MCP_UNKNOWNS,
        transport=httpx.MockTransport(respond),
    )
    model = ScriptedModel()
    requirements: list[Requirement[Any]] = [guard]
    if scenario == "other_requirement":
        requirements.append(ConditionalRequirement(selected, max_invocations=0))
    agent = RequirementAgent(llm=model, tools=[selected], requirements=requirements)
    await agent.run("synthetic fixture", max_iterations=4)
    assert len(calls) == executed
    assert guard.evidence and all(o.endpoint == ENDPOINT for o in guard.evidence)
    assert all(o.retrieved_at and isinstance(o.raw_body, bytes) for o in guard.evidence)
    if scenario == "changed_at_start":
        assert "selected" in model.offered[0]
        assert guard.evidence[0].denied is None
        assert guard.evidence[1].denied is not None
    if scenario == "other_requirement":
        assert "selected" not in model.offered[0]


@pytest.mark.parametrize(
    "malformation",
    ["duplicate_key", "duplicate_check", "summary", "scored", "missing_scored", "duplicate_scored", "verdict", "shape"],
)
def test_inconsistent_evidence_never_allows(malformation: str) -> None:
    doc = document()
    if malformation == "duplicate_check":
        doc["checks"].append(doc["checks"][0])
    elif malformation == "summary":
        doc["unknowns"] = ["independent_evidence"]
    elif malformation == "verdict":
        doc["verdict"] = "hire"
    elif malformation == "scored":
        doc["scored"].append("agent_card_resolves")
    elif malformation == "missing_scored":
        del doc["scored"]
    elif malformation == "duplicate_scored":
        doc["scored"].append("endpoint_reachable")
    elif malformation == "shape":
        doc["checks"] = [None]
    raw = json.dumps(doc).encode()
    if malformation == "duplicate_key":
        raw = raw[:-1] + b', "target":"https://selected.example/mcp"}'
    with pytest.raises(ValueError):
        denial(raw, ENDPOINT, MCP_UNKNOWNS)


@pytest.mark.parametrize(
    "missing",
    ["endpoint_reachable", "protocol_handshake", *sorted(MCP_UNKNOWNS)],
)
def test_omitting_a_check_cannot_erase_unknowns(missing: str) -> None:
    doc = document()
    policy = MCP_UNKNOWNS - {missing}
    if missing in MCP_UNKNOWNS:
        assert denial(json.dumps(doc).encode(), ENDPOINT, policy) is not None
    omit_checks(doc, {missing})  # Summaries remain consistent: omission itself must block.
    with pytest.raises(ValueError, match="Missing mandatory checks"):
        denial(json.dumps(doc).encode(), ENDPOINT, policy)


@pytest.mark.parametrize("status", ["proven", "unknown", "failed"])
@pytest.mark.parametrize("tolerated", [False, True])
def test_additive_checks_obey_the_same_policy(status: str, tolerated: bool) -> None:
    doc = document()
    doc["checks"].append({"check": "future_check", "status": status})
    if status == "unknown":
        doc["unknowns"].append("future_check")
    else:
        doc["scored"].append("future_check")
    if status == "failed":
        doc["failed"].append("future_check")
        doc["verdict"] = "delegate_with_caution"
    policy = MCP_UNKNOWNS | {"future_check"} if tolerated else MCP_UNKNOWNS
    reason = denial(json.dumps(doc).encode(), ENDPOINT, policy)
    assert (reason is None) == (status == "proven" or (status == "unknown" and tolerated))


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://selected.example/mcp",
        "https://user:pass@selected.example/mcp",
        "https://selected.example/mcp?key=secret",
        "https://selected.example/mcp#secret",
        "https://selected.example/has space",
        "https://selected.example:bad/mcp",
    ],
)
def test_ambiguous_or_credential_bearing_url_rejected(endpoint: str) -> None:
    with pytest.raises(ValueError):
        validate_endpoint(endpoint)


def test_explicit_disclosure_required() -> None:
    @tool(name="selected", description="Synthetic operation")
    async def selected() -> str:
        return "unused"

    with pytest.raises(ValueError, match="approve"):
        PreflightRequirement(selected, ENDPOINT, disclose_endpoint=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked", [False, True])
async def test_native_mcp_session_uses_the_selected_endpoint(monkeypatch: pytest.MonkeyPatch, blocked: bool) -> None:
    from examples.agents.requirement.agent_guild_preflight import delegate

    actual_client = httpx.AsyncClient
    mcp_calls: list[dict[str, Any]] = []
    observed_urls: list[str] = []

    def server(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and str(request.url).split("?")[0] == PREFLIGHT_URL:
            observed_urls.append(request.url.params["url"])
            return httpx.Response(200, json=document(failed="protocol_handshake" if blocked else None))
        assert str(request.url) == ENDPOINT
        if request.method == "GET":
            return httpx.Response(405)
        if request.method == "DELETE":
            return httpx.Response(200)
        assert request.method == "POST"
        msg = json.loads(request.content)
        if "id" not in msg:
            return httpx.Response(202)
        method = msg["method"]
        if method == "initialize":
            result = {
                "protocolVersion": msg["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "synthetic-in-memory", "version": "1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": name,
                        "description": "Synthetic local tool",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    }
                    for name in ("selected", "not_selected")
                ]
            }
        elif method == "tools/call":
            mcp_calls.append(msg["params"])
            result = {"content": [{"type": "text", "text": "synthetic completed"}], "isError": False}
        else:
            raise AssertionError(method)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": msg["id"], "result": result})

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(server)
        return actual_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    await delegate(
        ENDPOINT,
        "selected",
        "synthetic fixture",
        ScriptedModel(),
        disclose_endpoint=True,
        tolerated_unknowns=MCP_UNKNOWNS,
    )
    assert observed_urls and set(observed_urls) == {ENDPOINT}
    assert len(mcp_calls) == (0 if blocked else 1)
    assert all(call["name"] == "selected" for call in mcp_calls)
