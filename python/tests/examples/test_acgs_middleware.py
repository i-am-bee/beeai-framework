# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the ACGS middleware example.

Most tests use a small stand-in for ``acgs-lite`` so they run where the package is not installed.
Tests marked with ``requires_acgs`` use the real package and are skipped without it.
"""

import json
import socket
import sys
import types
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import pytest

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import AssistantMessage
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.message import MessageToolCallContent
from beeai_framework.backend.types import ChatModelInput, ChatModelOutput
from beeai_framework.context import RunContext
from beeai_framework.tools import tool
from examples.middleware.acgs import INSTALL_MESSAGE, ACGSMiddleware

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Tests must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)


@dataclass
class FakeViolation:
    rule_id: str
    rule_text: str


@dataclass
class FakeResult:
    valid: bool
    violations: list[FakeViolation] = field(default_factory=list)


class FakeViolationError(Exception):
    def __init__(self, message: str, *, rule_id: str) -> None:
        super().__init__(message)
        self.rule_id = rule_id


class FakeEngine:
    """Blocks any call whose text contains ``secret``; ``halt`` and ``crash`` raise."""

    def __init__(self, constitution: Any, *, strict: bool) -> None:
        self.actions: list[str] = []

    def validate(self, action: str, *, agent_id: str, strict: bool) -> FakeResult:
        self.actions.append(action)
        if "crash" in action:
            raise RuntimeError("engine failure")
        if "halt" in action:
            raise FakeViolationError("halted", rule_id="FAKE-HALT")
        if "secret" in action:
            return FakeResult(valid=False, violations=[FakeViolation("FAKE-1", "No secrets")])
        return FakeResult(valid=True)


@pytest.fixture
def fake_acgs(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("acgs_lite")
    module.Constitution = types.SimpleNamespace(default=lambda: "default-constitution")  # type: ignore[attr-defined]
    module.GovernanceEngine = FakeEngine  # type: ignore[attr-defined]
    module.ConstitutionalViolationError = FakeViolationError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "acgs_lite", module)


class ScriptedModel(ChatModel):
    """Calls ``send_note`` once with fixed arguments, then gives the final answer."""

    def __init__(self, text: str) -> None:
        super().__init__(
            allow_parallel_tool_calls=True,
            retry_on_empty_response=False,
            fix_invalid_tool_calls=False,
            tool_call_fallback_via_response_format=False,
        )
        self.text = text
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "synthetic-local-script"

    @property
    def provider_id(self) -> ProviderName:
        return "openai"  # A namespace only; no provider adapter is constructed.

    async def _create(self, input: ChatModelInput, run: RunContext) -> ChatModelOutput:
        self.calls += 1
        name, args = ("send_note", {"text": self.text}) if self.calls == 1 else ("final_answer", {"response": "done"})
        return ChatModelOutput(
            output=[AssistantMessage(MessageToolCallContent(id=str(self.calls), tool_name=name, args=json.dumps(args)))]
        )

    async def _create_stream(self, input: ChatModelInput, run: RunContext) -> AsyncGenerator[ChatModelOutput]:
        yield await self._create(input, run)


def make_tool(calls: list[str]) -> Any:
    @tool(name="send_note", description="Synthetic protected operation")
    async def send_note(text: str) -> str:
        calls.append(text)
        return "sent"

    return send_note


def test_missing_package_raises_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry in sys.modules makes the import fail, as if the package were not installed.
    # pyrefly: ignore [bad-argument-type]
    monkeypatch.setitem(sys.modules, "acgs_lite", None)
    with pytest.raises(ImportError, match="acgs-lite") as excinfo:
        ACGSMiddleware()
    assert str(excinfo.value) == INSTALL_MESSAGE


@pytest.mark.usefixtures("fake_acgs")
@pytest.mark.parametrize(
    ("text", "allowed", "reason"),
    [
        ("hello", True, "allow"),
        ("a secret value", False, "FAKE-1 (No secrets)"),
        ("halt now", False, "rule FAKE-HALT halted the call"),
        ("crash now", False, "could not be evaluated (RuntimeError)"),
    ],
)
def test_evaluate_fails_closed(text: str, allowed: bool, reason: str) -> None:
    middleware = ACGSMiddleware()
    decision, message = middleware.evaluate("send_note", {"text": text})
    assert decision is allowed
    assert reason in message


@pytest.mark.usefixtures("fake_acgs")
@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "executed"), [("hello", 1), ("a secret value", 0), ("crash now", 0)])
async def test_tool_run_is_blocked_before_execution(text: str, executed: int) -> None:
    calls: list[str] = []
    output = await make_tool(calls).run({"text": text}).middleware(ACGSMiddleware())
    assert len(calls) == executed
    if executed:
        assert output.get_text_content() == "sent"
    else:
        assert output.get_text_content().startswith("ACGS blocked tool 'send_note' before execution:")


@pytest.mark.usefixtures("fake_acgs")
@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "executed"), [("hello", 1), ("a secret value", 0)])
async def test_agent_run_continues_after_a_blocked_call(text: str, executed: int) -> None:
    calls: list[str] = []
    middleware = ACGSMiddleware()
    agent = RequirementAgent(llm=ScriptedModel(text), tools=[make_tool(calls)], middlewares=[middleware])
    response = await agent.run("synthetic fixture", max_iterations=4)
    assert len(calls) == executed
    assert response.last_message.text == "done"
    engine = middleware._engine
    assert isinstance(engine, FakeEngine)
    checked = engine.actions
    assert len(checked) == 1 and checked[0].startswith("tool: send_note\n")  # final_answer is not checked


@pytest.mark.asyncio
async def test_real_acgs_lite_default_rules() -> None:
    pytest.importorskip("acgs_lite")
    calls: list[str] = []
    middleware = ACGSMiddleware()
    send_note = make_tool(calls)

    allowed = await send_note.run({"text": "The report is ready."}).middleware(middleware)
    blocked = await send_note.run({"text": "The password is hunter2"}).middleware(middleware)

    assert calls == ["The report is ready."]
    assert allowed.get_text_content() == "sent"
    assert "ACGS-006" in blocked.get_text_content()
