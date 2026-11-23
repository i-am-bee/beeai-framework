# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the spend budget middleware example. They need ``llm-agent-budget`` and are skipped without it."""

import asyncio
import importlib
import importlib.util
import json
import socket
import sys
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import AssistantMessage, UserMessage
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.message import MessageToolCallContent
from beeai_framework.backend.types import ChatModelCost, ChatModelInput, ChatModelOutput
from beeai_framework.context import RunContext
from beeai_framework.tools import tool
from beeai_framework.tools.handoff import HandoffTool
from examples.middleware.spend_budget import INSTALL_MESSAGE, SpendBudgetMiddleware

agent_budget: Any = importlib.import_module("agent_budget") if importlib.util.find_spec("agent_budget") else None

pytestmark = pytest.mark.unit
requires_agent_budget = pytest.mark.skipif(agent_budget is None, reason="llm-agent-budget is not installed")


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Tests must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)


class ScriptedModel(ChatModel):
    """Each call costs ``cost_usd``. Calls the tools in ``script`` one per call, then gives the final answer."""

    def __init__(self, cost_usd: float, *, script: list[str] | None = None, fail: bool = False) -> None:
        super().__init__(
            allow_parallel_tool_calls=True,
            retry_on_empty_response=False,
            fix_invalid_tool_calls=False,
            tool_call_fallback_via_response_format=False,
        )
        self.cost_usd = cost_usd
        self.script = script or []
        self.fail = fail
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "synthetic-local-script"

    @property
    def provider_id(self) -> ProviderName:
        return "openai"  # A namespace only; no provider adapter is constructed.

    async def _create(self, input: ChatModelInput, run: RunContext) -> ChatModelOutput:
        self.calls += 1
        await asyncio.sleep(0)
        if self.fail:
            raise RuntimeError("provider error")
        name, args = ("final_answer", {"response": "done"})
        if self.calls <= len(self.script):
            name, args = self.script[self.calls - 1], {"task": f"step {self.calls}"}
        return ChatModelOutput(
            output=[
                AssistantMessage(MessageToolCallContent(id=str(self.calls), tool_name=name, args=json.dumps(args)))
            ],
            cost=ChatModelCost(total_cost_usd=self.cost_usd),
        )

    async def _create_stream(self, input: ChatModelInput, run: RunContext) -> AsyncGenerator[ChatModelOutput]:
        yield await self._create(input, run)


@tool(name="ping", description="Synthetic tool")
async def ping(task: str) -> str:
    return "pong"


def budget_error(error: BaseException | None) -> BaseException | None:
    """The BudgetError in an error's cause chain (BeeAI wraps errors raised by middleware listeners)."""
    while error is not None:
        if isinstance(error, agent_budget.BudgetError):
            return error
        if isinstance(error, BaseExceptionGroup):
            return next((found for e in error.exceptions if (found := budget_error(e))), None)
        error = error.__cause__
    return None


def test_missing_package_raises_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry in sys.modules makes the import fail, as if the package were not installed.
    # pyrefly: ignore [bad-argument-type]
    monkeypatch.setitem(sys.modules, "agent_budget", None)
    with pytest.raises(ImportError) as excinfo:
        SpendBudgetMiddleware(object())
    assert str(excinfo.value) == INSTALL_MESSAGE


@requires_agent_budget
@pytest.mark.asyncio
async def test_model_call_books_reported_cost() -> None:
    budget = agent_budget.Budget(1.00)
    middleware = SpendBudgetMiddleware(budget, reserve_usd=0.10)
    model = ScriptedModel(0.0123)

    await model.run([UserMessage("hi")]).middleware(middleware)

    assert budget.spent_usd == pytest.approx(0.0123)
    assert budget.reserved_usd == 0
    assert not SpendBudgetMiddleware._holds and not SpendBudgetMiddleware._bound


@requires_agent_budget
@pytest.mark.asyncio
async def test_call_without_reported_cost_is_booked_at_the_reservation() -> None:
    budget = agent_budget.Budget(1.00)
    await ScriptedModel(0.0).run([UserMessage("hi")]).middleware(SpendBudgetMiddleware(budget, reserve_usd=0.10))
    assert budget.spent_usd == pytest.approx(0.10)

    priced = agent_budget.Budget(1.00)
    middleware = SpendBudgetMiddleware(priced, reserve_usd=0.10, cost=lambda output: 0.004)
    await ScriptedModel(0.0).run([UserMessage("hi")]).middleware(middleware)
    assert priced.spent_usd == pytest.approx(0.004)


@requires_agent_budget
@pytest.mark.asyncio
async def test_failed_call_is_booked_at_the_reservation() -> None:
    budget = agent_budget.Budget(1.00)
    middleware = SpendBudgetMiddleware(budget, reserve_usd=0.10)
    with pytest.raises(Exception, match="Chat Model error"):
        await ScriptedModel(0.5, fail=True).run([UserMessage("hi")]).middleware(middleware)
    assert budget.spent_usd == pytest.approx(0.10)
    assert budget.reserved_usd == 0
    assert not SpendBudgetMiddleware._holds and not SpendBudgetMiddleware._bound


@requires_agent_budget
@pytest.mark.asyncio
async def test_model_call_that_does_not_fit_is_never_sent() -> None:
    # Each call costs $0.03 and reserves $0.04, so the fourth call no longer fits in $0.10.
    budget = agent_budget.Budget(0.10)
    model = ScriptedModel(0.03)
    middleware = SpendBudgetMiddleware(budget, reserve_usd=0.04)

    for _ in range(3):
        await model.run([UserMessage("hi")]).middleware(middleware)
    with pytest.raises(Exception) as excinfo:
        await model.run([UserMessage("hi")]).middleware(middleware)

    assert isinstance(budget_error(excinfo.value), agent_budget.BudgetExhausted)
    assert model.calls == 3
    assert budget.spent_usd == pytest.approx(0.09)
    assert budget.reserved_usd == 0
    assert not (SpendBudgetMiddleware._holds or SpendBudgetMiddleware._bound)


@requires_agent_budget
@pytest.mark.asyncio
async def test_agent_in_a_tool_loop_stops_at_the_cap() -> None:
    budget = agent_budget.Budget(0.10)
    model = ScriptedModel(0.03, script=["ping"] * 50)
    agent = RequirementAgent(llm=model, tools=[ping], middlewares=[SpendBudgetMiddleware(budget, reserve_usd=0.04)])

    with pytest.raises(Exception) as excinfo:
        await agent.run("synthetic fixture", max_iterations=50)

    assert isinstance(budget_error(excinfo.value), agent_budget.BudgetExhausted)
    assert model.calls == 3  # without the middleware the loop runs 50 model calls
    assert budget.spent_usd == pytest.approx(0.09)
    assert budget.reserved_usd == 0
    assert not (SpendBudgetMiddleware._holds or SpendBudgetMiddleware._bound or SpendBudgetMiddleware._parents)


@requires_agent_budget
@pytest.mark.asyncio
async def test_parallel_agent_runs_share_one_budget() -> None:
    budget = agent_budget.Budget(0.05, name="team")
    middleware = SpendBudgetMiddleware(budget, reserve_usd=0.01, run_usd=0.02)
    models = [ScriptedModel(0.003) for _ in range(3)]
    agents = [RequirementAgent(llm=model, middlewares=[middleware]) for model in models]

    results = await asyncio.gather(*(agent.run("synthetic fixture") for agent in agents), return_exceptions=True)

    finished = [r for r in results if not isinstance(r, BaseException)]
    refused = [r for r in results if isinstance(r, BaseException)]
    assert len(finished) == 2 and len(refused) == 1  # two $0.02 slices fit in $0.05, the third does not
    assert isinstance(refused[0], agent_budget.BudgetExhausted)
    assert sorted(model.calls for model in models) == [0, 1, 1]
    assert budget.spent_usd == pytest.approx(0.006)
    assert budget.reserved_usd == 0  # unused slices went back
    assert [round(r.spent_usd, 9) for r in middleware.receipts] == [0.003, 0.003]


@requires_agent_budget
@pytest.mark.asyncio
async def test_handoff_to_a_budgeted_agent_is_counted_once() -> None:
    team = agent_budget.Budget(1.00, name="team")
    outer = SpendBudgetMiddleware(team, reserve_usd=0.10)
    inner = SpendBudgetMiddleware(team, reserve_usd=0.10, run_usd=0.20)
    helper_model = ScriptedModel(0.002)
    helper = RequirementAgent(name="Helper", llm=helper_model, middlewares=[inner])
    lead_model = ScriptedModel(0.001, script=["helper"])
    lead = RequirementAgent(
        llm=lead_model,
        tools=[HandoffTool(helper, name="Helper", description="Delegate a task.")],
        middlewares=[outer],
    )

    response = await lead.run("synthetic fixture", max_iterations=4)

    assert response.last_message.text == "done"
    assert helper_model.calls == 1 and lead_model.calls == 2
    assert team.spent_usd == pytest.approx(0.004)  # 2 x $0.001 + $0.002, nothing booked twice
    assert team.reserved_usd == 0
    assert [round(r.spent_usd, 9) for r in inner.receipts] == [0.002]  # booked on the helper's own slice
    assert not (SpendBudgetMiddleware._holds or SpendBudgetMiddleware._bound or SpendBudgetMiddleware._parents)
