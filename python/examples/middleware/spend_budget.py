import asyncio
from collections.abc import Callable
from typing import Any, ClassVar

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel, ChatModelOutput
from beeai_framework.context import RunContext, RunContextFinishEvent, RunContextStartEvent, RunMiddlewareProtocol
from beeai_framework.emitter import EmitterOptions, EventMeta
from beeai_framework.emitter.utils import create_internal_event_matcher

INSTALL_MESSAGE = (
    "The 'llm-agent-budget' package is NOT installed. Run 'pip install -r examples/middleware/requirements.txt'."
)


class SpendBudgetMiddleware(RunMiddlewareProtocol):
    """Example middleware that puts a hard USD cap on the model calls of a run.

    Install the budget library separately, for example with ``pip install "llm-agent-budget>=0.1.3,<0.2"``.
    Before each model call it reserves ``reserve_usd`` from the budget. If that does not fit, the call
    raises ``BudgetExhausted`` and never reaches the provider. When the call finishes, the hold is
    replaced by what the call cost: ``ChatModelOutput.cost``, which the LiteLLM adapters fill in, or
    your own ``cost`` function. A call that reports no cost (for example a model missing from
    LiteLLM's price map) is booked at ``reserve_usd``, so an unknown price cannot slip past the cap.
    So is a call that fails, because the provider may already have charged for it (for example a
    reply that is then rejected as an invalid tool call). A refused call raises an error whose
    cause chain holds the ``BudgetExhausted``; the provider is not called.

    With ``run_usd``, every run gets its own slice of a shared budget: the slice is reserved when the
    run starts, and what the run did not spend goes back when it ends. Agents running at the same
    time can then never be admitted for more than the shared budget holds. Model calls made by
    nested runs (for example a handoff to another agent) count against the run they belong to.
    """

    # Shared by every instance, by run id: the runs in progress and their parents, the runs a middleware
    # is bound to, and the holds of model calls in flight. A model call is reserved once, on the
    # nearest enclosing run that has this middleware, even when a nested agent has its own.
    # They are class-level, not per instance, because a handoff starts the nested agent with its own
    # middleware instance: only a shared map lets that instance find the parent run and charge the
    # parent's budget. Keys are run ids, so concurrent runs never share an entry.
    _parents: ClassVar[dict[str, str | None]] = {}
    _bound: ClassVar[dict[str, tuple["SpendBudgetMiddleware", Any]]] = {}
    _holds: ClassVar[dict[str, tuple["SpendBudgetMiddleware", Any]]] = {}

    def __init__(
        self,
        budget: Any,
        *,
        reserve_usd: float = 0.05,
        run_usd: float | None = None,
        cost: Callable[[ChatModelOutput], float] | None = None,
    ) -> None:
        try:
            # pyrefly: ignore [missing-import]
            import agent_budget
        except ImportError as error:
            raise ImportError(INSTALL_MESSAGE) from error

        if not isinstance(budget, agent_budget.Budget):
            raise TypeError("budget must be an agent_budget.Budget")
        if reserve_usd <= 0:
            raise ValueError("reserve_usd must be greater than zero")
        if run_usd is not None and run_usd <= 0:
            raise ValueError("run_usd must be greater than zero")
        self.budget = budget
        self.reserve_usd = reserve_usd
        self.run_usd = run_usd
        self.cost = cost
        self.receipts: list[Any] = []

    def bind(self, ctx: RunContext) -> None:
        if ctx.run_id in self._bound:
            return
        # Raises BudgetExhausted here, before the run starts, when the slice does not fit.
        name = f"{type(ctx.instance).__name__}:{ctx.run_id[:8]}"
        scope: Any = self.budget.child(name, usd=self.run_usd) if self.run_usd else self.budget
        self._bound[ctx.run_id] = (self, scope)

        @ctx.emitter.on(
            create_internal_event_matcher("start"),
            # A low priority reserves after other listeners that may still answer the call themselves.
            options=EmitterOptions(match_nested=True, is_blocking=True, priority=-100),
        )
        async def handle_start(data: Any, meta: EventMeta) -> None:
            if not isinstance(data, RunContextStartEvent) or not isinstance(meta.creator, RunContext):
                return
            run = meta.creator
            if not isinstance(run.instance, ChatModel):
                self._parents[run.run_id] = run.parent_id
                return
            if run.run_id in self._holds:
                return
            owner = run.run_id if run.run_id in self._bound else run.parent_id
            while owner is not None and owner not in self._bound:
                owner = self._parents.get(owner)
            middleware, budget = self._bound.get(owner or "", (self, scope))
            try:
                self._holds[run.run_id] = (middleware, budget.reserve(middleware.reserve_usd))
            except Exception:
                if run.run_id == ctx.run_id:  # this run now ends without a finish event
                    unbind()
                raise

        @ctx.emitter.on(
            create_internal_event_matcher("finish"),
            options=EmitterOptions(match_nested=True, is_blocking=True),
        )
        async def handle_finish(data: Any, meta: EventMeta) -> None:
            if not isinstance(data, RunContextFinishEvent) or not isinstance(meta.creator, RunContext):
                return
            run_id = meta.creator.run_id
            self._parents.pop(run_id, None)
            if (held := self._holds.pop(run_id, None)) is not None:
                middleware, hold = held
                output = data.output if data.error is None else None
                hold.settle(
                    middleware._call_cost(output) if isinstance(output, ChatModelOutput) else middleware.reserve_usd
                )
            if run_id == ctx.run_id:
                unbind()

        def unbind() -> None:
            ctx.emitter.off(callback=handle_start)
            ctx.emitter.off(callback=handle_finish)
            self._bound.pop(ctx.run_id, None)
            if scope is not self.budget:
                self.receipts.append(scope.close())

    def _call_cost(self, output: ChatModelOutput) -> float:
        if self.cost is not None:
            return self.cost(output)
        reported = output.cost.total_cost_usd
        return reported if reported > 0 else self.reserve_usd


async def main() -> None:
    # pyrefly: ignore [missing-import]
    from agent_budget import Budget

    # $0.05 for the whole team, at most $0.02 per agent run, at most $0.01 per model call.
    budget = Budget(0.05, name="team")
    middleware = SpendBudgetMiddleware(budget, reserve_usd=0.01, run_usd=0.02)
    agent = RequirementAgent(
        llm=ChatModel.from_name("openai:gpt-4.1-mini"),
        role="Assistant",
        instructions="Answer in one sentence.",
        middlewares=[middleware],
    )

    questions = ["What is a budget?", "What is a token?", "What is a cap?"]
    agents = [await agent.clone() for _ in questions]
    results = await asyncio.gather(*(a.run(q) for a, q in zip(agents, questions, strict=True)), return_exceptions=True)
    for question, result in zip(questions, results, strict=True):
        print(question, "->", result if isinstance(result, BaseException) else result.last_message.text)

    # Two $0.02 slices fit in $0.05; the third run is refused before it calls the model.
    for receipt in middleware.receipts:
        print(f"{receipt.scope}: spent ${receipt.spent_usd:.6f}, returned ${receipt.released_usd:.6f}")
    print(f"team: spent ${budget.spent_usd:.6f} of ${budget.limit_usd:.2f}, ${budget.reserved_usd:.6f} still held")


if __name__ == "__main__":
    asyncio.run(main())
