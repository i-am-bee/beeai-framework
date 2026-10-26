import asyncio
import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from beeai_framework.agents.requirement.utils._tool import FinalAnswerTool
from beeai_framework.context import RunContext, RunContextStartEvent, RunMiddlewareProtocol
from beeai_framework.emitter import EmitterOptions, EventMeta
from beeai_framework.emitter.utils import create_internal_event_matcher
from beeai_framework.tools import Tool, tool
from beeai_framework.tools.types import StringToolOutput

INSTALL_MESSAGE = "The 'acgs-lite' package is NOT installed. Run 'pip install -r examples/middleware/requirements.txt'."


class ACGSMiddleware(RunMiddlewareProtocol):
    """Example pre-tool middleware that checks tool calls against ACGS rules.

    Install the rule engine separately, for example with ``pip install "acgs-lite>=3.0.1,<4"``.
    Rules are evaluated locally. This example fails closed: a matching blocking rule, a rule that
    asks for human review, or any error while evaluating denies the call. A denied tool does not
    run; the reason comes back as the tool output, so the agent run continues.
    """

    def __init__(self, *, constitution: Any = None, agent_id: str = "beeai") -> None:
        try:
            # pyrefly: ignore [missing-import]
            import acgs_lite
        except ImportError as error:
            raise ImportError(INSTALL_MESSAGE) from error

        self.constitution = constitution if constitution is not None else acgs_lite.Constitution.default()
        self.agent_id = agent_id
        self._engine = acgs_lite.GovernanceEngine(self.constitution, strict=False)
        self._violation_error: type[Exception] = acgs_lite.ConstitutionalViolationError
        self._bound_run_ids: set[str] = set()

    def bind(self, ctx: RunContext) -> None:
        if ctx.run_id in self._bound_run_ids:
            return
        self._bound_run_ids.add(ctx.run_id)

        @ctx.emitter.on(
            create_internal_event_matcher("start"),
            # A low priority runs the check after other listeners that may still change the input.
            options=EmitterOptions(match_nested=True, is_blocking=True, priority=-100),
        )
        async def handle_start(data: Any, meta: EventMeta) -> None:
            if not isinstance(data, RunContextStartEvent) or not isinstance(meta.creator, RunContext):
                return
            tool_instance = meta.creator.instance
            if not isinstance(tool_instance, Tool) or isinstance(tool_instance, FinalAnswerTool):
                return

            allowed, reason = self.evaluate(tool_instance.name, data.input.get("input"))
            if allowed:
                return

            data.output = StringToolOutput(
                result=f"ACGS blocked tool '{tool_instance.name}' before execution: {reason}"
            )

        @ctx.emitter.on(
            create_internal_event_matcher("finish"),
            options=EmitterOptions(match_nested=False),
        )
        async def handle_finish(_: Any, __: EventMeta) -> None:
            ctx.emitter.off(callback=handle_start)
            ctx.emitter.off(callback=handle_finish)
            self._bound_run_ids.discard(ctx.run_id)

    def evaluate(self, tool_name: str, tool_input: Any) -> tuple[bool, str]:
        """Return ``(allowed, reason)`` for one tool call. Never raises: errors deny the call."""
        try:
            result = self._engine.validate(
                self._format_call(tool_name, tool_input), agent_id=self.agent_id, strict=False
            )
        except self._violation_error as error:
            return False, f"rule {getattr(error, 'rule_id', None) or 'unknown'} halted the call"
        except Exception as error:
            return False, f"the rules could not be evaluated ({type(error).__name__}), so the call was denied"

        if result.valid and not result.violations:
            return True, "allow"
        rules = "; ".join(f"{v.rule_id} ({v.rule_text})" for v in result.violations)
        return False, f"matched {rules}" if rules else "the call was reported as invalid"

    @staticmethod
    def _format_call(tool_name: str, tool_input: Any) -> str:
        if isinstance(tool_input, BaseModel):
            tool_input = tool_input.model_dump(mode="json")
        elif isinstance(tool_input, Mapping):
            tool_input = dict(tool_input)
        arguments = json.dumps(tool_input, default=str, sort_keys=True, ensure_ascii=False)
        return f"tool: {tool_name}\narguments: {arguments}"


@tool(name="send_email", description="Example email tool protected by ACGS")
async def send_email(to: str, body: str) -> str:
    return f"Tool would send an email to {to}"


async def main() -> None:
    middleware = ACGSMiddleware()

    allowed = await send_email.run({"to": "team@example.com", "body": "The report is ready."}).middleware(middleware)
    print(allowed.get_text_content())

    blocked = await send_email.run({"to": "team@example.com", "body": "The password is hunter2"}).middleware(middleware)
    print(blocked.get_text_content())


if __name__ == "__main__":
    asyncio.run(main())
