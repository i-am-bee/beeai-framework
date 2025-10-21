import asyncio
from typing import Any

from beeai_framework.backend import AssistantMessage, ChatModel, ChatModelOutput, UserMessage
from beeai_framework.context import RunContext, RunContextStartEvent, RunMiddlewareProtocol
from beeai_framework.emitter import EmitterOptions, EventMeta
from beeai_framework.emitter.utils import create_internal_event_matcher


class OverrideResponseMiddleware(RunMiddlewareProtocol):
    """Middleware that sets the result value for a given runnable without executing it"""

    def __init__(self, result: Any) -> None:
        self._result = result

    def bind(self, ctx: RunContext) -> None:
        """Calls once the target is about to be run."""

        ctx.emitter.on(
            create_internal_event_matcher("start", ctx.instance),
            self._run,
            # ensures that this callback will be the first invoked
            EmitterOptions(is_blocking=True, priority=1),
        )

    async def _run(self, data: RunContextStartEvent, meta: EventMeta) -> None:
        """Set output property to the result which prevents an execution of the target handler."""

        data.output = self._result


async def main() -> None:
    middleware = OverrideResponseMiddleware(ChatModelOutput(output=[AssistantMessage("BeeAI is the best!")]))
    response = await ChatModel.from_name("ollama:granite3.3").run([UserMessage("Hello!")]).middleware(middleware)
    print(response.get_text_content())  # "BeeAI is the best!"


if __name__ == "__main__":
    asyncio.run(main())
