import asyncio
import re
import sys
import traceback
from typing import Any

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.events import RequirementAgentStartEvent
from beeai_framework.backend import ChatModel, UserMessage
from beeai_framework.context import RunContext, RunMiddlewareProtocol
from beeai_framework.emitter import EventMeta
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory


class StrictContentFilterMiddleware(RunMiddlewareProtocol):
    """
    Alternative implementation using abort controller for immediate termination.
    """

    def __init__(self, banned_words: list[str] | None = None) -> None:
        super().__init__()
        self.banned_words = [word.lower() for word in (banned_words or ["inappropriate"])]
        self._banned_pattern = re.compile("|".join(map(re.escape, self.banned_words)))
        self._cleanup_functions: list[Any] = []

    def bind(self, ctx: RunContext) -> None:
        # Clean up any existing event listeners
        while self._cleanup_functions:
            self._cleanup_functions.pop(0)()

        # Listen for agent start events to check content
        cleanup = ctx.emitter.on("start", lambda data, meta: self._check_and_abort(data, meta, ctx))
        self._cleanup_functions.append(cleanup)

    def _check_and_abort(self, data: RequirementAgentStartEvent, meta: EventMeta, ctx: RunContext) -> None:
        """Check content and abort execution if inappropriate."""
        if isinstance(data, RequirementAgentStartEvent):
            for message in data.state.memory.messages:
                if isinstance(message, UserMessage) and message.text and self._contains_banned_content(message.text):
                    print("🛑 StrictContentFilter: Aborting due to inappropriate content")
                    ctx.abort("Content policy violation")
                    return

    def _contains_banned_content(self, text: str) -> bool:
        """Check if text contains any banned words."""
        text_lower = text.lower()
        return bool(self._banned_pattern.search(text_lower))


async def main() -> None:
    """
    Example demonstrating enhanced content filtering capabilities.
    """

    agent = RequirementAgent(
        llm=ChatModel.from_name("ollama:granite4:micro"),
        memory=UnconstrainedMemory(),
        middlewares=[StrictContentFilterMiddleware(banned_words=["crap"])],
    )

    print("\n=== Testing Strict Content Filter (Abort Execution) ===")
    try:
        result = await agent.run("What a crap weather today.")
        print("Response:", result.last_message.text)
    except Exception as e:
        print(f"Execution aborted: {e}")

    print("\n=== Testing Clean Input (Should Work) ===")
    try:
        result = await agent.run("What is the capital of France?")
        print("Response:", result.last_message.text)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
