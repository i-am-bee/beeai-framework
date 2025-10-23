import asyncio
import re
import sys
import traceback
from typing import Any

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.types import RequirementAgentOutput, RequirementAgentRunState
from beeai_framework.backend import AssistantMessage, ChatModel
from beeai_framework.context import RunContext, RunContextStartEvent, RunMiddlewareProtocol
from beeai_framework.emitter import EventMeta
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory


class ContentFilterMiddleware(RunMiddlewareProtocol):
    """
    Content filter middleware that can stop agent execution
    and return custom messages when inappropriate content is detected.
    """

    def __init__(self, banned_words: list[str] | None = None, custom_response: str | None = None) -> None:
        super().__init__()
        self.banned_words = [word.lower() for word in (banned_words or ["inappropriate"])]
        self._banned_pattern = re.compile("|".join(map(re.escape, self.banned_words)))
        self.custom_response = custom_response or "I cannot process that request due to content policy restrictions."
        self._cleanup_functions: list[Any] = []

    def bind(self, ctx: RunContext) -> None:
        # Clean up any existing event listeners
        while self._cleanup_functions:
            self._cleanup_functions.pop(0)()

        # Listen for run context start events to intercept before agent execution
        cleanup = ctx.emitter.on("run.agent.requirement.start", self._on_run_start)
        self._cleanup_functions.append(cleanup)

    def _on_run_start(self, data: RunContextStartEvent, _: EventMeta) -> None:
        """Intercept run start events to filter input before agent execution."""
        run_params = data.input
        if "input" in run_params:
            input_data = run_params["input"]

            # Check if input contains banned content
            if self._contains_banned_content(input_data):
                print("🚫 Content blocked: Input contains inappropriate content")

                # Create a custom output to short-circuit execution
                custom_output = RequirementAgentOutput(
                    output=[AssistantMessage(self.custom_response)],
                    output_structured=None,
                    state=RequirementAgentRunState(result="", memory=UnconstrainedMemory(), iteration=0),
                )

                # Set the output on the event to prevent normal execution
                data.output = custom_output

    def _contains_banned_content(self, text: str) -> bool:
        """Check if text contains any banned words."""
        text_lower = text.lower()
        return bool(self._banned_pattern.search(text_lower))


async def main() -> None:
    """
    Example demonstrating enhanced content filtering capabilities.
    """

    agent = RequirementAgent(
        llm=ChatModel.from_name("ollama:granite3.3:8b"),
        memory=UnconstrainedMemory(),
        middlewares=[
            ContentFilterMiddleware(
                banned_words=["crap", "inappropriate"],
                custom_response="I understand you're frustrated, but I can only help with appropriate requests. How can I assist you today?",
            )
        ],
    )

    print("=== Testing Content Filter ===")
    try:
        result = await agent.run("This crap weather is making me sad. What is 2 + 2?")
        print("Response:", result.last_message.text)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
