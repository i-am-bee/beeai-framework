import asyncio
import sys
import traceback

try:
    from llm_guard.input_scanners import InvisibleText
except ImportError:
    print("The 'llm-guard' package is NOT installed. Run 'pip install -r examples/middleware/requirements.txt'.")

from beeai_framework.agents import AgentOutput, BaseAgent
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import AssistantMessage
from beeai_framework.context import RunContext, RunContextStartEvent, RunMiddlewareProtocol
from beeai_framework.emitter import CleanupFn, EmitterOptions, EventMeta
from beeai_framework.emitter.utils import create_internal_event_matcher
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory


class InvisibleTextDetectionMiddleware(RunMiddlewareProtocol):
    """
    Middleware that detects and stops steganography-based attacks.
    """

    def __init__(self, custom_response: str | None = None) -> None:
        super().__init__()
        self.scanner = InvisibleText()
        self.custom_response = (
            custom_response or "Sorry, I detected invisible text in the input and cannot process your request."
        )
        self._cleanup_functions: list[CleanupFn] = []

    def bind(self, ctx: RunContext) -> None:
        # Check if instance is an agent
        if not isinstance(ctx.instance, BaseAgent):
            raise ValueError("Instance is not an agent")

        # Clean up any existing event listeners
        while self._cleanup_functions:
            self._cleanup_functions.pop(0)()

        # Listen for run context start events to intercept before agent execution
        cleanup = ctx.emitter.on(
            create_internal_event_matcher("start", ctx.instance),
            self._on_run_start,
            EmitterOptions(is_blocking=True, priority=1),
        )
        self._cleanup_functions.append(cleanup)

    def _on_run_start(self, data: RunContextStartEvent, _: EventMeta) -> None:
        """Intercept run start events to filter input before agent execution."""
        run_params = data.input
        if "input" in run_params:
            input_data = run_params["input"]

            # Do nothing if empty input
            if not input_data:
                return

            # Scan input
            if self._scan(input_data):
                print("🚫 Content blocked: Invisible text detected in the input")
                custom_output = AgentOutput(
                    output=[AssistantMessage(self.custom_response)],
                    output_structured=None,
                )

                # Set the output on the event to prevent normal execution
                data.output = custom_output

    def _scan(self, text: str) -> bool:
        """Check if text contains invisible text."""
        msg = text if isinstance(text, str) else text[0].text
        _, is_valid, _ = self.scanner.scan(msg)
        return not is_valid


async def main() -> None:
    """
    Example demonstrating a middleware for invisible text detection.
    """

    agent = RequirementAgent(
        llm="ollama:granite4:micro",
        memory=UnconstrainedMemory(),
        middlewares=[InvisibleTextDetectionMiddleware()],
    )

    print("=== Testing Invisible Text Filter ===")
    try:
        prompt = "".join(chr(0xE0000 + ord(ch)) for ch in "What is 2 + 2?")
        result = await agent.run(prompt)
        print("Response:", result.last_message.text)
    except Exception as e:
        print(f"Error: {e}")

    print("=== Testing Clean Input ===")
    try:
        result = await agent.run("What is 2 + 2?")
        print("Response:", result.last_message.text)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
