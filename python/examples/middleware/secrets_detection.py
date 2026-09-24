import asyncio
import sys
import traceback

try:
    # pyrefly: ignore [missing-import]
    from llm_guard.input_scanners import Secrets
except ImportError:
    print("The 'llm-guard' package is NOT installed. Run 'pip install -r examples/middleware/requirements.txt'.")

from typing import Literal, TypeAlias

from beeai_framework.agents import AgentOutput, BaseAgent
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import AnyMessage, AssistantMessage, UserMessage
from beeai_framework.context import RunContext, RunContextStartEvent, RunMiddlewareProtocol
from beeai_framework.emitter import CleanupFn, EmitterOptions, EventMeta
from beeai_framework.emitter.utils import create_internal_event_matcher
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory

RedactMode: TypeAlias = Literal["partial", "all", "hash"]


class SecretsDetectionMiddleware(RunMiddlewareProtocol):
    """
    Middleware that detects secrets, sanitizing (permissive) or
    blocking (enforcement) inputs containing secrets.
    """

    def __init__(
        self, redact_mode: RedactMode = "partial", permissive: bool = False, custom_response: str | None = None
    ) -> None:
        super().__init__()
        self.scanner = Secrets(redact_mode=redact_mode)
        self.permissive = permissive
        self.custom_response = (
            custom_response or "Sorry, I detected a secret in the input and cannot process your request."
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
            sanitized_data, contains_secret = self._scan(input_data)
            if contains_secret:
                if self.permissive:
                    print("🛡️ Content redacted: Secrets were detected and masked in the input")
                    if isinstance(input_data, str):
                        data.input["input"] = sanitized_data
                    else:
                        data.input["input"][-1] = sanitized_data
                else:
                    print("🚫 Content blocked: Secrets detected in the input")
                    custom_output = AgentOutput(
                        output=[AssistantMessage(self.custom_response)],
                        output_structured=None,
                    )

                    # Set the output on the event to prevent normal execution
                    data.output = custom_output

    def _scan(self, text: str | list[AnyMessage]) -> tuple[str, bool]:
        """Check if text contains a secret."""
        msg = text if isinstance(text, str) else text[0].text
        sanitized_data, is_valid, _ = self.scanner.scan(msg)
        return sanitized_data, not is_valid


async def main() -> None:
    """
    Example demonstrating a middleware for secrets detection and redaction.
    """

    agent = RequirementAgent(
        llm="ollama:granite4:micro",
        memory=UnconstrainedMemory(),
        middlewares=[SecretsDetectionMiddleware()],
    )

    encoded_str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRhIjp7fX0.bVBhvll6IaeR3aUdoOeyR8YZe2S2DfhGAxTGfd9enLw"

    print("=== Testing Secret Detection (enforcing mode) ===")
    try:
        result = await agent.run(encoded_str)
        print("Response:", result.last_message.text)
    except Exception as e:
        print(f"Error: {e}")

    agent = RequirementAgent(
        llm="ollama:granite4:micro",
        memory=UnconstrainedMemory(),
        middlewares=[SecretsDetectionMiddleware(permissive=True)],
    )

    print("=== Testing Secret Detection (masking) ===")
    try:
        result = await agent.run(encoded_str)
        print("Response:", result.last_message.text)
    except Exception as e:
        print(f"Error: {e}")

    print("=== Testing Clean Input ===")
    try:
        result = await agent.run("What is 2 + 2?")
        print("Response:", result.last_message.text)
    except Exception as e:
        print(f"Error: {e}")

    print("=== Testing Clean Input (list of messages) ===")
    try:
        result = await agent.run([UserMessage("What is 2 + 2?")])
        print("Response:", result.last_message.text)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
