import asyncio
import sys
import traceback

from litellm.exceptions import APIConnectionError, NotFoundError, RateLimitError, ServiceUnavailableError

from beeai_framework.adapters.ollama import OllamaChatModel
from beeai_framework.backend import FallbackChatModel, UserMessage
from beeai_framework.errors import FrameworkError


def should_fallback(error: Exception) -> bool:
    # The error is wrapped by the framework, the original exception raised by the provider is its cause.
    cause = error.get_cause() if isinstance(error, FrameworkError) else error

    # Fall back only when the model is unavailable. Other errors (e.g., an invalid request)
    # would fail for the next model as well, so they are raised immediately.
    return isinstance(cause, NotFoundError | RateLimitError | ServiceUnavailableError | APIConnectionError)


async def main() -> None:
    llm = FallbackChatModel(
        [OllamaChatModel("non-existing-model"), OllamaChatModel("granite4:micro")],
        should_fallback=should_fallback,
    )

    response = await llm.run([UserMessage("What is the capital of France?")])
    print("LLM 🤖 : ", response.get_text_content())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
