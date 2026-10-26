import asyncio
import sys
import traceback

from litellm.exceptions import NotFoundError

from beeai_framework.adapters.ollama import OllamaChatModel
from beeai_framework.backend import FallbackChatModel, UserMessage
from beeai_framework.errors import FrameworkError


def should_fallback(error: FrameworkError) -> bool:
    # Keep the default behavior (rate limits, server errors, timeouts, and connection errors)
    # and also fall back when the model is not available on the provider.
    # The original exception raised by the provider is the cause of the framework error.
    return FallbackChatModel.default_should_fallback(error) or isinstance(error.get_cause(), NotFoundError)


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
