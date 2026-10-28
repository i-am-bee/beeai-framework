import asyncio
import sys
import traceback

from beeai_framework.adapters.ollama import OllamaChatModel
from beeai_framework.backend import ChatModelFallbackEvent, FallbackChatModel, UserMessage
from beeai_framework.emitter import EventMeta
from beeai_framework.errors import FrameworkError


async def main() -> None:
    llm = FallbackChatModel(
        [
            # Nothing listens on this port, so the request falls back to the next model.
            OllamaChatModel("llama3.1", base_url="http://localhost:1"),
            OllamaChatModel("granite4:micro"),
        ]
    )

    def on_fallback(data: ChatModelFallbackEvent, event: EventMeta) -> None:
        print(f"Model '{data.failed_model.model_id}' failed ({data.error.get_cause()!r}), falling back.")

    response = await llm.run([UserMessage("What is the capital of France?")]).on("fallback", on_fallback)
    print("LLM 🤖 : ", response.get_text_content())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
