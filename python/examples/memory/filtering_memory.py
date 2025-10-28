import asyncio
import re
import sys
import traceback

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel, UserMessage
from beeai_framework.backend.message import AnyMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory


class FilteringMemory(UnconstrainedMemory):
    """
    A memory wrapper that filters out banned words from user messages.
    """

    def __init__(self, banned_words: list[str] | str | None = None) -> None:
        super().__init__()
        if banned_words is None:
            banned_words = ["crap"]
        elif isinstance(banned_words, str):
            banned_words = [banned_words]
        self.banned_words = [word.lower() for word in banned_words]
        self.baneed_patterns = [re.compile(re.escape(w), re.IGNORECASE) for w in self.banned_words]

    async def add(self, message: AnyMessage, index: int | None = None) -> None:
        """Override add to filter user messages."""
        index = len(self._messages) if index is None else max(0, min(index, len(self._messages)))
        if isinstance(message, UserMessage) and message.text:
            original_text = message.text
            filtered_text = self._filter_text(original_text)

            if filtered_text != original_text:
                print("🛡️ ContentFilterMemory: Filtered banned words from input")
                print(f"   Original: '{original_text}'")
                print(f"   Filtered: '{filtered_text}'")

                # Create a new message with filtered content
                filtered_message = UserMessage(content=filtered_text, meta=message.meta)
                return await super().add(filtered_message, index)

        return await super().add(message, index)

    def _filter_text(self, text: str) -> str:
        """Filter out banned words from text (case-insensitive)."""
        filtered_text = text
        # Use regex to replace each banned word case-insensitively
        for pattern in self.baneed_patterns:
            filtered_text = pattern.sub("[FILTERED]", filtered_text)
        return filtered_text


async def main() -> None:
    """
    Example demonstrating the FilteringMemory with RequirementAgent.
    Tests with both a clean input and an input containing banned words.
    """
    chat_model = ChatModel.from_name("ollama:granite4:micro")

    # Create agent with filtering memory
    agent = RequirementAgent(
        llm=chat_model,
        memory=FilteringMemory(["crap", "damn", "heck"]),  # Use filtering memory with multiple banned words
    )

    print("=== Testing with clean input ===")
    result1 = await agent.run("What is the capital of France?")
    print("Answer:", result1.last_message.text)
    print()

    print("=== Testing with input containing 'crap' ===")
    result2 = await agent.run("This crap weather is making me sad. What is 2 + 2?")
    print("Answer:", result2.last_message.text)
    print()

    print("=== Testing with input containing 'CRAP' (uppercase) ===")
    result3 = await agent.run("CRAP! I forgot my keys. Can you help me remember the capital of Spain?")
    print("Answer:", result3.last_message.text)
    print()

    print("=== Testing with input container 'heck' ===")
    result4 = await agent.run("What the heck is going on, can you help me remember the capital of Brazil?")
    print("Answer:", result4.last_message.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
