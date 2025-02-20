import asyncio

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AssistantMessage, SystemMessage, UserMessage
from beeai_framework.memory.summarize_memory import SummarizeMemory


async def main() -> None:
    try:
        # Initialize the LLM with parameters
        llm = ChatModel.from_name(
            "ollama:granite3.1-dense:8b",
            # ChatModelParameters(temperature=0\),
        )

        # Create summarize memory instance
        memory = SummarizeMemory(llm)

        # Add messages
        await memory.add_many(
            [
                SystemMessage("You are a guide through France."),
                UserMessage("What is the capital?"),
                AssistantMessage("Paris"),
                UserMessage("What language is spoken there?"),
            ]
        )

        # Print results
        print(f"Is Empty: {memory.is_empty()}")
        print(f"Message Count: {len(memory.messages)}")

        if memory.messages:
            print(f"Summary: {memory.messages[0].get_texts()[0].get('text')}")

    except Exception as e:
        print(f"An error occurred: {e!s}")
        import traceback

        print(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
