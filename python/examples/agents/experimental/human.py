import asyncio
import sys
import traceback

from beeai_framework import ReActAgent, TokenMemory
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.agents import AgentExecutionConfig
from beeai_framework.errors import FrameworkError
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from examples.helpers.io import ConsoleReader
from examples.tools.experimental.human import HumanTool, HumanToolInput


async def main() -> None:
    # Initialize LLM (test against llama as requested)
    llm = OllamaChatModel("llama3.1")

    # Create the console reader once, share it with HumanTool
    reader = ConsoleReader()

    # Initialize ReActAgent with shared reader for HumanTool
    agent = ReActAgent(
        llm=llm,
        memory=TokenMemory(llm),
        tools=[
            OpenMeteoTool(),
            HumanTool(
                HumanToolInput(
                    reader=reader,
                )
            ),
        ],
    )

    # Main loop
    for prompt in reader:
        # Run the agent and observe events
        response = (
            await agent.run(
                prompt=prompt,
                execution=AgentExecutionConfig(max_retries_per_step=3, total_max_retries=10, max_iterations=20),
            )
            .on(
                "update",
                lambda data, event: reader.write(f"Agent({data.update.key}) 🤖 : ", data.update.parsed_value),
            )
            .on(
                "error",  # Log errors
                lambda data, event: reader.write("Agent 🤖 : ", FrameworkError.ensure(data.error).explain()),
            )
            .on(
                "retry",  # Retry notifications
                lambda data, event: reader.write("Agent 🤖 : ", "Retrying the action..."),
            )
        )

        # Print the final response
        reader.write("Agent 🤖 : ", response.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
