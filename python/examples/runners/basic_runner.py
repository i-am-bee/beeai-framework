import asyncio

from beeai_framework.agents.runners.base import (
    BeeRunnerToolInput,
    BeeRunnerToolResult,
    RunnerIteration,
)
from beeai_framework.agents.runners.default.prompts import (
    AssistantPromptTemplate,
)
from beeai_framework.agents.runners.default.runner import DefaultRunner
from beeai_framework.agents.types import (
    BeeAgentExecutionConfig,
    BeeInput,
    BeeIterationResult,
    BeeMeta,
    BeeRunInput,
    BeeRunOptions,
)
from beeai_framework.backend.message import AssistantMessage
from beeai_framework.llms.llm import LLM
from beeai_framework.memory import TokenMemory
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool


# Main async function for testing
async def main() -> None:
    llm = LLM(
        model="llama3.1",  # Use llama3.1 for better performance
        parameters={
            "temperature": 0,
            "repeat_penalty": 1.0,
            "num_predict": 2048,
        },
    )

    input = BeeInput(
        llm=llm,
        tools=[OpenMeteoTool()],
        memory=TokenMemory(llm),
        execution=BeeAgentExecutionConfig(max_iterations=10, max_retries_per_step=3, total_max_retries=10),
    )

    meta = BeeMeta(iteration=0)

    runner = DefaultRunner(
        input=input, options=BeeRunOptions(execution=input.execution, signal=None), run=None
    )  # TODO Figure out run

    await runner.init(BeeRunInput("What is the current weather in White Plains?"))

    final_answer: str | None = None

    while final_answer is None:
        iteration: RunnerIteration = await runner.create_iteration()

        # Run tool if the iteration state includes tool call
        if iteration.state.tool_name and iteration.state.tool_input:
            tool_result: BeeRunnerToolResult = await runner.tool(
                input=BeeRunnerToolInput(
                    state=BeeIterationResult(
                        tool_name=iteration.state.tool_name, tool_input=iteration.state.tool_input
                    ),
                    emitter=None,
                    meta=meta,
                    signal=None,
                )
            )

            iteration.state.tool_output = tool_result.output.get_text_content()
            assistant_prompt = AssistantPromptTemplate.render(iteration.state.to_template())
            await runner.memory.add(AssistantMessage(content=assistant_prompt))

        elif iteration.state.final_answer:
            assistant_prompt = AssistantPromptTemplate.render(iteration.state.to_template())
            final_answer = iteration.state.final_answer

    print(final_answer)


if __name__ == "__main__":
    asyncio.run(main())  # Runs the main coroutine
