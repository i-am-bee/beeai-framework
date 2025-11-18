import asyncio
from typing import Unpack

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AnyMessage, AssistantMessage, SystemMessage, UserMessage
from beeai_framework.context import RunMiddlewareType
from beeai_framework.runnable import RunnableOptions, RunnableOutput
from beeai_framework.workflows.v3.step import WorkflowBuilder
from beeai_framework.workflows.v3.workflow import Workflow, end_step, step


class PromptChainWorkflow(Workflow):
    def __init__(self, middlewares: list[RunMiddlewareType] | None = None) -> None:
        super().__init__(middlewares=middlewares)
        self.revised_response: str | None = None

    async def start(
        self,
        input: list[AnyMessage],
        /,
        **kwargs: Unpack[RunnableOptions],
    ) -> None:
        self.input = input

    @step
    async def answer(self, input: list[AnyMessage]) -> str:
        result = await ChatModel.from_name("ollama:ibm/granite4").run(self.input)
        self.response = result.get_text_content()
        return result.get_text_content()

    @step
    async def review(self, response: str) -> str:
        result = await ChatModel.from_name("ollama:ibm/granite4").run(
            [
                *self.input,
                AssistantMessage(response),
                UserMessage(
                    "Read the last agent response and produce a short (2 to 3 items max.) list of suggested improvements."
                ),
            ],
        )
        return result.get_text_content()

    @step
    async def revise_answer(self, suggested_improvements: str) -> str:
        result = await ChatModel.from_name("ollama:ibm/granite4").run(
            [
                SystemMessage(suggested_improvements),
                *self.input,
            ]
        )
        return result.get_text_content()

    @end_step
    async def end(self, response: str) -> RunnableOutput:
        return RunnableOutput(output=[AssistantMessage(response)])

    def build(self, start: WorkflowBuilder) -> None:
        """
        Build out the workflow.
        """
        start.then(self.answer).then(self.review).then(self.revise_answer).then(self.end)


async def main() -> None:
    workflow = PromptChainWorkflow()
    run_output = await workflow.run([UserMessage("How is a black dwarf formed?")])
    print(run_output.output[-1].text)


if __name__ == "__main__":
    asyncio.run(main())
