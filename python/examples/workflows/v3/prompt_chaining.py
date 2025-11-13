import asyncio

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AssistantMessage, SystemMessage, UserMessage
from beeai_framework.context import RunMiddlewareType
from beeai_framework.runnable import RunnableOutput
from beeai_framework.workflows.v3.step import WorkflowStep
from beeai_framework.workflows.v3.workflow import Workflow, step


class PromptChainWorkflow(Workflow):
    def __init__(self, middlewares: list[RunMiddlewareType] | None = None) -> None:
        super().__init__(middlewares=middlewares)
        self.revised_response: str | None = None

    @step
    async def answer(self) -> str:
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
    async def revise_answer(self, suggested_improvements: str) -> None:
        result = await ChatModel.from_name("ollama:ibm/granite4").run(
            [
                SystemMessage(suggested_improvements),
                *self.input,
            ]
        )
        self.revised_response = result.get_text_content()

    def build(self, start: WorkflowStep) -> None:
        """
        Build out the workflow from the start node.
        """
        start.then(self.answer).then(self.review).then(self.revise_answer)

    def finalize(self) -> RunnableOutput:
        """
        Produce a runnable output from the internal state of the flow
        """
        return RunnableOutput(output=[AssistantMessage(self.revised_response or "")])


async def main() -> None:
    workflow = PromptChainWorkflow()
    run_output = await workflow.run([UserMessage("How is a black dwarf formed?")])
    print(run_output.output[-1].text)


if __name__ == "__main__":
    asyncio.run(main())
