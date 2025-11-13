import asyncio

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AssistantMessage, SystemMessage, UserMessage
from beeai_framework.context import RunMiddlewareType
from beeai_framework.workflows.v3.step import WorkflowStep
from beeai_framework.workflows.v3.workflow import Workflow, step


class PromptChainWorkflow(Workflow):

    def __init__(self, middlewares: list[RunMiddlewareType] | None = None) -> None:
        super().__init__(middlewares=middlewares)

        self.response: str | None = None
        self.improvements: str | None = None

    @step
    async def answer(self) -> None:
        result = await ChatModel.from_name("ollama:ibm/granite4").run(self.messages)
        self.response = result.get_text_content()

    @step
    async def review(self) -> None:
        result = await ChatModel.from_name("ollama:ibm/granite4").run(
            [
                *self.messages,
                AssistantMessage(self.response or ""),
                UserMessage(
                    "Read the last agent response and produce a short (2 to 3 items max.) list of suggested improvements."
                ),
            ],
        )
        self.improvements = result.get_text_content()

    @step
    async def revise_answer(self) -> None:
        result = await ChatModel.from_name("ollama:ibm/granite4").run(
            [
                SystemMessage(self.improvements or ""),
                *self.messages,
            ]
        )
        self.messages.append(
            AssistantMessage(result.get_text_content()),
        )

    @step
    async def end(self) -> None:
        print(self.messages[-1].text)

    def build(self, start: WorkflowStep) -> None:
        """Build out the workflow"""
        start.then(self.answer).then(self.review).then(self.revise_answer).then(self.end)


async def main() -> None:

    workflow = PromptChainWorkflow()
    await workflow.run([UserMessage("How is a black dwarf formed?")])


if __name__ == "__main__":
    asyncio.run(main())
