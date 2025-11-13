import asyncio

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import SystemMessage, UserMessage
from beeai_framework.context import RunMiddlewareType
from beeai_framework.workflows.v3.step import WorkflowStep
from beeai_framework.workflows.v3.workflow import Workflow, step


class ConcurrentWorkflow(Workflow):
    def __init__(self, middlewares: list[RunMiddlewareType] | None = None) -> None:
        super().__init__(middlewares=middlewares)
        self.responses: list[str] = []

    @step
    async def answer_persona_a(self) -> None:
        result = await ChatModel.from_name("ollama:ibm/granite4").run(
            [SystemMessage(content="You are an irritated and unhelpful AI. Respond accordingly."), *self.messages]
        )
        self.responses.append(result.get_text_content())

    @step
    async def answer_persona_b(self) -> None:
        result = await ChatModel.from_name("ollama:ibm/granite4").run(
            [
                SystemMessage(content="You are a somewhat deranged AI bent on global domination. Respond accordingly."),
                *self.messages,
            ]
        )
        self.responses.append(result.get_text_content())

    @step
    async def answer_persona_c(self) -> None:
        result = await ChatModel.from_name("ollama:ibm/granite4").run(
            [
                SystemMessage(
                    content="You are a evil AI but you want the humans to think you are not plotting against them. Respond accordingly."
                ),
                *self.messages,
            ]
        )
        self.responses.append(result.get_text_content())

    @step
    async def end(self) -> None:
        print(len(self.responses))
        for resp in self.responses:
            print("==========")
            print(resp)

    # Handles return values and parameters between steps
    # Fork and join
    # Return a runnable output

    def build(self, start: WorkflowStep) -> None:
        start.then([self.answer_persona_a, self.answer_persona_b, self.answer_persona_c]).then(self.end)


async def main() -> None:
    workflow = ConcurrentWorkflow()
    await workflow.run([UserMessage("How should I invest $10K??")], context={})


if __name__ == "__main__":
    asyncio.run(main())
