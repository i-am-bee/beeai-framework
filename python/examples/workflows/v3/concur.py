import asyncio

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AssistantMessage, SystemMessage, UserMessage
from beeai_framework.context import RunMiddlewareType
from beeai_framework.runnable import RunnableOutput
from beeai_framework.workflows.v3.step import WorkflowStep
from beeai_framework.workflows.v3.workflow import Workflow, step


class ConcurrentWorkflow(Workflow):
    def __init__(self, middlewares: list[RunMiddlewareType] | None = None) -> None:
        super().__init__(middlewares=middlewares)
        self.responses: list[str] = []

    @step
    async def answer_persona_a(self) -> None:
        result = await ChatModel.from_name("ollama:ibm/granite4").run(
            [SystemMessage(content="You are an irritated and unhelpful AI. Respond accordingly."), *self.input]
        )
        self.responses.append(result.get_text_content())

    @step
    async def answer_persona_b(self) -> None:
        result = await ChatModel.from_name("ollama:ibm/granite4").run(
            [
                SystemMessage(content="You are a somewhat deranged AI bent on global domination. Respond accordingly."),
                *self.input,
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
                *self.input,
            ]
        )
        self.responses.append(result.get_text_content())

    def build(self, start: WorkflowStep) -> None:
        start.then([self.answer_persona_a, self.answer_persona_b, self.answer_persona_c])

    def finalize(self) -> RunnableOutput:
        return RunnableOutput(output=[AssistantMessage("\n".join(r for r in self.responses))])


async def main() -> None:
    workflow = ConcurrentWorkflow()
    run_output = await workflow.run([UserMessage("How should I invest $10K??")], context={})
    print(run_output.output[-1].text)


if __name__ == "__main__":
    asyncio.run(main())
