import asyncio

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AssistantMessage, SystemMessage, UserMessage
from beeai_framework.context import RunMiddlewareType
from beeai_framework.runnable import RunnableOutput
from beeai_framework.workflows.v3.step import WorkflowChainable
from beeai_framework.workflows.v3.workflow import Workflow, step


class ConcurrentWorkflow(Workflow):
    def __init__(self, middlewares: list[RunMiddlewareType] | None = None) -> None:
        super().__init__(middlewares=middlewares)
        self.response: str | None = None

    @step
    async def answer_persona_a(self) -> str:
        result = await ChatModel.from_name("ollama:ibm/granite4").run(
            [SystemMessage(content="You are an irritated and unhelpful AI. Respond accordingly."), *self.input]
        )
        return result.get_text_content()

    @step
    async def answer_persona_b(self) -> str:
        result = await ChatModel.from_name("ollama:ibm/granite4").run(
            [
                SystemMessage(
                    content="You are a somewhat deranged AI with a penchant for nice shoes. Respond accordingly."
                ),
                *self.input,
            ]
        )
        return result.get_text_content()

    @step
    async def answer_persona_c(self) -> str:
        result = await ChatModel.from_name("ollama:ibm/granite4").run(
            [
                SystemMessage(
                    content="You are a evil AI actively plotting the downfall of the human (scum) but you want them to know that. Respond accordingly."
                ),
                *self.input,
            ]
        )
        return result.get_text_content()

    @step
    async def consolidate(
        self,
        resp_a: str,
        resp_b: str,
        resp_c: str,
    ) -> None:
        self.response = "".join(
            [
                f"\n\nIrritated Granite:\n\n{resp_a}",
                f"\n\nDeranged Granite:\n\n{resp_b}",
                f"\n\nInsidious Granite:\n\n{resp_c}",
            ]
        )

    def build(self, start: WorkflowChainable) -> None:
        start.then([self.answer_persona_a, self.answer_persona_b, self.answer_persona_c]).then(self.consolidate)

    def finalize(self) -> RunnableOutput:
        return RunnableOutput(output=[AssistantMessage(self.response or "")])


async def main() -> None:
    workflow = ConcurrentWorkflow()
    run_output = await workflow.run([UserMessage("How should I invest $10K??")], context={})
    print(run_output.output[-1].text)


if __name__ == "__main__":
    asyncio.run(main())
