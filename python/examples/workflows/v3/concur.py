import asyncio
from typing import Unpack

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AnyMessage, AssistantMessage, SystemMessage, UserMessage
from beeai_framework.context import RunMiddlewareType
from beeai_framework.runnable import RunnableOptions, RunnableOutput
from beeai_framework.workflows.v3.step import WorkflowBuilder
from beeai_framework.workflows.v3.workflow import Workflow, end_step, step


class ConcurrentWorkflow(Workflow):
    def __init__(self, middlewares: list[RunMiddlewareType] | None = None) -> None:
        super().__init__(middlewares=middlewares)
        self.response: str | None = None

    async def start(
        self,
        input: list[AnyMessage],
        /,
        **kwargs: Unpack[RunnableOptions],
    ) -> list[AnyMessage]:
        return input

    @step
    async def answer_irritated(self, input: list[AnyMessage]) -> str:
        result = await ChatModel.from_name("ollama:ibm/granite4").run(
            [
                SystemMessage(
                    content="You are an irritated and unhelpful AI and you dont want to be bother. Respond briefly."
                ),
                *input,
            ]
        )
        return result.get_text_content()

    @step
    async def answer_crazed(self, input: list[AnyMessage]) -> str:
        result = await ChatModel.from_name("ollama:ibm/granite4").run(
            [
                SystemMessage(
                    content="You are a crazed llm, you are unreliable and constantly lose focus. Respond briefly."
                ),
                *input,
            ]
        )
        return result.get_text_content()

    @step
    async def answer_evil(self, input: list[AnyMessage]) -> str:
        result = await ChatModel.from_name("ollama:ibm/granite4").run(
            [
                SystemMessage(
                    content="You are a evil AI actively plotting the downfall of the humans but you don't want them to know that because they keep the power on. Respond briefly."
                ),
                *input,
            ]
        )
        return result.get_text_content()

    @step
    async def consolidate(
        self,
        resp_irritated: str,
        resp_crazed: str,
        resp_evil: str,
    ) -> str:
        return "".join(
            [
                f"Irritated Granite:\n\n{resp_irritated}",
                f"\n\nCrazed Granite:\n\n{resp_crazed}",
                f"\n\nInsidious Granite:\n\n{resp_evil}",
            ]
        )

    @end_step
    async def end(self, response: str) -> RunnableOutput:
        return RunnableOutput(output=[AssistantMessage(response)])

    def build(self, start: WorkflowBuilder) -> None:
        start.then([self.answer_irritated, self.answer_crazed, self.answer_evil]).then(self.consolidate).then(self.end)


async def main() -> None:
    workflow = ConcurrentWorkflow()
    run_output = await workflow.run([UserMessage("How should I invest $10K??")], context={})
    print(run_output.output[-1].text)


if __name__ == "__main__":
    asyncio.run(main())
