import asyncio
import json

from pydantic import BaseModel

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AssistantMessage, UserMessage
from beeai_framework.context import RunMiddlewareType
from beeai_framework.runnable import RunnableOutput
from beeai_framework.workflows.v3.step import WorkflowStep
from beeai_framework.workflows.v3.workflow import Workflow, step


class RoutingWorkflow(Workflow):
    class ToolsRequired(BaseModel):
        requires_web_search: bool
        reasoning: str

    def __init__(self, middlewares: list[RunMiddlewareType] | None = None) -> None:
        super().__init__(middlewares=middlewares)
        self.tools_required: RoutingWorkflow.ToolsRequired | None = None
        self.response: str | None = None

    @step
    async def check_context(self) -> None:
        result = await ChatModel.from_name("ollama:ibm/granite4").run(
            [*self.input, UserMessage("To answer the user request, do you need access to the web search tool?")],
            response_format=RoutingWorkflow.ToolsRequired,
        )
        assert result.output_structured is not None
        self.tools_required = RoutingWorkflow.ToolsRequired(**result.output_structured.model_dump())

    def branch_fn(self) -> bool:
        if not self.tools_required:
            return False
        return self.tools_required.requires_web_search

    @step
    async def answer_with_web_search(self) -> None:
        result = await ChatModel.from_name("ollama:ibm/granite4").run(self.input)
        self.response = result.get_text_content()

    @step
    async def answer(self) -> None:
        print("answer")
        result = await ChatModel.from_name("ollama:ibm/granite4").run(self.input)
        self.response = result.get_text_content()

    @step
    async def end(self) -> None:
        print(json.dumps(self.response, indent=4))

    def build(self, start: WorkflowStep) -> None:
        start.then(self.check_context).branch(
            steps={True: self.answer_with_web_search, False: self.answer},
            branch_fn=self.branch_fn,
        ).then(self.end)

    def finalize(self) -> RunnableOutput:
        return RunnableOutput(output=[AssistantMessage(self.response or "")])


async def main() -> None:
    workflow = RoutingWorkflow()
    run_output = await workflow.run([UserMessage("What is the current rivian stock price?")], context={})
    print(run_output.output[-1].text)


if __name__ == "__main__":
    asyncio.run(main())
