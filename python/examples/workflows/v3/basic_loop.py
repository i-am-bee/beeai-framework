import asyncio
from typing import Literal, Unpack

from pydantic import BaseModel

from beeai_framework.adapters.openai.backend.chat import OpenAIChatModel
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AnyMessage, AssistantMessage, SystemMessage, UserMessage
from beeai_framework.backend.types import ChatModelParameters
from beeai_framework.context import RunMiddlewareType
from beeai_framework.runnable import RunnableOptions, RunnableOutput
from beeai_framework.workflows.v3.step import WorkflowBuilder
from beeai_framework.workflows.v3.workflow import Workflow, end_step, step


class EvalOptimizeWorkflow(Workflow):
    class ResponseEval(BaseModel):
        evaluation: Literal["pass", "fail"]
        feedback: str

    def __init__(
        self,
        middlewares: list[RunMiddlewareType] | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(middlewares=middlewares)

        assert base_url
        assert api_key

        self.base_llm = ChatModel.from_name("ollama:ibm/granite4")
        self.eval_llm = OpenAIChatModel(
            model_id="openai/gpt-oss-20b",
            api_key=api_key,
            base_url=base_url,
            parameters=ChatModelParameters(max_tokens=1024, temperature=0.0),
            settings={"extra_headers": {"RITS_API_KEY": api_key}},
        )
        self.response_eval: EvalOptimizeWorkflow.ResponseEval | None = None

    async def start(
        self,
        input: list[AnyMessage],
        /,
        **kwargs: Unpack[RunnableOptions],
    ) -> None:
        self.input = input
        self.attempts = 3

    @step
    async def answer(self) -> None:
        messages = [*self.input]
        if self.response_eval and self.response_eval.evaluation == "fail":
            messages.insert(
                0,
                SystemMessage(
                    f"Use the following feedback when formulating your response.\n\nFeedback:\n{self.response_eval.feedback}"
                ),
            )

        result = await self.base_llm.run(messages)
        self.response = result.get_text_content()

    @step
    async def eval(self) -> None:
        result = await self.eval_llm.run(
            [
                SystemMessage(content="Evaluate the correctness of the assistant's response."),
                *self.input,
                AssistantMessage(self.response),
            ],
            response_format=EvalOptimizeWorkflow.ResponseEval,
        )

        assert result.output_structured is not None
        self.response_eval = EvalOptimizeWorkflow.ResponseEval(**result.output_structured.model_dump())
        self.attempts -= 1

    async def try_again(self) -> bool:
        if self.response_eval is None or self.attempts == 0:
            return False

        return self.response_eval.evaluation == "fail"

    @end_step
    async def end(self) -> RunnableOutput:
        return RunnableOutput(output=[AssistantMessage(self.response)])

    def build(self, start: WorkflowBuilder) -> None:
        start.then(self.answer).then(self.eval).branch(
            branch_fn=self.try_again, steps={True: self.answer, False: self.end}
        )


async def main() -> None:
    workflow = EvalOptimizeWorkflow(
        base_url="XXXX",
        api_key="XXXX",
    )
    messages: list[AnyMessage] = [UserMessage("How many 'r' is strawberry?")]
    run_output = await workflow.run(
        messages,
        context={},
    )
    messages.extend(run_output.output)
    print("\n".join(m.text for m in messages))


if __name__ == "__main__":
    asyncio.run(main())
