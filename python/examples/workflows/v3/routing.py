import asyncio
import json
from typing import Any

from pydantic import BaseModel

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AnyMessage, UserMessage
from beeai_framework.workflows.v3.workflow import Workflow, create_step


async def main() -> None:

    class ToolsRequired(BaseModel):
        requires_web_search: bool
        reasoning: str

    async def start(messages: list[AnyMessage], context: dict[str, Any]) -> None:
        print("start workflow")

    async def check_context(messages: list[AnyMessage], context: dict[str, Any]) -> None:
        print("check_context")
        result = await ChatModel.from_name("ollama:ibm/granite4").run(
            [*messages, UserMessage("To answer the user request, do you need access to the web search tool?")],
            response_format=ToolsRequired,
        )
        assert result.output_structured is not None
        context["requires_web_search"] = ToolsRequired(**result.output_structured.model_dump()).requires_web_search
        context["reasoning"] = ToolsRequired(**result.output_structured.model_dump()).reasoning

    async def answer_web_search(messages: list[AnyMessage], context: dict[str, Any]) -> None:
        print("answer_web_search")
        result = await ChatModel.from_name("ollama:ibm/granite4").run(messages)
        context["response"] = result.get_text_content()

    async def answer(messages: list[AnyMessage], context: dict[str, Any]) -> None:
        print("answer")
        result = await ChatModel.from_name("ollama:ibm/granite4").run(messages)
        context["response"] = result.get_text_content()

    async def end(messages: list[AnyMessage], context: dict[str, Any]) -> None:
        print("end")
        print(json.dumps(context, indent=4))

    # Define steps
    start_step = create_step(start)
    check_context_step = create_step(check_context)
    answer_web_search_step = create_step(answer_web_search)
    answer_step = create_step(answer)
    end_step = create_step(end)

    workflow = Workflow()
    workflow.start(start_step)

    start_step.then(check_context_step).branch(
        next_steps={True: answer_web_search_step, False: answer_step},
        branch_fn=lambda messages, context: context["requires_web_search"],
    )
    answer_web_search_step.then(end_step)
    answer_step.then(end_step)

    await workflow.run([UserMessage("What is the current rivian stock price?")], context={})


if __name__ == "__main__":
    asyncio.run(main())
