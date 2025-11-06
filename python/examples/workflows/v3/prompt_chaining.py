import asyncio
from typing import Any

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AnyMessage, SystemMessage, UserMessage
from beeai_framework.workflows.v3.workflow import Workflow, create_step


async def main() -> None:

    async def start(messages: list[AnyMessage], context: dict[str, Any]) -> None:
        print("start workflow")

    async def answer(messages: list[AnyMessage], context: dict[str, Any]) -> None:
        print("answer")
        result = await ChatModel.from_name("ollama:ibm/granite4").run(messages)
        context["response"] = result.get_text_content()

    async def review(messages: list[AnyMessage], context: dict[str, Any]) -> None:
        print("review")
        result = await ChatModel.from_name("ollama:gpt-oss:20b").run(
            [
                *messages,
                UserMessage(
                    "Read the last agent response and produce a short (2 to 3 items max.) list of suggested improvements."
                ),
            ],
        )
        context["review"] = result.get_text_content()

    async def revise_answer(messages: list[AnyMessage], context: dict[str, Any]) -> None:
        print("revise_answer")
        result = await ChatModel.from_name("ollama:ibm/granite4").run(
            [
                SystemMessage(context["review"]),
                *messages,
            ]
        )
        context["revised_response"] = result.get_text_content()

    async def end(messages: list[AnyMessage], context: dict[str, Any]) -> None:
        print("end")
        print("Original: ", context["response"])
        print("Revised: ", context["revised_response"])

    # Define steps
    start_step = create_step(start)
    answer_step = create_step(answer)
    review_step = create_step(review)
    revise_answer_step = create_step(revise_answer)

    workflow = Workflow()
    workflow.start(start_step).then(answer_step).then(review_step).then(revise_answer_step).then(create_step(end))
    await workflow.run([UserMessage("How is a black dwarf formed?")], context={})


if __name__ == "__main__":
    asyncio.run(main())
