import asyncio

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.events import RequirementAgentFinalAnswerEvent
from beeai_framework.backend import ChatModel
from beeai_framework.emitter import EventMeta


async def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.3:8b")
    llm.parameters.stream = True

    agent = RequirementAgent(
        llm=llm,
        instructions="Try to always respond in one sentence.",
    )

    def handle_final_answer(data: RequirementAgentFinalAnswerEvent, meta: EventMeta) -> None:
        print(data.delta)

    await agent.run("Calculate 4+5").on("final_answer", handle_final_answer)


if __name__ == "__main__":
    asyncio.run(main())
