import asyncio

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.events import RequirementAgentFinalAnswerEvent
from beeai_framework.backend import ChatModel
from beeai_framework.emitter import EventMeta


async def main() -> None:
    llm = ChatModel.from_name("ollama:granite4:latest")
    llm.parameters.stream = True
    agent = RequirementAgent(
        llm=llm,
        instructions="Try to always respond in one sentence.",
    )

    def on_new_token(data: RequirementAgentFinalAnswerEvent, meta: EventMeta) -> None:
        print("Update", data.delta, len(data.delta))

    response = await agent.run("Calculate 4+5").on("final_answer", on_new_token)
    print(response.last_message.text)


if __name__ == "__main__":
    asyncio.run(main())
