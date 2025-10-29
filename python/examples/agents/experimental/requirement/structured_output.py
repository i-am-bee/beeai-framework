import asyncio

from pydantic import BaseModel, Field

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.utils.models import to_list_model


async def main() -> None:
    agent = RequirementAgent(llm=ChatModel.from_name("ollama:granite4:micro"))

    class Character(BaseModel):
        first_name: str
        last_name: str
        age: int
        bio: str
        country: str

    Characters = to_list_model(Character, Field(min_length=5, max_length=5))  # noqa: N806

    response = await agent.run("Generate fictional characters", expected_output=Characters).middleware(
        GlobalTrajectoryMiddleware()
    )

    for index, character in response.output_structured:
        print("Index:", index)
        print("-> Full Name:", character.first_name, character.last_name)
        print("-> Age:", character.age)
        print("-> Country:", character.country)
        print("-> Bio: ", character.bio)


if __name__ == "__main__":
    asyncio.run(main())
