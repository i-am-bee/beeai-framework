import asyncio
from typing import Any

from pydantic import BaseModel, Field

from beeai_framework.tools.tool import Tool


class RiddleToolInput(BaseModel):
    riddle_number: int = Field(description="Index of riddle to retrieve.")


class RiddleTool(Tool[RiddleToolInput]):
    name = "Riddle"
    description = "It selects a riddle to test your knowledge."
    input_schema = RiddleToolInput

    data = (
        "What has hands but can't clap?",
        "What has a face and two hands but no arms or legs?",
        "What gets wetter the more it dries?",
        "What has to be broken before you can use it?",
        "What has a head, a tail, but no body?",
        "The more you take, the more you leave behind. What am I?",
        "What goes up but never comes down?",
    )

    def _run(self, input: RiddleToolInput, _: Any | None = None) -> None:
        index = input.riddle_number % (len(self.data))
        riddle = self.data[index]
        return riddle


async def main() -> None:
    tool = RiddleTool()
    input = RiddleToolInput(riddle_number=1)
    result = tool.run(input)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
