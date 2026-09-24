import asyncio
import json
import sys
import traceback

from pydantic import BaseModel, Field

from beeai_framework.backend import ChatModel, UserMessage
from beeai_framework.errors import FrameworkError


async def main() -> None:
    model = ChatModel.from_name("ollama:granite4:micro")

    class ProfileSchema(BaseModel):
        first_name: str = Field(..., min_length=1)
        last_name: str = Field(..., min_length=1)
        address: str
        age: int
        hobby: str

    response = await model.run(
        [UserMessage("Generate a profile of a citizen of Europe.")], response_format=ProfileSchema
    )
    assert isinstance(response.output_structured, ProfileSchema)
    print(json.dumps(response.output_structured.model_dump(), indent=4))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
