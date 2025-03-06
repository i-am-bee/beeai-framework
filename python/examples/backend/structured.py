import asyncio
import json
import sys
import traceback

from pydantic import BaseModel, Field

from beeai_framework import UserMessage
from beeai_framework.backend.chat import ChatModel
from beeai_framework.errors import FrameworkError


async def main() -> None:
    model = ChatModel.from_name("ollama:llama3.1")

    class ProfileSchema(BaseModel):
        first_name: str = Field(..., min_length=1)
        last_name: str = Field(..., min_length=1)
        address: str
        age: int = Field(..., min_length=1)
        hobby: str

    class ErrorSchema(BaseModel):
        error: str

    class SchemUnion(ProfileSchema, ErrorSchema):
        pass

    response = await model.create_structure(
        schema=SchemUnion,
        messages=[UserMessage("Generate a profile of a citizen of Europe.")],
    )

    print(
        json.dumps(
            response.object.model_dump() if isinstance(response.object, BaseModel) else response.object, indent=4
        )
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
