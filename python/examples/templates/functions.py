import sys
import traceback
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from beeai_framework.errors import FrameworkError
from beeai_framework.template import PromptTemplate, PromptTemplateInput


def main() -> None:
    class AuthorMessage(BaseModel):
        text: str
        author: str | None = None
        created_at: str | None = None

    def format_meta(data: dict[str, Any]) -> str:
        if data.get("author") is None and data.get("created_at") is None:
            return ""

        author = data.get("author") or "anonymous"
        created_at = data.get("created_at") or datetime.now(UTC).strftime("%A, %B %d, %Y at %I:%M:%S %p")

        return f"\nThis message was created at {created_at} by {author}."

    template: PromptTemplate[AuthorMessage] = PromptTemplate(
        PromptTemplateInput(
            schema=AuthorMessage,
            functions={
                "format_meta": lambda data: format_meta(data),
            },
            template="""Message: {{text}}{{format_meta}}""",
        )
    )

    # Message: Hello from 2024!
    # This message was created at 2024-01-01T00:00:00+00:00 by John.
    message = template.render(
        text="Hello from 2024!", author="John", created_at=datetime(2024, 1, 1, tzinfo=UTC).isoformat()
    )
    print(message)

    # Message: Hello from the present!
    message = template.render(text="Hello from the present!")
    print(message)


if __name__ == "__main__":
    try:
        main()
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
