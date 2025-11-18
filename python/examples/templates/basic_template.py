import sys
import traceback

from pydantic import BaseModel

from beeai_framework.errors import FrameworkError
from beeai_framework.template import PromptTemplate, PromptTemplateInput


def main() -> None:
    class UserMessage(BaseModel):
        label: str
        input: str

    template: PromptTemplate[UserMessage] = PromptTemplate(
        PromptTemplateInput(
            schema=UserMessage,
            template="""{{label}}: {{input}}""",
        )
    )

    prompt = template.render(label="Query", input="What interesting things happened on this day in history?")

    print(prompt)


if __name__ == "__main__":
    try:
        main()
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
