import sys
import traceback

from pydantic import BaseModel

from beeai_framework.errors import FrameworkError
from beeai_framework.template import PromptTemplate, PromptTemplateInput


def main() -> None:
    class Response(BaseModel):
        duration: int

    class ExpectedDuration(BaseModel):
        expected: int
        responses: list[Response]

    template: PromptTemplate[ExpectedDuration] = PromptTemplate(
        PromptTemplateInput(
            schema=ExpectedDuration,
            template="""Expected Duration: {{expected}}ms; Retrieved: {{#responses}}{{duration}}ms {{/responses}}""",
            defaults={"expected": 5},
        )
    )

    # Expected Duration: 5ms; Retrieved: 3ms 5ms 6ms
    output = template.render(responses=[Response(duration=3), Response(duration=5), Response(duration=6)])
    print(output)


if __name__ == "__main__":
    try:
        main()
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
