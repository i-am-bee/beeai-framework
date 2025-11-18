import sys
import traceback

from pydantic import BaseModel, Field

from beeai_framework.errors import FrameworkError
from beeai_framework.template import PromptTemplate, PromptTemplateInput


def main() -> None:
    class ColorsObject(BaseModel):
        colors: list[str] = Field(..., min_length=1)

    template: PromptTemplate[ColorsObject] = PromptTemplate(
        PromptTemplateInput(
            schema=ColorsObject,
            template="""Colors: {{#colors}}{{.}}, {{/colors}}""",
        )
    )

    # Colors: Green, Yellow,
    output = template.render(colors=["Green", "Yellow"])
    print(output)


if __name__ == "__main__":
    try:
        main()
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
