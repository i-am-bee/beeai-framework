import sys
import traceback
from typing import Any

from pydantic import BaseModel

from beeai_framework.errors import FrameworkError
from beeai_framework.template import PromptTemplate, PromptTemplateInput


def main() -> None:
    class OriginalSchema(BaseModel):
        name: str
        objective: str

    original: PromptTemplate[OriginalSchema] = PromptTemplate(
        PromptTemplateInput(
            schema=OriginalSchema,
            template="""You are a helpful assistant called {{name}}. Your objective is to {{objective}}.""",
        )
    )

    def customizer(temp_input: PromptTemplateInput[Any]) -> PromptTemplateInput[Any]:
        new_temp = temp_input.model_copy()
        new_temp.template = f"""{temp_input.template} Your answers must be concise."""
        new_temp.defaults["name"] = "Bee"
        return new_temp

    modified = original.fork(customizer=customizer)

    # You are a helpful assistant called Bee. Your objective is to fulfill the user needs. Your answers must be concise.
    prompt = modified.render(objective="fulfill the user needs")
    print(prompt)


if __name__ == "__main__":
    try:
        main()
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
