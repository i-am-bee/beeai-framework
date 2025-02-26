from pydantic import BaseModel

from beeai_framework.template import PromptTemplate, PromptTemplateInput


class UserMessage(BaseModel):
    label: str
    input: str


template = PromptTemplate(
    PromptTemplateInput(
        schema=UserMessage,
        template="""{{label}}: {{input}}""",
    )
)

prompt = template.render(UserMessage(label="Query", input="What interesting things happened on this day in history?"))

print(prompt)
