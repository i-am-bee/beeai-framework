from typing import Any

from openai import BaseModel

from beeai_framework.template import PromptTemplate, PromptTemplateInput


class AutoflowReasonPromptInput(BaseModel):
    actions: list[dict[str, Any]]
    context: str | None = None


AutoflowReasonPrompt = PromptTemplate(
    PromptTemplateInput(
        schema=AutoflowReasonPromptInput,
        template="""You are a supervisor who decides what actions to do next to resolve the user's task.

You can use one of the following actions.

{{#actions}}
- {{name}} ({{description}})
{{/actions}}
{{#context}}

This is additional context that you can use to help you resolve the task: {{.}}

{{/context}}

After you accomplished everything what the user asked for you must provide a final answer via the 'final_answer' action.
Plan your actions in logical order to resolve the task correctly.

Now begin!
""",
    )
)


class AutoflowResponsePromptInput(BaseModel):
    step: str
    input: str
    output: str


AutoflowResponsePrompt = PromptTemplate(
    PromptTemplateInput(
        schema=AutoflowResponsePromptInput,
        template="""Action Name: {{step}}
Action Input: {{input}}
Action Output: {{output}}
""",
    )
)
