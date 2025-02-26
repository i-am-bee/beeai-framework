import os
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from beeai_framework.template import PromptTemplate, PromptTemplateInput

os.environ["USER"] = "BeeAI"


class UserQuery(BaseModel):
    query: str


template = PromptTemplate(
    PromptTemplateInput(
        schema=UserQuery,
        functions={
            "format_date": lambda: datetime.now(ZoneInfo("US/Eastern")).strftime("%A, %B %d, %Y at %I:%M:%S %p"),
            "current_user": lambda: os.environ["USER"],
        },
        template="""
{{format_date}}
{{current_user}}: {{query}}
""",
    )
)
