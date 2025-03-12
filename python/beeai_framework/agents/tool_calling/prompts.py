# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import UTC, datetime

from pydantic import BaseModel

from beeai_framework.template import PromptTemplate, PromptTemplateInput


class ToolCallingAgentSystemPromptInput(BaseModel):
    instructions: str | None = None


ToolCallingAgentSystemPrompt = PromptTemplate(
    PromptTemplateInput(
        schema=ToolCallingAgentSystemPromptInput,
        functions={
            "formatDate": lambda data: datetime.now(tz=UTC).strftime("%A, %B %d, %Y at %I:%M:%S %p"),
        },
        defaults={"role": "You are a helpful assistant.", "instructions": ""},
        template="""{{role}}
When the user sends a message, figure out a solution and provide a final answer.

# Best practices
- Use markdown syntax to format code snippets, links, JSON, tables, images, and files.
- When the message is unclear, ask for a clarification.
- When the user wants to chitchat, always respond politely.

# Date and Time
The current date and time is: {{formatDate}}

{{#instructions}}
# Additional instructions
{{.}}
{{/instructions}}
""",
    )
)
