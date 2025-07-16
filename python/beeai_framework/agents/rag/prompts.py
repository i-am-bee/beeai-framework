# Copyright 2025 © BeeAI a Series of LF Projects, LLC
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


from pydantic import BaseModel

from beeai_framework.template import PromptTemplate, PromptTemplateInput


class PromptRefactoringMessage(BaseModel):
    user_query: str
    clarifications: str | None = None


QUERY_IMPROVEMENT_PROMPT = PromptTemplate(
    PromptTemplateInput(
        schema=PromptRefactoringMessage,
        template="""Refactor the following user prompt to optimize it for semantic search in a vector store:

Original prompt:
"{{user_query}}"

{{user_clarifications}}

Guidelines for refactoring:
- Remove ambiguous or overly broad terms; be specific and descriptive.
- Focus on key concepts, entities, or relationships that can be meaningfully embedded.
- Rephrase vague language into more concrete terminology.
- Prefer nouns and phrases over full questions, unless context is essential.
- Eliminate filler or conversational language that doesn’t add semantic value.

Provide the rewritten prompt as a single concise sentence or phrase that captures the semantic intent clearly and efficiently.
IMPORTANT: ONLY return the new prompt, no explanation, no quotation marks and nothing else, if you can't change the prompt, return the original prompt
""",  # noqa: E501
    )
)
