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


from typing import Any

import wikipediaapi

from pydantic import BaseModel, Field

from beeai_framework.tools import ToolError
from beeai_framework.tools.search import SearchToolOutput, SearchToolResult
from beeai_framework.tools.tool import Tool
from beeai_framework.utils import BeeLogger

class WikipediaSearchToolInput(BaseModel):
    query: str = Field(description="Search query, name of the Wikipedia page.")

class WikipediaSearchToolResult(SearchToolResult):
    pass

class WikipediaSearchToolOutput(SearchToolOutput):
    pass


class WikipediaSearchTool(Tool[WikipediaSearchToolInput]):
    name = "Wikipedia"
    description = "Search factual and historical information, including biography, history, politics, geography, society, culture, science, technology, people, animal species, mathematics, and other subjects."
    input_schema = WikipediaSearchToolInput
    client = wikipediaapi.Wikipedia(user_agent="beeai-framework https://github.com/i-am-bee/beeai-framework", language='en')

    def __init__(self, full_text: bool = False) -> None:
        super().__init__()
        self.full_text = full_text

    def _run(self, input: WikipediaSearchToolInput, _: Any | None = None) -> WikipediaSearchToolOutput:
        try:
            page_py = self.client.page(input.query)
            if(page_py.exists()):
                description_output = page_py.text if self.full_text else page_py.summary 
                search_results: list[WikipediaSearchToolResult] = [
                    WikipediaSearchToolResult(
                        title=input.query or "", description=description_output or "", url=page_py.fullurl or ""
                    )
                ]
                return WikipediaSearchToolOutput(search_results)
            else:
                raise Exception("No Wikipedia page matched the search term: %s." % (input.query))
        except Exception as e:
            raise ToolError("Error performing Wikipedia search: ") from e
