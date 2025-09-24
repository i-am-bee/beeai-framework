# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio

from pydantic import BaseModel

from beeai_framework.backend.message import AnyMessage, AssistantMessage, UserMessage
from beeai_framework.workflows.v2.decorators.after import after
from beeai_framework.workflows.v2.decorators.end import end
from beeai_framework.workflows.v2.decorators.fork import fork
from beeai_framework.workflows.v2.decorators.join import join
from beeai_framework.workflows.v2.decorators.start import start
from beeai_framework.workflows.v2.workflow import Workflow


class Page(BaseModel):
    link: str
    content: str


class WebScrapperWorkflow(Workflow):
    def __init__(self) -> None:
        super().__init__()
        self.url_to_content = {
            "https://www.wikipedia.org": "A free online encyclopedia with millions of articles.",
            "https://www.github.com": "A platform for hosting and collaborating on Git repositories.",
            "https://www.python.org": "Official home of the Python programming language.",
            "https://www.stackoverflow.com": "A community for programmers to ask and answer coding questions.",
            "https://www.nasa.gov": "NASA's official site with news about space missions and discoveries.",
        }

    @start
    async def extract_links(self, messages: list[AnyMessage]) -> list[str]:
        return list(self.url_to_content.keys())

    @after(extract_links)
    @fork
    async def scrape_link(self, link: str) -> str:
        print(f"Scrape {link}")
        await asyncio.sleep(2)
        content = self.url_to_content[link]
        return content

    @after(scrape_link)
    @join
    async def post_process(self, links: list[str], content: list[str]) -> list[Page]:
        return [Page(link=link, content=content) for link, content in zip(links, content, strict=False)]

    @after(post_process)
    @end
    async def finalize(self, pages: list[Page]) -> list[AnyMessage]:
        pages_txt = "\n\n".join([f"Link: {p.link}\n# Content: {p.content}" for p in pages])
        return [AssistantMessage(f"Here are all scrapped pages\n{pages_txt}")]


# Async main function
async def main() -> None:
    workflow = WebScrapperWorkflow()
    output = await workflow.run(
        [
            UserMessage(
                "Imagine we receive a signal from an intelligent extraterrestrial civilization. How should we interpret it, what assumptions should we question, and what could be the global implications of responding?"
            )
        ]
    )
    print(output.last_message.text)


# Entry point
if __name__ == "__main__":
    asyncio.run(main())
