# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
import re
import sys
import traceback
from typing import Annotated

from beeai_framework.adapters.agentstack.backend.chat import AgentStackChatModel
from beeai_framework.adapters.agentstack.context import AgentStackContext
from beeai_framework.adapters.agentstack.serve.server import AgentStackMemoryManager, AgentStackServer
from beeai_framework.adapters.agentstack.serve.types import BaseAgentStackExtensions
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.events import RequirementAgentSuccessEvent
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from beeai_framework.backend import AssistantMessage
from beeai_framework.context import RunContext, RunMiddlewareProtocol
from beeai_framework.emitter import EmitterOptions, EventMeta
from beeai_framework.errors import FrameworkError
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.think import ThinkTool

try:
    from agentstack_sdk.a2a.extensions import Citation, CitationExtensionServer, CitationExtensionSpec
    from agentstack_sdk.a2a.extensions.ui.agent_detail import AgentDetail
    from agentstack_sdk.a2a.types import AgentMessage
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [agentstack] not found.\nRun 'pip install \"beeai-framework[agentstack]\"' to install."
    ) from e


class PlatformCitationMiddleware(RunMiddlewareProtocol):
    def __init__(self) -> None:
        self._context: AgentStackContext | None = None

    def bind(self, ctx: RunContext) -> None:
        self._context = AgentStackContext.get()
        # add emitter with the highest priority to ensure citations are sent before any other event handling
        ctx.emitter.on("success", self._handle_success, options=EmitterOptions(priority=10, is_blocking=True))

    async def _handle_success(self, data: RequirementAgentSuccessEvent, meta: EventMeta) -> None:
        assert self._context is not None
        citation_ext = self._context.extensions.get("citation")

        # check it is the final step
        if data.state.answer is not None:
            citations, clean_text = extract_citations(data.state.answer.text)

            if citations:
                await self._context.context.yield_async(
                    AgentMessage(metadata=citation_ext.citation_metadata(citations=citations))  # type: ignore[attr-defined]
                )
                # replace an assistant message with an updated text without citation links
                data.state.answer = AssistantMessage(content=clean_text)


def main() -> None:
    agent = RequirementAgent(
        llm=AgentStackChatModel(preferred_models=["openai/gpt-5"]),
        tools=[WikipediaTool(), ThinkTool()],
        instructions=(
            "You are an AI assistant focused on retrieving information from online sources."
            "Mandatory Search: Always search for the topic on Wikipedia and always search for related current news."
            "Mandatory Output Structure: Return the result in two separate sections with headings:"
            " 1. Basic Information (primarily utilizing data from Wikipedia, if relevant)."
            " 2. News (primarily utilizing current news results). "
            "Mandatory Citation: Always include a source link for all given information, especially news."
        ),
        requirements=[
            ConditionalRequirement(ThinkTool, force_at_step=1, consecutive_allowed=False),
            ConditionalRequirement(WikipediaTool, min_invocations=1),
        ],
        description="Search for information based on a given phrase.",
        middlewares=[
            GlobalTrajectoryMiddleware(),
            PlatformCitationMiddleware(),
        ],  # add platform middleware to get citations from the platform
    )

    # define custom extensions
    class CustomExtensions(BaseAgentStackExtensions):
        citation: Annotated[CitationExtensionServer, CitationExtensionSpec()]

    # Runs HTTP server that registers to Agent Stack
    server = AgentStackServer(memory_manager=AgentStackMemoryManager())  # use platform memory
    server.register(
        agent,
        name="Information retrieval",
        detail=AgentDetail(interaction_mode="single-turn", user_greeting="What can I search for you?"),
        extensions=CustomExtensions,
    )
    server.serve()


# function to extract citations from text and return clean text without citation links
def extract_citations(text: str) -> tuple[list[Citation], str]:
    citations, offset = [], 0
    pattern = r"\[([^\]]+)\]\(([^)]+)\)"

    for match in re.finditer(pattern, text):
        content, url = match.groups()
        start = match.start() - offset

        citations.append(
            Citation(
                url=url,
                title=url.split("/")[-1].replace("-", " ").title() or content[:50],
                description=content[:100] + ("..." if len(content) > 100 else ""),
                start_index=start,
                end_index=start + len(content),
            )
        )
        offset += len(match.group(0)) - len(content)

    return citations, re.sub(pattern, r"\1", text)


if __name__ == "__main__":
    try:
        main()
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
