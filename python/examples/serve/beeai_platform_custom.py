# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Annotated, Any

from beeai_sdk.a2a.extensions import CitationExtensionServer, CitationExtensionSpec

from beeai_framework.adapters.beeai_platform.backend.chat import BeeAIPlatformChatModel
from beeai_framework.adapters.beeai_platform.context import BeeAIPlatformContext
from beeai_framework.adapters.beeai_platform.serve.server import BeeAIPlatformMemoryManager, BeeAIPlatformServer
from beeai_framework.adapters.beeai_platform.serve.types import BaseBeeAIPlatformExtensions
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.events import RequirementAgentSuccessEvent
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from beeai_framework.backend import AssistantMessage
from beeai_framework.context import RunContext, RunMiddlewareProtocol
from beeai_framework.emitter import EmitterOptions
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.think import ThinkTool

try:
    from beeai_sdk.a2a.extensions import Citation
    from beeai_sdk.a2a.extensions.ui.agent_detail import AgentDetail
    from beeai_sdk.a2a.types import AgentMessage
    from beeai_sdk.server.context import RunContext as BeeAIRunContext
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [beeai-platform] not found.\nRun 'pip install \"beeai-framework[beeai-platform]\"' to install."
    ) from e


def main() -> None:
    class PlatformMiddleware(RunMiddlewareProtocol):
        def bind(self, ctx: RunContext) -> None:
            try:
                context = BeeAIPlatformContext.get()

                def send_citation(data: Any, event: Any) -> None:
                    # get citation extension
                    citation_ext = context.extensions.get("citation")
                    platform_context: BeeAIRunContext = context.context

                    # check it is the final step
                    if isinstance(data, RequirementAgentSuccessEvent) and data.state.answer is not None:
                        citations, clean_text = extract_citations(data.state.answer.text)

                        if citations:
                            platform_context.yield_sync(
                                AgentMessage(metadata=citation_ext.citation_metadata(citations=citations))  # type: ignore[attr-defined]
                            )
                            # replace assistant message with an updated text without citation links
                            data.state.answer = AssistantMessage(content=clean_text)

                # add emitter with the highest priority to ensure citations are sent before any other event handling
                ctx.emitter.on("success", send_citation, options=EmitterOptions(priority=10, is_blocking=True))
            except Exception as e:
                print(e)

    agent = RequirementAgent(
        llm=BeeAIPlatformChatModel(preferred_models=["openai/gpt-5", "ollama/granite3.3:8b"]),
        tools=[WikipediaTool(), DuckDuckGoSearchTool(), ThinkTool()],
        instructions=(
            "You are an AI assistant focused on retrieving information from online sources."
            "Mandatory Search: Always search for the topic on Wikipedia and always search for related current news."
            "Mandatory Output Structure: Return the result in two separate sections wit headings:"
            " 1. Basic Information (primarily utilizing data from Wikipedia, if relevant)."
            " 2. News (primarily utilizing current news results). "
            "Mandatory Citation: Always include a source link for all given information, especially news."
        ),
        requirements=[
            ConditionalRequirement(ThinkTool, force_at_step=1, consecutive_allowed=False),
            ConditionalRequirement(WikipediaTool, min_invocations=1),
            ConditionalRequirement(DuckDuckGoSearchTool, min_invocations=1),
        ],
        description="Search for information based on a given phrase.",
        middlewares=[
            GlobalTrajectoryMiddleware(),
            PlatformMiddleware(),
        ],  # add platform middleware to obtain citations from the platform
    )

    # define custom extensions
    class CitationExtensions(BaseBeeAIPlatformExtensions):
        citation: Annotated[CitationExtensionServer, CitationExtensionSpec()]

    # Runs HTTP server that registers to BeeAI platform
    server = BeeAIPlatformServer(
        config={"configure_telemetry": False}, memory_manager=BeeAIPlatformMemoryManager()
    )  # use platform memory
    server.register(
        agent,
        name="Information retrieval",
        detail=AgentDetail(interaction_mode="single-turn", user_greeting="What can I search for you?"),
        extensions=CitationExtensions,
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
    main()
