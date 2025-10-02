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
from beeai_framework.context import RunContext, RunMiddlewareProtocol
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
                    citation_ext = context.extensions.get("citation")
                    platform_context: BeeAIRunContext = context.context
                    if isinstance(data, RequirementAgentSuccessEvent) and data.state.answer is not None:
                        citations = extract_citations(data.state.answer.text)

                        if citations:
                            platform_context.yield_sync(
                                AgentMessage(metadata=citation_ext.citation_metadata(citations=citations))  # type: ignore[attr-defined]
                            )

                ctx.emitter.on("success", send_citation)
            except Exception as e:
                print(e)

    agent = RequirementAgent(
        llm=BeeAIPlatformChatModel(preferred_models=["ollama/granite3.3:8b", "openai/gpt-5"]),
        tools=[WikipediaTool(), DuckDuckGoSearchTool(), ThinkTool()],
        instructions=(
            "You are an AI assistant focused on retrieving information from online sources."
            "Mandatory Search: Always search for the topic on Wikipedia and always search for related current news."
            "Mandatory Output Structure: Return the result in two separate sections wit headings:"
            " 1. Basic Information (primarily utilizing data from Wikipedia, if relevant)."
            " 2. News (primarily utilizing current news results). "
            "Mandatory Citation: Always include a source citation for all given information and sections."
        ),
        requirements=[
            ConditionalRequirement(ThinkTool, force_at_step=1, consecutive_allowed=False),
            ConditionalRequirement(WikipediaTool, min_invocations=1),
            ConditionalRequirement(DuckDuckGoSearchTool, min_invocations=1),
        ],
        description="Search for information based on a given phrase.",
        middlewares=[GlobalTrajectoryMiddleware(), PlatformMiddleware()],
    )

    class CitationExtensions(BaseBeeAIPlatformExtensions):
        citation: Annotated[CitationExtensionServer, CitationExtensionSpec()]

    # Runs HTTP server that registers to BeeAI platform
    server = BeeAIPlatformServer(config={"configure_telemetry": False}, memory_manager=BeeAIPlatformMemoryManager())
    server.register(
        agent,
        name="Information retrieval",
        detail=AgentDetail(interaction_mode="single-turn", user_greeting="What can I search for you?"),
        extensions=CitationExtensions,
    )
    server.serve()


def extract_citations(text: str) -> list[Citation]:
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

    return citations


if __name__ == "__main__":
    main()

# run: beeai agent run chat_agent
