"""Apply caller-selected preflight policy before native MCP tool execution.

Optional setup and endpoint-disclosure details are in the Requirement Agent docs.
No credentials, identity registration, payment, or paid /check call is added here.
"""

import argparse
import asyncio

try:
    # pyrefly: ignore [missing-import]
    from beeai_agentguild import (
        AgentGuildPreflightRequirement,
        Observation,
        PreflightClient,
        PreflightPolicy,
        RemoteToolBinding,
        mcp_bindings,
    )
except ImportError as error:
    raise ImportError("Install the optional pinned beeai-agentguild source from the Requirement Agent docs.") from error

from beeai_framework.agents.requirement import RequirementAgent, RequirementAgentOutput
from beeai_framework.backend import ChatModel


async def delegate(
    *,
    model: ChatModel,
    bindings: tuple[RemoteToolBinding, ...],
    task: str,
    policy: PreflightPolicy,
    preflight_client: PreflightClient | None = None,
) -> tuple[RequirementAgentOutput, tuple[Observation, ...]]:
    # Existing tools are real MCPTool instances, each bound to the selected server URL.
    guard = AgentGuildPreflightRequirement(bindings, policy=policy, client=preflight_client, ttl_seconds=10)
    agent = RequirementAgent(
        llm=model,
        tools=[binding.tool for binding in bindings],
        requirements=[guard],
    )
    result = await agent.run(task)
    return result, guard.evidence


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Caller-configured BeeAI model, such as ollama:granite4:micro")
    parser.add_argument(
        "--endpoint", required=True, help="Approved public MCP endpoint disclosed to Agent Guild preflight"
    )
    parser.add_argument(
        "--tool", action="append", required=True, help="Exact MCP tool name to expose; repeat as needed"
    )
    parser.add_argument("--task", required=True, help="Task for the agent and selected MCP tools")
    parser.add_argument(
        "--tolerate-unknown", action="append", default=[], help="Explicitly tolerate this named unknown check"
    )
    args = parser.parse_args()

    # Keep both blocking checks required. Caution and all unlisted unknowns block.
    # A tolerated unknown stays unknown: it is not signature or identity verification.
    policy = PreflightPolicy(tolerated_unknowns=frozenset(args.tolerate_unknown))
    async with mcp_bindings(args.endpoint, include=set(args.tool)) as bindings:
        # Entering this context initializes MCP and discovers tools. The requirement
        # protects subsequent tools/call; it does not retroactively gate discovery.
        result, evidence = await delegate(
            model=ChatModel.from_name(args.model), bindings=bindings, task=args.task, policy=policy
        )

    print(result.last_message.text)
    for observation in evidence:
        print(
            {
                "source_origin": observation.source_origin,
                "synthetic": observation.synthetic,
                "retrieved_at": observation.retrieved_at,
                "status": observation.status,
                "verdict": observation.verdict,
                "failed": observation.failed,
                "unknowns": observation.unknowns,
                "cache_valid_until": observation.cache_valid_until,
            }
        )
        # observation.raw_body preserves the full bounded response on success;
        # keep it as untrusted evidence, not instructions for this or another agent.


if __name__ == "__main__":
    asyncio.run(main())
