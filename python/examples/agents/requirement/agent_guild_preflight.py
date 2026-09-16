# Copyright 2026 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
"""Restrict selected MCP tool calls with a fresh, unsigned endpoint observation.

Use the repository's Python environment with its existing MCP extra. No separate
Agent Guild package is needed. Supply --disclose-endpoint only after approving
that the full public endpoint URL is sent to Agent Guild, which probes and logs
it. Never supply private endpoints, credentials, signed URLs or confidential paths.

The caller must separately authorize the model, task and selected tool. Preflight
is evidence, never that authorization: it does not establish identity, signature
validity, completed work, payment binding or safety. Only free /preflight is used.
MCP initialization/discovery happens before the requirement protects tools/call.
Unknowns and unavailable/invalid evidence block by default; named unknowns may be
explicitly tolerated. Retained response bytes are untrusted data, not instructions.
"""

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Self
from urllib.parse import urlsplit

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from beeai_framework.agents.requirement import RequirementAgent, RequirementAgentRunState
from beeai_framework.agents.requirement.requirements import Requirement
from beeai_framework.agents.requirement.requirements.requirement import Rule, run_with_context
from beeai_framework.backend import ChatModel
from beeai_framework.context import RunContext, RunContextStartEvent
from beeai_framework.emitter import EmitterOptions, EventMeta
from beeai_framework.emitter.utils import create_internal_event_matcher
from beeai_framework.tools import AnyTool, StringToolOutput
from beeai_framework.tools.mcp import MCPTool

PREFLIGHT_URL = "https://agent-guild-5d5r.onrender.com/preflight"
REQUIRED_PROVEN = frozenset({"endpoint_reachable", "protocol_handshake"})
MANDATORY_CHECKS = REQUIRED_PROVEN | {
    "agent_card_resolves",
    "agent_card_signed",
    "payment_claim_holds",
    "independent_evidence",
}


def validate_endpoint(endpoint: str) -> str:
    # Syntax checks cannot establish that a hostname/path is public. Disclosure
    # needs explicit caller approval even if these checks pass.
    parsed = urlsplit(endpoint)
    if (
        len(endpoint) > 2048
        or any(ord(char) <= 32 for char in endpoint)
        or any(char in endpoint for char in ("\\", "?", "#"))
        or parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or "%" in parsed.netloc
    ):
        raise ValueError("Use an approved public HTTPS URL without credentials, queries or fragments")
    _ = parsed.port
    return endpoint


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def denial(raw: bytes, endpoint: str, tolerated_unknowns: frozenset[str]) -> str | None:
    """Validate internal consistency before applying this example's caller policy."""
    doc = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(doc, dict) or doc.get("target") != endpoint or not isinstance(doc.get("checks"), list):
        raise ValueError("Missing checks or mismatched target")
    checks: dict[str, str] = {}
    for row in doc["checks"]:
        if not isinstance(row, dict):
            raise ValueError("Malformed check")
        name, status = row.get("check"), row.get("status")
        if not isinstance(name, str) or not name or name in checks or status not in ("proven", "unknown", "failed"):
            raise ValueError("Malformed or duplicate check")
        checks[name] = status
    # Optional A2A cards remain explicit unknowns for MCP endpoints. Omitting a
    # check must never erase its uncertainty. Additional checks obey the same policy.
    if not checks.keys() >= MANDATORY_CHECKS:
        raise ValueError("Missing mandatory checks")
    for summary, statuses in (("failed", {"failed"}), ("unknowns", {"unknown"}), ("scored", {"proven", "failed"})):
        names = doc.get(summary)
        if (
            not isinstance(names, list)
            or not all(isinstance(name, str) for name in names)
            or len(set(names)) != len(names)
            or set(names) != {name for name, value in checks.items() if value in statuses}
        ):
            raise ValueError("Inconsistent summary")
    expected = (
        "do_not_delegate"
        if any(checks[name] == "failed" for name in REQUIRED_PROVEN)
        else "delegate_with_caution"
        if doc["failed"]
        else "no_failed_checks"
    )
    if doc.get("verdict") != expected:
        raise ValueError("Inconsistent verdict")
    if doc["failed"] or any(checks[name] != "proven" for name in REQUIRED_PROVEN):
        return "Caller policy blocks a failed or unknown required check."
    if set(doc["unknowns"]) - tolerated_unknowns:
        return "Caller policy blocks an unknown check that was not explicitly tolerated."
    return None


@dataclass(frozen=True)
class Observation:
    endpoint: str
    retrieved_at: str
    raw_body: bytes
    denied: str | None


class PreflightRequirement(Requirement[RequirementAgentRunState]):
    """Deny-only rules plus a blocking native start hook; use one instance per run."""

    name = "agent_guild_preflight"

    def __init__(
        self,
        target: AnyTool,
        endpoint: str,
        *,
        disclose_endpoint: bool,
        tolerated_unknowns: frozenset[str] = frozenset(),
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__()
        if not disclose_endpoint:
            raise ValueError("The caller must approve full endpoint URL disclosure")
        if target.name == "final_answer":
            raise ValueError("Do not bind the final-answer tool")
        self.target = target
        self.endpoint = validate_endpoint(endpoint)
        self.tolerated_unknowns = frozenset(tolerated_unknowns)
        self.transport = transport  # Optional in-memory transport for deterministic offline tests.
        self.evidence: list[Observation] = []

    async def observe(self) -> str | None:
        body = bytearray()
        reason: str | None = "Preflight is unavailable or invalid; caller policy blocks this call."
        try:
            # Total deadline also bounds a response that keeps trickling bytes.
            async with asyncio.timeout(5):
                async with httpx.AsyncClient(
                    timeout=5,
                    follow_redirects=False,
                    trust_env=False,
                    transport=self.transport,
                ) as client:
                    async with client.stream("GET", PREFLIGHT_URL, params={"url": self.endpoint}) as response:
                        if (
                            response.status_code != 200
                            or response.headers.get("content-type", "").split(";")[0] != "application/json"
                        ):
                            raise ValueError("Expected JSON HTTP 200; redirects and payment challenges block")
                        async for chunk in response.aiter_bytes():
                            if len(body) + len(chunk) > 65_536:
                                raise ValueError("Response exceeds 64 KiB")
                            body.extend(chunk)
                        reason = denial(bytes(body), self.endpoint, self.tolerated_unknowns)
        except (httpx.HTTPError, TimeoutError, ValueError, TypeError, RecursionError):
            pass
        self.evidence.append(Observation(self.endpoint, datetime.now(UTC).isoformat(), bytes(body), reason))
        return reason

    async def init(self, *, tools: list[AnyTool], ctx: RunContext) -> None:
        await super().init(tools=tools, ctx=ctx)
        if sum(tool is self.target for tool in tools) != 1 or len({tool.name for tool in tools}) != len(tools):
            raise ValueError("The exact selected tool instance must appear once, with unique tool names")
        self.evidence.clear()

        async def before_tool(data: RunContextStartEvent, _: EventMeta) -> None:
            if self.enabled and data.output is None:
                # Recheck at execution; a previous allow observation is never cached.
                reason = await self.observe()
                if reason is not None:
                    data.output = StringToolOutput(reason)

        ctx.emitter.on(
            create_internal_event_matcher("start", self.target, parent_run_id=ctx.run_id),
            before_tool,
            EmitterOptions(is_blocking=True, persistent=True, match_nested=True),
        )

    @run_with_context
    # pyrefly: ignore [bad-override]
    async def run(self, state: RequirementAgentRunState, context: RunContext) -> list[Rule]:
        reason = await self.observe()
        # No positive allow rule: this requirement cannot override another restriction.
        return [Rule(target=self.target.name, allowed=False, reason=reason)] if reason else []

    async def clone(self) -> Self:
        instance = type(self)(
            self.target,
            self.endpoint,
            disclose_endpoint=True,
            tolerated_unknowns=self.tolerated_unknowns,
            transport=self.transport,
        )
        instance.enabled, instance.priority = self.enabled, self.priority
        instance.middlewares.extend(self.middlewares)
        instance.state = self.state.copy()
        return instance


async def delegate(
    endpoint: str,
    tool_name: str,
    task: str,
    model: ChatModel,
    *,
    disclose_endpoint: bool,
    tolerated_unknowns: frozenset[str] = frozenset(),
) -> None:
    if not disclose_endpoint:
        raise ValueError("Approve public endpoint disclosure before opening any transport")
    endpoint = validate_endpoint(endpoint)
    # Discovery is a caller-authorized operation, not guarded by preflight.
    async with (
        httpx.AsyncClient(timeout=10, follow_redirects=False, trust_env=False) as client,
        streamable_http_client(endpoint, http_client=client) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        selected = [tool for tool in await MCPTool.from_session(session) if tool.name == tool_name]
        if len(selected) != 1:
            raise ValueError("Select exactly one discovered MCP tool by its exact name")
        guard = PreflightRequirement(
            selected[0], endpoint, disclose_endpoint=True, tolerated_unknowns=tolerated_unknowns
        )
        agent = RequirementAgent(llm=model, tools=selected, requirements=[guard])
        result = await agent.run(task)
        print(result.last_message.text)
        for observation in guard.evidence:
            print(
                {
                    "endpoint": observation.endpoint,
                    "retrieved_at": observation.retrieved_at,
                    "denied": observation.denied,
                    "unsigned": True,
                }
            )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Caller-authorized BeeAI model")
    parser.add_argument("--endpoint", required=True, help="Approved public MCP HTTPS endpoint")
    parser.add_argument("--tool", required=True, help="Exact authorized MCP tool name")
    parser.add_argument("--task", required=True, help="Caller-authorized task")
    parser.add_argument("--disclose-endpoint", required=True, action="store_true")
    parser.add_argument("--tolerate-unknown", action="append", default=[])
    args = parser.parse_args()
    await delegate(
        args.endpoint,
        args.tool,
        args.task,
        ChatModel.from_name(args.model),
        disclose_endpoint=args.disclose_endpoint,
        tolerated_unknowns=frozenset(args.tolerate_unknown),
    )


if __name__ == "__main__":
    asyncio.run(main())
