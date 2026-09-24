# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import contextlib
from collections.abc import Sequence
from typing import Literal
from weakref import WeakKeyDictionary

from pydantic import BaseModel, ConfigDict

from beeai_framework.agents.errors import AgentError
from beeai_framework.agents.requirement.prompts import (
    RequirementAgentSystemPromptInput,
    RequirementAgentToolTemplateDefinition,
)
from beeai_framework.agents.requirement.requirements.events import RequirementInitEvent, requirement_event_types
from beeai_framework.agents.requirement.requirements.requirement import Requirement, Rule
from beeai_framework.agents.requirement.types import RequirementAgentRequest, RequirementAgentRunState
from beeai_framework.agents.requirement.utils._tool import FinalAnswerTool
from beeai_framework.backend import SystemMessage
from beeai_framework.context import RunContext
from beeai_framework.template import PromptTemplate
from beeai_framework.tools import AnyTool
from beeai_framework.tools.tool import Tool
from beeai_framework.utils.lists import _append_if_not_exists, remove_by_reference
from beeai_framework.utils.strings import to_json, to_safe_word


class RequirementsReasoner:
    def __init__(
        self,
        *,
        tools: Sequence[AnyTool],
        final_answer: FinalAnswerTool,
        context: RunContext,
    ) -> None:
        self._tools = [*tools, final_answer]
        self._entries: list[Requirement[RequirementAgentRunState]] = []
        self._context = context
        self.final_answer = final_answer

    async def update(self, requirements: Sequence[Requirement[RequirementAgentRunState]]) -> None:
        self._entries.clear()

        for requirement in requirements:
            self._entries.append(requirement)

        for entry in self._entries:
            emitter = self._context.emitter.child(
                group_id=to_safe_word(entry.name), creator=entry, events=requirement_event_types
            )
            emitter.namespace.append("requirement")

            tools = list(self._tools)
            await emitter.emit("init", RequirementInitEvent(tools=tools))
            await entry.init(tools=tools, ctx=self._context)

    def _find_tool_by_name(self, name: str) -> AnyTool:
        tool: AnyTool | None = next((t for t in self._tools if t.name == name), None)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found in ({','.join(t.name for t in self._tools)}).")
        return tool

    async def create_request(
        self,
        state: RequirementAgentRunState,
        *,
        force_tool_call: bool,
        extra_rules: list[Rule] | None = None,
    ) -> RequirementAgentRequest:
        hidden: list[AnyTool] = []
        allowed: list[AnyTool] = []
        all_tools: list[AnyTool] = list(self._tools)
        reason_by_tool: WeakKeyDictionary[AnyTool, str | None] = WeakKeyDictionary()

        prevent_stop: bool = False
        prevent_step_refs: list[RuleEntry] = []

        forced: AnyTool | None = None
        forced_level: int = 0
        rules_by_tool: dict[str, list[RuleEntry]] = {t.name: [] for t in self._tools}

        # Group rules
        for requirement in [entry for entry in self._entries if entry.enabled]:
            generated_rules = await requirement.run(state)
            for rule in generated_rules:
                tool = self._find_tool_by_name(rule.target)
                rules_by_tool[tool.name].append(
                    RuleEntry(priority=requirement.priority, rule=rule, requirement=requirement)
                )

        # Add extra rules
        for rule in extra_rules or []:
            if rule.target not in rules_by_tool:
                raise ValueError(f"Tool '{rule.target}' not found.")

            rules = rules_by_tool[rule.target]
            priority = max(rules, key=lambda v: v.priority).priority + 1 if rules else 1
            rules.append(RuleEntry(priority=priority, rule=rule, requirement=None))

        # Aggregate rules and infer the required tool
        for tool_name, rules in rules_by_tool.items():
            tool = self._find_tool_by_name(tool_name)
            rules.sort(key=lambda x: x.priority, reverse=True)  # DESC

            max_priority = rules[0].priority if rules else 1
            is_allowed = True
            is_forced = False
            is_hidden = False
            is_prevent_stop = False

            for rule_entry in rules:
                rule = rule_entry.rule
                if not rule.allowed:
                    is_allowed = False
                if rule.hidden:
                    is_hidden = True
                if rule.forced:
                    is_forced = True
                if rule.prevent_stop:
                    is_prevent_stop = True
                    prevent_step_refs.append(rule_entry)
                if rule.reason:
                    reason_by_tool[tool] = rule.reason

            if is_allowed and is_hidden:
                is_allowed = False

            if is_allowed:
                _append_if_not_exists(allowed, tool)
                if is_forced and (not forced or forced_level < max_priority):
                    forced = tool
                    forced_level = max_priority
            if is_hidden:
                _append_if_not_exists(hidden, tool)
            if is_prevent_stop:
                prevent_stop = True

        if forced is not None:
            allowed.clear()
            _append_if_not_exists(allowed, forced)
            _append_if_not_exists(allowed, self.final_answer)

        if prevent_stop and not isinstance(forced, FinalAnswerTool):
            with contextlib.suppress(ValueError):
                remove_by_reference(allowed, self.final_answer)

        if not allowed:
            raise AgentError(
                "One of the generated rules is preventing the agent from continuing. "
                "This indicates that the provided requirements may conflict with each other. "
                "See the following rules and their attached requirements that are preventing the agent from continuing."
                f"\n{to_json(prevent_step_refs, indent=2, sort_keys=False)}"
            )

        tool_choice: Literal["required"] | AnyTool = forced if forced is not None else "required"
        if len(allowed) == 1:
            tool_choice = allowed[0]

        return RequirementAgentRequest(
            tools=all_tools,
            allowed_tools=allowed,
            reason_by_tool=reason_by_tool,
            tool_choice=tool_choice if isinstance(tool_choice, Tool) or force_tool_call or prevent_stop else "auto",
            final_answer=self.final_answer,
            hidden_tools=hidden,
            can_stop=not prevent_stop,
        )


def _create_system_message(
    *, template: PromptTemplate[RequirementAgentSystemPromptInput], request: RequirementAgentRequest
) -> SystemMessage:
    return SystemMessage(
        template.render(
            tools=[
                RequirementAgentToolTemplateDefinition.from_tool(
                    tool,
                    allowed=tool in request.allowed_tools,
                    reason=request.reason_by_tool.get(tool, None),
                )
                for tool in request.tools
                if tool not in request.hidden_tools
            ],
            final_answer_name=request.final_answer.name,
            final_answer_schema=to_json(
                request.final_answer.input_schema.model_json_schema(mode="validation"), indent=2, sort_keys=False
            )
            if request.final_answer.custom_schema
            else None,
            final_answer_instructions=request.final_answer.instructions,
        )
    )


class RuleEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rule: Rule
    requirement: Requirement[RequirementAgentRunState] | None
    priority: int
