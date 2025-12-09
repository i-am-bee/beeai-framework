// Copyright 2025 © BeeAI a Series of LF Projects, LLC
// SPDX-License-Identifier: Apache-2.0

import {
  RequirementAgentSystemPromptInput,
  RequirementAgentToolTemplateDefinition,
} from "@/agents/requirement/prompts.js";
import {
  RequirementInitEvent,
  requirementEventTypes,
} from "@/agents/requirement/requirements/events";
import { Requirement, Rule } from "@/agents/requirement/requirements/requirement";
import { RequirementAgentRequest, RequirementAgentRunState } from "@/agents/requirement/types.js";
import { FinalAnswerTool } from "@/agents/requirement/utils/_tool";
import { AgentError } from "@/agents/errors";
import { SystemMessage } from "@/backend";
import { RunContext } from "@/context.js";
import { PromptTemplate } from "@/template.js";
import { AnyTool, Tool } from "@/tools";
import { appendIfNotExists, removeByReference } from "@/utils/lists";
import { toJson, toSafeWord } from "@/utils/strings";
import { RequirementAgent } from "@/agents/requirement/agent.js";

export class RequirementsReasoner {
  private _tools: AnyTool[];
  private _entries: Requirement<RequirementAgentRunState>[] = [];
  private _context: RunContext<RequirementAgent>;
  public readonly finalAnswer: FinalAnswerTool;

  constructor(options: {
    tools: readonly AnyTool[];
    finalAnswer: FinalAnswerTool;
    context: RunContext<RequirementAgent>;
  }) {
    this._tools = [...options.tools, options.finalAnswer];
    this._context = options.context;
    this.finalAnswer = options.finalAnswer;
  }

  async update(requirements: readonly Requirement<RequirementAgentRunState>[]): Promise<void> {
    this._entries = [];

    for (const requirement of requirements) {
      this._entries.push(requirement);
    }

    for (const entry of this._entries) {
      const emitter = this._context.emitter.child({
        groupId: toSafeWord(entry.name),
        creator: entry,
        events: requirementEventTypes,
      });
      emitter.namespace.push("requirement");

      const tools = [...this._tools];
      await emitter.emit("init", new RequirementInitEvent({ tools }));
      await entry.init({ tools, ctx: this._context });
    }
  }

  private _findToolByName(name: string): AnyTool {
    const tool = this._tools.find((t) => t.name === name);
    if (!tool) {
      const toolNames = this._tools.map((t) => t.name).join(",");
      throw new Error(`Tool '${name}' not found in (${toolNames}).`);
    }
    return tool;
  }

  async createRequest(
    state: RequirementAgentRunState,
    options: {
      forceToolCall: boolean;
      extraRules?: Rule[];
    },
  ): Promise<RequirementAgentRequest> {
    const hidden: AnyTool[] = [];
    const allowed: AnyTool[] = [];
    const allTools: AnyTool[] = [...this._tools];
    const reasonByTool = new WeakMap<AnyTool, string | null>();

    let preventStop = false;
    const preventStepRefs: RuleEntry[] = [];

    let forced: AnyTool | null = null;
    let forcedLevel = 0;
    const rulesByTool: Record<string, RuleEntry[]> = Object.fromEntries(
      this._tools.map((t) => [t.name, []]),
    );

    // Group rules
    for (const requirement of this._entries.filter((entry) => entry.enabled)) {
      const generatedRules = await requirement.run(state);
      for (const rule of generatedRules) {
        const tool = this._findToolByName(rule.target);
        rulesByTool[tool.name].push({
          priority: requirement.priority,
          rule,
          requirement,
        });
      }
    }

    // Add extra rules
    for (const rule of options.extraRules || []) {
      if (!(rule.target in rulesByTool)) {
        throw new Error(`Tool '${rule.target}' not found.`);
      }

      const rules = rulesByTool[rule.target];
      const priority = rules.length > 0 ? Math.max(...rules.map((v) => v.priority)) + 1 : 1;
      rules.push({
        priority,
        rule,
        requirement: null,
      });
    }

    // Aggregate rules and infer the required tool
    for (const [toolName, rules] of Object.entries(rulesByTool)) {
      const tool = this._findToolByName(toolName);
      rules.sort((a, b) => b.priority - a.priority); // DESC

      const maxPriority = rules.length > 0 ? rules[0].priority : 1;
      let isAllowed = true;
      let isForced = false;
      let isHidden = false;
      let isPreventStop = false;

      for (const ruleEntry of rules) {
        const rule = ruleEntry.rule;
        if (!rule.allowed) {
          isAllowed = false;
        }
        if (rule.hidden) {
          isHidden = true;
        }
        if (rule.forced) {
          isForced = true;
        }
        if (rule.preventStop) {
          isPreventStop = true;
          preventStepRefs.push(ruleEntry);
        }
        if (rule.reason) {
          reasonByTool.set(tool, rule.reason);
        }
      }

      if (isAllowed && isHidden) {
        isAllowed = false;
      }

      if (isAllowed) {
        appendIfNotExists(allowed, tool);
        if (isForced && (!forced || forcedLevel < maxPriority)) {
          forced = tool;
          forcedLevel = maxPriority;
        }
      }
      if (isHidden) {
        appendIfNotExists(hidden, tool);
      }
      if (isPreventStop) {
        preventStop = true;
      }
    }

    if (forced) {
      allowed.length = 0;
      appendIfNotExists(allowed, forced);
      appendIfNotExists(allowed, this.finalAnswer);
    }

    if (preventStop && !(forced instanceof FinalAnswerTool)) {
      try {
        removeByReference(allowed, this.finalAnswer);
      } catch {
        // Suppress error equivalent to contextlib.suppress(ValueError)
      }
    }

    if (allowed.length === 0) {
      const preventStepRefsJson = toJson(preventStepRefs, {
        indent: 2,
        sortKeys: false,
      });
      throw new AgentError(
        "One of the generated rules is preventing the agent from continuing. " +
          "This indicates that the provided requirements may conflict with each other. " +
          "See the following rules and their attached requirements that are preventing the agent from continuing." +
          `\n${preventStepRefsJson}`,
      );
    }

    let toolChoice: "required" | AnyTool = forced || "required";
    if (allowed.length === 1) {
      toolChoice = allowed[0];
    }

    return {
      tools: allTools,
      allowedTools: allowed,
      reasonByTool,
      toolChoice:
        toolChoice instanceof Tool || options.forceToolCall || preventStop ? toolChoice : "auto",
      finalAnswer: this.finalAnswer,
      hiddenTools: hidden,
      canStop: !preventStop,
    };
  }
}

export function createSystemMessage(options: {
  template: PromptTemplate<RequirementAgentSystemPromptInput>;
  request: RequirementAgentRequest;
}): SystemMessage {
  const { template, request } = options;

  return new SystemMessage(
    template.render({
      tools: request.tools
        .filter((tool) => !request.hiddenTools.includes(tool))
        .map((tool) =>
          RequirementAgentToolTemplateDefinition.fromTool(tool, {
            allowed: request.allowedTools.includes(tool),
            reason: request.reasonByTool.get(tool) || null,
          }),
        ),
      finalAnswerName: request.finalAnswer.name,
      finalAnswerSchema: request.finalAnswer.customSchema
        ? toJson(
            request.finalAnswer.inputSchema.modelJsonSchema({
              mode: "validation",
            }),
            { indent: 2, sortKeys: false },
          )
        : null,
      finalAnswerInstructions: request.finalAnswer.instructions,
    }),
  );
}

export interface RuleEntry {
  rule: Rule;
  requirement: Requirement<RequirementAgentRunState> | null;
  priority: number;
}
