/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AnyTool, Tool } from "@/tools/base.js";
import { RunContext } from "@/context.js";
import { AgentError } from "@/agents/base.js";
import { SystemMessage } from "@/backend/message.js";
import { PromptTemplate } from "@/template.js";
import {
  RequirementAgentRequest,
  RequirementAgentRunState,
  RequirementAgentSystemPromptInputSchema,
} from "@/agents/requirement/types.js";
import {
  Requirement,
  RequirementCallbacks,
  Rule,
} from "@/agents/requirement/requirements/requirement.js";
import { FinalAnswerTool } from "./tool.js";
import { ChatModelToolChoice } from "@/backend/chat.js";
import { toCamelCase } from "remeda";

interface RuleEntry {
  rule: Rule;
  requirement: Requirement | null;
  priority: number;
}

/**
 * Manages requirements and generates requests for the agent
 */
export class RequirementsReasoner {
  protected tools: AnyTool[];
  protected entries: Requirement[] = [];
  protected context: RunContext<any>;
  public readonly finalAnswer: FinalAnswerTool;

  constructor(tools: AnyTool[], finalAnswer: FinalAnswerTool, context: RunContext<any>) {
    this.tools = [...tools, finalAnswer];
    this.finalAnswer = finalAnswer;
    this.context = context;
  }

  async update(requirements: Requirement[]): Promise<void> {
    this.entries = [];

    for (const requirement of requirements) {
      this.entries.push(requirement);
    }

    for (const entry of this.entries) {
      const emitter = this.context.emitter.child<RequirementCallbacks>({
        groupId: toCamelCase(entry.name),
        creator: entry,
      });
      emitter.namespace.push("requirement");

      const tools = [...this.tools];
      await emitter.emit("init", { tools });
      await entry.init(tools, this.context);
    }
  }

  protected findToolByName(name: string): AnyTool {
    const tool = this.tools.find((t) => t.name === name);
    if (!tool) {
      throw new Error(`Tool '${name}' not found in (${this.tools.map((t) => t.name).join(", ")}).`);
    }
    return tool;
  }

  async createRequest(
    state: RequirementAgentRunState,
    forceTool: boolean,
    extraRules: Rule[] = [],
  ): Promise<RequirementAgentRequest> {
    const hidden: AnyTool[] = [];
    const allowed: AnyTool[] = [];
    const allTools: AnyTool[] = [...this.tools];
    const reasonByTool = new WeakMap<AnyTool, string | undefined>();

    let preventStop = false;
    const preventStepRefs: RuleEntry[] = [];

    let forced: AnyTool | null = null;
    let forcedLevel = 0;
    const rulesByTool = new Map<string, RuleEntry[]>();

    // Initialize rules map
    for (const tool of this.tools) {
      rulesByTool.set(tool.name, []);
    }

    // Group rules from requirements
    for (const requirement of this.entries.filter((e) => e.enabled)) {
      const generatedRules = await requirement.run(state);
      for (const rule of generatedRules) {
        const tool = this.findToolByName(rule.target);
        const rules = rulesByTool.get(tool.name) || [];
        rules.push({
          priority: requirement.priority,
          rule,
          requirement,
        });
        rulesByTool.set(tool.name, rules);
      }
    }

    // Add extra rules
    for (const rule of extraRules) {
      if (!rulesByTool.has(rule.target)) {
        throw new Error(`Tool '${rule.target}' not found.`);
      }

      const rules = rulesByTool.get(rule.target)!;
      const priority = rules.length > 0 ? Math.max(...rules.map((r) => r.priority)) + 1 : 1;
      rules.push({
        priority,
        rule,
        requirement: null,
      });
    }

    // Aggregate rules and infer required tool
    for (const [toolName, rules] of rulesByTool.entries()) {
      const tool = this.findToolByName(toolName);
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
        if (!allowed.includes(tool)) {
          allowed.push(tool);
        }
        if (isForced && (!forced || forcedLevel < maxPriority)) {
          forced = tool;
          forcedLevel = maxPriority;
        }
      }
      if (isHidden) {
        if (!hidden.includes(tool)) {
          hidden.push(tool);
        }
      }
      if (isPreventStop) {
        preventStop = true;
      }
    }

    // If forced tool is set, only allow it and final answer
    if (forced) {
      allowed.length = 0;
      if (!allowed.includes(forced)) {
        allowed.push(forced);
      }
      if (!allowed.includes(this.finalAnswer as any)) {
        allowed.push(this.finalAnswer as any);
      }
    }

    // If prevent stop is set and forced is not final answer, remove final answer
    if (preventStop && forced !== this.finalAnswer) {
      const finalAnswerIndex = allowed.indexOf(this.finalAnswer);
      if (finalAnswerIndex !== -1) {
        allowed.splice(finalAnswerIndex, 1);
      }
    }

    if (allowed.length === 0) {
      throw new AgentError(
        "One of the generated rules is preventing the agent from continuing. " +
          "This indicates that the provided requirements may conflict with each other. " +
          "See the following rules and their attached requirements that are preventing the agent from continuing.\n" +
          JSON.stringify(preventStepRefs, null, 2),
      );
    }

    let toolChoice: ChatModelToolChoice = forced || "required";
    if (allowed.length === 1) {
      toolChoice = allowed[0];
    }

    // Override to auto if not forcing and not preventing stop
    if (!(toolChoice instanceof Tool) && !forceTool && !preventStop) {
      toolChoice = "auto";
    }

    return {
      tools: allTools,
      allowedTools: allowed,
      reasonByTool,
      toolChoice,
      finalAnswer: this.finalAnswer,
      hiddenTools: hidden,
      canStop: !preventStop,
    };
  }
}

/**
 * Create system message from template and request
 */
export async function createSystemMessage(
  template: PromptTemplate<typeof RequirementAgentSystemPromptInputSchema>,
  request: RequirementAgentRequest,
): Promise<SystemMessage> {
  return new SystemMessage(
    template.render({
      tools: await Promise.all(
        request.tools
          .filter((tool) => !request.hiddenTools.includes(tool))
          .map(async (tool) => ({
            name: tool.name,
            description: tool.description,
            inputSchema: JSON.stringify(await tool.inputSchema(), null, 2),
            allowed: String(request.allowedTools.includes(tool)),
            reason: request.reasonByTool.get(tool),
          })),
      ),
      finalAnswerName: request.finalAnswer.name,
      finalAnswerSchema: request.finalAnswer.customSchema
        ? JSON.stringify(request.finalAnswer.inputSchema, null, 2)
        : undefined,
      finalAnswerInstructions: request.finalAnswer.instructions, // default
      role: undefined, // default
      instructions: undefined, // default
      notes: undefined, // default
    }),
  );
}
