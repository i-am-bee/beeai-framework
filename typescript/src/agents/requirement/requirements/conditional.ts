/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Requirement, RequirementError, Rule } from "./requirement.js";
import {
  TargetType,
  MultiTargetType,
  extractTargets,
  extractTargetName,
  targetSeenIn,
  assertAllRulesFound,
} from "./utils.js";
import { AnyTool } from "@/tools/base.js";
import { RunContext } from "@/context.js";
import { ValueError } from "@/errors.js";
import type { RequirementAgentRunState } from "@/agents/requirement/types.js";
import { isTruthy } from "remeda";

export interface ConditionalRequirementOptions {
  name?: string;
  forceAtStep?: number;
  onlyBefore?: MultiTargetType;
  onlyAfter?: MultiTargetType;
  forceAfter?: MultiTargetType;
  forcePreventStop?: boolean;
  minInvocations?: number;
  maxInvocations?: number;
  onlySuccessInvocations?: boolean;
  consecutiveAllowed?: boolean;
  priority?: number;
  customChecks?: ((state: RequirementAgentRunState) => boolean)[];
  enabled?: boolean;
  reason?: string;
}

export class ConditionalRequirement extends Requirement {
  protected source: TargetType;
  protected sourceTool?: AnyTool;
  protected before: Set<string>;
  protected after: Set<string>;
  protected forceAfterSet: Set<string>;
  protected minInvocations: number;
  protected maxInvocations: number;
  protected forceAtStepValue?: number;
  protected onlySuccessInvocations: boolean;
  protected consecutiveAllowedValue: boolean;
  protected customChecks: ((state: RequirementAgentRunState) => boolean)[];
  protected forcePreventStop: boolean;
  protected reasonValue?: string;

  constructor(target: TargetType, options: ConditionalRequirementOptions = {}) {
    super(options.name || `Condition${extractTargetName(target)}`);

    this.source = target;
    this.enabled = options.enabled ?? true;

    if (options.priority !== undefined) {
      this.priority = options.priority;
    }

    this.before = extractTargets(options.onlyBefore);
    this.after = extractTargets(options.onlyAfter);
    this.forceAfterSet = extractTargets(options.forceAfter);
    this.minInvocations = options.minInvocations ?? 0;
    this.maxInvocations = options.maxInvocations ?? Infinity;
    this.forceAtStepValue = options.forceAtStep;
    this.onlySuccessInvocations = options.onlySuccessInvocations ?? true;
    this.consecutiveAllowedValue = options.consecutiveAllowed ?? true;
    this.customChecks = options.customChecks ?? [];
    this.forcePreventStop = options.forcePreventStop ?? true;
    this.reasonValue = options.reason;

    this.checkInvariant();
  }

  protected checkInvariant(): void {
    if (this.minInvocations < 0) {
      throw new ValueError("The 'minInvocations' argument must be non negative!");
    }

    if (this.maxInvocations < 0) {
      throw new ValueError("The 'maxInvocations' argument must be non negative!");
    }

    if (this.minInvocations > this.maxInvocations) {
      throw new ValueError(
        "The 'minInvocations' argument must be less than or equal to 'maxInvocations'!",
      );
    }

    const sourceName = extractTargetName(this.source);
    if (this.before.has(sourceName)) {
      throw new ValueError(`Referencing self in 'before' is not allowed (${sourceName})!`);
    }

    if (this.forceAfterSet.has(sourceName)) {
      throw new ValueError(`Referencing self in 'forceAfter' is not allowed (${sourceName})!`);
    }

    const beforeAfterIntersection = new Set([...this.before].filter((x) => this.after.has(x)));
    if (beforeAfterIntersection.size > 0) {
      throw new ValueError(
        `Tool specified as 'before' and 'after' at the same time: ${Array.from(beforeAfterIntersection).join(", ")}!`,
      );
    }

    const beforeForceAfterIntersection = new Set(
      [...this.before].filter((x) => this.forceAfterSet.has(x)),
    );
    if (beforeForceAfterIntersection.size > 0) {
      throw new ValueError(
        `Tool specified as 'before' and 'forceAfter' at the same time: ${Array.from(beforeForceAfterIntersection).join(", ")}!`,
      );
    }

    if (this.forceAtStepValue !== undefined && this.forceAtStepValue < 1) {
      throw new ValueError("The 'forceAtStep' argument must be >= 1!");
    }
  }

  async init(tools: AnyTool[], ctx: RunContext<any>): Promise<void> {
    await super.init(tools, ctx);

    const allTargets = new Set([
      ...this.before,
      ...this.after,
      ...this.forceAfterSet,
      extractTargetName(this.source),
    ]);
    assertAllRulesFound(allTargets, tools);

    for (const tool of tools) {
      if (targetSeenIn(tool, new Set([extractTargetName(this.source)]))) {
        if (this.sourceTool && this.sourceTool !== tool) {
          throw new ValueError(
            `More than one occurrence of ${extractTargetName(this.source)} has been found!`,
          );
        }
        this.sourceTool = tool;
      }
    }

    if (!this.sourceTool) {
      throw new ValueError(`Source tool ${extractTargetName(this.source)} was not found!`);
    }

    if (targetSeenIn(this.sourceTool, this.before)) {
      throw new ValueError(`Referencing self in 'before' is not allowed: ${this.sourceTool.name}!`);
    }

    if (this.consecutiveAllowedValue && targetSeenIn(this.sourceTool, this.forceAfterSet)) {
      throw new ValueError(
        `Referencing self in 'forceAfter' is not allowed: ${this.sourceTool.name}. ` +
          `It would prevent an infinite loop. Consider setting 'consecutiveAllowed' to false.`,
      );
    }
  }

  reset(): this {
    this.before.clear();
    this.after.clear();
    this.forceAfterSet.clear();
    return this;
  }

  async run(state: RequirementAgentRunState): Promise<Rule[]> {
    const sourceTool = this.sourceTool;
    if (!sourceTool) {
      throw new RequirementError("Source was not found!", this);
    }

    // Get steps from state (assuming state has steps property)
    const steps = this.onlySuccessInvocations
      ? state.steps.filter((step) => !step.error)
      : state.steps;

    const lastStepTool = steps.at(-1)?.tool;

    const invocations = steps.filter((step) => step.tool === sourceTool).length;

    const resolve = (allowed: boolean): Rule[] => {
      const currentStep = steps.length + 1;
      if (!allowed && this.forceAtStepValue === currentStep) {
        throw new RequirementError(
          `Tool '${sourceTool.name}' cannot be executed at step ${this.forceAtStepValue} ` +
            `because it has not met all requirements.`,
          this,
        );
      }

      const forced = allowed
        ? Boolean(
            targetSeenIn(lastStepTool, this.forceAfterSet) || this.forceAtStepValue === currentStep,
          )
        : false;

      return [
        {
          target: sourceTool.name,
          allowed,
          forced,
          hidden: false,
          preventStop: this.minInvocations > invocations || (forced && this.forcePreventStop),
          reason: !allowed ? this.reasonValue : undefined,
        },
      ];
    };

    // Check consecutive constraint
    if (!this.consecutiveAllowedValue && sourceTool === lastStepTool) {
      return resolve(false);
    }

    // Check max invocations
    if (invocations >= this.maxInvocations) {
      return resolve(false);
    }

    // Check after/before constraints
    if (this.after.size > 0) {
      const stepsAsToolCalls = steps.map((step) => step.tool).filter(isTruthy);
      const afterToolsRemaining = new Set(this.after);

      for (const stepTool of stepsAsToolCalls) {
        if (targetSeenIn(stepTool, this.forceAfterSet)) {
          return resolve(false);
        }

        const matcher = targetSeenIn(stepTool as any, this.after);
        if (matcher) {
          afterToolsRemaining.delete(matcher);
        }
      }

      if (afterToolsRemaining.size > 0) {
        return resolve(false);
      }
    }

    // Check custom checks
    for (const check of this.customChecks) {
      if (!check(state)) {
        return resolve(false);
      }
    }

    return resolve(true);
  }

  async clone(): Promise<this> {
    const instance = await super.clone();
    instance.before = new Set(this.before);
    instance.after = new Set(this.after);
    instance.forceAfterSet = new Set(this.forceAfterSet);
    instance.minInvocations = this.minInvocations;
    instance.maxInvocations = this.maxInvocations;
    instance.customChecks = [...this.customChecks];
    instance.onlySuccessInvocations = this.onlySuccessInvocations;
    instance.forceAtStepValue = this.forceAtStepValue;
    instance.consecutiveAllowedValue = this.consecutiveAllowedValue;
    instance.source = this.source;
    instance.sourceTool = this.sourceTool;
    instance.forcePreventStop = this.forcePreventStop;
    instance.reasonValue = this.reasonValue;
    return instance;
  }
}
