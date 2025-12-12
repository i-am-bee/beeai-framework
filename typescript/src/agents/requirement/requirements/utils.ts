/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AnyTool, AnyToolClass } from "@/tools/base.js";

export type TargetType = string | AnyTool | AnyToolClass;
export type MultiTargetType = TargetType | TargetType[];

/**
 * Extract target name from various input types
 */
export function extractTargetName(target: TargetType): string {
  if (typeof target === "string") {
    return target;
  }
  if (typeof target === "function") {
    return target.name;
  }
  return target.name;
}

/**
 * Extract set of target names from various input types
 */
export function extractTargets(targets?: MultiTargetType): Set<string> {
  if (!targets) {
    return new Set();
  }

  const targetArray = Array.isArray(targets) ? targets : [targets];
  return new Set(targetArray.map(extractTargetName));
}

/**
 * Check if a tool matches any target in the set
 */
export function targetSeenIn(
  tool: AnyTool | null | undefined,
  targets: Set<string>,
): string | null {
  if (!tool) {
    return null;
  }

  for (const target of targets) {
    if (tool.name === target || tool.constructor.name === target) {
      return target;
    }
  }

  return null;
}

/**
 * Assert that all targets exist in the tools list
 */
export function assertAllRulesFound(targets: Set<string>, tools: AnyTool[]): void {
  const toolNames = new Set(tools.map((t) => t.name));
  const toolClassNames = new Set(tools.map((t) => t.constructor.name));

  for (const target of targets) {
    if (!toolNames.has(target) && !toolClassNames.has(target)) {
      throw new Error(
        `Target '${target}' not found in tools. Available: ${Array.from(toolNames).join(", ")}`,
      );
    }
  }
}
