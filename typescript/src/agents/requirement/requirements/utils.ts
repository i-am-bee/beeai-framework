/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AnyTool, AnyToolClass, Tool } from "@/tools/base.js";
import { ValueError } from "@/errors.js";
import { isString } from "remeda";
import { isConstructor } from "@/internals/helpers/prototype.js";

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
export function extractTargets(targets?: MultiTargetType): Set<TargetType> {
  if (!targets) {
    return new Set();
  }

  const targetArray = Array.isArray(targets) ? targets : [targets];
  return new Set(targetArray);
}

/**
 * Check if a tool matches any target in the set
 */
export function targetSeenIn(
  target: AnyTool | null | undefined,
  haystack: Set<TargetType> | TargetType,
): TargetType | null {
  if (!target) {
    return null;
  }

  if (!(haystack instanceof Set)) {
    haystack = new Set([haystack]);
  }

  for (const needle of haystack.values()) {
    if (isString(needle) && needle === target.name) {
      return needle;
    }
    if (needle instanceof Tool && needle === target) {
      return needle;
    }
    if (isConstructor(needle) && target instanceof needle) {
      return needle;
    }
  }

  return null;
}

/**
 * Assert that all targets exist in the tools list
 */
export function assertAllRulesFound(targets: Set<TargetType>, tools: AnyTool[]): void {
  for (const target of targets) {
    let found = false;
    for (const tool of tools) {
      if (targetSeenIn(tool, target)) {
        found = true;
        break;
      }
    }
    if (!found) {
      throw new ValueError(
        `Tool '${target}' is specified as 'source', 'before', 'after' or 'force_after' but not found.`,
      );
    }
  }
}
