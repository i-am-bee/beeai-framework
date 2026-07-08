/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { ToolCallPart } from "ai";

export interface ToolCallCheckerConfig {
  maxStrikeLength?: number;
  maxTotalOccurrences?: number;
  windowSize?: number;
}

/**
 * Detects cycles in tool calls
 */
export class ToolCallChecker {
  protected config: Required<ToolCallCheckerConfig>;
  protected strikeHistory: ToolCallPart[] = [];
  protected occurrencesHistory: ToolCallPart[] = [];
  public cycleFound = false;
  public enabled = true;

  constructor(config: ToolCallCheckerConfig = {}) {
    this.config = {
      maxStrikeLength: config.maxStrikeLength ?? 1,
      maxTotalOccurrences: config.maxTotalOccurrences ?? 5,
      windowSize: config.windowSize ?? 10,
    };
  }

  register(value: ToolCallPart): void {
    if (!this.enabled) {
      return;
    }

    // Check for consecutive strikes
    const lastCall = this.strikeHistory.at(-1);
    if (lastCall && this.isSameToolCall(lastCall, value)) {
      this.strikeHistory.push(value);
      if (this.strikeHistory.length > this.config.maxStrikeLength) {
        this.cycleFound = true;
      }
    } else {
      this.strikeHistory = [value];
    }

    // Check for total occurrences in window
    this.occurrencesHistory.push(value);
    if (this.occurrencesHistory.length > this.config.windowSize) {
      this.occurrencesHistory.shift();
    }

    const occurrences = this.occurrencesHistory.filter((call) =>
      this.isSameToolCall(call, value),
    ).length;
    if (occurrences > this.config.maxTotalOccurrences) {
      this.cycleFound = true;
    }
  }

  reset(current?: ToolCallPart): void {
    this.strikeHistory = current ? [current] : [];
    this.occurrencesHistory = current ? [current] : [];
    this.cycleFound = false;
  }

  protected isSameToolCall(a: ToolCallPart, b: ToolCallPart): boolean {
    return a.toolName === b.toolName && JSON.stringify(a.input) === JSON.stringify(b.input);
  }
}
