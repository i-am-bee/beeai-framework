/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { z, ZodSchema } from "zod";
import { Tool, StringToolOutput, BaseToolRunOptions, BaseToolOptions } from "@/tools/base.js";
import { RunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";

const ThinkSchema = z.object({
  thoughts: z.string().describe("Precisely describe what you are thinking about."),
});

type ThinkInput = z.infer<typeof ThinkSchema>;

export interface ThinkToolOptions extends BaseToolOptions<StringToolOutput> {
  extraInstructions?: string;
  toolOutput?: string | ((input: ThinkInput) => string);
  schema?: ZodSchema;
}

export class ThinkTool extends Tool<StringToolOutput, ThinkToolOptions> {
  public name = "think";
  public description =
    "Use when you want to think through a problem, clarify your assumptions, or break down complex steps before acting or responding.";

  public readonly emitter: Emitter<any>;

  protected _inputSchema: ZodSchema;
  protected _toolOutput: string | ((input: ThinkInput) => string);
  protected _extraInstructions: string;

  constructor(options: ThinkToolOptions = {}) {
    super(options);
    this._inputSchema = options.schema ?? ThinkSchema;
    this._toolOutput = options.toolOutput ?? "OK";
    this._extraInstructions = options.extraInstructions ?? "";

    if (this._extraInstructions) {
      this.description += ` ${this._extraInstructions}`;
    }

    this.emitter = this.createEmitter();
  }

  inputSchema(): ZodSchema {
    return this._inputSchema;
  }

  async _run(
    input: ThinkInput,
    _options: BaseToolRunOptions,
    _context: RunContext<typeof this>,
  ): Promise<StringToolOutput> {
    const output =
      typeof this._toolOutput === "function" ? this._toolOutput(input) : this._toolOutput;
    return new StringToolOutput(output);
  }

  protected createEmitter(): Emitter<any> {
    return Emitter.root.child({
      namespace: ["tool", "think"],
      creator: this,
    });
  }
}
