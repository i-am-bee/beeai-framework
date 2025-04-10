/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { ToolEmitter, StringToolOutput, Tool, ToolInput } from "@/tools/base.js";
import { z } from "zod";
import { create, all, evaluate, ImportOptions, ImportObject, ConfigOptions } from "mathjs";
import { Emitter } from "@/emitter/emitter.js";

export interface CalculatorToolInput {
  config?: ConfigOptions;
  imports?: {
    entries: ImportObject | ImportObject[];
    options?: ImportOptions;
  };
}

/**
 * Waring: The CalculatorTool enbales the agent (and by proxy the user) to execute arbirtary
 * expressions via mathjs.
 *
 * Please consider the security and stability risks documented at
 * https://mathjs.org/docs/expressions/security.html before using this tool.
 */
export class CalculatorTool extends Tool<StringToolOutput> {
  name = "Calculator";
  description = `A calculator tool that performs basic arithmetic operations like addition, subtraction, multiplication, and division. 
Only use the calculator tool if you need to perform a calculation.`;

  public readonly emitter: ToolEmitter<ToolInput<this>, StringToolOutput> = Emitter.root.child({
    namespace: ["tool", "calculator"],
    creator: this,
  });

  inputSchema() {
    return z.object({
      expression: z
        .string()
        .min(1)
        .describe(
          `The mathematical expression to evaluate (e.g., "2 + 3 * 4"). Use Mathjs basic expression syntax. Constants only.`,
        ),
    });
  }

  protected limitedEvaluate: typeof evaluate;

  constructor({ config, imports, ...options }: CalculatorToolInput = {}) {
    super(options);
    const math = create(all, config);
    this.limitedEvaluate = math.evaluate;
    // Disable use of potentially vulnerable functions
    math.import(
      {
        // most important (hardly any functional impact)
        import: function () {
          throw new Error("Function import is disabled");
        },
        createUnit: function () {
          throw new Error("Function createUnit is disabled");
        },
        reviver: function () {
          throw new Error("Function reviver is disabled");
        },

        // extra (has functional impact)
        evaluate: function () {
          throw new Error("Function evaluate is disabled");
        },
        parse: function () {
          throw new Error("Function parse is disabled");
        },
        simplify: function () {
          throw new Error("Function simplify is disabled");
        },
        derivative: function () {
          throw new Error("Function derivative is disabled");
        },
        resolve: function () {
          throw new Error("Function resolve is disabled");
        },
      },
      { override: true, ...imports?.options },
    );
  }

  protected async _run({ expression }: ToolInput<this>) {
    const result = this.limitedEvaluate(expression);
    return new StringToolOutput(result);
  }
}
