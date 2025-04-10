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

import {
  BaseToolOptions,
  BaseToolRunOptions,
  ToolEmitter,
  StringToolOutput,
  Tool,
  ToolInput,
} from "@/tools/base.js";
import { FrameworkError } from "@/errors.js";
import { z } from "zod";
import { validate } from "@/internals/helpers/general.js";
import { callCodeInterpreter, CodeInterpreterOptions } from "./python/python.js";
import { RunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";
import { ToolError } from "@/tools/base.js";
import { merge } from "remeda";
import { omitUndefined } from "@/internals/helpers/object.js";

export class CustomToolCreateError extends FrameworkError {}
export class CustomToolExecuteError extends FrameworkError {}

const toolOptionsSchema = z
  .object({
    codeInterpreter: z.object({
      url: z.string().url(),
      connectionOptions: z.any().optional(),
    }),
    sourceCode: z.string().min(1),
    name: z.string().min(1),
    description: z.string().min(1),
    inputSchema: z.any(),
  })
  .passthrough();

type CustomToolEnv = Record<string, string | boolean | null | undefined | number>;

export type CustomToolOptions = z.output<typeof toolOptionsSchema> &
  BaseToolOptions & { env?: CustomToolEnv };

export type CustomToolRunOptions = Pick<CustomToolOptions, "env"> & BaseToolRunOptions;

export class CustomTool extends Tool<StringToolOutput, CustomToolOptions, CustomToolRunOptions> {
  name: string;
  description: string;

  public readonly emitter: ToolEmitter<ToolInput<this>, StringToolOutput> = Emitter.root.child({
    namespace: ["tool", "custom"],
    creator: this,
  });

  public inputSchema() {
    return this.options.inputSchema;
  }

  static {
    this.register();
  }

  public constructor(options: CustomToolOptions) {
    validate(options, toolOptionsSchema);
    super(options);
    this.name = options.name;
    this.description = options.description;
  }

  protected async _run(
    input: any,
    options: Partial<CustomToolRunOptions>,
    run: RunContext<typeof this>,
  ) {
    try {
      const result = await callCodeInterpreter({
        url: `${this.options.codeInterpreter.url}/v1/execute-custom-tool`,
        body: {
          tool_source_code: this.options.sourceCode,
          tool_input_json: JSON.stringify(input),
          env: merge(omitUndefined(this.options.env ?? {}), omitUndefined(options.env ?? {})),
        },
        signal: run.signal,
      });

      if (result.stderr) {
        throw new CustomToolExecuteError(result.stderr);
      }

      return new StringToolOutput(result.tool_output_json);
    } catch (e) {
      if (e instanceof ToolError) {
        throw new CustomToolExecuteError(e.message, [e]);
      } else {
        throw e;
      }
    }
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>): void {
    super.loadSnapshot(snapshot);
  }

  static async fromSourceCode(
    { env, ...codeInterpreter }: CodeInterpreterOptions & Pick<CustomToolOptions, "env">,
    sourceCode: string,
  ) {
    try {
      const result = await callCodeInterpreter({
        url: `${codeInterpreter.url}/v1/parse-custom-tool`,
        body: {
          tool_source_code: sourceCode,
        },
      });

      if (result.error_messages) {
        throw new CustomToolCreateError(result.error_messages.join("\n"));
      }

      return new CustomTool({
        codeInterpreter,
        sourceCode,
        name: result.tool_name,
        description: result.tool_description,
        inputSchema: JSON.parse(result.tool_input_schema_json),
        env,
      });
    } catch (e) {
      if (e instanceof ToolError) {
        throw new CustomToolCreateError(e.message, [e]);
      } else {
        throw e;
      }
    }
  }
}
