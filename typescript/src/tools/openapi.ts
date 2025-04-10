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

import { join } from "path";

import {
  BaseToolOptions,
  BaseToolRunOptions,
  StringToolOutput,
  Tool,
  ToolError,
  ToolEmitter,
} from "@/tools/base.js";
import { Callback, Emitter } from "@/emitter/emitter.js";
import { GetRunContext } from "@/context.js";
import { ValueError } from "@/errors.js";
import { SchemaObject } from "ajv";
import { parse } from "yaml";
import { isEmpty, isTruthy, clone, toCamelCase } from "remeda";

export interface OpenAPIToolOptions extends BaseToolOptions {
  name?: string;
  description?: string;
  openApiSchema: any;
  url?: string;
  fetchOptions?: RequestInit;
}

export interface OpenAPIEvents {
  beforeFetch: Callback<{ options: RequestInit; url: URL }>;
  afterFetch: Callback<{ data: OpenAPIToolOutput; url: URL }>;
}

export class OpenAPIToolOutput extends StringToolOutput {
  constructor(
    public readonly status: number,
    public readonly statusText: string,
    public readonly result = "",
  ) {
    super();
    this.status = status;
    this.statusText = statusText;
    this.result = result ?? "";
  }
}

export class OpenAPITool extends Tool<OpenAPIToolOutput, OpenAPIToolOptions> {
  public readonly name: string;
  public readonly description: string;
  public readonly url: string;

  public readonly openApiSchema: any;

  inputSchema(): SchemaObject {
    const getReferencedObject = (json: any, refPath: string) => {
      const pathSegments = refPath.split("/");
      let currentObject = json;
      for (const segment of pathSegments) {
        if (segment === "#") {
          continue;
        }
        currentObject = currentObject[segment];
      }
      return currentObject;
    };

    return {
      type: "object",
      required: ["path", "method"],
      oneOf: Object.entries(this.openApiSchema.paths).flatMap(([path, pathSpec]: [string, any]) =>
        Object.entries(pathSpec).map(([method, methodSpec]: [string, any]) => ({
          additionalProperties: false,
          properties: {
            path: {
              const: path,
              description:
                "Do not replace variables in path, instead of, put them to the parameters object.",
            },
            method: { const: method, description: methodSpec.summary || methodSpec.description },
            ...(methodSpec.requestBody?.content?.["application/json"]?.schema
              ? {
                  body: methodSpec.requestBody?.content?.["application/json"]?.schema,
                }
              : {}),
            ...(methodSpec.parameters
              ? {
                  parameters: {
                    type: "object",
                    additionalProperties: false,
                    required: methodSpec.parameters
                      .filter((p: any) => p.required === true)
                      .map((p: any) => p.name),
                    properties: methodSpec.parameters.reduce(
                      (acc: any, p: any) =>
                        !p.$ref
                          ? { ...acc, [p.name]: { ...p.schema, description: p.name } }
                          : {
                              ...acc,
                              [getReferencedObject(this.openApiSchema, p.$ref)?.name]: {
                                ...getReferencedObject(this.openApiSchema, p.$ref)?.schema,
                                description: getReferencedObject(this.openApiSchema, p.$ref)?.name,
                              },
                            },
                      {},
                    ),
                  },
                }
              : {}),
          },
        })),
      ),
    } as const satisfies SchemaObject;
  }

  public readonly emitter: ToolEmitter<Record<string, any>, OpenAPIToolOutput, OpenAPIEvents>;

  static {
    this.register();
  }

  public constructor(options: OpenAPIToolOptions) {
    super(options);

    this.openApiSchema = parse(options.openApiSchema);

    this.url = this.options.url || this.openApiSchema.servers?.find((s: any) => s?.url)?.url;
    if (!this.url) {
      throw new ValueError(
        "OpenAPI schema hasn't any server with url specified. Pass it manually.",
      );
    }

    this.name = options.name ?? this.openApiSchema?.info?.title?.trim();
    if (!this.name) {
      throw new ValueError("OpenAPI schema hasn't 'name' specified. Pass it manually.");
    }
    this.emitter = Emitter.root.child({
      namespace: ["tool", "web", "openAPI", toCamelCase(this.name)],
      creator: this,
    });

    this.description =
      options.description ||
      this.openApiSchema?.info?.description ||
      "Performs REST API requests to the servers and retrieves the response. The server API interfaces are defined in OpenAPI schema. \n" +
        "Only use the OpenAPI tool if you need to communicate to external servers.";

    if (!this.description) {
      throw new ValueError("OpenAPI schema hasn't 'description' specified. Pass it manually.");
    }
  }

  protected async _run(
    input: Record<string, any>,
    _options: Partial<BaseToolRunOptions>,
    run: GetRunContext<typeof this>,
  ) {
    let path: string = input.path || "";
    const url = new URL(this.url);
    Object.keys(input.parameters ?? {}).forEach((key) => {
      const value = input.parameters[key];
      const newPath = path.replace(`{${key}}`, value);
      if (newPath == path) {
        url.searchParams.append(key, value);
      } else {
        path = newPath;
      }
    });
    url.pathname = join(url.pathname, path);
    const options: RequestInit = {
      ...this.options.fetchOptions,
      signal: AbortSignal.any([run.signal, this.options.fetchOptions?.signal].filter(isTruthy)),
      body: !isEmpty(input.body) ? input.body : undefined,
      method: input.method.toLowerCase(),
      headers: { Accept: "application/json", ...this.options.fetchOptions?.headers },
    };
    await run.emitter.emit("beforeFetch", { options: options, url: url });
    try {
      const response = await fetch(url.toString(), options);
      const text = await response.text();
      const output = new OpenAPIToolOutput(response.status, response.statusText, text);
      await run.emitter.emit("afterFetch", { data: output, url });
      return output;
    } catch (err) {
      throw new ToolError(`Request to ${url} has failed.`, [err]);
    }
  }
  createSnapshot() {
    return {
      ...super.createSnapshot(),
      openApiSchema: clone(this.openApiSchema),
      name: this.name,
      description: this.description,
      url: this.url,
    };
  }
}
