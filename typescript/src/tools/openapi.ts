/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
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
import { isTruthy, clone, toCamelCase } from "remeda";

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

  protected getReferencedObject(refPath: string) {
    let currentObject = this.openApiSchema;
    for (const segment of refPath.split("/")) {
      if (segment === "#") {
        continue;
      }
      currentObject = currentObject[segment];
    }
    return currentObject;
  }

  protected getParameters(path: string, method: string): any[] {
    const pathSpec = this.openApiSchema.paths?.[path];
    const parameters = new Map<string, any>();
    for (const parameter of [
      ...(pathSpec?.parameters ?? []),
      ...(pathSpec?.[method]?.parameters ?? []),
    ]) {
      const resolved = parameter.$ref ? this.getReferencedObject(parameter.$ref) : parameter;
      // An operation overrides a shared parameter with the same name and location.
      parameters.set(JSON.stringify([resolved.name, resolved.in]), resolved);
    }
    return Array.from(parameters.values());
  }

  inputSchema(): SchemaObject {
    return {
      type: "object",
      required: ["path", "method"],
      oneOf: Object.entries(this.openApiSchema.paths).flatMap(([path, pathSpec]: [string, any]) =>
        Object.entries(pathSpec)
          .filter(([method]) =>
            ["get", "put", "post", "delete", "options", "head", "patch", "trace"].includes(method),
          )
          .map(([method, methodSpec]: [string, any]) => {
            const parameters = this.getParameters(path, method);
            return {
              additionalProperties: false,
              properties: {
                path: {
                  const: path,
                  description:
                    "Do not replace variables in path, instead of, put them to the parameters object.",
                },
                method: {
                  const: method,
                  description: methodSpec.summary || methodSpec.description,
                },
                ...(methodSpec.requestBody?.content?.["application/json"]?.schema
                  ? {
                      body: methodSpec.requestBody?.content?.["application/json"]?.schema,
                    }
                  : {}),
                ...(parameters.length > 0
                  ? {
                      parameters: {
                        type: "object",
                        additionalProperties: false,
                        required: parameters
                          .filter((p: any) => p.required === true)
                          .map((p: any) => p.name),
                        properties: Object.fromEntries(
                          parameters.map((p: any) => [
                            p.name,
                            { ...p.schema, description: p.name },
                          ]),
                        ),
                      },
                    }
                  : {}),
              },
            };
          }),
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
    const headers = new Headers(this.options.fetchOptions?.headers);
    const cookies: string[] = [];
    const parameters = this.getParameters(path, input.method.toLowerCase());
    const locations = new Map<string, string>(
      parameters.map((parameter: any) => [parameter.name, parameter.in]),
    );
    Object.keys(input.parameters ?? {}).forEach((key) => {
      const value = input.parameters[key];
      const placeholder = `{${key}}`;
      const location = locations.get(key) ?? (path.includes(placeholder) ? "path" : "query");
      if (location === "header") {
        // These header parameters are ignored by OpenAPI; configured headers take precedence.
        if (
          !["accept", "content-type", "authorization"].includes(key.toLowerCase()) &&
          !headers.has(key)
        ) {
          headers.set(key, String(value));
        }
        return;
      }
      if (location === "cookie") {
        cookies.push(`${key}=${encodeURIComponent(String(value))}`);
        return;
      }
      if (location !== "path") {
        url.searchParams.append(key, value);
        return;
      }
      const serialized = String(value);
      // Dot segments would be normalized away and step outside the operation path.
      if (serialized === "." || serialized === "..") {
        throw new ToolError(`Invalid value for path parameter '${key}': ${serialized}`);
      }
      // Encode so the value stays inside its segment; fill every occurrence of the placeholder.
      path = path.split(placeholder).join(encodeURIComponent(serialized));
    });
    // A value can still complete a dot segment together with the template (e.g. "/{id}.." with
    // an empty id), so check the substituted path before it is normalized.
    if (path.split("/").some((segment) => segment === "." || segment === "..")) {
      throw new ToolError(`Invalid path after substituting parameters: ${path}`);
    }
    url.pathname = join(url.pathname, path);
    if (cookies.length > 0) {
      headers.set("Cookie", [headers.get("Cookie"), ...cookies].filter(isTruthy).join("; "));
    }
    if (!headers.has("Accept")) {
      headers.set("Accept", "application/json");
    }
    if (input.body !== undefined && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }
    const options: RequestInit = {
      ...this.options.fetchOptions,
      signal: AbortSignal.any([run.signal, this.options.fetchOptions?.signal].filter(isTruthy)),
      body: input.body !== undefined ? JSON.stringify(input.body) : undefined,
      method: input.method.toLowerCase(),
      headers,
    };
    await run.emitter.emit("beforeFetch", { options: options, url: url });
    let output: OpenAPIToolOutput;
    try {
      const response = await fetch(url.toString(), options);
      const text = await response.text();
      output = new OpenAPIToolOutput(response.status, response.statusText, text);
      await run.emitter.emit("afterFetch", { data: output, url });
    } catch (err) {
      throw new ToolError(`Request to ${url} has failed.`, [err]);
    }

    if (output.status < 200 || output.status >= 300) {
      // A failed response can arrive after a write has already taken effect.
      // The operation path, not the resolved URL, so a credential in the server URL is not echoed
      let message = `${String(input.method).toUpperCase()} ${input.path || ""} failed with HTTP ${output.status}`;
      if (output.result) {
        message += `: ${output.result.slice(0, 500)}`;
      }
      throw new ToolError(message, [], { isRetryable: false });
    }
    return output;
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
