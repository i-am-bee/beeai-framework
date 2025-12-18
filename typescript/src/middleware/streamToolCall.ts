/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Middleware, RunContext, RunInstance } from "@/context.js";
import { Callback, Emitter, EventMeta } from "@/emitter/emitter.js";
import { ChatModel, ChatModelEvents } from "@/backend/chat.js";
import { ChatModelOutput } from "@/backend/chat.js";
import { Tool } from "@/tools/base.js";
import { parseBrokenJson } from "@/internals/helpers/schema.js";
import { isPlainObject, isString } from "remeda";
import { hasProp } from "@/internals/helpers/object.js";
import { EmitterOptions, InferCallbackValue } from "@/emitter/types.js";

/**
 * Event emitted when the middleware detects an update to the target tool's arguments
 */
export interface StreamToolCallMiddlewareUpdateEvent<T = any> {
  /** The validated and structured tool input */
  outputStructured: T | null;
  /** The current value of the target field */
  output: string;
  /** The incremental change since the last update */
  delta: string;
}

/**
 * Callbacks for StreamToolCallMiddleware events
 */
export interface StreamToolCallMiddlewareCallbacks<T = any> {
  update: Callback<StreamToolCallMiddlewareUpdateEvent<T>>;
}

/**
 * Options for configuring StreamToolCallMiddleware
 */
export interface StreamToolCallMiddlewareOptions {
  /** The tool to monitor for streaming updates */
  target: Tool<any>;
  /** The field name in the tool's input schema to stream */
  key: string;
  /** Whether to apply middleware to nested run contexts */
  matchNested?: boolean;
  /** Whether to force streaming on the ChatModel */
  forceStreaming?: boolean;
}

/**
 * Middleware for handling streaming tool calls in a ChatModel.
 *
 * This middleware observes and listens to ChatModel stream updates and parses
 * the tool calls on demand so that they can be consumed as soon as possible.
 *
 * @example
 * ```typescript
 * const middleware = new StreamToolCallMiddleware({
 *   target: thinkTool,
 *   key: "thoughts",
 *   matchNested: false,
 *   forceStreaming: true,
 * });
 *
 * middleware.emitter.on("update", (event) => {
 *   console.log("Delta:", event.delta);
 *   console.log("Structured:", event.outputStructured);
 * });
 *
 * await llm.run(messages, { tools: [thinkTool] }).middleware(middleware);
 * ```
 */
export class StreamToolCallMiddleware<T = any> extends Middleware<RunInstance> {
  private readonly target: Tool<any>;
  private readonly key: string;
  private readonly matchNested: boolean;
  private readonly forceStreaming: boolean;
  private readonly cleanups: (() => void)[] = [];

  private output = new ChatModelOutput([]);
  private buffer = "";
  private delta = "";

  public readonly emitter: Emitter<StreamToolCallMiddlewareCallbacks<T>>;

  constructor(options: StreamToolCallMiddlewareOptions) {
    super();

    this.target = options.target;
    this.key = options.key;
    this.matchNested = options.matchNested ?? false;
    this.forceStreaming = options.forceStreaming ?? false;

    this.emitter = Emitter.root.child<StreamToolCallMiddlewareCallbacks<T>>({
      namespace: ["middleware", "streamToolCall"],
    });
  }

  bind(ctx: RunContext<RunInstance>): void {
    // Reset state
    this.output = new ChatModelOutput([]);
    this.buffer = "";
    this.delta = "";

    // Listen to ChatModel start event
    this.cleanups.push(
      ctx.instance.emitter.match(
        (meta) => meta.creator instanceof ChatModel && meta.name === "start",
        this.handleStart.bind(this),
        this.createEmitterOptions(),
      ),
    );

    // Listen to ChatModel newToken event
    this.cleanups.push(
      ctx.instance.emitter.match(
        (meta) => meta.creator instanceof ChatModel && meta.name === "newToken",
        this.handleNewToken.bind(this),
        this.createEmitterOptions(),
      ),
    );

    // Listen to ChatModel success event
    this.cleanups.push(
      ctx.instance.emitter.match(
        (meta) => meta.creator instanceof ChatModel && meta.name === "success",
        this.handleSuccess.bind(this),
        this.createEmitterOptions(),
      ),
    );
  }

  unbind(): void {
    // Clean up previous bindings
    while (this.cleanups.length > 0) {
      const fn = this.cleanups.shift()!;
      fn();
    }
  }

  protected createEmitterOptions(): EmitterOptions {
    return {
      matchNested: this.matchNested,
      isBlocking: true,
    };
  }

  private async process(toolName: string, args: any): Promise<void> {
    if (toolName !== this.target.name) {
      return;
    }

    const parsedArgs = isString(args) ? parseBrokenJson(args, { pair: ["{", "}"] }) : args;
    const outputStructured = (await this.target.parse(parsedArgs).catch(() => null)) as T;
    if (!outputStructured) {
      return;
    }

    let output = "";
    if (hasProp(outputStructured, this.key as keyof T)) {
      output = (outputStructured as any)[this.key] || "";
      if (!isString(output)) {
        output = JSON.stringify(output);
      }
      this.delta = output.slice(this.buffer.length);
      this.buffer = output;

      if (!this.delta) {
        return;
      }
    }

    await this.emitter.emit("update", {
      outputStructured,
      delta: this.delta,
      output,
    });
  }

  private async handleStart(
    data: InferCallbackValue<ChatModelEvents["start"]>,
    _meta: EventMeta,
  ): Promise<void> {
    if (this.forceStreaming) {
      data.input.stream = true;
      data.input.streamPartialToolCalls = true;
    }
  }

  private async handleSuccess(
    data: InferCallbackValue<ChatModelEvents["success"]>,
    _meta: EventMeta,
  ): Promise<void> {
    // If we haven't received any tokens yet, process the final output
    if (this.output.messages.length === 0) {
      await this.handleNewToken({ value: data.value, callbacks: { abort: () => {} } }, _meta);
    }
  }

  private async handleNewToken(
    data: InferCallbackValue<ChatModelEvents["newToken"]>,
    _meta: EventMeta,
  ): Promise<void> {
    this.output.merge(data.value);

    const toolCalls = this.output.getToolCalls();
    if (toolCalls.length > 0) {
      for (const toolCall of toolCalls) {
        await this.process(toolCall.toolName, toolCall.input);
      }
      return;
    }

    // Try to parse text content as a tool call
    const textContent = this.output.getTextContent();
    const toolCall = parseBrokenJson(textContent, { pair: ["{", "}"] });

    if (!toolCall || !isPlainObject(toolCall)) {
      return;
    }

    await this.process(toolCall.name as string, toolCall.parameters as any);
  }
}
