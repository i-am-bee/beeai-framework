/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Tool, ToolEmitter, ToolInput, BaseToolOptions, BaseToolRunOptions, JSONToolOutput, ToolError } from "@/tools/base.js";
import { z } from "zod";
import { Cache } from "@/cache/decoratorCache.js";
import { RunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";
import { getEnv } from "@/internals/env.js";

export interface FeedoToolOptions extends BaseToolOptions {
  usageKey?: string;
  private?: boolean;
}

export interface FeedoToolRunOptions extends BaseToolRunOptions {}

export class FeedoToolOutput extends JSONToolOutput<any> {
  static {
    this.register();
  }
}

export class FeedoSearchTool extends Tool<FeedoToolOutput, FeedoToolOptions, FeedoToolRunOptions> {
  name = "FeedoMemory";
  description = "Access the decentralized Feedo Memory Network to store, search, update, and delete permanent knowledge and context across sessions.";

  public readonly emitter: ToolEmitter<ToolInput<this>, FeedoToolOutput> = Emitter.root.child({
    namespace: ["tool", "search", "feedo"],
    creator: this,
  });

  @Cache()
  inputSchema() {
    return z.object({
      action: z.enum(["add", "search", "update", "delete"]).describe("The action to perform on the Feedo memory network."),
      query: z.string().optional().describe("The search query (required for 'search') or the text to add/update."),
      memory_id: z.string().optional().describe("The ID of the memory to update or delete."),
      topic: z.string().optional().describe("Optional category or topic when adding a memory."),
    });
  }

  public constructor(options: Partial<FeedoToolOptions> = {}) {
    super({
      ...options,
      usageKey: options.usageKey ?? getEnv("FEEDO_USAGE_KEY"),
      private: options.private ?? true,
    });

    if (!this.options.usageKey) {
      throw new Error(
        "FEEDO_USAGE_KEY is required to initialize Feedo memory.\n" +
        "You can generate a free testnet usage key at: https://feedo.ink"
      );
    }
  }

  static {
    this.register();
  }

  @Cache({ enumerable: false })
  protected async _createClient() {
    try {
      const { FeedoMemory } = await import("feedo-protocol-sdk/memory.js");
      return new FeedoMemory({
        usageKey: this.options.usageKey,
        private: this.options.private,
      });
    } catch (e: any) {
      throw new Error("Optional module [feedo-protocol-sdk] not found. Please install it using `npm install feedo-protocol-sdk`.");
    }
  }

  protected async _run(
    input: ToolInput<this>,
    _options: Partial<FeedoToolRunOptions>,
    _run: RunContext<this>,
  ): Promise<FeedoToolOutput> {
    const memory = await this._createClient();

    try {
      if (input.action === "add") {
        if (!input.query) {
          throw new Error("The 'query' field is required to add a memory.");
        }
        const metadata = input.topic ? { topic: input.topic } : {};
        const memId = await memory.add(input.query, { metadata });
        return new FeedoToolOutput({ result: `Successfully saved to Feedo memory with ID: ${memId}` });
      }

      if (input.action === "search") {
        if (!input.query) {
          throw new Error("The 'query' field is required to search memory.");
        }
        const results = await memory.search(input.query, { limit: 5 });
        if (!results || results.length === 0) {
          return new FeedoToolOutput({ result: "No relevant memories found.", memories: [] });
        }
        return new FeedoToolOutput({ result: `Found ${results.length} memories.`, memories: results });
      }

      if (input.action === "update") {
        if (!input.memory_id || !input.query) {
          throw new Error("Both 'memory_id' and 'query' are required to update a memory.");
        }
        const newId = await memory.update(input.memory_id, input.query);
        return new FeedoToolOutput({ result: `Memory successfully updated. New ID: ${newId}` });
      }

      if (input.action === "delete") {
        if (!input.memory_id) {
          throw new Error("The 'memory_id' field is required to delete a memory.");
        }
        await memory.delete(input.memory_id);
        return new FeedoToolOutput({ result: `Memory ${input.memory_id} successfully deleted.` });
      }

      throw new Error(`Unknown action: ${input.action}`);
    } catch (e: any) {
      throw new ToolError(`Error interacting with Feedo: ${e.message}`, [e]);
    }
  }
}
