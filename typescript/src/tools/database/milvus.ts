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
  Tool,
  ToolInput,
  ToolError,
  BaseToolOptions,
  BaseToolRunOptions,
  JSONToolOutput,
  ToolInputValidationError,
  ToolEmitter,
} from "@/tools/base.js";
import { Cache } from "@/cache/decoratorCache.js";
import { AnyToolSchemaLike } from "@/internals/helpers/schema.js";
import {
  ClientConfig,
  MilvusClient,
  SearchRes,
  ShowCollectionsResponse,
  DescribeCollectionResponse,
} from "@zilliz/milvus2-sdk-node";
import { z } from "zod";
import { Emitter } from "@/emitter/emitter.js";

export interface MilvusToolOptions extends BaseToolOptions {
  connection: ClientConfig;
}

export type MilvusSearchToolResult =
  | ShowCollectionsResponse
  | DescribeCollectionResponse
  | SearchRes;

export enum MilvusAction {
  ListCollections = "ListCollections",
  GetCollectionInfo = "GetCollectionInfo",
  Search = "Search",
  Insert = "Insert",
  Delete = "Delete",
}

export class MilvusDatabaseTool extends Tool<
  JSONToolOutput<MilvusSearchToolResult>,
  MilvusToolOptions
> {
  name = "MilvusDatabaseTool";

  description = `Can query data from a Milvus database. IMPORTANT: strictly follow this order of actions:
   1. ${MilvusAction.ListCollections} - List all the Milvus collections
   2. ${MilvusAction.GetCollectionInfo} - Get information about into a Milvus collection
   3. ${MilvusAction.Insert} - Insert data into a Milvus collection
   3. ${MilvusAction.Search} - Perform search on a Milvus collection
   4. ${MilvusAction.Delete} - Delete from a Milvus collection`;

  inputSchema() {
    return z.object({
      action: z
        .nativeEnum(MilvusAction)
        .describe(
          `The action to perform. ${MilvusAction.ListCollections} lists all collections, ${MilvusAction.GetCollectionInfo} fetches details for a specified collection, ${MilvusAction.Search} executes a vector search, ${MilvusAction.Insert} inserts new vectors, and ${MilvusAction.Delete} removes vectors.`,
        ),
      collectionName: z
        .string()
        .optional()
        .describe(
          `The name of the collection to query, required for ${MilvusAction.GetCollectionInfo}, ${MilvusAction.Search}, ${MilvusAction.Insert}, and ${MilvusAction.Delete}`,
        ),
      vector: z
        .array(z.number())
        .optional()
        .describe(`The vector to search for or insert, required for ${MilvusAction.Search}`),
      vectors: z
        .array(z.array(z.number()))
        .optional()
        .describe(`The vectors to insert, required for ${MilvusAction.Insert}`),
      topK: z.coerce
        .number()
        .int()
        .min(1)
        .max(1000)
        .default(10)
        .optional()
        .describe(
          `The number of nearest neighbors to return for ${MilvusAction.Search}. Maximum is 1000`,
        ),
      filter: z
        .string()
        .optional()
        .describe(`Optional filter expression for ${MilvusAction.Search}`),
      metadata: z
        .record(z.string(), z.any())
        .optional()
        .describe(`Additional metadata to insert with vectors for ${MilvusAction.Insert}`),
      ids: z
        .array(z.string().or(z.number()))
        .optional()
        .describe(`Array of IDs to delete for ${MilvusAction.Delete}`),
      searchOutput: z
        .array(z.string())
        .optional()
        .describe(`Fields to return in search results for ${MilvusAction.Search}`),
    });
  }

  public readonly emitter: ToolEmitter<ToolInput<this>, JSONToolOutput<MilvusSearchToolResult>> =
    Emitter.root.child({
      namespace: ["tool", "database", "milvus"],
      creator: this,
    });

  protected validateInput(
    schema: AnyToolSchemaLike,
    input: unknown,
  ): asserts input is ToolInput<this> {
    super.validateInput(schema, input);
    if (input.action === MilvusAction.GetCollectionInfo && !input.collectionName) {
      throw new ToolInputValidationError(
        `Collection name is required for ${MilvusAction.GetCollectionInfo}, ${MilvusAction.Search}, ${MilvusAction.Insert}, and ${MilvusAction.Delete} actions.`,
      );
    }
    if (input.action === MilvusAction.Search && (!input.collectionName || !input.vector)) {
      throw new ToolInputValidationError(
        `Both collection name and vector are required for ${MilvusAction.Search} action.`,
      );
    }
    if (input.action === MilvusAction.Insert && (!input.collectionName || !input.vectors)) {
      throw new ToolInputValidationError(
        `Both collection name and vectors are required for ${MilvusAction.Insert} action.`,
      );
    }
  }

  static {
    this.register();
  }

  @Cache()
  protected async client(): Promise<MilvusClient> {
    try {
      const client = new MilvusClient(this.options.connection);
      // operation expected to fail if connection is not established correctly.
      await client.listCollections();
      return client;
    } catch (error) {
      throw new ToolError(`Unable to connect to Milvus.`, [error], {
        isRetryable: false,
        isFatal: true,
      });
    }
  }

  protected async _run(
    input: ToolInput<this>,
    _options: Partial<BaseToolRunOptions>,
  ): Promise<JSONToolOutput<any>> {
    switch (input.action) {
      case MilvusAction.ListCollections: {
        const collections = await this.listCollections();
        return new JSONToolOutput(collections);
      }

      case MilvusAction.GetCollectionInfo: {
        if (!input.collectionName) {
          throw new ToolError("A collection name is required for Milvus GetCollectionInfo action");
        }
        const collectionInfo = await this.getCollectionInfo(input.collectionName);
        return new JSONToolOutput(collectionInfo);
      }

      case MilvusAction.Search: {
        if (!input.collectionName || !input.vector) {
          throw new ToolError("A collection name and vector are required for Milvus Search action");
        }
        const searchResults = await this.search(input);
        return new JSONToolOutput(searchResults);
      }

      case MilvusAction.Insert: {
        if (!input.collectionName || !input.vectors) {
          throw new ToolError(
            "A collection name and vectors are required for Milvus Insert action",
          );
        }
        const insertResults = await this.insert(input);
        return new JSONToolOutput(insertResults);
      }

      case MilvusAction.Delete: {
        if (!input.collectionName || !input.ids) {
          throw new ToolError("Collection name and ids are required for Milvus Delete action");
        }
        const deleteResults = await this.delete(input);
        return new JSONToolOutput(deleteResults);
      }

      default: {
        throw new ToolError(`Invalid action specified: ${input.action}`);
      }
    }
  }

  protected async listCollections(): Promise<string[]> {
    try {
      const client = await this.client();
      const response = await client.listCollections({});

      if (response && response.data && Array.isArray(response.data)) {
        return response.data.map((collection) => collection.name);
      } else {
        return [];
      }
    } catch (error) {
      throw new ToolError(`Failed to list collections from Milvus: ${error}`);
    }
  }

  protected async getCollectionInfo(collectionName: string): Promise<any> {
    const client = await this.client();
    const response = client.describeCollection({ collection_name: collectionName });
    return response;
  }

  protected async insert(input: ToolInput<this>): Promise<any> {
    const client = await this.client();
    const response = await client.insert({
      collection_name: input.collectionName as string,
      fields_data: input.vectors!.map((vector, index) => ({
        vector: vector,
        ...input.metadata?.[index],
      })),
    });
    return response;
  }

  protected async search(input: ToolInput<this>): Promise<any> {
    const client = await this.client();
    const response = await client.search({
      collection_name: input.collectionName as string,
      vector: input.vector,
      limit: input.topK || 10,
      filter: input.filter,
      output_fields: input.searchOutput,
    });
    return response.results;
  }

  protected async delete(input: ToolInput<this>): Promise<any> {
    const client = await this.client();
    const response = client.delete({
      collection_name: input.collectionName as string,
      filter: `id in [${input.ids?.join(",")}]`,
    });
    return response;
  }
}
