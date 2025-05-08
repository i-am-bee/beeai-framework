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
import { QdrantClient, QdrantClientParams } from "@qdrant/js-client-rest";
import { v4 as uuidv4 } from "uuid";

import { z } from "zod";
import { Emitter } from "@/emitter/emitter.js";

export interface QdrantToolOptions extends BaseToolOptions, QdrantClientParams {
  connection: QdrantClientParams;
}

export type QdrantSearchToolResult = any;

export enum QdrantAction {
  ListCollections = "ListCollections",
  GetCollectionInfo = "GetCollectionInfo",
  Search = "Search",
  Insert = "Insert",
  Delete = "Delete",
}

export class QdrantDatabaseTool extends Tool<
  JSONToolOutput<QdrantSearchToolResult>,
  QdrantToolOptions
> {
  name = "QdrantDatabaseTool";

  description = `Can query data from a Qdrant vector database. IMPORTANT: strictly follow this order of actions:
     1. ${QdrantAction.ListCollections} - List all the Qdrant collections
     2. ${QdrantAction.GetCollectionInfo} - Get information about a Qdrant collection
     3. ${QdrantAction.Insert} - Insert data into a Qdrant collection
     3. ${QdrantAction.Search} - Perform search on a Qdrant collection
     4. ${QdrantAction.Delete} - Delete from a Qdrant collection`;

  inputSchema() {
    return z.object({
      action: z
        .nativeEnum(QdrantAction)
        .describe(
          `The action to perform. ${QdrantAction.ListCollections} lists all collections, ${QdrantAction.GetCollectionInfo} fetches details for a specified collection, ${QdrantAction.Search} executes a vector search, ${QdrantAction.Insert} inserts new vectors, and ${QdrantAction.Delete} removes vectors.`,
        ),
      collectionName: z
        .string()
        .optional()
        .describe(
          `The name of the collection to query, required for ${QdrantAction.GetCollectionInfo}, ${QdrantAction.Search}, ${QdrantAction.Insert}, and ${QdrantAction.Delete}`,
        ),
      vector: z
        .array(z.number())
        .optional()
        .describe(`The vector to search for, required for ${QdrantAction.Search}.`),
      vectors: z
        .array(z.array(z.number()))
        .optional()
        .describe(`The vectors to insert, required for ${QdrantAction.Insert}.`),
      topK: z.coerce
        .number()
        .int()
        .default(10)
        .optional()
        .describe(`The number of nearest neighbors to return.`),
      filter: z
        .record(z.string(), z.any())
        .optional()
        .describe(`Optional filter for ${QdrantAction.Search}.`),
      payload: z
        .array(z.record(z.string(), z.any()))
        .optional()
        .describe(`Additional payload to insert with vectors.`),
      ids: z
        .array(z.string().or(z.number()))
        .optional()
        .describe(`Array of IDs to delete or insert.`),
    });
  }

  public readonly emitter: ToolEmitter<ToolInput<this>, JSONToolOutput<QdrantSearchToolResult>> =
    Emitter.root.child({
      namespace: ["tool", "database", "qdrant"],
      creator: this,
    });

  protected validateInput(
    schema: AnyToolSchemaLike,
    input: unknown,
  ): asserts input is ToolInput<this> {
    super.validateInput(schema, input);
    if (input.action != QdrantAction.ListCollections && !input.collectionName) {
      throw new ToolInputValidationError(
        `Collection name is required for ${QdrantAction.GetCollectionInfo}, ${QdrantAction.Search}, ${QdrantAction.Insert}, and ${QdrantAction.Delete} actions.`,
      );
    }
    if (input.action === QdrantAction.Search && (!input.collectionName || !input.vector)) {
      throw new ToolInputValidationError(`Vector is required for ${QdrantAction.Search} action.`);
    }
    if (input.action === QdrantAction.Insert && (!input.collectionName || !input.vectors)) {
      throw new ToolInputValidationError(`Vectors are required for ${QdrantAction.Insert} action.`);
    }
  }

  static {
    this.register();
  }

  @Cache()
  protected async client(): Promise<QdrantClient> {
    return new QdrantClient(this.options.connection);
  }

  protected async _run(
    input: ToolInput<this>,
    _options: Partial<BaseToolRunOptions>,
  ): Promise<JSONToolOutput<any>> {
    switch (input.action) {
      case QdrantAction.ListCollections: {
        const collections = await this.listCollections();
        return new JSONToolOutput(collections);
      }

      case QdrantAction.GetCollectionInfo: {
        if (!input.collectionName) {
          throw new ToolError("A collection name is required for Qdrant GetCollectionInfo action");
        }
        const collectionInfo = await this.getCollectionInfo(input.collectionName);
        return new JSONToolOutput(collectionInfo);
      }

      case QdrantAction.Search: {
        if (!input.collectionName || !input.vector) {
          throw new ToolError("A collection name and vector are required for Qdrant Search action");
        }
        const searchResults = await this.search(input);
        return new JSONToolOutput(searchResults);
      }

      case QdrantAction.Insert: {
        if (!input.collectionName || !input.vectors) {
          throw new ToolError(
            "A collection name and vectors are required for Qdrant Insert action",
          );
        }
        const insertResults = await this.insert(input);
        return new JSONToolOutput(insertResults);
      }

      case QdrantAction.Delete: {
        if (!input.collectionName || !input.ids) {
          throw new ToolError("Collection name and ids are required for Qdrant Delete action");
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
      const response = await client.getCollections();

      return response.collections.map((collection) => collection.name);
    } catch (error) {
      throw new ToolError(`Failed to list collections from Qdrant: ${error}`);
    }
  }

  protected async getCollectionInfo(collectionName: string): Promise<any> {
    const client = await this.client();
    const response = await client.getCollection(collectionName);
    return response;
  }

  protected async insert(input: ToolInput<this>): Promise<any> {
    const client = await this.client();

    const points = input.vectors!.map((vector, index) => ({
      id: input?.ids?.[index] ?? uuidv4(),
      vector: vector,
      payload: input?.payload?.[index] || {},
    }));

    const response = await client.upsert(input.collectionName as string, {
      points: points,
    });

    return response;
  }

  protected async search(input: ToolInput<this>): Promise<any> {
    const client = await this.client();

    const searchParams: any = {
      query: input.vector,
      limit: input.topK || 10,
      with_payload: true,
      filter: input.filter,
    };

    const response = await client.query(input.collectionName as string, searchParams);
    return response;
  }

  protected async delete(input: ToolInput<this>): Promise<any> {
    const client = await this.client();

    const response = await client.delete(input.collectionName as string, {
      points: input.ids!,
    });

    return response;
  }
}
