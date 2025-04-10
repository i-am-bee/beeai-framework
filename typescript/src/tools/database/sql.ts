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
import { z } from "zod";
import { Sequelize, Options } from "sequelize";
import { Provider, getMetadata } from "@/tools/database/metadata.js";
import { Cache } from "@/cache/decoratorCache.js";
import { ValidationError } from "ajv";
import { AnyToolSchemaLike } from "@/internals/helpers/schema.js";
import { Emitter } from "@/emitter/emitter.js";

interface ToolOptions extends BaseToolOptions {
  provider: Provider;
  connection: Options;
}

type ToolRunOptions = BaseToolRunOptions;

export const SQLToolAction = {
  GetMetadata: "GET_METADATA",
  Query: "QUERY",
} as const;

export class SQLToolOutput extends JSONToolOutput<any> {}

export class SQLTool extends Tool<SQLToolOutput, ToolOptions, ToolRunOptions> {
  name = "SQLTool";

  description = `Converts natural language to SQL query and executes it. IMPORTANT: strictly follow this order of actions:
   1. ${SQLToolAction.GetMetadata} - get database tables structure (metadata)
   2. ${SQLToolAction.Query} - execute the generated SQL query`;

  inputSchema() {
    return z.object({
      action: z
        .nativeEnum(SQLToolAction)
        .describe(
          `The action to perform. ${SQLToolAction.GetMetadata} get database tables structure, ${SQLToolAction.Query} execute the SQL query`,
        ),
      query: z
        .string()
        .optional()
        .describe(`The SQL query to be executed, required for ${SQLToolAction.Query} action`),
    });
  }

  public readonly emitter: ToolEmitter<ToolInput<this>, SQLToolOutput> = Emitter.root.child({
    namespace: ["tool", "database", "sql"],
    creator: this,
  });

  public constructor(options: ToolOptions) {
    super(options);
    if (!options.connection.dialect) {
      throw new ValidationError([
        {
          message: "Property is required",
          propertyName: "connection.dialect",
        },
      ]);
    }
    if (
      !options.connection.schema &&
      (options.provider === "oracle" || options.provider === "db2")
    ) {
      throw new ValidationError([
        {
          message: `Property is required for ${options.provider}`,
          propertyName: "connection.schema",
        },
      ]);
    }
    if (!options.connection.storage && options.provider === "sqlite") {
      throw new ValidationError([
        {
          message: `Property is required for ${options.provider}`,
          propertyName: "connection.storage",
        },
      ]);
    }
  }

  protected validateInput(
    schema: AnyToolSchemaLike,
    input: unknown,
  ): asserts input is ToolInput<this> {
    super.validateInput(schema, input);
    if (input.action === SQLToolAction.Query && !input.query) {
      throw new ToolInputValidationError(
        `SQL Query is required for ${SQLToolAction.Query} action.`,
      );
    }
  }

  static {
    this.register();
  }

  @Cache()
  protected async connection(): Promise<Sequelize> {
    try {
      const sequelize = new Sequelize(this.options.connection);

      await sequelize.authenticate();
      return sequelize;
    } catch (error) {
      throw new ToolError(`Unable to connect to database: ${error}`, [], {
        isRetryable: false,
        isFatal: true,
      });
    }
  }

  protected async _run(
    input: ToolInput<this>,
    _options: Partial<ToolRunOptions>,
  ): Promise<SQLToolOutput> {
    const { provider, connection } = this.options;
    const { schema } = connection;

    if (input.action === SQLToolAction.GetMetadata) {
      const sequelize = await this.connection();
      const metadata = await getMetadata(sequelize, provider, schema);
      return new SQLToolOutput(metadata);
    }

    if (input.action === SQLToolAction.Query) {
      return await this.executeQuery(input.query!, provider, schema);
    }

    throw new ToolError(`Invalid action specified: ${input.action}`);
  }

  protected async executeQuery(
    query: string,
    provider: Provider,
    schema: string | undefined,
  ): Promise<SQLToolOutput> {
    if (!this.isReadOnlyQuery(query)) {
      return new JSONToolOutput({
        success: false,
        error: "Invalid query. Only SELECT queries are allowed.",
      });
    }

    try {
      const sequelize = await this.connection();
      const [results] = await sequelize.query(query);
      if (Array.isArray(results) && results.length > 0) {
        return new JSONToolOutput({ success: true, results });
      }

      return new JSONToolOutput({
        success: false,
        message: `No rows selected`,
      });
    } catch (error) {
      const schemaHint = schema
        ? `Fully qualify the table names by appending the schema name "${schema}" as a prefix, for example: ${schema}.table_name`
        : "";
      const errorMessage = `Generate a correct query that retrieves data using the appropriate ${provider} dialect.
      ${schemaHint}
      The original request was: ${query}, and the error was: ${error.message}.`;
      throw new ToolError(errorMessage);
    }
  }

  private isReadOnlyQuery(query: string): boolean {
    const normalizedQuery = query.trim().toUpperCase();
    return (
      normalizedQuery.startsWith("SELECT") ||
      normalizedQuery.startsWith("SHOW") ||
      normalizedQuery.startsWith("DESC")
    );
  }

  public async destroy(): Promise<void> {
    // @ts-expect-error protected property
    const cache = Cache.getInstance(this, "connection");
    const entry = cache.get();

    if (entry) {
      cache.clear();

      try {
        await entry.data.close();
      } catch (error) {
        throw new ToolError(`Failed to close the database connection`, [error]);
      }
    }
  }
}
