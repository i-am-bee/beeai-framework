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

import { ToolError } from "@/tools/base.js";
import { Sequelize } from "sequelize";

export type Provider = "mysql" | "mariadb" | "postgres" | "mssql" | "db2" | "sqlite" | "oracle";

export interface Metadata {
  tableName: string;
  columnName: string;
  dataType: string;
}

export async function getMetadata(
  sequelize: Sequelize,
  provider: Provider,
  schema?: string,
): Promise<string> {
  try {
    const query = getMetadataQuery(provider, schema);

    const [metadata] = (await sequelize.query(query)) as [Metadata[], any];

    const tableMap = new Map<string, string[]>();

    metadata.forEach(({ tableName, columnName, dataType }) => {
      if (!tableMap.has(tableName)) {
        tableMap.set(tableName, [`Table '${tableName}' with columns: ${columnName} (${dataType})`]);
      } else {
        tableMap.get(tableName)!.push(`${columnName} (${dataType})`);
      }
    });

    return Array.from(tableMap.values())
      .map((columns) => columns.join(", "))
      .join("; ");
  } catch (error) {
    throw new ToolError(`Error initializing metadata: ${error}`, [], {
      isRetryable: false,
    });
  }
}

function getMetadataQuery(provider: Provider, schema?: string): string {
  const schemaName = schema ?? getDefaultSchema(provider);

  switch (provider) {
    case "mysql":
    case "mariadb":
      return `
          SELECT t.table_name AS tableName, c.column_name AS columnName, 
                 c.data_type AS dataType
          FROM information_schema.tables t
          JOIN information_schema.columns c ON t.table_name = c.table_name
          WHERE t.table_schema = DATABASE();
        `;

    case "postgres":
      return `
          SELECT t.table_name AS "tableName", c.column_name AS "columnName", 
                 c.data_type AS "dataType"
          FROM information_schema.tables t
          JOIN information_schema.columns c ON t.table_name = c.table_name
          WHERE t.table_schema = lower('${schemaName}');
        `;

    case "mssql":
      return `
          SELECT t.name AS tableName, c.name AS columnName,
                 ty.name AS dataType
           FROM sys.tables t
           JOIN sys.columns c ON t.object_id = c.object_id
           JOIN sys.types ty ON c.user_type_id = ty.user_type_id
           JOIN sys.schemas s ON t.schema_id = s.schema_id
           WHERE t.is_ms_shipped = 0 AND t.type = 'U'
                AND s.name = lower('${schemaName}');
        `;

    case "db2":
      return `
          SELECT t.tabname AS "tableName", c.colname AS "columnName", 
                 c.typename AS "dataType"
          FROM SYSCAT.TABLES t
          JOIN SYSCAT.COLUMNS c ON t.tabname = c.tabname
          WHERE t.tabschema = upper('${schemaName}');
        `;

    case "sqlite":
      return `
          SELECT tbl_name AS "tableName", name AS "columnName", type AS "dataType"
            FROM (
                SELECT name AS tbl_name
                FROM sqlite_master
                WHERE type = 'table'
            )
            JOIN pragma_table_xinfo(tbl_name);
        `;

    case "oracle":
      return `
          SELECT t.table_name AS "tableName", c.column_name AS "columnName", 
                 c.data_type AS "dataType"
          FROM all_tables t
          JOIN all_tab_columns c ON t.table_name = c.table_name
          WHERE t.owner = upper('${schemaName}');
        `;

    default:
      throw new ToolError(`Provider ${provider} is not supported`, [], {
        isFatal: true,
        isRetryable: false,
      });
  }
}

function getDefaultSchema(provider: Provider): string {
  switch (provider) {
    case "postgres":
      return "public";
    case "mssql":
      return "dbo";
    case "db2":
    case "oracle":
      throw new ToolError(`Schema name is required for ${provider}`, [], {
        isRetryable: false,
        isFatal: true,
      });
    default:
      return "";
  }
}
