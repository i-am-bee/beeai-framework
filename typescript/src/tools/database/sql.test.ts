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

import { beforeEach, expect, vi } from "vitest";
import { SQLTool } from "@/tools/database/sql.js";
import { getMetadata } from "@/tools/database/metadata.js";
import { Sequelize } from "sequelize";

vi.mock("@/tools/database/metadata.js", () => ({
  getMetadata: vi.fn(),
}));

describe("SQLTool", () => {
  const mockSequelize = {
    query: vi.fn(),
    close: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("Is a valid tool", () => {
    const tool = new SQLTool({
      provider: "sqlite",
      connection: {
        dialect: "sqlite",
        storage: "chinook.sqlite",
        logging: false,
      },
    });
    expect(tool).toBeInstanceOf(SQLTool);
    expect(SQLTool.isTool(tool)).toBe(true);
    expect(tool.name).toBeDefined();
    expect(tool.description).toBeDefined();
  });

  it("Executes a valid SELECT query", async () => {
    vi.mocked(getMetadata).mockResolvedValue("Table schema");

    mockSequelize.query.mockResolvedValueOnce([[{ id: 1, name: "John Doe" }]]);

    const tool = new SQLTool({
      provider: "sqlite",
      connection: {
        dialect: "sqlite",
        storage: "chinook.sqlite",
        logging: false,
      },
    });

    vi.spyOn(tool as any, "connection").mockResolvedValue(mockSequelize as unknown as Sequelize);

    const result = await tool.run({
      action: "QUERY",
      query: "SELECT * FROM users;",
    });

    expect(result.result.success).toBe(true);
    expect(result.result.results).toEqual([{ id: 1, name: "John Doe" }]);
    expect(mockSequelize.query).toHaveBeenCalledWith("SELECT * FROM users;");
  });

  it("Handles non-SELECT queries", async () => {
    const tool = new SQLTool({
      provider: "sqlite",
      connection: {
        dialect: "sqlite",
        storage: "chinook.sqlite",
        logging: false,
      },
    });

    const result = await tool.run({
      action: "QUERY",
      query: "DELETE FROM users;",
    });

    expect(result.result.success).toBe(false);
    expect(result.result.error).toBe("Invalid query. Only SELECT queries are allowed.");
  });

  it("Returns top-n query results", async () => {
    mockSequelize.query.mockResolvedValueOnce([
      [
        { id: 1, name: "John Doe" },
        { id: 2, name: "Jane Doe" },
      ],
    ]);

    const tool = new SQLTool({
      provider: "sqlite",
      connection: {
        dialect: "sqlite",
        storage: "chinook.sqlite",
        logging: false,
      },
    });

    vi.spyOn(tool as any, "connection").mockResolvedValue(mockSequelize as unknown as Sequelize);

    const result = await tool.run({
      action: "QUERY",
      query: "SELECT * FROM users;",
    });

    expect(result.result.results.length).toBe(2);
    expect(result.result.results).toEqual([
      { id: 1, name: "John Doe" },
      { id: 2, name: "Jane Doe" },
    ]);
  });
});
