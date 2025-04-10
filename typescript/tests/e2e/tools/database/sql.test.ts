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

import { SQLTool } from "@/tools/database/sql.js";
import { beforeEach, expect } from "vitest";

describe("SQLTool", () => {
  let instance: SQLTool;

  beforeEach(() => {
    instance = new SQLTool({
      provider: "sqlite",
      connection: {
        dialect: "sqlite",
        storage: "chinook.sqlite",
        logging: false,
      },
    });
  });

  it("Returns an error for invalid query", async () => {
    const response = await instance.run({
      action: "QUERY",
      query: "DELETE FROM users",
    });

    expect(response.result.success).toBe(false);
    expect(response.result.error).toContain("Only SELECT queries are allowed");
  });
});
