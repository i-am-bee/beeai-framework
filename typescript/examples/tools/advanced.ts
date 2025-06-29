/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { UnconstrainedCache } from "beeai-framework/cache/unconstrainedCache";

const tool = new OpenMeteoTool({
  cache: new UnconstrainedCache(),
  retryOptions: {
    maxRetries: 3,
  },
});
console.log(tool.name); // OpenMeteo
console.log(tool.description); // Retrieve current, past, or future weather forecasts for a location.
console.log(tool.inputSchema()); // (zod/json schema)

await tool.cache.clear();

const result = await tool.run({
  location: { name: "New York" },
  start_date: "2024-10-10",
  end_date: "2024-10-10",
  temperature_unit: "celsius",
});
console.log(result.isEmpty()); // false
console.log(result.result); // prints raw data
console.log(result.getTextContent()); // prints data as text
