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

const today = new Date().toISOString().split("T")[0];
const result = await tool.run({
  location: { name: "New York" },
  start_date: today,
  end_date: today,
  temperature_unit: "celsius",
});
console.log(result.isEmpty()); // false
console.log(result.result); // prints raw data
console.log(result.getTextContent()); // prints data as text
