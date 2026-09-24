import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";

const tool = new OpenMeteoTool();

const today = new Date().toISOString().split("T")[0];
const result = await tool.run({
  location: { name: "New York" },
  start_date: today,
  end_date: today,
});
console.log(result.getTextContent());
