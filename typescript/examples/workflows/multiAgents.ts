import "dotenv/config";
import { createConsoleReader } from "examples/helpers/io.js";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { WikipediaTool } from "beeai-framework/tools/search/wikipedia";
import { AgentWorkflow } from "beeai-framework/workflows/agent";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

const workflow = new AgentWorkflow("Smart assistant");
const llm = new OllamaChatModel("llama3.1");

workflow.addAgent({
  name: "Researcher",
  role: "A diligent researcher",
  instructions: "You look up and provide information about a specific topic.",
  tools: [new WikipediaTool()],
  llm,
});
workflow.addAgent({
  name: "WeatherForecaster",
  role: "A weather reporter",
  instructions: "You provide detailed weather reports.",
  tools: [new OpenMeteoTool()],
  llm,
});
workflow.addAgent({
  name: "DataSynthesizer",
  role: "A meticulous and creative data synthesizer",
  instructions: "You can combine disparate information into a final coherent summary.",
  llm,
});

const reader = createConsoleReader();
reader.write("Assistant 🤖 : ", "What location do you want to learn about?");
for await (const { prompt } of reader) {
  const { result } = await workflow
    .run([
      { prompt: "Provide a short history of the location.", context: prompt },
      {
        prompt: "Provide a comprehensive weather summary for the location today.",
        expectedOutput:
          "Essential weather details such as chance of rain, temperature and wind. Only report information that is available.",
      },
      {
        prompt: "Summarize the historical and weather data for the location.",
        expectedOutput:
          "A paragraph that describes the history of the location, followed by the current weather conditions.",
      },
    ])
    .observe((emitter) => {
      emitter.on("success", (data) => {
        reader.write(
          `Step '${data.step}' has been completed with the following outcome:\n`,
          data.state?.finalAnswer ?? "-",
        );
      });
    });

  reader.write(`Assistant 🤖`, result.finalAnswer);
  reader.write("Assistant 🤖 : ", "What location do you want to learn about?");
}
