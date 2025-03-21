import "dotenv/config";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { WikipediaTool } from "beeai-framework/tools/search/wikipedia";
import { AgentWorkflow } from "beeai-framework/workflows/agent";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

const workflow = new AgentWorkflow();

workflow.addAgent({
  name: "Researcher",
  role: "A diligent researcher",
  instructions: "You look up and provide information about a specific topic.",
  tools: [new WikipediaTool()],
  llm: new OllamaChatModel("llama3.1"),
});

workflow.addAgent({
  name: "WeatherForecaster",
  role: "A weather reporter",
  instructions: "You provide detailed weather reports.",
  tools: [new OpenMeteoTool()],
  llm: new OllamaChatModel("llama3.1"),
});

workflow.addAgent({
  name: "DataSynthesizer",
  role: "A meticulous and creative data synthesizer",
  instructions: "You can combine disparate information into a final coherent summary.",
  llm: new OllamaChatModel("llama3.1"),
});

const location = "Saint-Tropez";
await workflow
  .run([
    {
      prompt: `Provide a short history of ${location}.`,
    },
    {
      prompt: `Provide a comprehensive weather summary for ${location} today.`,
      expectedOutput:
        "Essential weather details such as chance of rain, temperature and wind. Only report information that is available.",
    },
    {
      prompt: `Summarize the historical and weather data for ${location}.`,
      expectedOutput: `A paragraph that describes the history of ${location}, followed by the current weather conditions.`,
    },
  ])
  .observe((emitter) => {
    emitter.on("success", (data) => {
      console.log(
        `-> ${data.step} has been completed with the following outcome\n`,
        data?.state.finalAnswer ?? "-",
      );
    });
  });
