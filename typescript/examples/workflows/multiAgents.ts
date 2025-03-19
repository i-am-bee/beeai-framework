import "dotenv/config";
import { createConsoleReader } from "examples/helpers/io.js";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { WikipediaTool } from "beeai-framework/tools/search/wikipedia";
import { AgentWorkflow, AgentWorkflowInput } from "beeai-framework/workflows/agent";
import { WatsonxChatModel } from "beeai-framework/adapters/watsonx/backend/chat";

const workflow = new AgentWorkflow("Smart assistant");
const llm = new WatsonxChatModel("meta-llama/llama-3-3-70b-instruct");

workflow.addAgent({
  name: "Researcher",
  role: "A diligent researcher.",
  instructions: "You are a researcher assistant. Respond only if you can provide a useful answer.",
  tools: [new WikipediaTool()],
  llm,
});
workflow.addAgent({
  name: "WeatherForecaster",
  role: "A weather reporter.",
  instructions: "You are a weather assistant. Respond only if you can provide a useful answer.",
  tools: [new OpenMeteoTool()],
  llm,
  execution: { maxIterations: 3 },
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
      new AgentWorkflowInput("Provide a short history of the location.", prompt),
      new AgentWorkflowInput(
        "Provide a comprehensive weather summary for the location today.",
        undefined,
        "Essential weather details such as chance of rain, temperature and wind. Only report information that is available.",
      ),
      new AgentWorkflowInput(
        "Summarize the historical and weather data for the location.",
        undefined,
        "A paragraph that describes the history of the location, followed by the current weather conditions.",
      ),
    ])
    .observe((emitter) => {
      emitter.on("success", (data) => {
        reader.write(`-> ${data.step}`, data.state?.finalAnswer ?? "-");
      });
    });

  reader.write(`Assistant 🤖`, result.finalAnswer);
}
