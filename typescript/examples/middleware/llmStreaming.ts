import "dotenv/config.js";
import { ChatModel } from "beeai-framework/backend/chat";
import { UserMessage } from "beeai-framework/backend/message";
import { StreamToolCallMiddleware } from "beeai-framework/middleware/streamToolCall";
import { ThinkTool } from "beeai-framework/tools/think";

const thinkTool = new ThinkTool();

// Create middleware to stream the location field from the weather tool
const middleware = new StreamToolCallMiddleware({
  target: thinkTool,
  key: "thoughts", // Field from ThinkToolInput schema
  matchNested: false, // Apply middleware to the model directly
  forceStreaming: true, // Enable streaming on the model
});

// Listen to update events
middleware.emitter.on("update", (event) => {
  console.info("Thoughts:", event.delta);
  // console.log("Received update:", {
  //   delta: event.delta,
  //   output: event.output,
  //   structured: event.outputStructured,
  // });
});

const llm = await ChatModel.fromName("ollama:llama3.1");
const response = await llm
  .create({
    messages: [new UserMessage("Why sky is blue?")],
    tools: [thinkTool],
    stream: true,
  })
  .middleware(middleware);

console.log("\nTool Call:", response.getToolCalls()[0]);
