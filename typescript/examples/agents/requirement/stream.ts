import "dotenv/config";

import { RequirementAgent } from "beeai-framework/agents/requirement/agent";
import { ChatModel } from "beeai-framework/backend/chat";

const agent = new RequirementAgent({
  llm: await ChatModel.fromName("watsonx:ibm/granite-3-3-8b-instruct", { stream: true }),
});

await agent
  .run({
    // pass the task
    prompt: "Write a step-by-step tutorial on how to bake bread.",
    // nudge the model to format an output
    expectedOutput:
      "The output should be an ordered list of steps. Each step should be ideally one sentence.",
  })
  .observe((emitter) =>
    emitter.on("finalAnswer", (data) => {
      console.log("Final Answer Chunk:", data.delta);
    }),
  );
