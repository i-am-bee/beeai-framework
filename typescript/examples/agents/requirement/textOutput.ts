import { RequirementAgent } from "beeai-framework/agents/requirement/agent";
import { ChatModel } from "beeai-framework/backend/chat";

const agent = new RequirementAgent({
  llm: await ChatModel.fromName("ollama:granite4:micro", { stream: true }),
  // tools: [new ThinkTool()],
});

const response = await agent
  .run({
    // pass the task
    prompt: "Write a step-by-step tutorial on how to bake bread. Make it short.",
    // nudge the model to format an output
    //expectedOutput:
    //  "The output should be an ordered list of steps. Each step should be ideally one sentence.",
  })
  .observe((emitter) =>
    emitter.on("finalAnswer", (data) => {
      console.info("Final Answer Stream", data.delta);
    }),
  );

console.log("=================");
console.log(response.result.text);
