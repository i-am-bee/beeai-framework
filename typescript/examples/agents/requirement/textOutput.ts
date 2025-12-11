import { RequirementAgent } from "beeai-framework/agents/requirement/agent";
import { ChatModel } from "beeai-framework/backend/chat";

const agent = new RequirementAgent({
  llm: await ChatModel.fromName("ollama:llama3.1"),
});

const response = await agent.run({
  // pass the task
  prompt: "Write a step-by-step tutorial on how to bake bread.",
  // nudge the model to format an output
  expectedOutput:
    "The output should be an ordered list of steps. Each step should be ideally one sentence.",
});

console.log(response.result.text);
