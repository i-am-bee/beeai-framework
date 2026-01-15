import "dotenv/config.js";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { RequirementAgent } from "beeai-framework/agents/requirement/agent";
import { ConditionalRequirement } from "beeai-framework/agents/requirement/requirements/conditional";
import { WatsonxChatModel } from "beeai-framework/adapters/watsonx/backend/chat";
import { ThinkTool } from "beeai-framework/tools/think";
import { RunContext } from "beeai-framework/context";
import { StreamToolCallMiddleware } from "beeai-framework/middleware/streamToolCall";

const llm = new WatsonxChatModel("meta-llama/llama-3-3-70b-instruct");
llm.parameters.stream = true;
llm.parameters.temperature = 0;

const agent = new RequirementAgent({
  llm,
  tools: [new ThinkTool()],
  memory: new UnconstrainedMemory(),
  requirements: [new ConditionalRequirement(ThinkTool, { forceAtStep: 1 })],
});

const response = await agent
  .run({
    prompt: "Why is the sky blue?",
  })
  .middleware(function streamUpdatesMiddleware(ctx: RunContext<typeof agent>) {
    const thinkTool = ctx.instance.meta.tools.find((tool) => tool instanceof ThinkTool)!;
    const streamMiddleware = new StreamToolCallMiddleware({
      target: thinkTool,
      key: "thoughts",
    });
    llm.middlewares.push(streamMiddleware);
    streamMiddleware.emitter.on("update", (data) => {
      console.info("Got update", data.delta);
    });
    ctx.emitter.on("success", async ({ state }) => {
      const lastStep = state.steps.at(-1)!;
      if (lastStep.tool instanceof ThinkTool) {
        if (streamMiddleware.isEmpty()) {
          console.info("is empty, manually streaming");
          const input = lastStep.input as { thoughts: string };
          await streamMiddleware.emitter.emit("update", {
            delta: input.thoughts,
            output: input.thoughts,
            outputStructured: lastStep.input,
          });
        }
      }
      // reset on every iteration to prevent overriding
      streamMiddleware.reset();
    });
  });

console.info(response.result.text);
