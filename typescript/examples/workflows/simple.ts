import { Workflow } from "beeai-framework/workflows/workflow";
import { z } from "zod";

const schema = z.object({
  hops: z.number().default(0),
});

const workflow = new Workflow({ schema })
  .addStep("a", async (state) => {
    state.hops += 1;
  })
  .addStep("b", () => (Math.random() > 0.5 ? Workflow.PREV : Workflow.END));

const response = await workflow.run({ hops: 0 }).observe((emitter) => {
  emitter.on("start", (data) => console.log(`-> start ${data.step}`));
  emitter.on("error", (data) => console.log(`-> error ${data.step}`));
  emitter.on("success", (data) => console.log(`-> finish ${data.step}`));
});

console.log(`Hops: ${response.result.hops}`);
console.log(`-> steps`, response.steps.map((step) => step.name).join(","));
