import { Workflow } from "beeai-framework/workflows/workflow";
import { z } from "zod";

const schema = z.object({
  threshold: z.number().min(0).max(1),
  counter: z.number().default(0),
});

const addFlow = new Workflow({ schema }).addStep("run", async (state) => {
  state.counter += 1;
  return Math.random() > 0.5 ? Workflow.SELF : Workflow.END;
});

const subtractFlow = new Workflow({
  schema,
}).addStep("run", async (state) => {
  state.counter -= 1;
  return Math.random() > 0.5 ? Workflow.SELF : Workflow.END;
});

const workflow = new Workflow({
  schema,
})
  .addStep("start", (state) =>
    Math.random() > state.threshold ? "delegateAdd" : "delegateSubtract",
  )
  .addStep("delegateAdd", addFlow.asStep({ next: Workflow.END }))
  .addStep("delegateSubtract", subtractFlow.asStep({ next: Workflow.END }));

const response = await workflow.run({ threshold: 0.5 }).observe((emitter) => {
  emitter.on("start", (data, event) =>
    console.log(`-> step ${data.step}`, event.trace?.parentRunId ? "(nested flow)" : ""),
  );
});
console.info(`Counter:`, response.result);
