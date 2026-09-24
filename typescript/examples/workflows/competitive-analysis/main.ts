import { workflow } from "./workflow.js";
import { InputSchema } from "./state.js";
import { createConsoleReader } from "examples/helpers/io.js";

const reader = createConsoleReader();
for await (const { prompt } of reader) {
  let input;
  try {
    input = InputSchema.parse(JSON.parse(prompt));
  } catch (e) {
    input = { industry: prompt };
  }

  reader.write(
    `Analyst 🤖:`,
    `🔬 Initiating competitor research workflow. I'll analyze your industry and build a detailed report based on real-time market data 📊`,
  );

  const response = await workflow
    .run({
      ...input,
      competitors: input.specifiedCompetitors,
    })
    .observe((emitter) => {
      emitter.on("start", (data) => {
        reader.write(``, `▶ Starting ${data.step} for ${data.run.state.industry}`);
      });
      emitter.on("success", (data) => {
        reader.write(``, `  ✔ Completed ${data.step}`);
        if (data.run.state.currentCompetitor) {
          reader.write(``, `  ✔ Analyzed: ${data.run.state.currentCompetitor}`);
        }
      });
      emitter.on("error", (data) => {
        reader.write(`Analyst 🤖:`, `Error in ${data.step}: ${data.error}`);
      });
    });

  reader.write(
    `Analyst 🤖:`,
    `✅ Analysis complete! Your competitive analysis report is ready for your review 📋\n=== Final Analysis ===\n${response.state.answer!.text}`,
  );
}
