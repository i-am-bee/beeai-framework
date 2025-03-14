import "dotenv/config";
import { z } from "zod";
import { Workflow } from "beeai-framework/workflows/workflow";
import { ReActAgent } from "beeai-framework/agents/react/agent";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { createConsoleReader } from "examples/helpers/io.js";
import { Message } from "beeai-framework/backend/message";
import { isEmpty } from "remeda";
import { LLMTool } from "beeai-framework/tools/llm";
import { GoogleSearchTool } from "beeai-framework/tools/search/googleSearch";
import { GroqChatModel } from "beeai-framework/adapters/groq/backend/chat";

const schema = z.object({
  input: z.string(),
  output: z.string().optional(),

  topic: z.string().optional(),
  notes: z.array(z.string()).default([]),
  plan: z.string().optional(),
  draft: z.string().optional(),
});

const workflow = new Workflow({
  schema,
  outputSchema: schema.required({ output: true }),
})
  .addStep("preprocess", async (state) => {
    const llm = new GroqChatModel("llama-3.3-70b-versatile");

    const { object: parsed } = await llm.createStructure({
      schema: schema.pick({ topic: true, notes: true }).or(
        z.object({
          error: z
            .string()
            .describe("Use when the input query does not make sense or you need clarification."),
        }),
      ),
      messages: [
        Message.of({
          role: `user`,
          text: [
            "Your task is to rewrite the user query so that it guides the content planner and editor to craft a blog post that perfectly aligns with the user's needs. Notes should be used only if the user complains about something.",
            "If the user query does ",
            "",
            ...[state.topic && ["# Previous Topic", state.topic, ""]],
            ...[!isEmpty(state.notes) && ["# Previous Notes", ...state.notes, ""]],
            "# User Query",
            state.input || "empty",
          ]
            .filter(Boolean)
            .join("\n"),
        }),
      ],
    });

    if ("error" in parsed) {
      state.output = parsed.error;
      return Workflow.END;
    }

    state.notes = parsed.notes ?? [];
    state.topic = parsed.topic;
  })
  .addStrictStep("planner", schema.required({ topic: true }), async (state) => {
    const llm = new GroqChatModel("llama-3.3-70b-versatile");
    const agent = new ReActAgent({
      llm,
      memory: new UnconstrainedMemory(),
      tools: [new GoogleSearchTool(), new LLMTool({ llm })],
    });

    agent.emitter.on("update", (data) => {
      console.info(data.update);
    });

    const { result } = await agent.run({
      prompt: [
        `You are a Content Planner. Your task is to write a content plan for "${state.topic}" topic in Markdown format.`,
        ``,
        `# Objectives`,
        `1. Prioritize the latest trends, key players, and noteworthy news.`,
        `2. Identify the target audience, considering their interests and pain points.`,
        `3. Develop a detailed content outline including introduction, key points, and a call to action.`,
        `4. Include SEO keywords and relevant sources.`,
        ``,
        ...[!isEmpty(state.notes) && ["# Notes", ...state.notes, ""]],
        `Provide a structured output that covers the mentioned sections.`,
      ].join("\n"),
    });

    state.plan = result.text;
  })
  .addStrictStep("writer", schema.required({ plan: true }), async (state) => {
    const llm = new GroqChatModel("llama-3.3-70b-versatile");
    const output = await llm.create({
      messages: [
        Message.of({
          role: `system`,
          text: [
            `You are a Content Writer. Your task is to write a compelling blog post based on the provided context.`,
            ``,
            `# Context`,
            `${state.plan}`,
            ``,
            `# Objectives`,
            `- An engaging introduction`,
            `- Insightful body paragraphs (2-3 per section)`,
            `- Properly named sections/subtitles`,
            `- A summarizing conclusion`,
            `- Format: Markdown`,
            ``,
            ...[!isEmpty(state.notes) && ["# Notes", ...state.notes, ""]],
            `Ensure the content flows naturally, incorporates SEO keywords, and is well-structured.`,
          ].join("\n"),
        }),
      ],
    });

    state.draft = output.getTextContent();
  })
  .addStrictStep("editor", schema.required({ draft: true }), async (state) => {
    const llm = new GroqChatModel("llama-3.3-70b-versatile");
    const output = await llm.create({
      messages: [
        Message.of({
          role: `system`,
          text: [
            `You are an Editor. Your task is to transform the following draft blog post to a final version.`,
            ``,
            `# Draft`,
            `${state.draft}`,
            ``,
            `# Objectives`,
            `- Fix Grammatical errors`,
            `- Journalistic best practices`,
            ``,
            ...[!isEmpty(state.notes) && ["# Notes", ...state.notes, ""]],
            ``,
            `IMPORTANT: The final version must not contain any editor's comments.`,
          ].join("\n"),
        }),
      ],
    });

    state.output = output.getTextContent();
  });

let lastResult = {} as Workflow.output<typeof workflow>;
const reader = createConsoleReader();
for await (const { prompt } of reader) {
  const { result } = await workflow
    .run({
      input: prompt,
      notes: lastResult?.notes,
      topic: lastResult?.topic,
    })
    .observe((emitter) => {
      emitter.on("start", ({ step, run }) => {
        reader.write(`-> ▶️ ${step}`, JSON.stringify(run.state).substring(0, 200).concat("..."));
      });
    });

  lastResult = result;
  reader.write("🤖 Answer", lastResult.output);
}
