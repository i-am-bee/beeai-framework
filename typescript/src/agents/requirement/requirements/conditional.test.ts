/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { ConditionalRequirement, ConditionalRequirementOptions } from "./conditional.js";
import { RequirementError } from "./requirement.js";
import { AnyTool, DynamicTool, StringToolOutput } from "@/tools/base.js";
import { UnconstrainedMemory } from "@/memory/unconstrainedMemory.js";
import { RequirementAgentRunState } from "@/agents/requirement/types.js";
import { z } from "zod";
import { FrameworkError } from "@/errors.js";

function createTool(name: string): AnyTool {
  return new DynamicTool({
    name,
    description: `Tool ${name}.`,
    inputSchema: z.object({}),
    async handler() {
      return new StringToolOutput(name);
    },
  });
}

const toolA = createTool("tool_a");
const toolB = createTool("tool_b");
const toolC = createTool("tool_c");

function createState(...tools: AnyTool[]): RequirementAgentRunState {
  return {
    answer: null,
    result: undefined,
    memory: new UnconstrainedMemory(),
    iteration: tools.length,
    steps: tools.map((tool, idx) => ({
      id: String(idx),
      iteration: idx,
      tool,
      input: {},
      output: new StringToolOutput(""),
      error: null,
    })),
    usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
  };
}

async function createRequirement(options: ConditionalRequirementOptions) {
  const requirement = new ConditionalRequirement(toolA, options);
  await requirement.init([toolA, toolB, toolC], undefined as any);
  return requirement;
}

describe("ConditionalRequirement", () => {
  describe("onlyBefore", () => {
    it("disables the target once a 'before' tool has been called", async () => {
      const requirement = await createRequirement({ onlyBefore: [toolB] });

      const [initial] = await requirement.run(createState());
      expect(initial.allowed).toBe(true);

      const [unrelated] = await requirement.run(createState(toolC));
      expect(unrelated.allowed).toBe(true);

      const [afterB] = await requirement.run(createState(toolB));
      expect(afterB).toMatchObject({
        target: "tool_a",
        allowed: false,
        forced: false,
        hidden: false,
      });

      const [laterAfterB] = await requirement.run(createState(toolB, toolC));
      expect(laterAfterB.allowed).toBe(false);
    });

    it("is combined with onlyAfter", async () => {
      const requirement = await createRequirement({ onlyBefore: [toolB], onlyAfter: [toolC] });

      expect((await requirement.run(createState()))[0].allowed).toBe(false);
      expect((await requirement.run(createState(toolC)))[0].allowed).toBe(true);
      expect((await requirement.run(createState(toolC, toolB)))[0].allowed).toBe(false);
      expect((await requirement.run(createState(toolB, toolC)))[0].allowed).toBe(false);
    });

    it("prevents forcing via forceAfter once a 'before' tool has been called", async () => {
      const requirement = await createRequirement({ onlyBefore: [toolB], forceAfter: [toolC] });

      const [forced] = await requirement.run(createState(toolC));
      expect(forced).toMatchObject({ allowed: true, forced: true });

      const [blocked] = await requirement.run(createState(toolB, toolC));
      expect(blocked).toMatchObject({ allowed: false, forced: false });
    });

    it("throws when forceAtStep cannot be satisfied", async () => {
      const requirement = await createRequirement({ onlyBefore: [toolB], forceAtStep: 2 });

      await expect(requirement.run(createState(toolB))).rejects.toThrow(RequirementError);
    });
  });

  describe("onlyAfter", () => {
    it("does not disable the target after a 'forceAfter' tool has been called", async () => {
      const requirement = await createRequirement({ onlyAfter: [toolC], forceAfter: [toolB] });

      const [rule] = await requirement.run(createState(toolC, toolB));
      expect(rule).toMatchObject({ allowed: true, forced: true });
    });
  });
});

describe("ConditionalRequirement onlyBefore targets", () => {
  it("accepts tool names", async () => {
    const requirement = await createRequirement({ onlyBefore: [toolB.name] });

    expect((await requirement.run(createState()))[0].allowed).toBe(true);
    expect((await requirement.run(createState(toolB)))[0].allowed).toBe(false);
  });

  it.each([
    { onlySuccessInvocations: true, allowed: true },
    { onlySuccessInvocations: false, allowed: false },
  ])(
    "handles a failed 'before' call with onlySuccessInvocations=$onlySuccessInvocations",
    async ({ onlySuccessInvocations, allowed }) => {
      const requirement = await createRequirement({ onlyBefore: [toolB], onlySuccessInvocations });
      const state = createState(toolB);
      state.steps[0].error = new FrameworkError("failed");

      expect((await requirement.run(state))[0].allowed).toBe(allowed);
    },
  );
});
