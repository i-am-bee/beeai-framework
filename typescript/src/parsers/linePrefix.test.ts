/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { LinePrefixParser, LinePrefixParserError } from "@/parsers/linePrefix.js";
import { z } from "zod";
import { JSONParserField, ZodParserField } from "@/parsers/field.js";
import { splitString } from "@/internals/helpers/string.js";
import { ValueError } from "@/errors.js";

describe("LinePrefixParser", () => {
  it.each([1, 5, 20, 50, 100, 1000])("Handles arbitrary chunk size", async (chunkSize) => {
    const parser = new LinePrefixParser({
      thought: {
        prefix: "Thought:",
        next: ["tool_name", "final_answer"],
        isStart: true,
        field: new ZodParserField(z.string().min(1)),
      },
      tool_name: {
        prefix: "Tool Name:",
        next: ["tool_caption"],
        field: new ZodParserField(z.enum(["Google"])),
      },
      tool_caption: {
        prefix: "Tool Caption:",
        next: ["tool_input"],
        field: new ZodParserField(z.string().min(1)),
      },
      tool_input: {
        prefix: "Tool Input:",
        next: ["tool_output"],
        field: new JSONParserField({
          schema: z.record(z.string()),
          base: {},
        }),
      },
      tool_output: {
        prefix: "Tool Output:",
        next: ["final_answer"],
        isEnd: true,
        field: new JSONParserField({
          schema: z.record(z.string()),
          base: {},
        }),
      },
      final_answer: {
        prefix: "Final Answer:",
        next: [],
        isStart: true,
        isEnd: true,
        field: new ZodParserField(z.string().min(1)),
      },
    } as const);

    // Gather all update events and then compare if they equal the same
    const updates = {
      partial: {},
      final: {},
    };
    parser.emitter.on("update", async ({ key, field }) => {
      Object.assign(updates.final, { [key]: field.raw });
    });
    parser.emitter.on("partialUpdate", async ({ key, field }) => {
      Object.assign(updates.partial, { [key]: field.raw });
    });

    const chunks = splitString(
      `Thought: I don't know who is president of USA.
But I can use GoogleSearch to find out.
Tool Name: Google
Tool Caption: Searching for USA president
Tool Input: {"query":"USA president"}
Tool Output: {"answer":"Joe Biden"}
Final Answer: Joe Biden is the current president of USA.`,
      { size: chunkSize, overlap: 0 },
    );
    for (const chunk of chunks) {
      await parser.add(chunk);
    }
    const state = await parser.end();

    expect(state).toStrictEqual({
      thought: `I don't know who is president of USA.
But I can use GoogleSearch to find out.`,
      tool_name: "Google",
      tool_caption: "Searching for USA president",
      tool_input: { query: "USA president" },
      tool_output: { answer: "Joe Biden" },
      final_answer: "Joe Biden is the current president of USA.",
    });
    expect(updates.partial).toStrictEqual(updates.final);
  });

  describe("Streaming", () => {
    it("JSON", async () => {
      const parser = new LinePrefixParser({
        start: {
          prefix: "Start:",
          field: new JSONParserField({
            schema: z.object({
              a: z.number().int(),
              b: z.number().int(),
            }),
            base: {},
          }),
          next: [],
          isStart: true,
          isEnd: true,
        },
      } as const);
      await parser.add("Start:");

      for (const field of [
        {
          value: `{"a`,
          expected: {},
        },
        {
          value: `":1,`,
          expected: { a: 1 },
        },
        {
          value: `"b`,
          expected: { a: 1 },
        },
        {
          value: `":`,
          expected: { a: 1, b: undefined },
        },
        {
          value: `2,`,
          expected: { a: 1, b: 2 },
        },
      ]) {
        await Promise.all([
          new Promise<void>((resolve) => {
            parser.emitter.on(
              "partialUpdate",
              (update) => {
                expect(update.delta).toStrictEqual(field.value);
                expect(update.field.getPartial()).toStrictEqual(field.expected);
                resolve();
              },
              {
                once: true,
                isBlocking: true,
              },
            );
          }),
          parser.add(field.value),
        ]);
      }
      await parser.end();
    });
  });

  describe("Fallback", () => {
    const getParser = () =>
      new LinePrefixParser(
        {
          thought: {
            prefix: "Thought:",
            field: new ZodParserField(z.string()),
            isStart: true,
            next: ["final_answer"],
          },
          final_answer: {
            prefix: "Final Answer:",
            field: new ZodParserField(z.string()),
            isEnd: true,
            next: [],
          },
        },
        {
          fallback: (value) =>
            [
              { key: "thought", value: "I now know the final answer." },
              {
                key: "final_answer",
                value,
              },
            ] as const,
        },
      );

    it("Process", async () => {
      const parser = getParser();
      await parser.add("2+2=4");
      await expect(parser.end()).resolves.toMatchInlineSnapshot(`
        {
          "final_answer": "2+2=4",
          "thought": "I now know the final answer.",
        }
      `);
    });

    it("Process #2", async () => {
      const parser = getParser();
      await parser.add("A\nB\nC");
      await expect(parser.end()).resolves.toMatchInlineSnapshot(`
        {
          "final_answer": "A
        B
        C",
          "thought": "I now know the final answer.",
        }
      `);
    });

    it("Process #3", async () => {
      const parser = getParser();
      await parser.add("Thought");
      await expect(parser.end()).resolves.toMatchInlineSnapshot(`
        {
          "final_answer": "Thought",
          "thought": "I now know the final answer.",
        }
      `);
    });
  });

  describe("Edge cases", () => {
    it("Prevents processing when line starts with a potential prefix", async () => {
      const parser = new LinePrefixParser({
        final_answer: {
          prefix: "Final Answer:",
          field: new ZodParserField(z.string()),
          isStart: true,
          isEnd: true,
          next: ["final_detail"],
        },
        final_detail: {
          prefix: "Final Detail:",
          field: new ZodParserField(z.string()),
          isStart: true,
          isEnd: true,
          next: ["final_answer"],
        },
      });

      const deltas: string[] = [];
      parser.emitter.on("partialUpdate", ({ key, delta }) => {
        expect(key).toStrictEqual("final_answer");
        deltas.push(delta);
      });
      await parser.add("Final Answer: 4\n");
      await parser.add("\n"); // we can stream for multiple lines
      expect(deltas.join("")).toStrictEqual("4\n");
      expect(deltas.length).toBe(2);
      await parser.add("Final ");
      expect(deltas.length).toBe(2);
      await parser.add("Detail");
      expect(deltas.length).toBe(2);
      await parser.end();
      expect(deltas.join("")).toStrictEqual("4\n\nFinal Detail");
      expect(deltas.length).toBe(3);
    });

    it.each([true, false])("Correctly interprets new lines", async (shouldThrow) => {
      const input = `Thought: Summarize the discussion. Final Answer: The discussion thread is about ...`;
      const parser = new LinePrefixParser({
        thought: {
          prefix: "Thought:",
          next: ["final_answer"],
          isEnd: !shouldThrow,
          isStart: true,
          field: new ZodParserField(z.string().min(1)),
        },
        final_answer: {
          prefix: "Final Answer:",
          next: [],
          isStart: true,
          isEnd: true,
          field: new ZodParserField(z.string().min(1)),
        },
      } as const);

      const chunks = splitString(input, { size: 25, overlap: 0 });
      for (const chunk of chunks) {
        await parser.add(chunk);
      }

      if (shouldThrow) {
        await expect(parser.end()).rejects.toThrowError(LinePrefixParserError);
      } else {
        await parser.end();
        expect(parser.finalState).toMatchObject({
          thought: `Summarize the discussion. Final Answer: The discussion thread is about ...`,
        });
      }
    });

    it("Ignores unrelated text and non-starting nodes", async () => {
      const parser = new LinePrefixParser(
        {
          first: {
            prefix: "First:",
            field: new ZodParserField(z.string()),
            isStart: true,
            isEnd: false,
            next: ["second"],
          },
          second: {
            prefix: "Second:",
            field: new ZodParserField(z.string()),
            isStart: false,
            isEnd: true,
            next: [],
          },
        },
        {
          waitForStartNode: true,
        },
      );

      await parser.add("      Random text\nthat\nshould be ignored.\n");
      await parser.add("  Second: This should be ignored too.\n");
      await parser.add("First: first\n");
      await parser.add("Second: second\n");
      await parser.end();
      expect(parser.finalState).toMatchInlineSnapshot(`
        {
          "first": "first",
          "second": "second
        ",
        }
      `);
    });

    it.each([true, false])(
      "Ignores other prefixes when the current node is a termination node",
      async (endOnRepeat) => {
        const parser = new LinePrefixParser(
          {
            thought: {
              prefix: "Thought:",
              field: new ZodParserField(z.string()),
              isStart: true,
              isEnd: true,
              next: ["answer"],
            },
            answer: {
              prefix: "Answer:",
              field: new ZodParserField(z.string()),
              isStart: true,
              isEnd: true,
              next: [],
            },
          },
          {
            endOnRepeat,
          },
        );

        await parser.add(`Answer: The answer is 39, see the intermediate steps that I did:\n`);
        await parser.add(`Thought: 3*(2+5+6)=? can be rewritten as 3*13\n`);
        await parser.add(`Analyze: 3*13=39\n`);
        await parser.add(`Outcome: 39`);
        await parser.end();
        expect(parser.finalState).toStrictEqual({
          answer: `The answer is 39, see the intermediate steps that I did:
Thought: 3*(2+5+6)=? can be rewritten as 3*13
Analyze: 3*13=39
Outcome: 39`,
        });
      },
    );

    it("Premature ends when existing node visited", async () => {
      const parser = new LinePrefixParser(
        {
          thought: {
            prefix: "Thought:",
            field: new ZodParserField(z.string()),
            isStart: true,
            isEnd: false,
            next: ["tool_input", "final_answer"],
          },
          tool_input: {
            prefix: "Tool Input:",
            field: new ZodParserField(z.string()),
            isStart: false,
            isEnd: true,
            next: ["tool_output"],
          },
          tool_output: {
            prefix: "Tool Output:",
            field: new ZodParserField(z.string()),
            isStart: false,
            isEnd: true,
            next: [],
          },
          final_answer: {
            prefix: "Final Answer:",
            field: new ZodParserField(z.string()),
            isStart: false,
            isEnd: true,
            next: [],
          },
        },
        {
          endOnRepeat: true,
        },
      );

      await parser.add(
        [`Thought: Hello!`, `Tool Input: {}`, `Thought: This is ignored.`].join("\n"),
      );
      expect(parser.isDone).toBe(true);
      expect(parser.finalState).toMatchInlineSnapshot(`
        {
          "thought": "Hello!",
          "tool_input": "{}",
        }
      `);
    });
  });

  describe("Chunk boundaries", () => {
    // The same grammar as the one used by the default ReAct runner
    const createReActParser = () =>
      new LinePrefixParser(
        {
          thought: {
            prefix: "Thought:",
            next: ["tool_name", "final_answer"],
            isStart: true,
            field: new ZodParserField(z.string().min(1)),
          },
          tool_name: {
            prefix: "Function Name:",
            next: ["tool_input"],
            field: new ZodParserField(z.pipeline(z.string().trim(), z.enum(["Search", "Calc"]))),
          },
          tool_input: {
            prefix: "Function Input:",
            next: ["tool_output"],
            isEnd: true,
            field: new JSONParserField({
              schema: z.object({}).passthrough(),
              base: {},
              matchPair: ["{", "}"],
            }),
          },
          tool_output: {
            prefix: "Function Output:",
            next: ["final_answer"],
            isEnd: true,
            field: new ZodParserField(z.string()),
          },
          final_answer: {
            prefix: "Final Answer:",
            next: [],
            isStart: true,
            isEnd: true,
            field: new ZodParserField(z.string().min(1)),
          },
        } as const,
        {
          waitForStartNode: true,
          endOnRepeat: true,
          fallback: (stash) =>
            stash
              ? [
                  { key: "thought", value: "I now know the final answer." },
                  { key: "final_answer", value: stash },
                ]
              : [],
        },
      );

    const parse = async (chunks: string[]) => {
      const parser = createReActParser();
      try {
        for (const chunk of chunks) {
          await parser.add(chunk);
        }
        return { state: await parser.end() };
      } catch (e) {
        if (e instanceof LinePrefixParserError) {
          return { error: e.reason };
        }
        throw e;
      }
    };

    it("Parses the ReAct output when the indentation of a line arrives as a separate chunk", async () => {
      expect(await parse(["Thought: a\n", " ", "Final Answer: b"])).toStrictEqual({
        state: { thought: "a", final_answer: "b" },
      });
    });

    it.each([
      ["whitespace-only chunk", ["Thought: a\n", " ", "Final Answer: b"]],
      ["multiple spaces", ["Thought: a\n", "  ", "Final Answer: b"]],
      ["tab", ["Thought: a\n", "\t", "Final Answer: b"]],
      ["newline and indentation in one chunk", ["Thought: a", "\n ", "Final Answer: b"]],
      ["indentation split from the prefix", ["Thought: a\n", "  ", "Final", " Answer: b"]],
      [
        "tool call",
        ["Thought: a\n", " ", "Function Name:", "  Search\n", " ", 'Function Input: {"q":1}\n'],
      ],
    ])("Recognizes a prefix after a whitespace-only chunk (%s)", async (_, chunks) => {
      const expected = await parse([chunks.join("")]);
      expect(expected).toHaveProperty("state");
      expect(await parse(chunks)).toStrictEqual(expected);
    });

    const inputs = [
      "Thought: a\n Final Answer: b",
      "Thought: a\n\t\nFinal Answer: b",
      '  Thought: need data\n  Function Name:  Search\n  Function Input: {"q": "cats"}\n',
      "Thought: first\n   second\n\nFinal Answer: done\n  more",
      "Preamble\nThought: x\nFinal Answer: y\n",
      "Just a plain answer",
    ];

    it.each(inputs)("Result does not depend on the split point: %j", async (input) => {
      const expected = await parse([input]);
      expect(expected).toHaveProperty("state");

      for (let i = 0; i <= input.length; i++) {
        expect(await parse([input.slice(0, i), input.slice(i)]), `split at ${i}`).toStrictEqual(
          expected,
        );

        for (let j = i; j <= input.length; j++) {
          expect(
            await parse([input.slice(0, i), input.slice(i, j), input.slice(j)]),
            `split at ${i} and ${j}`,
          ).toStrictEqual(expected);
        }
      }
    });

    it.each(inputs)("Result does not depend on random chunking: %j", async (input) => {
      const expected = await parse([input]);

      // deterministic pseudo-random generator (LCG) to keep the test reproducible
      let seed = 42;
      const random = () => {
        seed = (Math.imul(seed, 1103515245) + 12345) & 0x7fffffff;
        return seed / 0x7fffffff;
      };

      for (let run = 0; run < 100; run++) {
        const chunks: string[] = [];
        for (let i = 0; i < input.length; ) {
          const size = 1 + Math.floor(random() * 6);
          chunks.push(input.slice(i, i + size));
          i += size;
        }
        expect(await parse(chunks), JSON.stringify(chunks)).toStrictEqual(expected);
      }
    });
  });

  describe("Error handling", () => {
    it("Throws when no node is provided", () => {
      expect(() => new LinePrefixParser({})).toThrowError(ValueError);
    });

    it("Throws when no start node is provided", () => {
      expect(
        () =>
          new LinePrefixParser({
            a: {
              prefix: "A:",
              isStart: false,
              isEnd: true,
              next: [],
              field: new ZodParserField(z.string()),
            },
          }),
      ).toThrowErrorMatchingInlineSnapshot(`ValueError: At least one start node must be provided!`);
    });

    it("Throws when no start node is provided", () => {
      expect(
        () =>
          new LinePrefixParser({
            a: {
              prefix: "A:",
              isStart: true,
              isEnd: false,
              next: [],
              field: new ZodParserField(z.string()),
            },
          }),
      ).toThrowErrorMatchingInlineSnapshot(`ValueError: At least one end node must be provided!`);
    });

    it("Throws for non-existing transition", () => {
      expect(
        () =>
          new LinePrefixParser({
            a: {
              prefix: "A:",
              isStart: true,
              isEnd: true,
              next: ["b"] as any[],
              field: new ZodParserField(z.string()),
            },
          }),
      ).toThrowErrorMatchingInlineSnapshot(
        `ValueError: Node 'a' contains a transition to non-existing node 'b'.`,
      );
    });

    it("Throws when self-reference", () => {
      expect(
        () =>
          new LinePrefixParser({
            a: {
              prefix: "A:",
              isStart: true,
              isEnd: true,
              next: ["a"],
              field: new ZodParserField(z.string()),
            },
          }),
      ).toThrowErrorMatchingInlineSnapshot(`ValueError: Node 'a' cannot point to itself.`);
    });
  });
});
