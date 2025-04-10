/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { PromptTemplateError, PromptTemplate, ValidationPromptTemplateError } from "@/template.js";
import { z, ZodType } from "zod";

describe("Prompt Template", () => {
  describe("Rendering", () => {
    it.each([1, "1", true, false, "Hey!"])("Handles primitives (%s)", (value) => {
      const template = new PromptTemplate({
        schema: z.object({
          value: z.union([z.number(), z.string(), z.boolean()]),
        }),
        template: `Input: {{value}}!`,
      });

      expect(template.render({ value })).toStrictEqual(`Input: ${value}!`);
    });

    it.each([[[]], [[1, 2, 3]], [["a", "b", "c"]]])("Handles arrays (%s)", (values) => {
      const template = new PromptTemplate({
        schema: z.object({
          values: z.array(z.any()),
        }),
        template: `Input: {{#values}}{{.}}{{/values}}`,
      });

      expect(template.render({ values })).toStrictEqual(`Input: ${values.join("")}`);
    });

    it.each([[[{ name: "Tomas" }, { name: "Lukas" }, { name: "Alex" }]], [[]], [[{}]]])(
      "Handles plain objects (%s)",
      (values) => {
        const template = new PromptTemplate({
          schema: z.object({
            values: z.array(z.record(z.string())),
          }),
          template: `{{#values}}Name: {{name}}\n{{/values}}`,
        });

        expect(template.render({ values })).toMatch(
          values.map((value: any) => `Name: ${value?.name ?? ""}\n`).join(""),
        );
      },
    );

    it("Handles multiple occurrences of same variable", () => {
      const template = new PromptTemplate({
        schema: z.object({
          firstName: z.string(),
          lastName: z.string(),
        }),
        template: `First Name: {{firstName}}. Last Name {{lastName}}. Full Name: {{firstName}} {{lastName}}.`,
      });
      expect(template.render({ firstName: "A", lastName: "B" })).toMatchInlineSnapshot(
        `"First Name: A. Last Name B. Full Name: A B."`,
      );
    });

    it("Uses the value from defaults", () => {
      const template = new PromptTemplate({
        schema: z.object({
          name: z.string().default("Alex"),
        }),
        template: `{{name}}`,
      });
      expect(template.render({ name: undefined })).toStrictEqual("Alex");
    });
  });

  describe("Validation", () => {
    it.each([null, undefined])("Validates required variables (%s)", (value) => {
      const template = new PromptTemplate({
        schema: z.object({
          value: z.string(),
        }),
        template: `Input: {{value}}!`,
      });

      expect(() => template.render({ value: value as any })).toThrowError(PromptTemplateError);
    });

    it.each([null, undefined])("Validates only required variables (%s)", (value) => {
      const template = new PromptTemplate({
        schema: z.object({
          value: z.string().nullish(),
        }),
        template: `Input: {{value}}!`,
      });

      expect(template.render({ value })).toBeTruthy();
    });

    it.each([
      [
        {
          template: `{{#value}}{{.}}{{/value}}`,
          values: [true, false, {}, new Map(), new Set(), "", 1, false],
        },
      ],
      [
        {
          template: `{{value}}`,
          values: [{}, [], new Map(), new Set()],
        },
      ],
      [
        {
          template: `{{value}}`,
          values: [() => ({})],
        },
      ],
    ])("Throws wrong data-type is passed (%s)", ({ template: templateRaw, values }) => {
      const template = new PromptTemplate({
        schema: z.object({ value: z.string().min(1) }),
        template: templateRaw,
      });

      for (const value of values) {
        expect(() => template.render({ value: value as any })).toThrowError(
          ValidationPromptTemplateError,
        );
      }
    });
  });

  describe("Functions", () => {
    it("Trims", () => {
      const template = new PromptTemplate({
        template: `{{#trim}}{{names}},{{/trim}}`,
        schema: z.object({
          names: z.array(z.string()),
        }),
      });
      expect(template.render({ names: ["Tomas", "Alex"] })).toMatchInlineSnapshot(`"Tomas,Alex"`);
    });

    it("Custom function", () => {
      const template = new PromptTemplate({
        template: [`Name: {{name}}`, `User Name: {{userName}}`].join("\n"),
        schema: z.object({ name: z.string() }),
        functions: {
          userName: function () {
            return this.name.replaceAll(" ", "-").toLowerCase().trim();
          },
        },
      });
      expect(
        template.render({
          name: "John Doe",
        }),
      ).toMatchInlineSnapshot(`
        "Name: John Doe
        User Name: john-doe"
      `);
    });
  });

  describe("Customization", () => {
    const createTemplate = () => {
      return new PromptTemplate({
        template: `Hello <<name>>!`,
        schema: z.object({
          name: z.string(),
        }),
        customTags: ["<<", ">>"],
        escape: false,
      });
    };

    it("Clones", async () => {
      const template = createTemplate();
      const cloned = await template.clone();
      expect(cloned).toEqual(template);
    });

    it.each([
      <T extends ZodType>(template: PromptTemplate<T>) =>
        template.fork((config) => ({
          ...config,
          template: "Hi {{name}}!",
          customTags: ["{{", "}}"],
          functions: { formatDate: () => "Today" },
        })),
      <T extends ZodType>(template: PromptTemplate<T>) =>
        template.fork((config) => {
          config.template = "Hi {{name}}!";
          config.customTags = ["{{", "}}"];
        }),
    ])("Forks", (forkFn) => {
      const template = new PromptTemplate({
        template: `Hello <<name>>!`,
        schema: z.object({
          name: z.string(),
        }),
        customTags: ["<<", ">>"],
        escape: false,
      });
      const forked = forkFn(template);
      expect(template.render({ name: "Tomas" })).toEqual("Hello Tomas!");
      expect(forked.render({ name: "Tomas" })).toEqual("Hi Tomas!");
    });
  });
  test("Custom function", () => {
    const template = new PromptTemplate({
      schema: z.object({
        input: z.string(),
        meta: z
          .object({
            createdAt: z.string().datetime(),
          })
          .optional(),
      }),
      functions: {
        formatMeta: function () {
          return [`Created At: ${this.meta?.createdAt ?? "unknown"}`].filter(Boolean).join("\n");
        },
      },
      template: `Message: {{input}}\n\n{{formatMeta}}`,
    });

    expect(
      template.render({
        input: "Who are you?",
        meta: {
          createdAt: new Date("2024-09-10T17:55:44.947Z").toISOString(),
        },
      }),
    ).toMatchInlineSnapshot(`
      "Message: Who are you?

      Created At: 2024-09-10T17:55:44.947Z"
    `);
  });
});
