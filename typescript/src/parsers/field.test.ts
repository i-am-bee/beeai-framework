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

import { JSONParserField, ZodParserField } from "@/parsers/field.js";
import { z } from "zod";
import { splitString } from "@/internals/helpers/string.js";

describe("Parser Fields", () => {
  describe("JSON", () => {
    it("Object", async () => {
      const field = new JSONParserField({
        schema: z.record(z.any()),
        base: {},
      });
      const obj = { a: { b: { c: { d: 1 } } }, b: 2 };
      const content = JSON.stringify(obj);
      for (const chunk of splitString(content, { size: 5, overlap: 0 })) {
        field.write(chunk);
      }
      await field.end();
      expect(field.raw).toBe(content);
      expect(JSON.stringify(field.get())).toMatchInlineSnapshot(
        `"{"a":{"b":{"c":{"d":1}}},"b":2}"`,
      );
    });

    it("String", async () => {
      const field = new JSONParserField({
        schema: z.string(),
        base: "",
      });
      expect(field.getPartial()).toStrictEqual("");
      field.write(`"Hello\\nworld!"`);
      await field.end();
      expect(field.get()).toStrictEqual(`Hello\nworld!`);
    });

    it("Array of booleans", async () => {
      const field = new JSONParserField({
        schema: z.array(z.coerce.boolean()),
        base: [],
      });
      expect(field.getPartial()).toStrictEqual([]);
      field.write("[true,false,true]");
      expect(field.getPartial()).toStrictEqual([true, false, true]);
      await field.end();
      expect(field.get()).toStrictEqual([true, false, true]);
    });
  });

  describe("Invalid JSON", () => {
    it("Object", async () => {
      const field = new JSONParserField({
        schema: z.object({}).passthrough(),
        base: {},
        matchPair: ["{", "}"],
      });

      const validPart = `{"a":{"b":{"c":{"d":1}}},"b":2}`;
      const invalidPart = `{"a":{"b":{"c":{"d":1}}},"b":2,}`;

      const content = `Here is the object that you were asking for: ${invalidPart} Thank you!`;
      for (const chunk of splitString(content, { size: 4, overlap: 0 })) {
        field.write(chunk);
      }
      await field.end();
      expect(field.raw).toBe(invalidPart);
      expect(JSON.stringify(field.get())).toBe(validPart);
    });
  });

  it("String", async () => {
    const field = new ZodParserField(z.string());
    const content = "Hello world!";
    for (const chunk of splitString(content, { size: 2, overlap: 0 })) {
      field.write(chunk);
    }
    await field.end();
    expect(field.raw).toBe(content);
    expect(field.get()).toStrictEqual(content);
  });

  describe("Zod", () => {
    it("Enum", async () => {
      const values = {
        apple: "apple",
      };
      const field = new ZodParserField(z.pipeline(z.string().trim(), z.nativeEnum(values)));
      field.write(" apple\n");
      await field.end();
      expect(field.get()).toBe(values.apple);
    });

    it("Number", async () => {
      const field = new ZodParserField(z.coerce.number().int());
      expect(field.getPartial()).toBe("");
      field.write("1000");
      expect(field.getPartial()).toBe("1000");
      await field.end();
      expect(field.get()).toBe(1000);
    });
  });
});
