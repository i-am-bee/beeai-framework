/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { expect } from "vitest";
import { findFirstPair, splitString } from "./string.js";
import { ValueError } from "@/errors.js";

describe("String Utilities", () => {
  describe("splitString", () => {
    it("Works", () => {
      const text =
        "00 01 02 03 04 05 06 07 08 09 " +
        "10 11 12 13 14 15 16 17 18 19 " +
        "20 21 22 23 24 25 26 27 28 29 " +
        "30 31 32 33 34 35 36 37 38 39 ";
      const chunks = [...splitString(text, { size: 30, overlap: 15, trim: false })];
      expect(chunks).toEqual([
        "00 01 02 03 04 05 06 07 08 09 ",
        "05 06 07 08 09 10 11 12 13 14 ",
        "10 11 12 13 14 15 16 17 18 19 ",
        "15 16 17 18 19 20 21 22 23 24 ",
        "20 21 22 23 24 25 26 27 28 29 ",
        "25 26 27 28 29 30 31 32 33 34 ",
        "30 31 32 33 34 35 36 37 38 39 ",
      ]);
    });
    it("Handles edge-cases", () => {
      const text = `Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur ac viverra dolor, eu fringilla magna.`;
      const output = Array.from(splitString(text, { size: 35, overlap: 10 })).join("|");
      expect(output).toMatchInlineSnapshot(
        `"Lorem ipsum dolor sit amet, consect|t, consectetur adipiscing elit. Cur| elit. Curabitur ac viverra dolor, |ra dolor, eu fringilla magna."`,
      );
    });
  });

  describe("findFirstPair", () => {
    it.each([
      [["", ""]],
      [["", "."]],
      [[".", ""]],
      [[null, "XXX"]],
      [[null, false]],
      [[undefined, undefined]],
      [[[]]],
      [[]],
    ] as const)("Throws %s", (pair: any) => {
      expect(() => findFirstPair("Hello world!", pair)).toThrowError(ValueError);
    });

    it.each([
      {
        input: "This ?is?.",
        pair: ["?", "?"],
      },
      {
        input: 'This is the output {"a":1}.',
        pair: ["{", "}"],
      },
      {
        input: "[[[0]]]",
        pair: ["[", "]"],
      },
      {
        pair: ["```python-app", "```"],
        input: `Here is a simple Streamlit app that displays a header, a subheader, and a button. When the button is clicked, it displays a success message.

\`\`\`

\`\`\`python-app
import streamlit as st

st.header("My First Streamlit App")

st.subheader("This is a simple app")

if st.button("Click me"):
    st.success("Button clicked!")
\`\`\`

\`\`\`

Let me know if you'd like to add any features to this app!`,
      },
    ])("Works %#", ({ input, pair }) => {
      const generated = findFirstPair(input, pair as [string, string])!;
      expect(generated).toBeTruthy();
      expect({ inner: generated.inner, outer: generated.outer }).toMatchSnapshot();
    });
  });
});
