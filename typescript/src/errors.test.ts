/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { FrameworkError } from "@/errors.js";

describe("FrameworkError", () => {
  it("Correctly sets all properties", () => {
    const errMessage = "TOP LEVEL";

    const error = new FrameworkError(errMessage);
    expect(error.name).toBe("FrameworkError");
    expect(error.message).toBe(errMessage);
    expect(error.cause).toBe(undefined);
    expect(error.errors).toHaveLength(0);
  });

  it("Sets correct name when inheriting", () => {
    class CustomFrameworkError extends FrameworkError {}
    expect(CustomFrameworkError.name).toBe("CustomFrameworkError");
    expect(new CustomFrameworkError().name).toBe("CustomFrameworkError");

    class CustomCustomFrameworkError extends CustomFrameworkError {}
    expect(CustomCustomFrameworkError.name).toBe("CustomCustomFrameworkError");
    expect(new CustomCustomFrameworkError().name).toBe("CustomCustomFrameworkError");
  });

  it("Correctly traverse all errors", () => {
    const errors: Error[] = [
      new Error("1", { cause: new Error("1A") }),
      new Error("2"),
      new FrameworkError("3", [
        new Error("4"),
        new FrameworkError("5"),
        new FrameworkError("6", [new Error("7"), new Error("8")]),
        new TypeError("9"),
      ]),
      new Error("10"),
    ];

    const error = new FrameworkError("Error!", errors);
    expect(Array.from(error.traverseErrors()).map((err) => err.message)).toMatchInlineSnapshot(`
      [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
      ]
    `);
  });

  it("Correctly explain/format the error as a whole", () => {
    const error = new FrameworkError("Something went wrong!", [
      new Error("1", { cause: new Error("Something went wrong.") }),
      new Error("2"),
      new FrameworkError("3", [
        new Error("4"),
        new FrameworkError("5"),
        new FrameworkError("6", [new Error("7"), new Error("8")]),
        new TypeError("9"),
      ]),
    ]);
    expect(error.explain()).toMatchInlineSnapshot(`
      "FrameworkError: Something went wrong!
          Error: 1
          Cause: Error: Something went wrong.
              Error: 2
                  FrameworkError: 3
                      Error: 4
                          FrameworkError: 5
                              FrameworkError: 6
                                  Error: 7
                                      Error: 8
                                          TypeError: 9"
    `);
  });
});
