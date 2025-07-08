/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import dotenv from "dotenv";
import { FrameworkError } from "@/errors.js";
dotenv.config();
dotenv.config({
  path: ".env.test",
  override: true,
});
dotenv.config({
  path: ".env.test.local",
  override: true,
});

expect.addSnapshotSerializer({
  serialize(val: FrameworkError): string {
    return val.explain();
  },
  test(val): boolean {
    return val && val instanceof FrameworkError;
  },
});
