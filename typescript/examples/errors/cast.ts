/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { FrameworkError } from "beeai-framework/errors";

function getUser() {
  throw new Error("User was not found!");
}

try {
  getUser();
} catch (e) {
  const err = FrameworkError.ensure(e);
  console.log(err.dump());
  console.log(err.explain());
}
