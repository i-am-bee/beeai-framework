/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";

const tool = new OpenMeteoTool();
const result = await tool.run({
  location: { name: "New York" },
  start_date: "2024-10-10",
  end_date: "2024-10-10",
});
console.log(result.getTextContent());
