/**
 * Copyright 2024 IBM Corp.
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

import { DynamicTool, StringToolOutput } from "beeai-framework/tools/base";
import { z } from "zod";

const tool = new DynamicTool({
  name: "GenerateRandomNumber",
  description: "Generates a random number in the given interval.",
  inputSchema: z.object({
    min: z.number().int().min(0),
    max: z.number().int(),
  }),
  async handler(input) {
    const min = Math.min(input.min, input.max);
    const max = Math.max(input.max, input.min);

    const number = Math.floor(Math.random() * (max - min + 1)) + min;
    return new StringToolOutput(number.toString());
  },
});
