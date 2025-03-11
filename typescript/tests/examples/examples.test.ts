/**
 * Copyright 2025 IBM Corp.
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

import { expect } from "vitest";
import { exec } from "child_process";
import { glob } from "glob";
import { isTruthy } from "remeda";
import { getEnv } from "@/internals/env.js";
import { ExecException } from "node:child_process";

const execAsync = (command: string) =>
  new Promise<{ stdout: string; stderr: string }>((resolve, reject) =>
    exec(
      command,
      {
        shell: process.env.SHELL || "/bin/bash",
      },
      (error, stdout, stderr) => (error ? reject(error) : resolve({ stdout, stderr })),
    ),
  );

const includePattern = process.env.INCLUDE_PATTERN || `./examples/**/*.ts`;
const excludePattern = process.env.EXCLUDE_PATTERN || ``;

const exclude: string[] = [
  "examples/workflows/contentCreator.ts",
  "examples/workflows/competitive-analysis/**/*.ts",
  "examples/agents/experimental/remote.ts",
  "examples/playground/**/*.ts",
  // prevents 'Too many requests' error on Free Tier
  !getEnv("WATSONX_API_KEY") && [
    "examples/backend/providers/watson*.ts",
    "examples/agents/experimental/replan.ts",
    "examples/agents/experimental/streamlit.ts",
    "examples/agents/granite/*.ts",
    "examples/agents/granite/single_turn.ts",
  ],
  !getEnv("GROQ_API_KEY") && [
    "examples/agents/sql.ts",
    "examples/backend/providers/groq.ts",
    "examples/workflows/*",
  ],
  !getEnv("OPENAI_API_KEY") && [
    "examples/agents/bee_reusable.ts",
    "examples/backend/providers/openai.ts",
  ],
  !getEnv("AZURE_OPENAI_API_KEY") && ["examples/backend/providers/azure-openai.ts"],
  !getEnv("COHERE_API_KEY") && ["examples/backend/providers/langchain.ts"],
  !getEnv("CODE_INTERPRETER_URL") && ["examples/tools/custom/python.ts"],
  !getEnv("ELASTICSEARCH_NODE") && ["examples/agents/elasticsearch.ts"],
  !getEnv("AWS_REGION") && ["examples/backend/providers/amazon-bedrock.ts"],
  !getEnv("GOOGLE_APPLICATION_CREDENTIALS") && ["examples/backend/providers/vertexai.ts"],
  !getEnv("ANTHROPIC_API_KEY") && ["examples/backend/providers/anthropic.ts"],
  "examples/tools/custom/extending.ts", // DDG problems
]
  .filter(isTruthy)
  .flat(); // list of examples that are excluded

describe("E2E Examples", async () => {
  const exampleFiles = await glob(includePattern, {
    cwd: process.cwd(),
    dot: false,
    realpath: true,
    ignore: [exclude, excludePattern].flat(),
  });

  it.concurrent.each(exampleFiles)(`Run %s`, async (example) => {
    await execAsync(`yarn start -- ${example} <<< "Hello world"`)
      .then(({ stdout, stderr }) => {
        // eslint-disable-next-line no-console
        console.log("STDOUT:", stdout);
        expect(stderr).toBeFalsy();
      })
      .catch((_e) => {
        const error = _e as ExecException;

        // eslint-disable-next-line no-console
        console.error(error.message);
        // eslint-disable-next-line no-console
        console.error("STDOUT:", error.stdout);

        expect(error.stderr).toBeFalsy();
        expect(error.code).toBe(0);
      });
  });
});
