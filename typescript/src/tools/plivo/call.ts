/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  BaseToolOptions,
  BaseToolRunOptions,
  ToolEmitter,
  JSONToolOutput,
  Tool,
  ToolError,
  ToolInput,
} from "@/tools/base.js";
import { z } from "zod";
import { RunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";

type ToolOptions = BaseToolOptions;
type ToolRunOptions = BaseToolRunOptions;

export class PlivoMakeCallToolOutput extends JSONToolOutput<Record<string, any>> {}

export class PlivoMakeCallTool extends Tool<
  PlivoMakeCallToolOutput,
  ToolOptions,
  ToolRunOptions
> {
  name = "PlivoMakeCall";
  description =
    "Place an outbound phone call using Plivo that runs the answer XML at the given URL when answered.";

  inputSchema() {
    return z
      .object({
        to: z
          .string()
          .min(1)
          .describe("Destination phone number to call in E.164 format, e.g. +14150000001."),
        answer_url: z
          .string()
          .url()
          .describe(
            "Publicly reachable URL that returns Plivo answer XML (e.g. a <Speak> element) when the call is answered.",
          ),
      })
      .strip();
  }

  public readonly emitter: ToolEmitter<ToolInput<this>, PlivoMakeCallToolOutput> =
    Emitter.root.child({
      namespace: ["tool", "plivo", "call"],
      creator: this,
    });

  static {
    this.register();
  }

  public constructor(options: Partial<ToolOptions> = {}) {
    super(options);
  }

  protected async _run(
    { to, answer_url: answerUrl }: ToolInput<this>,
    _options: Partial<BaseToolRunOptions>,
    run: RunContext<this>,
  ) {
    const authId = process.env.PLIVO_AUTH_ID;
    const authToken = process.env.PLIVO_AUTH_TOKEN;
    const src = process.env.PLIVO_SRC;
    if (!authId || !authToken || !src) {
      throw new ToolError("PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, and PLIVO_SRC must be set.");
    }

    const response = await fetch(`https://api.plivo.com/v1/Account/${authId}/Call/`, {
      method: "POST",
      headers: {
        Authorization: `Basic ${Buffer.from(`${authId}:${authToken}`).toString("base64")}`,
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({ from: src, to, answer_url: answerUrl }),
      signal: run.signal,
    });

    if (!response.ok) {
      throw new ToolError("Request to Plivo Call API has failed!", [
        new Error(await response.text()),
      ]);
    }

    return new PlivoMakeCallToolOutput(await response.json());
  }
}
