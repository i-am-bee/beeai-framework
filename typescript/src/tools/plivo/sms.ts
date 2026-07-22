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

export class PlivoSendMessageToolOutput extends JSONToolOutput<Record<string, any>> {}

export class PlivoSendMessageTool extends Tool<
  PlivoSendMessageToolOutput,
  ToolOptions,
  ToolRunOptions
> {
  name = "PlivoSendMessage";
  description = "Send an SMS text message to a phone number using Plivo.";

  inputSchema() {
    return z
      .object({
        to: z.string().min(1).describe("Recipient phone number in E.164 format, e.g. +14150000001."),
        text: z.string().min(1).describe("The text body of the SMS message to send."),
      })
      .strip();
  }

  public readonly emitter: ToolEmitter<ToolInput<this>, PlivoSendMessageToolOutput> =
    Emitter.root.child({
      namespace: ["tool", "plivo", "sms"],
      creator: this,
    });

  static {
    this.register();
  }

  public constructor(options: Partial<ToolOptions> = {}) {
    super(options);
  }

  protected async _run(
    { to, text }: ToolInput<this>,
    _options: Partial<BaseToolRunOptions>,
    run: RunContext<this>,
  ) {
    const authId = process.env.PLIVO_AUTH_ID;
    const authToken = process.env.PLIVO_AUTH_TOKEN;
    const src = process.env.PLIVO_SRC;
    if (!authId || !authToken || !src) {
      throw new ToolError("PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, and PLIVO_SRC must be set.");
    }

    const response = await fetch(`https://api.plivo.com/v1/Account/${authId}/Message/`, {
      method: "POST",
      headers: {
        Authorization: `Basic ${Buffer.from(`${authId}:${authToken}`).toString("base64")}`,
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({ src, dst: to, text }),
      signal: run.signal,
    });

    if (!response.ok) {
      throw new ToolError("Request to Plivo Message API has failed!", [
        new Error(await response.text()),
      ]);
    }

    return new PlivoSendMessageToolOutput(await response.json());
  }
}
