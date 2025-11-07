/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { CustomMessage, Role, UserMessage } from "@/backend/message.js";
import { isPlainObject, isString, isTruthy } from "remeda";
import { getProp } from "@/internals/helpers/object.js";
import { TextPart } from "ai";
import { z } from "zod";
import { parseEnv } from "@/internals/env.js";

export function encodeCustomMessage(msg: CustomMessage): UserMessage {
  return new UserMessage([
    {
      type: "text",
      text: `#custom_role#${msg.role}#`,
    },
    ...(msg.content.slice() as TextPart[]),
  ]);
}

export function decodeCustomMessage(value: string) {
  const [_, id, role, ...content] = value.split("#");
  if (id !== "custom_role") {
    return;
  }
  return { role, content: content.join("#") };
}

function unmaskCustomMessage(msg: Record<string, any>) {
  if (msg.role !== Role.USER) {
    return;
  }

  for (const key of ["content", "text"]) {
    let value = msg[key];
    if (!value) {
      continue;
    }

    if (Array.isArray(value)) {
      value = value
        .map((val) => (val.type === "text" ? val.text || val.content : null))
        .filter(isTruthy)
        .join("");
    }

    const decoded = decodeCustomMessage(value);
    if (decoded) {
      msg.role = decoded.role;
      msg[key] = decoded.content;
      break;
    }
  }
}

export function vercelFetcher(customFetch?: typeof fetch): typeof fetch {
  return async (url, options) => {
    if (
      options &&
      isString(options.body) &&
      (getProp(options.headers, ["content-type"]) === "application/json" ||
        getProp(options.headers, ["Content-Type"]) === "application/json")
    ) {
      const body = JSON.parse(options.body);
      if (isPlainObject(body) && Array.isArray(body.messages)) {
        body.messages.forEach((msg) => {
          if (!isPlainObject(msg)) {
            return;
          }
          unmaskCustomMessage(msg);
        });
      }
      options.body = JSON.stringify(body);
    }

    const fetcher = customFetch ?? fetch;
    return await fetcher(url, options);
  };
}

export function parseHeadersFromEnv(env: string): Record<string, any> {
  return parseEnv(
    env,
    z.preprocess((value) => {
      return Object.fromEntries(
        String(value || "")
          .split(",")
          .filter((pair) => pair.includes("="))
          .map((pair) => pair.split("=")),
      );
    }, z.record(z.string())),
  );
}
