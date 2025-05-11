/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
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

import { CustomMessage, Role, UserMessage } from "@/backend/message.js";
import { isPlainObject, isString, isTruthy } from "remeda";
import { getProp } from "@/internals/helpers/object.js";
import { TextPart } from "ai";

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
      (getProp(options.headers, ["content-type"]) == "application/json" ||
        getProp(options.headers, ["Content-Type"]) == "application/json")
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
