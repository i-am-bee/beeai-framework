/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AssistantMessage, Message, UserMessage } from "@/backend/message.js";
import { Message as A2AMessage, Artifact } from "@a2a-js/sdk";
import { FilePart } from "ai";

export function convertA2AMessageToFrameworkMessage(input: A2AMessage | Artifact): Message {
  const msg =
    "kind" in input && input.kind === "message" && input.role === "user"
      ? new UserMessage([], input.metadata)
      : new AssistantMessage([], input.metadata);

  for (const part of input.parts) {
    if (part.kind === "text") {
      msg.content.push({ type: "text", text: part.text });
    } else if (part.kind === "data") {
      msg.content.push({ type: "text", text: JSON.stringify(part.data, null, 2) });
    } else if (part.kind === "file") {
      // TODO: handle non-publicly accessible URLs (always convert to base64)
      const fileData: FilePart =
        "bytes" in part.file
          ? {
              type: "file",
              data: part.file.bytes,
              mimeType: part.file.mimeType || "application/octet-stream",
              filename: part.file.name,
            }
          : {
              type: "file",
              data: part.file.uri,
              mimeType: part.file.mimeType || "application/octet-stream",
              filename: part.file.name,
            };

      msg.content.push(fileData);
    }
  }

  return msg;
}
