/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { randomUUID } from "node:crypto";
import { AssistantMessage, Message, UserMessage } from "@/backend/message.js";
import { Message as A2AMessage, Artifact, Part, Role } from "@a2a-js/sdk";
import { FilePart } from "ai";

type A2AMessageInput = Pick<A2AMessage, "role" | "parts"> &
  Partial<Omit<A2AMessage, "role" | "parts">>;

// a2a-sdk 1.x types are generated from protobuf and every field is required, unset strings are empty
export function createA2AMessage(input: A2AMessageInput): A2AMessage {
  return {
    messageId: input.messageId || randomUUID(),
    contextId: input.contextId ?? "",
    taskId: input.taskId ?? "",
    role: input.role,
    parts: input.parts,
    metadata: input.metadata,
    extensions: input.extensions ?? [],
    referenceTaskIds: input.referenceTaskIds ?? [],
  };
}

export function createA2APart(
  content: Part["content"],
  options: Partial<Omit<Part, "content">> = {},
): Part {
  return {
    content,
    metadata: options.metadata,
    filename: options.filename ?? "",
    mediaType: options.mediaType ?? "",
  };
}

interface ParsedDataUri {
  data: string;
  mediaType?: string;
}

/**
 * Convert an RFC 2397 `data:` URI to base64 data and its media type.
 *
 * Returns `undefined` when `uri` is not a well-formed data URI, so callers can fall
 * back to treating it as a regular, fetchable URL. Percent-encoded payloads are
 * re-encoded to base64 so callers always receive base64 data.
 */
function dataUriToBase64(uri: string): ParsedDataUri | undefined {
  if (!uri.startsWith("data:")) {
    return undefined;
  }
  const commaIndex = uri.indexOf(",");
  if (commaIndex === -1) {
    return undefined; // malformed data URI
  }
  let header = uri.slice("data:".length, commaIndex);
  const payload = uri.slice(commaIndex + 1);
  const isBase64 = header.endsWith(";base64");
  if (isBase64) {
    header = header.slice(0, -";base64".length);
  }
  const mediaType = header.split(";")[0] || undefined;
  try {
    if (isBase64) {
      const normalized = payload.replace(/\s/g, "");
      if (!/^[A-Za-z0-9+/]*={0,2}$/.test(normalized)) {
        return undefined; // not valid base64 -> treat as a regular URL
      }
      return { data: Buffer.from(normalized, "base64").toString("base64"), mediaType };
    }
    return {
      data: Buffer.from(decodeURIComponent(payload), "utf-8").toString("base64"),
      mediaType,
    };
  } catch {
    return undefined;
  }
}

export function convertA2AMessageToFrameworkMessage(input: A2AMessage | Artifact): Message {
  const msg =
    "role" in input && input.role === Role.ROLE_USER
      ? new UserMessage([], input.metadata)
      : new AssistantMessage([], input.metadata);

  for (const part of input.parts) {
    const content = part.content;
    if (content?.$case === "text") {
      msg.content.push({ type: "text", text: content.value });
    } else if (content?.$case === "data") {
      msg.content.push({ type: "text", text: JSON.stringify(content.value, null, 2) });
    } else if (content?.$case === "raw" || content?.$case === "url") {
      let fileData: FilePart;
      if (content.$case === "raw") {
        // a2a-sdk 1.x carries the raw bytes, the framework expects base64 file data
        fileData = {
          type: "file",
          data: Buffer.from(content.value).toString("base64"),
          mediaType: part.mediaType || "application/octet-stream",
          filename: part.filename || undefined,
        };
      } else {
        // Inline data: URIs as base64 so non-publicly-accessible content travels with
        // the message; leave regular (fetchable) URLs untouched.
        const parsed = dataUriToBase64(content.value);
        fileData = {
          type: "file",
          data: parsed ? parsed.data : content.value,
          mediaType: part.mediaType || parsed?.mediaType || "application/octet-stream",
          filename: part.filename || undefined,
        };
      }

      msg.content.push(fileData);
    }
  }

  return msg;
}
