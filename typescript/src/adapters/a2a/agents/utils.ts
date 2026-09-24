/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AssistantMessage, Message, UserMessage } from "@/backend/message.js";
import { Message as A2AMessage, Artifact } from "@a2a-js/sdk";
import { FilePart } from "ai";

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
    "kind" in input && input.kind === "message" && input.role === "user"
      ? new UserMessage([], input.metadata)
      : new AssistantMessage([], input.metadata);

  for (const part of input.parts) {
    if (part.kind === "text") {
      msg.content.push({ type: "text", text: part.text });
    } else if (part.kind === "data") {
      msg.content.push({ type: "text", text: JSON.stringify(part.data, null, 2) });
    } else if (part.kind === "file") {
      let fileData: FilePart;
      if ("bytes" in part.file) {
        fileData = {
          type: "file",
          data: part.file.bytes,
          mediaType: part.file.mimeType || "application/octet-stream",
          filename: part.file.name,
        };
      } else {
        // Inline data: URIs as base64 so non-publicly-accessible content travels with
        // the message; leave regular (fetchable) URLs untouched.
        const parsed = dataUriToBase64(part.file.uri);
        fileData = {
          type: "file",
          data: parsed ? parsed.data : part.file.uri,
          mediaType: part.file.mimeType || parsed?.mediaType || "application/octet-stream",
          filename: part.file.name,
        };
      }

      msg.content.push(fileData);
    }
  }

  return msg;
}
