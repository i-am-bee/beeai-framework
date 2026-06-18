/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { convertA2AMessageToFrameworkMessage } from "@/adapters/a2a/agents/utils.js";
import { Message as A2AMessage, FileWithBytes, FileWithUri } from "@a2a-js/sdk";
import { FilePart } from "ai";

const PNG_BASE64 = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]).toString("base64");

function messageWithFile(file: FileWithBytes | FileWithUri): A2AMessage {
  return {
    kind: "message",
    messageId: "msg-1",
    role: "user",
    parts: [{ kind: "file", file }],
  };
}

describe("convertA2AMessageToFrameworkMessage", () => {
  it("inlines data: URIs as base64", () => {
    const msg = convertA2AMessageToFrameworkMessage(
      messageWithFile({ uri: `data:image/png;base64,${PNG_BASE64}`, name: "logo.png" }),
    );
    const file = msg.content[0] as FilePart;
    expect(file.type).toBe("file");
    expect(file.data).toBe(PNG_BASE64);
    expect(file.mediaType).toBe("image/png");
    expect(file.filename).toBe("logo.png");
  });

  it("encodes percent-encoded data: URIs to base64", () => {
    const msg = convertA2AMessageToFrameworkMessage(
      messageWithFile({ uri: "data:text/plain,Hello%20World" }),
    );
    const file = msg.content[0] as FilePart;
    expect(Buffer.from(file.data as string, "base64").toString("utf-8")).toBe("Hello World");
    expect(file.mediaType).toBe("text/plain");
  });

  it("leaves regular (fetchable) URLs untouched", () => {
    const url = "https://example.com/image.png";
    const msg = convertA2AMessageToFrameworkMessage(
      messageWithFile({ uri: url, mimeType: "image/png", name: "logo.png" }),
    );
    const file = msg.content[0] as FilePart;
    expect(file.data).toBe(url);
  });

  it("leaves malformed base64 data: URIs as the original URI", () => {
    const uri = "data:image/png;base64,not valid base64!!!";
    const msg = convertA2AMessageToFrameworkMessage(messageWithFile({ uri }));
    const file = msg.content[0] as FilePart;
    expect(file.data).toBe(uri);
  });

  it("passes file bytes through unchanged", () => {
    const msg = convertA2AMessageToFrameworkMessage(
      messageWithFile({ bytes: PNG_BASE64, mimeType: "image/png", name: "logo.png" }),
    );
    const file = msg.content[0] as FilePart;
    expect(file.data).toBe(PNG_BASE64);
    expect(file.mediaType).toBe("image/png");
  });

  it("prefers the explicit mimeType over the data: URI media type", () => {
    const msg = convertA2AMessageToFrameworkMessage(
      messageWithFile({ uri: `data:image/png;base64,${PNG_BASE64}`, mimeType: "image/jpeg" }),
    );
    const file = msg.content[0] as FilePart;
    expect(file.mediaType).toBe("image/jpeg");
  });
});
