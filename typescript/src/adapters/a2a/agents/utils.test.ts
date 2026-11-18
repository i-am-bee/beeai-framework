/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  convertA2AMessageToFrameworkMessage,
  createA2AMessage,
  createA2APart,
} from "@/adapters/a2a/agents/utils.js";
import { Message as A2AMessage, Role } from "@a2a-js/sdk";
import { FilePart } from "ai";

const PNG_BASE64 = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]).toString("base64");

function messageWithFile(
  url: string,
  options: { mediaType?: string; filename?: string } = {},
): A2AMessage {
  return createA2AMessage({
    messageId: "msg-1",
    role: Role.ROLE_USER,
    parts: [createA2APart({ $case: "url", value: url }, options)],
  });
}

describe("convertA2AMessageToFrameworkMessage", () => {
  it("inlines data: URIs as base64", () => {
    const msg = convertA2AMessageToFrameworkMessage(
      messageWithFile(`data:image/png;base64,${PNG_BASE64}`, { filename: "logo.png" }),
    );
    const file = msg.content[0] as FilePart;
    expect(file.type).toBe("file");
    expect(file.data).toBe(PNG_BASE64);
    expect(file.mediaType).toBe("image/png");
    expect(file.filename).toBe("logo.png");
  });

  it("encodes percent-encoded data: URIs to base64", () => {
    const msg = convertA2AMessageToFrameworkMessage(
      messageWithFile("data:text/plain,Hello%20World"),
    );
    const file = msg.content[0] as FilePart;
    expect(Buffer.from(file.data as string, "base64").toString("utf-8")).toBe("Hello World");
    expect(file.mediaType).toBe("text/plain");
  });

  it("strips whitespace inside base64 payloads", () => {
    // RFC 2045 permits whitespace (e.g. line breaks) inside base64 payloads.
    const payload = `${PNG_BASE64.slice(0, 4)}\n ${PNG_BASE64.slice(4)}`;
    const msg = convertA2AMessageToFrameworkMessage(
      messageWithFile(`data:image/png;base64,${payload}`),
    );
    const file = msg.content[0] as FilePart;
    expect(file.data).toBe(PNG_BASE64);
  });

  it("accepts unpadded base64 payloads", () => {
    const unpadded = Buffer.from("hello").toString("base64").replace(/=+$/, "");
    const msg = convertA2AMessageToFrameworkMessage(
      messageWithFile(`data:text/plain;base64,${unpadded}`),
    );
    const file = msg.content[0] as FilePart;
    expect(Buffer.from(file.data as string, "base64").toString("utf-8")).toBe("hello");
  });

  it("leaves data: URIs with malformed percent-encoding as the original URI", () => {
    const uri = "data:text/plain,bad%2";
    const msg = convertA2AMessageToFrameworkMessage(messageWithFile(uri));
    const file = msg.content[0] as FilePart;
    expect(file.data).toBe(uri);
  });

  it("leaves regular (fetchable) URLs untouched", () => {
    const url = "https://example.com/image.png";
    const msg = convertA2AMessageToFrameworkMessage(
      messageWithFile(url, { mediaType: "image/png", filename: "logo.png" }),
    );
    const file = msg.content[0] as FilePart;
    expect(file.data).toBe(url);
  });

  it("leaves malformed base64 data: URIs as the original URI", () => {
    const uri = "data:image/png;base64,not valid base64!!!";
    const msg = convertA2AMessageToFrameworkMessage(messageWithFile(uri));
    const file = msg.content[0] as FilePart;
    expect(file.data).toBe(uri);
  });

  it("encodes raw file bytes as base64", () => {
    const msg = convertA2AMessageToFrameworkMessage(
      createA2AMessage({
        role: Role.ROLE_USER,
        parts: [
          createA2APart(
            { $case: "raw", value: Buffer.from(PNG_BASE64, "base64") },
            { mediaType: "image/png", filename: "logo.png" },
          ),
        ],
      }),
    );
    const file = msg.content[0] as FilePart;
    expect(file.data).toBe(PNG_BASE64);
    expect(file.mediaType).toBe("image/png");
    expect(file.filename).toBe("logo.png");
  });

  it("prefers the explicit mediaType over the data: URI media type", () => {
    const msg = convertA2AMessageToFrameworkMessage(
      messageWithFile(`data:image/png;base64,${PNG_BASE64}`, { mediaType: "image/jpeg" }),
    );
    const file = msg.content[0] as FilePart;
    expect(file.mediaType).toBe("image/jpeg");
  });

  it("converts text and data parts and keeps the role and metadata", () => {
    const user = convertA2AMessageToFrameworkMessage(
      createA2AMessage({
        role: Role.ROLE_USER,
        parts: [
          createA2APart({ $case: "text", value: "hello" }),
          createA2APart({ $case: "data", value: { city: "Prague" } }),
        ],
        metadata: { source: "test" },
      }),
    );
    expect(user.role).toBe("user");
    expect(user.meta).toMatchObject({ source: "test" });
    expect(user.content).toEqual([
      { type: "text", text: "hello" },
      { type: "text", text: JSON.stringify({ city: "Prague" }, null, 2) },
    ]);

    const agent = convertA2AMessageToFrameworkMessage(
      createA2AMessage({
        role: Role.ROLE_AGENT,
        parts: [createA2APart({ $case: "text", value: "hi" })],
      }),
    );
    expect(agent.role).toBe("assistant");
  });
});

describe("createA2AMessage", () => {
  it("fills the fields left out with their protobuf defaults", () => {
    const message = createA2AMessage({
      role: Role.ROLE_USER,
      parts: [],
      contextId: undefined,
      taskId: undefined,
    });
    expect(message.messageId).not.toBe("");
    expect(message).toMatchObject({
      contextId: "",
      taskId: "",
      extensions: [],
      referenceTaskIds: [],
      metadata: undefined,
    });
  });
});
