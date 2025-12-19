/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Serializable } from "@/internals/serializable.js";
import { shallowCopy } from "@/serializer/utils.js";
import { FilePart, ImagePart, TextPart, ToolCallPart, ToolResultPart } from "ai";
import { z } from "zod";
import { ValueError } from "@/errors.js";
import { hasProp, popProp, safeDefineProperty } from "@/internals/helpers/object.js";
import { Logger } from "@/logger/logger.js";
import { isString } from "remeda";
import { watchArray } from "@/internals/helpers/array.js";

export type MessageRole = "user" | "system" | "tool" | "assistant";
export type MessageContentPart = TextPart | ToolCallPart | ImagePart | FilePart | ToolResultPart;

export interface MessageMeta {
  [key: string]: any;
  createdAt?: Date;
}

export interface MessageInput {
  role: MessageRole;
  text: string; // TODO
  meta?: MessageMeta;
}

function isText(content: MessageContentPart): content is TextPart {
  return content.type === "text";
}
function isImage(content: MessageContentPart): content is ImagePart {
  return content.type === "image";
}
function isFile(content: MessageContentPart): content is FilePart {
  return content.type === "file";
}
function isToolCall(content: MessageContentPart): content is ToolCallPart {
  return content.type === "tool-call";
}
function isToolResult(content: MessageContentPart): content is ToolResultPart {
  return content.type === "tool-result";
}

export abstract class Message<
  T extends MessageContentPart = MessageContentPart,
  R extends string = MessageRole | string,
> extends Serializable {
  public abstract readonly role: R;
  public readonly content: T[];

  constructor(
    content: T | T[] | string,
    public readonly meta: MessageMeta = {},
    public id: string | undefined = undefined,
  ) {
    super();
    if (!meta?.createdAt) {
      meta.createdAt = new Date();
    }
    if (typeof content === "string") {
      this.content = [this.fromString(content)];
    } else {
      this.content = Array.isArray(content) ? content : [content];
    }
    for (const chunk of this.content) {
      this.assertContent(chunk);
    }
    this.content = watchArray(this.content, {
      onAdd: this.assertContent,
    });
  }

  protected assertContent(content: T): asserts content is T {}

  protected abstract fromString(input: string): T;

  static of({ role, text, meta }: MessageInput): Message {
    if (role === "user") {
      return new UserMessage(text, meta);
    } else if (role === "assistant") {
      return new AssistantMessage(text, meta);
    } else if (role === "system") {
      return new SystemMessage(text, meta);
    } else if (role === "tool") {
      return new ToolMessage(text, meta);
    } else {
      return new CustomMessage(role, text, meta);
    }
  }

  get text() {
    return this.getTexts()
      .map((c) => c.text)
      .join("");
  }

  getTexts() {
    return this.content.filter(isText) as TextPart[];
  }

  createSnapshot() {
    return { content: shallowCopy(this.content), meta: shallowCopy(this.meta), role: this.role };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }

  toPlain() {
    return { role: this.role, content: shallowCopy(this.content), id: this.id } as const;
  }

  merge(other: Message<T, R>) {
    this.id = this.id || other.id;
    Object.assign(this, other.meta);
    this.content.push(...other.content);
  }

  static fromChunks<M2 extends Message>(this: new (...args: any[]) => M2, chunks: M2[]): M2 {
    const instance = new this([]);
    chunks.forEach((chunk) => instance.merge(chunk));
    return instance;
  }

  [Symbol.iterator](): Iterator<T> {
    return this.content[Symbol.iterator]();
  }
}

export class AssistantMessage extends Message<TextPart | ToolCallPart | FilePart> {
  public readonly role = "assistant";

  static {
    this.register();
  }

  getToolCalls() {
    return this.content.filter(isToolCall);
  }

  protected fromString(text: string): TextPart {
    return { type: "text", text };
  }

  protected assertContent(
    content: TextPart | ToolCallPart | FilePart,
  ): asserts content is TextPart | ToolCallPart | FilePart {
    if (content.type === "tool-call") {
      const key = "args";
      if (!Object.getOwnPropertyDescriptor(content, key)) {
        const args = popProp(content as any, key);

        if (args !== null && !hasProp(content, "input")) {
          Logger.root.warn(
            `The '${key}' property in the AssistantMessage class is deprecated and will be removed. Use the 'input' property for Tool Call content chunks instead.`,
          );
          content.input = args;
        }
      }

      safeDefineProperty(content, key as keyof typeof content, () => {
        Logger.root.warn(
          `The '${key}' property in the Tool Call is deprecated and will be removed. Use the 'input' property instead.`,
        );
        return content.input;
      });
    }
  }
}

export class ToolMessage extends Message<ToolResultPart> {
  public readonly role = "tool";

  static {
    this.register();
  }

  getToolResults() {
    return this.content.filter(isToolResult);
  }

  protected fromString(text: string): ToolResultPart {
    const schema = z
      .object({
        type: z.literal("tool-result"),
        toolName: z.string(),
        toolCallId: z.string(),
      })
      .and(z.object({ output: z.any() }).or(z.object({ result: z.any() })));

    const { success, data } = schema.safeParse(text);
    if (!success) {
      throw new ValueError(`ToolMessage cannot be created from '${text}'!`);
    }

    return data as ToolResultPart;
  }

  protected assertContent(content: ToolResultPart): asserts content is ToolResultPart {
    const result = popProp(content as any, "result", null);
    const isError = popProp(content as any, "isError", false);

    if (result !== null && !hasProp(content, "output")) {
      Logger.root.warn(
        "The 'result' property in the ToolMessage class is deprecated and will be removed. Use the 'output' property for content chunks instead.",
      );
      if (isString(result)) {
        content.output = { type: isError ? "error-text" : "text", value: result };
      } else {
        content.output = { type: isError ? "error-json" : "json", value: result };
      }
    }

    safeDefineProperty(content, "result" as any, () => {
      Logger.root.warn(
        "The 'result' property of Tool Message is deprecated and will be removed. Use the 'output.value' property instead.",
      );
      return content.output.value as any;
    });
    safeDefineProperty(content, "isError" as any, () => {
      Logger.root.warn(
        "The 'isError' property of Tool Message is deprecated and will be removed. Use the 'output.type' property instead.",
      );
      return content.output.type.includes("error");
    });
  }
}

export class SystemMessage extends Message<TextPart> {
  public readonly role: MessageRole = "system";

  static {
    this.register();
  }

  protected fromString(text: string): TextPart {
    return { type: "text", text };
  }
}

export class UserMessage extends Message<TextPart | ImagePart | FilePart> {
  public readonly role = "user";

  static {
    this.register();
  }

  getImages() {
    return this.content.filter(isImage);
  }

  getFiles() {
    return this.content.filter(isFile);
  }

  protected fromString(text: string): TextPart {
    return { type: "text", text };
  }
}

export const Role = {
  ASSISTANT: "assistant",
  SYSTEM: "system",
  USER: "user",
} as const;

export class CustomMessage extends Message<MessageContentPart, string> {
  public role: string;

  constructor(role: string, content: MessageContentPart | string, meta: MessageMeta = {}) {
    super(content, meta);
    if (!role) {
      throw new ValueError(`Role "${role}" must be specified!`);
    }
    this.role = role;
  }

  protected fromString(input: string): MessageContentPart {
    return { type: "text", text: input };
  }
}
