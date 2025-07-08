/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AssistantMessage, Message, SystemMessage } from "@/backend/message.js";
import { BaseMemory } from "@/memory/base.js";
import { PromptTemplate } from "@/template.js";
import { shallowCopy } from "@/serializer/utils.js";
import { z } from "zod";
import { ChatModel } from "@/backend/chat.js";

export interface SummarizeMemoryInput {
  llm: ChatModel;
  template?: typeof SummarizeMemoryTemplate;
}

export const SummarizeMemoryTemplate = new PromptTemplate({
  schema: z.object({
    summary: z.string(),
  }),
  template: `Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.

Current summary:
{{summary}}`,
});

export class SummarizeMemory extends BaseMemory {
  protected summary = "";
  protected template;
  protected llm;

  constructor(config: SummarizeMemoryInput) {
    super();
    this.template = config.template ?? SummarizeMemoryTemplate;
    this.llm = config.llm;
  }

  static {
    this.register();
  }

  get messages(): Message[] {
    const currentSummary = this.summary;
    if (!currentSummary) {
      return [];
    }

    return [new AssistantMessage(currentSummary)];
  }

  // eslint-disable-next-line unused-imports/no-unused-vars
  async delete(message: Message) {
    return false;
  }

  async add(message: Message, _index?: number) {
    const response = await this.llm.create({
      messages: [
        new SystemMessage(
          this.template.render({
            summary: this.summary,
          }),
        ),
        new AssistantMessage(`New lines of conversation:
${message.role}: ${message.text}

New summary:
`),
      ],
    });
    this.summary = response.getTextContent();
  }

  reset() {
    this.summary = "";
  }

  createSnapshot() {
    return {
      summary: this.summary,
      template: this.template,
      llm: this.llm,
      messages: shallowCopy(this.messages),
    };
  }

  loadSnapshot(state: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, state);
  }
}
