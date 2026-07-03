/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

// request

export interface ResponsesRequestInputMessage {
  role: "user" | "assistant" | "system" | "developer";
  content: string | null;
  type: string;
}

export interface ResponsesRequestConversation {
  id: string;
}

export interface ResponsesRequestBody {
  model: string;
  input: string | ResponsesRequestInputMessage[];
  instructions?: string | null;
  conversation?: string | ResponsesRequestConversation | null;
  stream?: boolean | null;
}

// response

export interface ResponsesMessageContent {
  type: "output_text";
  text: string;
}

export interface ResponsesMessageOutput {
  type: "message";
  id: string;
  status: string;
  role: "assistant";
  content: ResponsesMessageContent[];
}

export interface ResponsesReasoningContent {
  type: "reasoning_text";
  text: string;
}

export interface ResponsesReasoningSummary {
  type: "summary_text";
  text: string;
}

export interface ResponsesReasoningOutput {
  type: "reasoning";
  id: string;
  status: string;
  summary?: ResponsesReasoningSummary | null;
  content?: ResponsesReasoningContent | null;
}

export interface ResponsesCustomToolCallOutput {
  type: "custom_tool_call";
  id: string;
  name: string;
  input: string;
  call_id: string;
}

export interface ResponsesUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
}

export type ResponsesResponseOutput =
  | ResponsesMessageOutput
  | ResponsesReasoningOutput
  | ResponsesCustomToolCallOutput;

export interface ResponsesResponse {
  id: string;
  object: "response";
  created: number;
  status: string;
  error?: string | null;
  instructions?: string | null;
  model: string;
  output?: ResponsesResponseOutput[] | null;
  usage?: ResponsesUsage | null;
}

// stream

export interface ResponsesStreamResponseCreated {
  type: "response.created";
  response: ResponsesResponse;
  sequence_number?: number;
}

export interface ResponsesStreamResponseInProgress {
  type: "response.in_progress";
  response: ResponsesResponse;
  sequence_number?: number;
}

export interface ResponsesStreamResponseCompleted {
  type: "response.completed";
  response: ResponsesResponse;
  sequence_number?: number;
}

export interface ResponsesStreamOutputItemAdded {
  type: "response.output_item.added";
  item: ResponsesResponseOutput;
  output_index: number;
  sequence_number?: number;
}

export interface ResponsesStreamOutputItemDone {
  type: "response.output_item.done";
  item: ResponsesResponseOutput;
  output_index: number;
  sequence_number?: number;
}

export interface ResponsesStreamPartOutputText {
  text: string;
  type: "output_text";
  annotations: unknown[];
}

export interface ResponsesStreamContentPartAdded {
  type: "response.content_part.added";
  content_index: number;
  item_id: string;
  output_index: number;
  part: ResponsesStreamPartOutputText;
  sequence_number?: number;
}

export interface ResponsesStreamContentPartDone {
  type: "response.content_part.done";
  content_index: number;
  item_id: string;
  output_index: number;
  part: ResponsesStreamPartOutputText;
  sequence_number?: number;
}

export interface ResponsesStreamOutputTextDelta {
  type: "response.output_text.delta";
  content_index: number;
  delta: string;
  item_id: string;
  output_index: number;
  sequence_number?: number;
}

export interface ResponsesStreamOutputTextDone {
  type: "response.output_text.done";
  content_index: number;
  text: string;
  item_id: string;
  output_index: number;
  sequence_number?: number;
}

export interface ResponsesStreamError {
  type: "error";
  code: string;
  message: string;
  param: string;
  sequence_number?: number;
}
