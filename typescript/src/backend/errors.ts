/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { FrameworkError } from "@/errors.js";

export class BackendError extends FrameworkError {}

export class ChatModelError extends BackendError {}

export class EmbeddingModelError extends BackendError {}
