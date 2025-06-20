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

import { VercelChatModel } from "@/adapters/vercel/backend/chat.js";
import {
	XaiClient,
	type XaiClientSettings,
} from "@/adapters/xai/backend/client.js";
import { getEnv } from "@/internals/env.js";
import type { XaiProvider } from "@ai-sdk/xai";

type XaiParameters = Parameters<XaiProvider["languageModel"]>;
export type XAIChatModelId = NonNullable<XaiParameters[0]>;
export type XAIChatModelSettings = NonNullable<XaiParameters[1]>;

export class XAIChatModel extends VercelChatModel {
	constructor(
		modelId: XAIChatModelId = getEnv("XAI_CHAT_MODEL", "grok-3-mini"),
		settings: XAIChatModelSettings = {},
		client?: XaiClientSettings | XaiClient,
	) {
		const model = XaiClient.ensure(client).instance.languageModel(
			modelId,
			settings,
		);
		super(model);
	}

	static {
		XAIChatModel.register();
	}
}
