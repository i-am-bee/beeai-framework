/**
 * Copyright 2025 IBM Corp.
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

import { ValueError } from "@/errors.js";
import { ClassConstructor } from "@/internals/types.js";
import { BackendProviders, ProviderDef, ProviderName } from "@/backend/constants.js";
import { capitalize } from "remeda";

export type FullModelName = `${ProviderName}:${string}`;

function findProviderDef(value: string): ProviderDef | null {
  return (
    Object.values(BackendProviders).find(
      (p) => p.name === value || p.module === value || p.aliases.includes(value),
    ) ?? null
  );
}

export function parseModel(name: string) {
  if (!name) {
    throw new ValueError("Neither 'provider' nor 'provider:model' was specified.");
  }
  const [providerId, ...rest] = name.split(":") as [ProviderName, ...string[]];
  const modelId = rest.join(":");

  const providerDef = findProviderDef(providerId);
  if (!providerDef) {
    throw new ValueError("Model does not contain provider name!");
  }
  return { providerId, modelId, providerDef };
}

export async function loadModel<T>(
  name: ProviderName | FullModelName,
  type: "embedding" | "chat",
): Promise<ClassConstructor<T>> {
  const { providerDef } = parseModel(name);
  const module = await import(`beeai-framework/adapters/${providerDef.module}/backend/${type}`);
  return module[`${providerDef.name}${capitalize(type)}Model`];
}
