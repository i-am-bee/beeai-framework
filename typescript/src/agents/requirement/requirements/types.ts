import type { Callback } from "@/emitter/types.js";
import type { AnyTool } from "@/tools/base.js";

export interface RequirementCallbacks {
  init?: Callback<{ tools: AnyTool[] }>;
}
