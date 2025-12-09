import type { RequirementAgentTemplates } from "@/agents/requirement/types.js";
import {
  RequirementAgentSystemPrompt,
  RequirementAgentTaskPrompt,
} from "@/agents/requirement/prompts.js";
import { mapObj } from "@/internals/helpers/object.js";
import { PromptTemplate } from "@/template.js";

export function createTemplates(overrides: any): RequirementAgentTemplates {
  const defaultTemplates: RequirementAgentTemplates = {
    system: RequirementAgentSystemPrompt,
    task: RequirementAgentTaskPrompt,
  } as const;

  return mapObj(defaultTemplates)((key, defaultTemplate: RequirementAgentTemplates[typeof key]) => {
    const override = overrides[key] ?? defaultTemplate;
    if (override instanceof PromptTemplate) {
      return override;
    }
    return override(defaultTemplate) ?? defaultTemplate;
  });
}
