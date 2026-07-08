import { z } from "zod";
import { PromptTemplate } from "beeai-framework/template";

export const competitorsSchema = z.object({
  competitors: z.array(z.string()),
  overview: z.string(),
});

export const competitorsPromptTemplate = new PromptTemplate({
  schema: z.object({
    industry: z.string(),
    specifiedCompetitors: z.array(z.string()).optional(),
  }),
  template: `Identify key players in the {{industry}} domain.

{{#specifiedCompetitors}}
Focus on these specific competitors: {{specifiedCompetitors}}
{{/specifiedCompetitors}}

Provide a list of top competitors and a brief industry overview.

Return JSON with:
{
  "competitors": ["list of competitors"],
  "overview": "High-level industry description"
}`,
});

export const findingsSchema = z.object({
  key_insights: z.array(z.string()),
  unique_capabilities: z.array(z.string()).optional(),
});

export const categorizationPromptTemplate = new PromptTemplate({
  schema: z.object({
    competitor: z.string(),
    searchResults: z.string(),
  }),
  template: `Analyze {{competitor}} based on these search results:
{{searchResults}}

Key Analysis Points:
- Core capabilities
- Unique technological approaches
- Market positioning
- Potential impact

Provide concise, actionable insights about the competitor.`,
});

export const reflectionSchema = z.object({
  critique: z.array(z.string()),
  suggestions: z.array(z.string()),
  should_iterate: z.boolean(),
});

export const reflectionPromptTemplate = new PromptTemplate({
  schema: z.object({
    analysis: z.string(),
    previous_feedback: z.array(z.string()).optional(),
  }),
  template: `Review this competitive analysis:
{{analysis}}

{{#previous_feedback}}
Previous feedback:
{{previous_feedback}}
{{/previous_feedback}}

Consider:
- Depth of competitor analysis
- Missing key capabilities
- Market positioning clarity
- Evidence and sources
- Actionable insights

Return JSON:
{
  "critique": ["Specific issues found"],
  "suggestions": ["Concrete improvement suggestions"],
  "should_iterate": boolean
}`,
});
