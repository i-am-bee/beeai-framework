import { z } from "zod";
import { Message } from "beeai-framework/backend/message";

export const InputSchema = z.object({
  industry: z.string(),
  specifiedCompetitors: z.array(z.string()).optional(),
});

export const StateSchema = z.object({
  industry: z.string(),
  specifiedCompetitors: z.array(z.string()).optional(),
  competitors: z.array(z.string()).default([]),
  searchQuery: z.string().default(""),
  webResearchResults: z.array(z.string()).default([]),
  sourcesGathered: z.array(z.string()).default([]),
  currentCompetitor: z.string().optional(),
  competitorFindings: z.record(z.string(), z.array(z.string())).default({}),
  researchLoopCount: z.number().default(0),
  runningSummary: z.string().optional(),
  answer: z.instanceof(Message).optional(),
  reflectionFeedback: z.array(z.string()).optional(),
  reflectionIteration: z.number().default(0),
  maxReflectionIterations: z.number().default(3),
});

export type State = z.infer<typeof StateSchema>;
