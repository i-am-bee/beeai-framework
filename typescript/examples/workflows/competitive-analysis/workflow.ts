import { Workflow } from "beeai-framework/workflows/workflow";
import {
  categorizationPromptTemplate,
  competitorsPromptTemplate,
  competitorsSchema,
  findingsSchema,
  reflectionPromptTemplate,
  reflectionSchema,
} from "./prompts.js";
import { State, StateSchema } from "./state.js";
import { deduplicateAndFormatSources, formatSources, getChatLLM, tavilySearch } from "./utils.js";
import { isEmpty, mapValues } from "remeda";
import { AssistantMessage, SystemMessage } from "beeai-framework/backend/message";

export enum Steps {
  GENERATE_COMPETITORS = "GENERATE_COMPETITORS",
  SELECT_COMPETITOR = "SELECT_COMPETITOR",
  WEB_RESEARCH = "WEB_RESEARCH",
  CATEGORIZE_FINDINGS = "CATEGORIZE_FINDINGS",
  FINALIZE_SUMMARY = "FINALIZE_SUMMARY",
  REFLECTION = "REFLECTION",
}

async function generateCompetitors(state: State) {
  if (state.specifiedCompetitors?.length) {
    state.competitors = state.specifiedCompetitors;
    state.runningSummary = `Competitive analysis of ${state.specifiedCompetitors.join(", ")}`;
    state.competitorFindings = mapValues(state.specifiedCompetitors, () => []);
    return;
  }

  const llm = getChatLLM();
  const result = await llm.createStructure({
    schema: competitorsSchema,
    messages: [
      new SystemMessage(
        competitorsPromptTemplate.render({
          industry: state.industry,
          specifiedCompetitors: undefined,
        }),
      ),
    ],
  });

  state.competitors = result.object.competitors;
  state.runningSummary = result.object.overview;
  state.competitorFindings = mapValues(state.competitors, () => []);
}

async function selectCompetitor(state: State) {
  const unprocessedCompetitors = state.competitors.filter((competitor) =>
    isEmpty(state.competitorFindings[competitor]),
  );

  if (isEmpty(unprocessedCompetitors)) {
    return Steps.FINALIZE_SUMMARY;
  }

  const currentCompetitor = unprocessedCompetitors[0];
  const searchTerms = `${currentCompetitor} comprehensive overview key capabilities`;

  state.currentCompetitor = currentCompetitor;
  state.searchQuery = searchTerms;
  return Steps.WEB_RESEARCH;
}

async function webResearch(state: State) {
  const searchResults = await tavilySearch(state.searchQuery);
  const searchResultsString = deduplicateAndFormatSources(searchResults, 1000, true);

  state.sourcesGathered.push(formatSources(searchResults));
  state.researchLoopCount += 1;
  state.webResearchResults.push(searchResultsString);
}

async function categorizeFindings(state: State) {
  if (!state.currentCompetitor) {
    throw new Error("No competitor selected for analysis");
  }

  if (state.webResearchResults.length === 0) {
    throw new Error("No research results available for analysis");
  }

  const llm = getChatLLM();
  const result = await llm.createStructure({
    schema: findingsSchema,
    messages: [
      new SystemMessage(
        categorizationPromptTemplate.render({
          competitor: state.currentCompetitor,
          searchResults: state.webResearchResults[state.webResearchResults.length - 1],
        }),
      ),
    ],
  });

  state.competitorFindings = {
    ...state.competitorFindings,
    [state.currentCompetitor]: [
      ...(result.object.key_insights || []),
      ...(result.object.unique_capabilities || []),
    ],
  };
  return Steps.SELECT_COMPETITOR;
}

async function finalizeSummary(state: State) {
  const competitorSections = Object.entries(state.competitorFindings)
    .map(([competitor, findings]) => {
      return `## ${competitor}\n${findings.map((insight) => `* ${insight}`).join("\n")}`;
    })
    .join("\n\n");

  const finalSummary = `# Competitive Analysis: ${state.industry}

${state.runningSummary || "Comprehensive Competitive Landscape Overview"}

${competitorSections}

### Sources
${state.sourcesGathered.join("\n")}`;

  state.answer = new AssistantMessage(finalSummary);
  return Steps.REFLECTION;
}

async function reflectAndImprove(state: State) {
  if (!state.answer) {
    throw new Error("No analysis to reflect upon");
  }

  const llm = getChatLLM();
  const result = await llm.createStructure({
    schema: reflectionSchema,
    messages: [
      new SystemMessage(
        reflectionPromptTemplate.render({
          analysis: state.answer.text,
          previous_feedback: state.reflectionFeedback,
        }),
      ),
    ],
  });

  const feedback = [...(result.object.critique || []), ...(result.object.suggestions || [])];

  if (result.object.should_iterate && state.reflectionIteration < state.maxReflectionIterations) {
    state.reflectionFeedback = feedback;
    state.reflectionIteration = state.reflectionIteration + 1;
    return Steps.GENERATE_COMPETITORS;
  }

  state.answer = new AssistantMessage(
    `${state.answer.text}\n\n## Reflection Notes\n${feedback.map((f) => `* ${f}`).join("\n")}`,
  );
  return Workflow.END;
}

export const workflow = new Workflow({ schema: StateSchema })
  .addStep(Steps.GENERATE_COMPETITORS, generateCompetitors)
  .addStep(Steps.SELECT_COMPETITOR, selectCompetitor)
  .addStep(Steps.WEB_RESEARCH, webResearch)
  .addStep(Steps.CATEGORIZE_FINDINGS, categorizeFindings)
  .addStep(Steps.FINALIZE_SUMMARY, finalizeSummary)
  .addStep(Steps.REFLECTION, reflectAndImprove);
