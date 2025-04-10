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

import { wikiSearchResult } from "wikipedia";

vitest.mock("wikipedia", () => {
  const pages = [
    {
      title: "Tomáš Dvořák (ice hockey)",
      pageid: "1",
    },
    {
      title: "Dvorak",
      pageid: "2",
    },
    {
      title: "Tomáš",
      pageid: "3",
    },
    {
      title: "List of Czech sportspeople (section Ice hockey)",
      pageid: "4",
    },
  ];

  return {
    default: {
      default: {
        setLang(lang: string) {
          return lang;
        },
        async search() {
          return {
            results: pages,
            suggestion: "",
          } as wikiSearchResult;
        },
        async page(titleOrId: string | number) {
          const page = pages.find((page) => page.title === titleOrId || page.pageid === titleOrId);
          if (!page) {
            throw new Error("No page found.");
          }

          return {
            ...page,
            content: async () => "Content",
            infobox: async () => ({ text: "Infobox" }),
          };
        },
      },
    },
  };
});

import { verifyDeserialization } from "@tests/e2e/utils.js";
import { WikipediaTool } from "@/tools/search/wikipedia.js";

describe("Wikipedia", () => {
  it("Retrieves a correct page", async () => {
    const instance = new WikipediaTool();
    const response = await instance.run({
      query: "tomas dvorak ice hockey",
    });
    expect(response.results.length).toBe(1);
    expect(response.results[0].title).toBe("Tomáš Dvořák (ice hockey)");
  });

  it("Serializes", async () => {
    const instance = new WikipediaTool({
      extraction: {
        fields: { infobox: {} },
      },
    });
    await instance.run({ query: "Prague" });
    const serialized = await instance.serialize();
    const deserialized = await WikipediaTool.fromSerialized(serialized);
    verifyDeserialization(instance, deserialized);
  });
});
