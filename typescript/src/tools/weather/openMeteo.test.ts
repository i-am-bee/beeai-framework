/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { OpenMeteoTool } from "@/tools/weather/openMeteo.js";
import { afterEach, beforeEach, expect, vi } from "vitest";

const CAMBRIDGE_RESULTS = [
  {
    name: "Cambridge",
    latitude: 52.2,
    longitude: 0.12,
    country: "United Kingdom",
    country_code: "GB",
    admin1: "England",
  },
  {
    name: "Cambridge",
    latitude: 42.37,
    longitude: -71.11,
    country: "United States",
    country_code: "US",
    admin1: "Massachusetts",
  },
];

describe("OpenMeteoTool geocoding", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn(async (url: string) =>
      Response.json(url.includes("geocoding-api") ? { results: CAMBRIDGE_RESULTS } : {}),
    );
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  const run = (location: { name: string; country?: string }) =>
    new OpenMeteoTool().run({ location, start_date: "2026-10-20", temperature_unit: "celsius" });

  const requested = () => {
    const [geocode, forecast] = fetchMock.mock.calls.map(([url]) => new URL(url));
    return { geocode: geocode.searchParams, forecast: forecast.searchParams };
  };

  it.each([
    ["country code", "US"],
    ["country name", "united states"],
    ["region", "Massachusetts"],
  ])("picks the candidate matching the %s", async (_label, country) => {
    await run({ name: "Cambridge", country });

    const { geocode, forecast } = requested();
    expect(geocode.has("country")).toBe(false);
    expect(geocode.get("count")).toBe("100");
    expect(forecast.get("latitude")).toBe("42.37");
  });

  it("falls back to the top candidate when nothing matches the country", async () => {
    await run({ name: "Cambridge", country: "UK" });

    expect(requested().forecast.get("latitude")).toBe("52.2");
  });

  describe("with a qualifier after a comma", () => {
    const geocodeNames = () =>
      fetchMock.mock.calls
        .map(([url]) => new URL(url))
        .filter((url) => url.hostname.startsWith("geocoding-api"))
        .map((url) => url.searchParams.get("name"));

    const answerFullName = (results: unknown[]) =>
      fetchMock.mockImplementation(async (url: string) => {
        if (!url.includes("geocoding-api")) {
          return Response.json({});
        }
        const name = new URL(url).searchParams.get("name") ?? "";
        return Response.json({ results: name.includes(",") ? results : CAMBRIDGE_RESULTS });
      });

    it("keeps the API's own match without retrying", async () => {
      answerFullName([CAMBRIDGE_RESULTS[1]]);
      await run({ name: "Cambridge, MA" });

      expect(geocodeNames()).toEqual(["Cambridge, MA"]);
      expect(fetchMock.mock.calls.at(-1)?.[0]).toContain("latitude=42.37");
    });

    it("retries with the bare name and matches the qualifier", async () => {
      answerFullName([]);
      await run({ name: "Cambridge, Massachusetts" });

      expect(geocodeNames()).toEqual(["Cambridge, Massachusetts", "Cambridge"]);
      expect(fetchMock.mock.calls.at(-1)?.[0]).toContain("latitude=42.37");
    });

    it("falls back to the top candidate when the qualifier matches nothing", async () => {
      answerFullName([]);
      await run({ name: "Cambridge, UK" });

      expect(geocodeNames()).toEqual(["Cambridge, UK", "Cambridge"]);
      expect(fetchMock.mock.calls.at(-1)?.[0]).toContain("latitude=52.2");
    });
  });

  it("asks for a single candidate without a country", async () => {
    await run({ name: "Cambridge" });

    const { geocode, forecast } = requested();
    expect(geocode.get("count")).toBe("1");
    expect(forecast.get("latitude")).toBe("52.2");
  });
});
