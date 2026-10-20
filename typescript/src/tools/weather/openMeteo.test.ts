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

  it("asks for a single candidate without a country", async () => {
    await run({ name: "Cambridge" });

    const { geocode, forecast } = requested();
    expect(geocode.get("count")).toBe("1");
    expect(forecast.get("latitude")).toBe("52.2");
  });
});
