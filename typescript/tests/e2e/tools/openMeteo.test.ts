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

import { beforeEach, expect } from "vitest";
import { OpenMeteoTool } from "@/tools/weather/openMeteo.js";
import { ToolInputValidationError } from "@/tools/base.js";

describe("OpenMeteo", () => {
  let instance: OpenMeteoTool;

  beforeEach(() => {
    instance = new OpenMeteoTool();
  });

  it("Runs", async () => {
    const response = await instance.run(
      {
        location: {
          name: "Boston",
        },
        start_date: "2024-11-06",
        temperature_unit: "celsius",
      },
      {
        signal: AbortSignal.timeout(60 * 1000),
        retryOptions: {},
      },
    );

    expect(response.isEmpty()).toBe(false);
    expect(response.result).toMatchObject({
      current: expect.any(Object),
      current_units: expect.any(Object),
      daily: expect.any(Object),
      daily_units: expect.any(Object),
    });
  });

  it("Custom filter", async () => {
    instance = new OpenMeteoTool({
      responseFilter: {
        excludedKeys: [
          "latitude",
          "longitude",
          "timezone_abbreviation",
          "elevation",
          "utc_offset_seconds",
          "generationtime_ms",
        ],
      },
    });

    const now = new Date();
    const dateString = new Date(Date.UTC(now.getFullYear(), now.getMonth(), now.getDate()))
      .toISOString()
      .split("T")[0];
    const response = await instance.run(
      {
        location: {
          name: "Anchorage",
        },
        start_date: dateString,
        temperature_unit: "fahrenheit",
      },
      {
        signal: AbortSignal.timeout(60 * 1000),
        retryOptions: {},
      },
    );

    expect(response.isEmpty()).toBe(false);

    expect(response.result).toMatchObject({
      timezone: expect.any(String),
      current: expect.any(Object),
      current_units: expect.any(Object),
      daily: expect.any(Object),
      daily_units: expect.any(Object),
      hourly: expect.any(Object),
      hourly_units: expect.any(Object),
    });
  });

  it("Correct Date", async () => {
    const now = new Date();
    const dateString = new Date(Date.UTC(now.getFullYear(), now.getMonth(), now.getDate()))
      .toISOString()
      .split("T")[0];
    const response = await instance.run(
      {
        location: {
          name: "Reykjavík",
        },
        start_date: dateString,
        temperature_unit: "fahrenheit",
      },
      {
        signal: AbortSignal.timeout(60 * 1000),
        retryOptions: {},
      },
    );

    expect(response.isEmpty()).toBe(false);
    expect(response.result.daily).toHaveProperty("time", [dateString]);
  });

  it("Throws", async () => {
    await expect(
      instance.run({
        location: { name: "Prague" },
        start_date: "123",
      }),
    ).rejects.toThrowError(ToolInputValidationError);
  });

  it("Throws for unknown location", async () => {
    await expect(
      instance.run({
        location: { name: "ABCDEFGH" },
        start_date: "2024-01-01",
      }),
    ).rejects.toThrowErrorMatchingInlineSnapshot(`ToolError: Location 'ABCDEFGH' was not found.`);
  });

  it("Throws on bad end_date < start_date", async () => {
    await expect(
      instance.run(
        {
          location: {
            name: "Boston",
          },
          start_date: "2024-11-06",
          end_date: "2024-11-05",
          temperature_unit: "celsius",
        },
        {
          signal: AbortSignal.timeout(60 * 1000),
          retryOptions: {},
        },
      ),
    ).rejects.toThrowError(ToolInputValidationError);
  });
});
