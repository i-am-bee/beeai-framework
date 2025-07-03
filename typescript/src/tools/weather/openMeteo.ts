/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  BaseToolOptions,
  BaseToolRunOptions,
  ToolEmitter,
  JSONToolOutput,
  Tool,
  ToolError,
  ToolInput,
  ToolInputValidationError,
} from "@/tools/base.js";
import { z } from "zod";
import { createURLParams } from "@/internals/fetcher.js";
import { isNullish, omit, pick, pickBy } from "remeda";
import { Cache } from "@/cache/decoratorCache.js";
import { RunContext } from "@/context.js";
import { getProp, setProp } from "@/internals/helpers/object.js";
import { Emitter } from "@/emitter/emitter.js";

export interface ResponseFilter {
  excludedKeys: (keyof OpenMeteoToolResponse)[];
}

interface ToolOptions extends BaseToolOptions {
  apiKey?: string;
  responseFilter?: ResponseFilter;
}

type ToolRunOptions = BaseToolRunOptions;

interface Location {
  latitude: number;
  longitude: number;
}

interface LocationSearch {
  name: string;
  country?: string;
  language?: string;
}

export interface OpenMeteoToolResponse {
  latitude?: number;
  longitude?: number;
  generationtime_ms?: number;
  utc_offset_seconds?: number;
  timezone?: string;
  timezone_abbreviation?: string;
  elevation?: number;
  current_units?: Record<string, string>;
  current?: Record<string, string>;
  hourly_units?: Record<string, string>;
  hourly?: Record<string, any[]>;
  daily_units?: Record<string, string>;
  daily?: Record<string, any[]>;
}

export class OpenMeteoToolOutput extends JSONToolOutput<OpenMeteoToolResponse> {}

export class OpenMeteoTool extends Tool<OpenMeteoToolOutput, ToolOptions, ToolRunOptions> {
  name = "OpenMeteo";
  description = `Retrieve current, past, or future weather forecasts for a location.`;

  inputSchema() {
    return z
      .object({
        location: z.union([
          z
            .object({
              name: z.string().min(1),
              country: z.string().optional(),
              language: z.string().default("English"),
            })
            .strip(),
          z
            .object({
              latitude: z.coerce.number(),
              longitude: z.coerce.number(),
            })
            .strip(),
        ]),
        start_date: z
          .string()
          .date()
          .describe("Start date for the weather forecast in the format YYYY-MM-DD (UTC)"),
        end_date: z
          .string()
          .date()
          .describe("End date for the weather forecast in the format YYYY-MM-DD (UTC)")
          .optional(),
        temperature_unit: z.enum(["celsius", "fahrenheit"]).default("celsius"),
      })
      .strip();
  }

  public readonly emitter: ToolEmitter<ToolInput<this>, OpenMeteoToolOutput> = Emitter.root.child({
    namespace: ["tool", "weather", "openMeteo"],
    creator: this,
  });

  static {
    this.register();
  }

  public constructor(options: Partial<ToolOptions> = {}) {
    super({
      ...options,
      responseFilter: options?.responseFilter ?? {
        excludedKeys: [
          "latitude",
          "longitude",
          "generationtime_ms",
          "utc_offset_seconds",
          "timezone",
          "timezone_abbreviation",
          "elevation",
          "hourly",
          "hourly_units",
        ],
      },
    });
  }

  protected preprocessInput(rawInput: unknown) {
    super.preprocessInput(rawInput);

    const fixDate = (key: keyof ToolInput<this>) => {
      const value = getProp(rawInput, [key]);
      if (value) {
        setProp(rawInput, [key], value.substring(0, 10));
      }
    };

    fixDate("start_date");
    fixDate("end_date");
  }

  protected async _run(
    { location, start_date: startDate, end_date: endDate, ...input }: ToolInput<this>,
    _options: Partial<BaseToolRunOptions>,
    run: RunContext<this>,
  ) {
    const { apiKey } = this.options;

    const prepareParams = async () => {
      const extractLocation = async (): Promise<Location> => {
        if ("name" in location) {
          const response = await this._geocode(location, run.signal);
          return pick(response, ["latitude", "longitude"]);
        }
        return location;
      };

      function validateAndSetDates(
        startDateStr: string,
        endDateStr?: string,
      ): { start: Date; end: Date } {
        const now = new Date();
        const start = startDateStr
          ? new Date(startDateStr)
          : new Date(Date.UTC(now.getFullYear(), now.getMonth(), now.getDate()));

        if (endDateStr) {
          const end = new Date(endDateStr);
          if (end < start) {
            throw new ToolInputValidationError(
              `The 'end_date' (${endDateStr}) has to occur on or after the 'start_date' (${startDateStr}).`,
            );
          }
          return { start: start, end: end };
        } else {
          // If endDate is undefined, set it to the start date
          return { start: start, end: new Date(start) };
        }
      }

      const { start, end } = validateAndSetDates(startDate, endDate);

      const toDateString = (date: Date) => date.toISOString().split("T")[0];

      return createURLParams({
        ...pickBy(input, (v) => !isNullish(v) && v !== ""),
        ...(await extractLocation()),
        start_date: toDateString(start),
        end_date: toDateString(end),
        current: ["temperature_2m", "rain", "relative_humidity_2m", "wind_speed_10m"],
        daily: ["temperature_2m_max", "temperature_2m_min", "rain_sum"],
        hourly: ["temperature_2m", "relative_humidity_2m", "rain"],
        timezone: "UTC",
      });
    };

    const params = await prepareParams();
    const response = await fetch(`https://api.open-meteo.com/v1/forecast?${params}`, {
      headers: {
        ...(apiKey && {
          Authorization: `Bearer ${apiKey}`,
        }),
      },
      signal: run.signal,
    });

    if (!response.ok) {
      throw new ToolError("Request to OpenMeteo API has failed!", [
        new Error(await response.text()),
      ]);
    }

    let data: OpenMeteoToolResponse = await response.json();

    if (this.options?.responseFilter?.excludedKeys) {
      data = omit(data, this.options.responseFilter.excludedKeys);
    }

    return new OpenMeteoToolOutput(data);
  }

  @Cache()
  protected async _geocode(location: LocationSearch, signal: AbortSignal) {
    const { apiKey } = this.options;

    const params = createURLParams({
      name: location.name,
      language: location.language,
      country: location.country,
      format: "json",
      count: 1,
    });
    const response = await fetch(`https://geocoding-api.open-meteo.com/v1/search?${params}`, {
      headers: {
        ...(apiKey && {
          Authorization: `Bearer ${apiKey}`,
        }),
      },
      signal,
    });
    if (!response.ok) {
      throw new ToolError(`Failed to GeoCode provided location (${location.name}).`, [
        new Error(await response.text()),
      ]);
    }

    const { results } = await response.json();
    if (!results || results.length === 0) {
      throw new ToolError(`Location '${location.name}' was not found.`);
    }
    return results[0];
  }
}
