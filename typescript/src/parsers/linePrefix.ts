/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Emitter } from "@/emitter/emitter.js";
import { entries } from "remeda";
import { ValueError } from "@/errors.js";
import { Serializable } from "@/internals/serializable.js";
import { shallowCopy } from "@/serializer/utils.js";
import { Cache } from "@/cache/decoratorCache.js";
import { ParserField } from "@/parsers/field.js";
import { Callback, InferCallbackValue } from "@/emitter/types.js";
import { ZodError } from "zod";
import { ValueOf } from "@/internals/types.js";
import { LinePrefixParserError } from "@/parsers/errors.js";
export * from "@/parsers/errors.js";

export interface ParserNode<T extends string, P extends ParserField<any, any>> {
  prefix: string;
  next: readonly T[];
  isStart?: boolean;
  isEnd?: boolean;
  field: P;
}

type Customizer<T extends NonNullable<unknown>, T2 extends NonNullable<unknown> = T> = (
  nodes: T,
  options: Options<T>,
) => { nodes: T2; options: Options<T2> };

type StringKey<T extends NonNullable<unknown>> = Extract<keyof T, string>;

interface Callbacks<T extends Record<string, ParserNode<StringKey<T>, ParserField<any, any>>>> {
  update: Callback<
    {
      [K in StringKey<T>]: {
        key: K;
        value: ParserField.inferValue<T[K]["field"]>;
        field: T[K]["field"];
      };
    }[StringKey<T>]
  >;
  partialUpdate: Callback<
    {
      [K in StringKey<T>]: {
        key: K;
        delta: string;
        value: ParserField.inferPartialValue<T[K]["field"]>;
        field: T[K]["field"];
      };
    }[StringKey<T>]
  >;
}

interface Line {
  value: string;
  newLine: boolean;
}

const NEW_LINE_CHARACTER = "\n" as const;

interface ExtractedLine<T extends NonNullable<unknown>> {
  key: StringKey<T>;
  value: string;
  partial: boolean;
}

const trimLeftSpaces = (value: string) => value.replace(/^\s*/g, "");
const linesToString = (lines: Line[]) =>
  lines.reduce(
    (acc, { newLine, value }) => `${acc}${newLine ? NEW_LINE_CHARACTER.concat(value) : value}`,
    "",
  );

interface Options<T extends NonNullable<unknown>> {
  fallback?: (value: string) => readonly { key: StringKey<T>; value: string }[];
  endOnRepeat?: boolean;
  waitForStartNode?: boolean;
  silentNodes?: StringKey<T>[];
}

type Input<K extends string = string> = Record<string, ParserNode<K, ParserField<any, any>>>;

export class LinePrefixParser<T extends Input<StringKey<T>>> extends Serializable {
  public readonly emitter = new Emitter<Callbacks<T>>({
    creator: this,
    namespace: ["agent", "parser", "line"],
  });

  protected readonly lines: Line[] = [];
  protected readonly excludedLines: Line[] = [];
  protected done = false;
  protected lastNodeKey: StringKey<T> | null = null;

  public readonly finalState = {} as LinePrefixParser.infer<T>;
  public readonly partialState = {} as Partial<Record<StringKey<T>, string>>;

  get isDone() {
    return this.done;
  }

  constructor(
    protected readonly nodes: T,
    protected readonly options: Options<T> = {},
  ) {
    super();

    let hasStartNode = false;
    let hasEndNode = false;

    for (const [key, { next, isStart, isEnd }] of entries(nodes ?? {})) {
      hasStartNode = Boolean(hasStartNode || isStart);
      hasEndNode = Boolean(hasEndNode || isEnd);

      for (const nextKey of next) {
        if (key === nextKey) {
          throw new ValueError(`Node '${key}' cannot point to itself.`);
        }
        if (!(nextKey in nodes)) {
          throw new ValueError(
            `Node '${key}' contains a transition to non-existing node '${nextKey}'.`,
          );
        }
      }
    }

    if (!hasStartNode) {
      throw new ValueError(`At least one start node must be provided!`);
    }
    if (!hasEndNode) {
      throw new ValueError(`At least one end node must be provided!`);
    }
  }

  fork<R extends T = T>(customizer: Customizer<T, R>) {
    const { nodes, options } = customizer(this.nodes, this.options);
    return new LinePrefixParser(nodes, options);
  }

  async add(chunk: string) {
    if (!chunk || this.done) {
      return;
    }

    chunk.split(NEW_LINE_CHARACTER).forEach((line, i, arr) => {
      const isFirstLine = i === 0;

      if (isFirstLine) {
        if (this.lines.length === 0) {
          this.lines.push({ newLine: false, value: line });
        } else {
          this.lines.at(-1)!.value += line;
        }
      } else {
        this.lines.push({ newLine: arr.length > 1, value: line });
      }
    });

    while (this.lines.length > 0) {
      const line = this.lines[0];
      const isLastLine = this.lines.length === 1;

      const lastNode = this.lastNodeKey ? this.nodes[this.lastNodeKey] : null;
      const isTerminationNode = lastNode
        ? Boolean(lastNode.isEnd && lastNode.next.length === 0)
        : false;

      const parsedLine =
        isTerminationNode || (lastNode && !line.newLine) ? null : this.extractLine(line.value);

      if (isLastLine && (parsedLine?.partial || !line.value)) {
        break;
      }
      this.lines.shift();

      if (parsedLine && !parsedLine.partial) {
        if (lastNode) {
          if (!lastNode.next.includes(parsedLine.key)) {
            if (parsedLine.key in this.finalState && this.options.endOnRepeat && lastNode.isEnd) {
              await this.end();
              return;
            }

            this.throwWithContext(
              `Transition from '${this.lastNodeKey}' to '${parsedLine.key}' does not exist!`,
              LinePrefixParserError.Reason.InvalidTransition,
              { line },
            );
          }

          await this.emitFinalUpdate(this.lastNodeKey!, lastNode.field);
        } else if (!this.nodes[parsedLine.key].isStart) {
          if (!this.options.waitForStartNode) {
            this.throwWithContext(
              `Parsed text line corresponds to a node "${parsedLine.key}" which is not a start node!`,
              LinePrefixParserError.Reason.NotStartNode,
              { line },
            );
          }
          this.excludedLines.push(line);
          continue;
        }

        const node = this.nodes[parsedLine.key];
        node.field.write(parsedLine.value);
        await this.emitPartialUpdate({
          key: parsedLine.key,
          value: node.field.getPartial(),
          delta: parsedLine.value,
          field: node.field,
        });
        this.lastNodeKey = parsedLine.key;
      } else if (this.lastNodeKey) {
        if (!this.nodes[this.lastNodeKey].field.raw) {
          line.value = trimLeftSpaces(line.value);
        }
        if (line.newLine) {
          line.value = `${NEW_LINE_CHARACTER}${line.value}`;
        }

        const node = this.nodes[this.lastNodeKey];
        node.field.write(line.value);
        await this.emitPartialUpdate({
          key: this.lastNodeKey,
          value: node.field.getPartial(),
          delta: line.value,
          field: node.field,
        });
      } else {
        this.excludedLines.push(line);
      }
    }
  }

  protected throwWithContext(
    message: string,
    reason: ValueOf<typeof LinePrefixParserError.Reason>,
    extra: { line?: Line; errors?: Error[] } = {},
  ): never {
    throw new LinePrefixParserError(
      [`The generated output does not adhere to the schema.`, message].join(NEW_LINE_CHARACTER),
      extra.errors ?? [],
      {
        reason,
        context: {
          lines: linesToString(this.lines.concat(extra.line ? [extra.line] : [])),
          excludedLines: linesToString(this.excludedLines),
          finalState: this.finalState,
          partialState: this.partialState,
        },
      },
    );
  }

  async end() {
    if (this.done) {
      return this.finalState;
    }

    if (!this.lastNodeKey && this.options.fallback) {
      const stash = linesToString([
        ...this.excludedLines.splice(0, Infinity),
        ...this.lines.splice(0, Infinity),
      ]);

      const nodes = this.options.fallback(stash);
      await this.add(
        nodes.map((node) => `${this.nodes[node.key].prefix}${node.value}`).join(NEW_LINE_CHARACTER),
      );
    }

    this.done = true;

    if (!this.lastNodeKey) {
      this.throwWithContext(
        "Nothing valid has been parsed yet!",
        LinePrefixParserError.Reason.NoDataReceived,
      );
    }

    const stash = linesToString(this.lines);
    this.lines.length = 0;

    const field = this.nodes[this.lastNodeKey].field!;
    if (stash) {
      field.write(stash);
      await this.emitPartialUpdate({
        key: this.lastNodeKey,
        value: field.getPartial(),
        delta: stash,
        field,
      });
    }
    await this.emitFinalUpdate(this.lastNodeKey, field);

    const currentNode = this.nodes[this.lastNodeKey];
    if (!currentNode.isEnd) {
      this.throwWithContext(
        `Node '${this.lastNodeKey}' is not an end node.`,
        LinePrefixParserError.Reason.NotEndNode,
      );
    }

    await Promise.allSettled(Object.values(this.nodes).map(({ field }) => field.end()));
    return this.finalState;
  }

  protected async emitPartialUpdate(data: InferCallbackValue<Callbacks<T>["partialUpdate"]>) {
    if (data.key in this.finalState) {
      this.throwWithContext(
        `Cannot update partial event for completed key '${data.key}'`,
        LinePrefixParserError.Reason.AlreadyCompleted,
      );
    }
    if (!(data.key in this.partialState)) {
      this.partialState[data.key] = "";
    }
    this.partialState[data.key] += data.delta;
    if (!this.options.silentNodes?.includes(data.key)) {
      await this.emitter.emit("partialUpdate", data);
    }
  }

  protected async emitFinalUpdate(key: StringKey<T>, field: ParserField<any, any>) {
    if (key in this.finalState) {
      this.throwWithContext(
        `Duplicated key '${key}'`,
        LinePrefixParserError.Reason.AlreadyCompleted,
      );
    }

    try {
      const value = field.get();
      this.finalState[key] = value;
      if (!this.options.silentNodes?.includes(key)) {
        await this.emitter.emit("update", {
          key,
          field,
          value,
        });
      }
    } catch (e) {
      if (e instanceof ZodError) {
        this.throwWithContext(
          `Value for '${key}' cannot be retrieved because it's value does not adhere to the appropriate schema.`,
          LinePrefixParserError.Reason.InvalidSchema,
          { errors: [e] },
        );
      }
      throw e;
    }
  }

  @Cache()
  protected get normalizedNodes() {
    return entries(this.nodes)
      .sort(([_, a], [__, b]) => a.prefix.length - b.prefix.length)
      .map(([key, value]) => {
        return [key, { lowerCasePrefix: value.prefix.toLowerCase(), ref: value }] as const;
      });
  }

  protected extractLine(line: string): ExtractedLine<T> | null {
    const trimmedLine = trimLeftSpaces(line);
    if (!trimmedLine) {
      return null;
    }

    for (const [key, node] of this.normalizedNodes) {
      const partial = node.lowerCasePrefix.length > trimmedLine.length;
      const [a, b] = partial
        ? [node.lowerCasePrefix, trimmedLine]
        : [trimmedLine, node.lowerCasePrefix];

      if (a.toLowerCase().startsWith(b.toLowerCase())) {
        return {
          key: key as StringKey<T>,
          value: partial ? trimmedLine : trimLeftSpaces(a.substring(b.length)),
          partial,
        };
      }
    }
    return null;
  }

  createSnapshot() {
    return {
      nodes: shallowCopy(this.nodes),
      lines: shallowCopy(this.lines),
      emitter: this.emitter,
      done: this.done,
      lastNodeKey: this.lastNodeKey,
      options: this.options,
    };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
}

// eslint-disable-next-line @typescript-eslint/no-namespace
export namespace LinePrefixParser {
  export type infer<T extends Record<string, ParserNode<StringKey<T>, ParserField<any, any>>>> = {
    [K in keyof T]: ParserField.inferValue<T[K]["field"]>;
  };
  export type inferPartial<
    T extends Record<string, ParserNode<StringKey<T>, ParserField<any, any>>>,
  > = {
    [K in StringKey<T>]: ParserField.inferPartialValue<T[K]["field"]>;
  };

  export type inferCallback<T> =
    T extends LinePrefixParser<any> ? (T["emitter"] extends Emitter<infer L> ? L : never) : never;

  export type inferOutput<T> =
    T extends LinePrefixParser<infer P> ? LinePrefixParser.infer<P> : never;

  export type inferPartialOutput<T> =
    T extends LinePrefixParser<infer P> ? LinePrefixParser.inferPartial<P> : never;

  export type define<T extends Record<string, ParserField<any, any>>> = {
    [K in StringKey<T>]: ParserNode<StringKey<T>, T[K]>;
  };
}
