import { defineConfig } from "tsup";
import packageJson from "./package.json" with { type: "json" };
import swc, { JscConfig } from "@swc/core";
import path from "node:path";

import tsConfig from "./tsconfig.json" with { type: "json" };
import { JscTarget } from "@swc/types";

export default defineConfig({
  entry: ["src/**/*.{ts,js}", "!src/**/*.test.ts"],
  tsconfig: "./tsconfig.json",
  sourcemap: true,
  dts: true,
  format: ["esm", "cjs"],
  plugins: [
    {
      name: "fix-cjs-imports",
      renderChunk(code) {
        if (this.format === "cjs") {
          const regexCjs = /require\((?<quote>['"])(?<import>\.[^'"]+)\.js['"]\)/g;
          const regexEsm = /from(?<space>[\s]*)(?<quote>['"])(?<import>\.[^'"]+)\.js['"]/g;
          return {
            code: code
              .replace(regexCjs, "require($<quote>$<import>.cjs$<quote>)")
              .replace(regexEsm, "from$<space>$<quote>$<import>.cjs$<quote>"),
          };
        }
      },
    },
    {
      name: "override-swc",
      esbuildOptions: (options) => {
        const plugin = options.plugins?.find((p) => p.name === "swc");
        if (plugin) {
          // Original Source: https://github.com/egoist/tsup/blob/49c11c3073ce977a01c84e7848fc070d5de0a652/src/esbuild/swc.ts#L14-L67
          // Reason: tsup does not provide a way to modify 'jsc' config
          plugin.setup = (build) => {
            // Force esbuild to keep class names as well
            build.initialOptions.keepNames = true;

            build.onLoad({ filter: /\.[jt]sx?$/ }, async (args: any) => {
              const isTs = /\.tsx?$/.test(args.path);

              const jsc: JscConfig = {
                parser: {
                  syntax: isTs ? "typescript" : "ecmascript",
                  decorators: true,
                },
                transform: {
                  legacyDecorator: true,
                  decoratorMetadata: true,
                },
                baseUrl: path.resolve(__dirname, tsConfig.compilerOptions.baseUrl || "."),
                paths: tsConfig.compilerOptions.paths,
                keepClassNames: true,
                preserveAllComments: true,
                target: (tsConfig.compilerOptions.target || "es2022").toLowerCase() as JscTarget,
              };

              const result = await swc.transformFile(args.path, {
                jsc,
                sourceMaps: true,
                configFile: false,
                swcrc: false,
              });

              let code = result.code;
              if (result.map) {
                const map: { sources: string[] } = JSON.parse(result.map);
                // Make sure sources are relative path
                map.sources = map.sources.map((source) => {
                  return path.isAbsolute(source)
                    ? path.relative(path.dirname(args.path), source)
                    : source;
                });
                code += `//# sourceMappingURL=data:application/json;base64,${Buffer.from(
                  JSON.stringify(map),
                ).toString("base64")}`;
              }
              return {
                contents: code,
              };
            });
          };
        }
      },
    },
  ],
  treeshake: true,
  shims: true,
  skipNodeModulesBundle: true,
  legacyOutput: false,
  bundle: false,
  splitting: false,
  silent: false,
  clean: true,
  define: {
    __LIBRARY_VERSION: JSON.stringify(packageJson.version),
  },
});
