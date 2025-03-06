import { defineConfig } from "vitest/config";
import tsConfigPaths from "vite-tsconfig-paths";
import packageJson from "./package.json" with { type: "json" };

export default defineConfig({
  test: {
    globals: true,
    passWithNoTests: true,
    testTimeout: 120 * 1000,
    printConsoleTrace: true,
    setupFiles: ["./tests/setup.ts"],
    deps: {
      interopDefault: false,
    },
  },
  define: {
    __LIBRARY_VERSION: JSON.stringify(packageJson.version),
  },
  plugins: [
    tsConfigPaths({
      projects: ["tsconfig.json"],
    }),
  ],
});
