/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

// @ts-check

import eslint from "@eslint/js";
import tseslint from "typescript-eslint";
import prettierConfig from "eslint-config-prettier";
import unusedImports from "eslint-plugin-unused-imports";
import markdown from "@eslint/markdown";
import stylisticJs from "@stylistic/eslint-plugin-js";

export default tseslint.config(
  {
    ignores: ["node_modules/**", "dist/**"],
  },
  eslint.configs.recommended,
  ...tseslint.configs.strict,
  ...tseslint.configs.stylistic,
  {
    files: ["**/*.md/**"],
    languageOptions: {
      parserOptions: {
        project: null,
      },
    },
    rules: {
      "@typescript-eslint/no-unused-vars": "off",
      "@typescript-eslint/no-non-null-assertion": "off",
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-empty-function": "off",
    },
  },
  {
    ignores: ["**/*.md/**"],
    languageOptions: {
      parserOptions: {
        project: "./tsconfig.json",
      },
    },
    plugins: {
      "unused-imports": unusedImports,
      // @ts-expect-error wrong types
      "@stylistic/js": stylisticJs,
    },
    rules: {
      "no-console": ["error"],
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              group: ["../", "src/"],
              message: "Relative imports are not allowed.",
            },
          ],
        },
      ],
      "@typescript-eslint/no-non-null-asserted-optional-chain": "off",
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-unused-vars": "off",
      "@typescript-eslint/no-empty-function": "off",
      "@typescript-eslint/no-extraneous-class": "off",
      "@typescript-eslint/no-empty-interface": "off",
      "@typescript-eslint/no-non-null-assertion": "off",
      "@typescript-eslint/no-floating-promises": "error",
      "unused-imports/no-unused-imports": "error",
      "unused-imports/no-unused-vars": [
        "warn",
        {
          vars: "all",
          varsIgnorePattern: "^_",
          args: "after-used",
          argsIgnorePattern: "^_",
        },
      ],
      "@typescript-eslint/ban-ts-comment": "off",
      "@typescript-eslint/no-empty-object-type": "off",
      "quote-props": ["error", "consistent"],
      // Force a new line after the copyright header; otherwise, nwa reports them as incorrectly formatted
      "@stylistic/js/lines-around-comment": [
        "error",
        {
          applyDefaultIgnorePatterns: false,
          afterBlockComment: true,
          beforeBlockComment: false,
          ignorePattern: /^(?![\s\S]*Copyright \d+ © BeeAI a Series of LF Projects, LLC)[\s\S]+$/u
            .source,
        },
      ],
      "@typescript-eslint/class-literal-property-style": "off",
    },
  },
  {
    files: ["examples/**/*.{ts,js}"],
    languageOptions: {
      parserOptions: {
        project: "./tsconfig.examples.json",
      },
      globals: {
        setTimeout: "readonly",
      },
    },
    rules: {
      "no-console": "off",
      "@typescript-eslint/no-unused-vars": "off",
      "unused-imports/no-unused-vars": "off",
      "@typescript-eslint/no-non-null-assertion": "off",
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              group: ["@/"],
              message: "Use 'beeai-framework' instead.",
            },
          ],
        },
      ],
    },
  },
  {
    files: ["examples/playground/**/*.ts"],
    languageOptions: {
      parserOptions: {
        project: "./tsconfig.examples.json",
      },
    },
    rules: {
      "no-restricted-imports": "off",
    },
  },
  ...markdown.configs.processor,
  prettierConfig,
  {
    rules: {
      curly: ["error", "all"],
    },
  },
);
