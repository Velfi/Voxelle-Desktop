import eslint from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  eslint.configs.recommended,
  ...tseslint.configs.recommended,
  {
    ignores: ["dist/", "src-tauri/"],
  },
  {
    files: ["scripts/**"],
    languageOptions: {
      globals: {
        process: "readonly",
        console: "readonly",
      },
    },
  },
  {
    rules: {
      // Allow underscore-prefixed unused vars (common convention for intentionally unused bindings)
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
        },
      ],
      // Allow empty catch blocks (used for swallowing localStorage errors etc.)
      "no-empty": ["error", { allowEmptyCatch: true }],
    },
    // Don't error on eslint-disable comments for missing rules (e.g. react-hooks/exhaustive-deps)
    linterOptions: {
      reportUnusedDisableDirectives: "warn",
    },
  },
);
