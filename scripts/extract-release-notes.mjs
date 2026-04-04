#!/usr/bin/env node
// Extract release notes for a specific version from CHANGELOG.md.
// Usage: node scripts/extract-release-notes.mjs v0.2.6

import { readFileSync } from "node:fs";
import { join } from "node:path";

const tag = process.argv[2];
if (!tag) {
  console.error("Usage: node scripts/extract-release-notes.mjs <tag>");
  process.exit(1);
}

const version = tag.replace(/^v/, "");
const changelog = readFileSync(join(import.meta.dirname, "..", "CHANGELOG.md"), "utf-8");

// Split on version headings, find the one matching our version
const sections = changelog.split(/^(?=## )/m);
const escaped = version.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const section = sections.find((s) => new RegExp(`^## ${escaped}\\b`).test(s));

if (!section) {
  console.error(`No changelog section found for version ${version}`);
  process.exit(1);
}

// Strip the ## heading line, print the rest
const body = section.replace(/^## [^\n]*\n/, "").trim();
if (body) console.log(body);
