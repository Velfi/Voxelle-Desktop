#!/usr/bin/env node
// Preview the next changelog section without modifying any files.
// Usage: node scripts/preview-changelog.mjs [version]

import { readFragments, buildSection } from "./compile-changelog.mjs";

const version = process.argv[2] || "UNRELEASED";
const fragments = readFragments();

if (fragments.length === 0) {
  console.log("No changelog fragments found in .changes/");
  process.exit(0);
}

console.log(buildSection(version, fragments));
