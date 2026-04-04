#!/usr/bin/env node
// Compile .changes/*.md fragments into CHANGELOG.md and delete them.
// Usage: node scripts/compile-changelog.mjs <version>

import { readdirSync, readFileSync, writeFileSync, unlinkSync, existsSync } from "node:fs";
import { join, basename } from "node:path";

const CATEGORIES = ["added", "changed", "fixed", "removed"];
const CHANGES_DIR = join(import.meta.dirname, "..", ".changes");

/** Parse a fragment file into { category, body }. Throws on invalid format. */
export function parseFragment(filePath) {
  const raw = readFileSync(filePath, "utf-8").trim();
  const match = raw.match(/^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$/);
  if (!match) {
    throw new Error(`Invalid fragment format in ${basename(filePath)} — expected YAML frontmatter`);
  }

  const frontmatter = match[1];
  const body = match[2].trim();
  const catMatch = frontmatter.match(/^category:\s*(.+)$/m);
  if (!catMatch) {
    throw new Error(`Missing 'category' in frontmatter of ${basename(filePath)}`);
  }

  const category = catMatch[1].trim().toLowerCase();
  if (!CATEGORIES.includes(category)) {
    throw new Error(
      `Invalid category '${category}' in ${basename(filePath)} — must be one of: ${CATEGORIES.join(", ")}`,
    );
  }

  return { category, body };
}

/** Read all fragment files (excluding README.md) from .changes/. */
export function readFragments() {
  const files = readdirSync(CHANGES_DIR)
    .filter((f) => f.endsWith(".md") && f !== "README.md")
    .sort();
  return files.map((f) => ({ file: f, ...parseFragment(join(CHANGES_DIR, f)) }));
}

/** Build a changelog section string from parsed fragments. */
export function buildSection(version, fragments) {
  const date = new Date().toISOString().slice(0, 10);
  const lines = [`## ${version} — ${date}`, ""];

  if (fragments.length === 0) {
    lines.push("No notable changes.", "");
    return lines.join("\n");
  }

  const grouped = Object.groupBy(fragments, (f) => f.category);

  for (const cat of CATEGORIES) {
    const entries = grouped[cat];
    if (!entries?.length) continue;
    lines.push(`### ${cat.charAt(0).toUpperCase() + cat.slice(1)}`);
    for (const entry of entries) {
      lines.push(`- ${entry.body}`);
    }
    lines.push("");
  }

  return lines.join("\n");
}

// --- CLI entry point ---
if (process.argv[1] === import.meta.filename) {
  const version = process.argv[2];
  if (!version) {
    console.error("Usage: node scripts/compile-changelog.mjs <version>");
    process.exit(1);
  }

  const fragments = readFragments();
  if (fragments.length === 0) {
    console.warn("Warning: no changelog fragments found in .changes/");
  }

  const section = buildSection(version, fragments);
  const changelogPath = join(import.meta.dirname, "..", "CHANGELOG.md");
  const existing = existsSync(changelogPath) ? readFileSync(changelogPath, "utf-8") : "";
  const header = existing ? "" : "# Changelog\n\n";
  writeFileSync(changelogPath, header + section + "\n" + existing, "utf-8");
  console.log(`Updated CHANGELOG.md with ${fragments.length} entries for ${version}`);

  // Delete compiled fragments
  for (const f of fragments) {
    unlinkSync(join(CHANGES_DIR, f.file));
  }
}
