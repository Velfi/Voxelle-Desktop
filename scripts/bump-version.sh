#!/usr/bin/env bash
# Bump app version, compile changelog, commit, tag (vX.Y.Z), and push.
# Usage: ./scripts/bump-version.sh 0.1.2
# Run from anywhere inside the repo (uses git root).

set -euo pipefail

die() {
  echo "error: $*" >&2
  exit 1
}

[[ "${1:-}" ]] || die "usage: ${0##*/} <version>   (example: 0.1.2)"

VER="$1"
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel 2>/dev/null)" \
  || die "not inside a git repository"
cd "$ROOT"

if git rev-parse -q --verify "refs/tags/v${VER}" >/dev/null; then
  die "tag v${VER} already exists"
fi

# Update version in package.json, tauri.conf.json, and Cargo.toml
node - "$VER" "$ROOT" <<'JS'
const fs = require("node:fs");
const path = require("node:path");

const [ver, root] = process.argv.slice(2);
const pkg = path.join(root, "package.json");
const tc = path.join(root, "src-tauri", "tauri.conf.json");
const cargo = path.join(root, "src-tauri", "Cargo.toml");

for (const p of [pkg, tc]) {
  const data = JSON.parse(fs.readFileSync(p, "utf-8"));
  data.version = ver;
  fs.writeFileSync(p, JSON.stringify(data, null, 2) + "\n", "utf-8");
}

const text = fs.readFileSync(cargo, "utf-8");
const replaced = text.replace(/^version = "[^"]+"/m, `version = "${ver}"`);
if (replaced === text) {
  console.error("expected exactly one [package] version line in Cargo.toml");
  process.exit(1);
}
fs.writeFileSync(cargo, replaced, "utf-8");
JS

# Compile changelog fragments into CHANGELOG.md
node "$ROOT/scripts/compile-changelog.mjs" "$VER"

git add package.json src-tauri/tauri.conf.json src-tauri/Cargo.toml CHANGELOG.md .changes/
git commit -m "Release ${VER}" -- package.json src-tauri/tauri.conf.json src-tauri/Cargo.toml CHANGELOG.md .changes/
git tag -a "v${VER}" -m "v${VER}"

echo "Pushing branch and tag v${VER}..."
git push origin HEAD
git push origin "refs/tags/v${VER}"

echo "Done: version ${VER}, tag v${VER}"
