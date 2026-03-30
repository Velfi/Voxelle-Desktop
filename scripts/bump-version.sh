#!/usr/bin/env bash
# Bump app version in package.json, tauri.conf.json, and Cargo.toml; commit, tag (vX.Y.Z), and push.
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

python3 - "$VER" "$ROOT" <<'PY'
import json, pathlib, re, sys

ver, root = sys.argv[1], pathlib.Path(sys.argv[2])

pkg = root / "package.json"
tc = root / "src-tauri" / "tauri.conf.json"
cargo = root / "src-tauri" / "Cargo.toml"

for path in (pkg, tc):
    data = json.loads(path.read_text(encoding="utf-8"))
    data["version"] = ver
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

text = cargo.read_text(encoding="utf-8")
new, n = re.subn(
    r"^version = \"[^\"]+\"",
    f'version = "{ver}"',
    text,
    count=1,
    flags=re.MULTILINE,
)
if n != 1:
    sys.exit(f"expected exactly one [package] version line in Cargo.toml, got {n}")
cargo.write_text(new, encoding="utf-8")
PY

git add package.json src-tauri/tauri.conf.json src-tauri/Cargo.toml
git commit -m "Bump version to ${VER}" -- package.json src-tauri/tauri.conf.json src-tauri/Cargo.toml
git tag -a "v${VER}" -m "v${VER}"

echo "Pushing branch and tag v${VER}..."
git push origin HEAD
git push origin "refs/tags/v${VER}"

echo "Done: version ${VER}, tag v${VER}"
