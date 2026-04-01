# Agent notes — Voxelle Desktop (VD)

Guidance for humans and coding agents working in this repo. This app is desktop-only, no web, no android, no ios.

## Topics

- [Main UI](docs/agents/ui-layout.md) — sidebar/status-bar layout and long-wait UX rules
- [Viewport & picking](docs/agents/viewport-picking.md) — screen-to-world ray mapping, pixel-grid conventions, NDC pitfalls
- [Performance metrics](docs/agents/performance.md) — how to capture and share a performance snapshot
- [Code change hints](docs/agents/code-change-hints.md) — mutex ordering, mesh/edit paths, IPC allowlist, macOS undo
- [Rust → webview events](docs/agents/tauri-events.md) — when and how to emit Tauri events, collaboration broadcast rules
- [Phased viewport tools](docs/agents/phased-tools.md) — `useStrokePhase` hook, multi-step gesture conventions
- [`.voxelle` file format](docs/agents/file-format.md) — v4 wire format, outer/inner encoding
- [Bundled resources](docs/agents/bundled-resources.md) — `bundle.resources` path resolution in dev vs release

## Other docs

- [Headless server mode](docs/HEADLESS_SERVER.md) — integration test setup, flags, health endpoint
- [`.voxelle` format v4 spec](docs/VOXELLE_FORMAT_V4.md) — canonical format specification

## Format

Extend these files with project conventions as they solidify (e.g. build commands, test expectations). Keep entries short and scannable.
