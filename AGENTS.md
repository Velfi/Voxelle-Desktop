# Agent notes — Voxelle Desktop (VD)

Guidance for humans and coding agents working in this repo. This app is desktop-only product work.

## Guidelines

- Agents should choose the most robust solution, even if it's more code.

## Topics

- [Main UI](docs/agents/ui-layout.md) — current sidebar/HUD/status-bar responsibilities and long-wait UX rules
- [Viewport & picking](docs/agents/viewport-picking.md) — authoritative viewport pixel mapping and pointer-to-world conventions
- [Performance metrics](docs/agents/performance.md) — how to capture the built-in performance snapshot and what it includes
- [Code change hints](docs/agents/code-change-hints.md) — lock ordering, edit/mesh paths, IPC allowlist, autosave, raytrace, macOS undo
- [Rust → webview events](docs/agents/tauri-events.md) — when to emit events and how shared collab state must propagate
- [Phased viewport tools](docs/agents/phased-tools.md) — `useStrokePhase` and current multi-step tool patterns
- [`.voxelle` file format](docs/agents/file-format.md) — current VX5/VX4 container behavior and dense wire payload notes
- [Bundled resources](docs/agents/bundled-resources.md) — current compile-time embedded assets and generated avatar module behavior

## Other docs

- [Headless server mode](docs/HEADLESS_SERVER.md) — integration test startup, flags, health endpoint
- [Format/container notes](docs/VOXELLE_FORMAT_V4.md) — detailed container and wire-format behavior, including current VX5 saves

## Format

Keep entries short, practical, and tied to the code that exists today. Prefer updating these focused docs when a project convention becomes recurring enough to matter.
