# Contributing to Voxelle Desktop

Technical reference for developers working on this codebase.

## Tech stack

- **Frontend**: React + TypeScript, bundled with Vite
- **Backend**: Rust via [Tauri](https://tauri.app/), GPU rendering with wgpu
- **IPC**: Tauri commands (Rust ↔ webview) and events (Rust → webview)

## Dev setup

### Prerequisites

- [Rust](https://rustup.rs/) (stable)
- Node.js 18+
- Tauri CLI: `npm install` installs it as a dev dependency

### Run in development

```sh
npm install
npm run tauri dev
```

### Build a release binary

```sh
npm run tauri build
```

## Project layout

```
src/                        React/TypeScript frontend
  App.tsx                   Root component
  ToolsSidebar.tsx          Tool panel (draw, sculpt, select, etc.)
  hooks/                    useFlyMode, useWalkMode, useKeyboardShortcuts, …
src-tauri/
  src/
    lib.rs                  App entry point — invoke_handler! registration
    state.rs                ViewerState and shared enums
    commands/               Tauri IPC command handlers (one file per domain)
    camera.rs               Fly/walk camera logic
    render/                 wgpu renderer — pipelines, frame loop, overlays
    native_menu.rs          macOS/Windows native menu
    frame_loop.rs           Per-frame update loop
  permissions/
    voxelle.toml            IPC allowlist — must be updated for new commands
docs/
  VOXELLE_FORMAT_V4.md      Canonical .voxelle file format specification
  HEADLESS_SERVER.md        Headless/integration-test mode
  agents/                   Focused reference docs (viewport, events, etc.)
AGENTS.md                   Index of agent/contributor reference docs
```

## Adding a Tauri command

Every new `#[tauri::command]` function must be registered in **two places**:

1. `invoke_handler!` macro in `src-tauri/src/lib.rs`
2. `src-tauri/permissions/voxelle.toml` — IPC allowlist

Skipping either step will cause a silent permission denial at runtime.

## Agent / contributor docs

The [AGENTS.md](AGENTS.md) file indexes focused reference documents covering:

- [Main UI layout](docs/agents/ui-layout.md)
- [Viewport & picking](docs/agents/viewport-picking.md)
- [Performance metrics](docs/agents/performance.md)
- [Code change hints](docs/agents/code-change-hints.md) (mutex ordering, mesh paths, undo)
- [Rust → webview events](docs/agents/tauri-events.md)
- [Phased viewport tools](docs/agents/phased-tools.md)
- [.voxelle file format](docs/agents/file-format.md)
- [Bundled resources](docs/agents/bundled-resources.md)

## Headless / integration testing

The app can run without a visible window and expose a local HTTP health endpoint for test harnesses. See [docs/HEADLESS_SERVER.md](docs/HEADLESS_SERVER.md).

## Recommended IDE

VS Code with the [Tauri extension](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) and [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer).
