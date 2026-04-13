# Contributing to Voxelle Desktop

Technical reference for developers working on this codebase.

## Tech stack

- Frontend: React 18 + TypeScript + Vite 6
- Native shell: Tauri 2
- Renderer: Rust + `wgpu`
- IPC: Tauri commands plus Rust-emitted events consumed from the webview

## Dev setup

### Prerequisites

- Node.js 18+
- Rust toolchain from `rustup`
- Tauri CLI is installed through `npm install`

### Common commands

```sh
npm install
npm run tauri dev
```

```sh
npm run lint
npm run typecheck
npm run build
npm run tauri build
```

Rust-side benchmarks live in `src-tauri/benches/`:

```sh
cd src-tauri
cargo bench --bench greedy_mesh
cargo bench --bench smooth_mesh
cargo bench --bench preview_generation
cargo bench --bench collab_encoding
```

## Project layout

```text
src/                         React frontend
  App.tsx                    Root shell and high-level app state
  ToolsSidebar.tsx           Left tools/sidebar UI
  InspectorSidebar.tsx       Right inspector/session sidebar
  StatusBar.tsx              Bottom status bar
  ViewportHUD.tsx            Floating viewport HUDs for phased tools and work progress
  hooks/                     Input, generators, collab listeners, fly/walk, tool state
  toolOptions/               Tool-specific option panes
src-tauri/
  src/
    lib.rs                   App bootstrap, invoke registration, menu/event wiring
    commands/                Tauri command handlers by domain
    collab/                  Host/client networking, roster, snapshots, presence
    load_pipeline.rs         Load/open/new-project pipeline and progress events
    edit_pipeline.rs         Native edit application and menu sync helpers
    frame_loop.rs            Work-progress emits and viewport loop helpers
    render/                  Native renderer and GPU pipelines
    voxelle/                 File format, scene/object helpers, decode/encode
    state.rs                 Shared app/viewer state
  permissions/
    voxelle.toml             IPC allowlist for webview `invoke`
docs/
  agents/                    Focused engineering notes for recurring repo pitfalls
  HEADLESS_SERVER.md         Headless readiness server for integration tests
  VOXELLE_FORMAT_V4.md       Container and wire-format notes, including current VX5 saves
AGENTS.md                    Short index into the focused docs above
```

## Adding a Tauri command

Every new `#[tauri::command]` must be wired in two places:

1. `invoke_handler!` in [src-tauri/src/lib.rs](/Users/zelda/Documents/Voxelle Desktop/src-tauri/src/lib.rs)
2. `commands.allow` in [src-tauri/permissions/voxelle.toml](/Users/zelda/Documents/Voxelle Desktop/src-tauri/permissions/voxelle.toml)

If one side is missing, the webview call will be denied at runtime.

## Current persistence behavior

- Normal file saves use `encode_payload_v5`
- Collaboration snapshots still use `encode_payload_v4`
- Autosaves, recent files, and last-session state are stored under the app data directory via [src-tauri/src/commands/file_io.rs](/Users/zelda/Documents/Voxelle Desktop/src-tauri/src/commands/file_io.rs)

## Agent / contributor docs

The focused docs in [AGENTS.md](/Users/zelda/Documents/Voxelle Desktop/AGENTS.md) cover the project-specific things that most often trip people up: viewport coordinate mapping, long-running progress UX, phased tool state, file format details, event propagation, and native resource/loading behavior.

## Headless / integration testing

The app can start with the main window hidden and expose a localhost health endpoint for test harnesses. See [docs/HEADLESS_SERVER.md](/Users/zelda/Documents/Voxelle Desktop/docs/HEADLESS_SERVER.md).
