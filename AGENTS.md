# Agent notes — Voxelle Desktop (VD)

Guidance for humans and coding agents working in this repo.

## Performance metrics

When investigating slowness, regressions, or GPU/mesh behavior, **include a performance snapshot** in issues or chat:

1. In the app menu: **Debug → Copy Performance Data to Clipboard**.
2. Paste the full text block into your message (issue, PR description, or agent conversation).

That snapshot includes viewport FPS, physical viewport size, mesh index/vertex buffer stats, voxel count, grid size, file label, platform, and a UTC timestamp. It is produced in [`src-tauri/src/lib.rs`](src-tauri/src/lib.rs) (`performance_report_text`).

If clipboard copy fails, say so and manually note: approximate FPS (shown in the viewport UI), OS, scene size, and what you were doing (e.g. editing vs orbiting).

## Hints for code changes

- **3D / mesh / editing**: [`src-tauri/src/render/`](src-tauri/src/render/), [`src-tauri/src/greedy_mesh.rs`](src-tauri/src/greedy_mesh.rs), [`src-tauri/src/voxel_edit.rs`](src-tauri/src/voxel_edit.rs), [`src-tauri/src/lib.rs`](src-tauri/src/lib.rs) (Tauri commands, `ViewerState`).
- **Frontend**: [`src/App.tsx`](src/App.tsx) — viewport events, `invoke` to Rust.

## Format

Extend this file with project conventions as they solidify (e.g. build commands, test expectations). Keep entries short and scannable.
