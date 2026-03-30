# Agent notes — Voxelle Desktop (VD)

Guidance for humans and coding agents working in this repo.

## Performance metrics

When investigating slowness, regressions, or GPU/mesh behavior, **include a performance snapshot** in issues or chat:

1. In the app menu: **Debug → Copy Performance Data to Clipboard**.
2. Paste the full text block into your message (issue, PR description, or agent conversation).

That snapshot includes viewport FPS, physical viewport size, mesh index/vertex buffer stats, voxel count, grid size, file label, platform, a UTC timestamp, and **last voxel edit timings** (apply, prepare, viewer lock wait, brick upload, mesh total, **mesh sub-phases** when relevant—spatial cache delta, cold cache init, greedy remesh, chunk GPU buffers, full chunked rebuild, non-incremental pipeline, preview clear—then route and total). It is produced in [`src-tauri/src/lib.rs`](src-tauri/src/lib.rs) (`performance_report_text`).

If clipboard copy fails, say so and manually note: approximate FPS (shown in the viewport UI), OS, scene size, and what you were doing (e.g. editing vs orbiting).

## Hints for code changes

- **3D / mesh / editing**: [`src-tauri/src/render/`](src-tauri/src/render/), [`src-tauri/src/greedy_mesh.rs`](src-tauri/src/greedy_mesh.rs), [`src-tauri/src/voxel_edit.rs`](src-tauri/src/voxel_edit.rs), [`src-tauri/src/lib.rs`](src-tauri/src/lib.rs) (Tauri commands, `ViewerState`).
- **CPU mesh benchmarks** (Criterion): from `src-tauri/`, run `cargo bench --bench greedy_mesh` (see [`benches/greedy_mesh.rs`](src-tauri/benches/greedy_mesh.rs)).
- **Edit perf**: scene bounds use incremental updates when possible ([`greedy_mesh::mesh_bounds_expand_with_voxel`](src-tauri/src/greedy_mesh.rs), strict-interior removes skip full scans); chunk GPU buffers may reuse via `COPY_DST` + `write_buffer` when the new mesh fits.
- **Frontend**: [`src/App.tsx`](src/App.tsx) — viewport events, `invoke` to Rust.

## Format

Extend this file with project conventions as they solidify (e.g. build commands, test expectations). Keep entries short and scannable.
