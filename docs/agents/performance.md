# Performance metrics

When investigating slowness, regressions, or GPU/mesh behavior, **include a performance snapshot** in issues or chat:

1. In the app menu: **Debug → Copy performance info**.
2. Paste the full text block into your message (issue, PR description, or agent conversation).

That snapshot includes viewport FPS, physical viewport size, mesh index/vertex buffer stats, voxel count, grid size, file label, platform, a UTC timestamp, and **last voxel edit timings** (apply, prepare, viewer lock wait, brick upload, mesh total, **mesh sub-phases** when relevant—spatial cache delta, cold cache init, greedy remesh, chunk GPU buffers, full chunked rebuild, non-incremental pipeline, preview clear—then route and total). It is produced in [`src-tauri/src/lib.rs`](src-tauri/src/lib.rs) (`performance_report_text`).

If clipboard copy fails, say so and manually note: approximate FPS (shown in the viewport UI), OS, scene size, and what you were doing (e.g. editing vs orbiting).
