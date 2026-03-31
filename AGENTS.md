# Agent notes — Voxelle Desktop (VD)

Guidance for humans and coding agents working in this repo. This app is desktop-only, no web, no android, no ios.

## Main UI

| Name | Role |
| --- | --- |
| **Tools sidebar** (left) | Controls and inputs for making art (modes, brushes, palette-style tools as they land). Implemented in [`src/App.tsx`](src/App.tsx) as the left `aside`. |
| **Inspector sidebar** (right) | Project metadata and management: hierarchy / outliner-style tools, properties, and related panels. Implemented as the right `aside`. |
| **Status bar** (bottom) | High-level feedback on what the app is doing (current file, load/collab state, optional FPS). The `footer.app-status-bar` row. |
| **Tool Panel** (bottom left)| Tool options and selection. Implemented in [`src/App.tsx`](src/App.tsx) as the bottom left `aside`. |

**Long waits:** Whenever the user might wait more than a few seconds (load, save, mesh rebuild, visibility refresh, etc.), the status bar should explain **why**—not a spinner with no copy. Emit meaningful phases (and optional fraction) via the usual events (e.g. `voxelle-work-progress`, `voxelle-load-progress` from [`src-tauri/src/lib.rs`](src-tauri/src/lib.rs)); don’t leave the UI silent during noticeable work.

## Viewport and screen-to-world ray mapping

The native [`WgpuViewer`](src-tauri/src/render/mod.rs) swapchain is sized by [`viewer_resize`](src-tauri/src/lib.rs) from the **`.viewport`** div in [`src/App.tsx`](src/App.tsx) (not the full window: chrome + sidebars shrink the div). **Picking and hover** must map the pointer into **the same pixel grid** the shader uses (`proj` / `view_proj` and [`voxel_edit::screen_to_world_ray`](src-tauri/src/voxel_edit.rs)).

**Conventions:**

- **Origin**: Treat `(sx, sy)` passed to `voxel_pick_probe`, `voxel_edit_at_screen`, and `sync_preview_input` as **physical pixels** in the **viewport-sized render target** (the sub-rect copied into the swapchain): `(0,0)` = top-left of that texture, **+X** right, **+Y** down (same as NDC→pixel in `screen_to_world_ray`).
- **Do not** assume `(clientX - rect.left) * devicePixelRatio` matches that grid. CSS layout, `getBoundingClientRect`, and `devicePixelRatio` can disagree slightly with the **actual** drawable; that skews NDC. Use **proportional** mapping with **`(clientX - rect.left) / rect.width`** and **`(clientY - rect.top) / rect.height`** (same **`getBoundingClientRect()`** as in `sendResize`), scaled by **`physW` × `physH`**. Do **not** use **`offsetX / clientWidth`**: `clientWidth` / `clientHeight` are integers and can disagree with **fractional** `rect.width` / `rect.height`, which **breaks aspect ratio** (picking looks fine at the **center** of the viewport and **diverges toward the edges**). When converting CSS size to physical pixels for `viewer_resize`, derive **one** dimension with `Math.round` and the other from **aspect** (`height = round(width * (rh/rw))`) so **pw/ph** matches the element aspect.
- **Source of truth for `physW` × `physH`**: [`get_viewport_pixel_size`](src-tauri/src/lib.rs) (includes viewport and **surface** sizes). **Surface** width/height must match the native drawable (wgpu `frame.texture`); [`get_surface_pixel_size`](src-tauri/src/lib.rs) exposes swapchain pixels only. The **`viewport-pixel-size`** event fires after [`viewer_resize`](src-tauri/src/lib.rs) and when the surface size changes after a frame, so the webview can stay aligned without guessing `innerWidth × devicePixelRatio` alone. [`sendResize`](src/App.tsx) maps `viewportX` / `viewportY` **proportionally** (`rect.left / innerWidth`, etc.) into that surface’s pixel space—**not** only `rect.left × dpr`, which can drift when the drawable differs from the CSS estimate.
- **Voxel center convention:** voxel indices `(x,y,z)` are cell centers in world space; only add `+0.5` when a helper specifically needs corner-to-center conversion.
- **Preview/debug parity:** debug projection markers must use the same mode-specific target as preview (`preview_add_cell` for Add; `preview_remove_cell` for Paint/Remove/Select), or they can show false offsets.

## Performance metrics

When investigating slowness, regressions, or GPU/mesh behavior, **include a performance snapshot** in issues or chat:

1. In the app menu: **Debug → Copy performance info**.
2. Paste the full text block into your message (issue, PR description, or agent conversation).

That snapshot includes viewport FPS, physical viewport size, mesh index/vertex buffer stats, voxel count, grid size, file label, platform, a UTC timestamp, and **last voxel edit timings** (apply, prepare, viewer lock wait, brick upload, mesh total, **mesh sub-phases** when relevant—spatial cache delta, cold cache init, greedy remesh, chunk GPU buffers, full chunked rebuild, non-incremental pipeline, preview clear—then route and total). It is produced in [`src-tauri/src/lib.rs`](src-tauri/src/lib.rs) (`performance_report_text`).

If clipboard copy fails, say so and manually note: approximate FPS (shown in the viewport UI), OS, scene size, and what you were doing (e.g. editing vs orbiting).

## Headless server mode (tests)

For integration tests, VD can start with the main window hidden and a localhost HTTP health server. Flags, env vars, stdout readiness line, `/health`, and caveats (GPU still needs a display stack) are documented in [`docs/HEADLESS_SERVER.md`](docs/HEADLESS_SERVER.md). Rust entry: [`src-tauri/src/headless_server.rs`](src-tauri/src/headless_server.rs).

## `.voxelle` file format (v4)

Canonical spec: [`docs/VOXELLE_FORMAT_V4.md`](docs/VOXELLE_FORMAT_V4.md). Implementation: [`encode_payload_v4`](src-tauri/src/voxelle/format.rs) / [`decode_payload`](src-tauri/src/voxelle/format.rs).

- **Outer:** `VX4` magic + gzip + CRC32 of the uncompressed inner.
- **Inner (small):** BSON with `version: 4`, voxel rows, `scene`, `objects`, `activeObjectId`, `fileMeta`, etc.
- **Inner (large, ≥ `V3_WIRE_VOXEL_THRESHOLD` voxels):** `VX3` magic + BSON header + dense body. **`wire_version` 3** = 20-byte records. **`wire_version` 4** = 20- or **24**-byte records (decoder picks by body length); 24-byte = 20-byte prefix + `object_id` u32, header includes `objects` and `activeObjectId`. **`wire_version` 5** (briefly used) = treat like 24-byte v4. There is **no** separate wire “v5” in current writers.

## Hints for code changes

- **3D / mesh / editing**: [`src-tauri/src/render/`](src-tauri/src/render/), [`src-tauri/src/greedy_mesh.rs`](src-tauri/src/greedy_mesh.rs), [`src-tauri/src/voxel_edit.rs`](src-tauri/src/voxel_edit.rs), [`src-tauri/src/lib.rs`](src-tauri/src/lib.rs) (Tauri commands, `ViewerState`).
- **CPU mesh benchmarks** (Criterion): from `src-tauri/`, run `cargo bench --bench greedy_mesh` (see [`benches/greedy_mesh.rs`](src-tauri/benches/greedy_mesh.rs)).
- **Edit perf**: scene bounds use incremental updates when possible ([`greedy_mesh::mesh_bounds_expand_with_voxel`](src-tauri/src/greedy_mesh.rs), strict-interior removes skip full scans); chunk GPU buffers may reuse via `COPY_DST` + `write_buffer` when the new mesh fits.
- **Frontend**: [`src/App.tsx`](src/App.tsx) — viewport events, `invoke` to Rust. **Screen → voxel picking**: follow [Viewport and screen-to-world ray mapping](#viewport-and-screen-to-world-ray-mapping) so pointer math matches the GPU swapchain.
- **In-app updater** (**Check for Updates…** in the app menu): After an update is found, confirmation must not use `@tauri-apps/plugin-dialog` `confirm()` from the webview — it parents the alert to the main window; on macOS that sheet can take **Return** as OK immediately after menu navigation. Use the Rust command [`confirm_app_update_dialog`](src-tauri/src/lib.rs) (app-modal dialog, no webview parent) and `invoke` it from the [`voxelle-check-updates`](src/App.tsx) listener.
- **Tauri IPC allowlist**: Custom commands registered in [`src-tauri/src/lib.rs`](src-tauri/src/lib.rs) must also appear in [`src-tauri/permissions/voxelle.toml`](src-tauri/permissions/voxelle.toml) under `commands.allow`, or `invoke` from the webview will be denied. If a new command “does nothing,” check that list first (and avoid empty `.catch` handlers that hide the error).
- **macOS Edit → Undo/Redo**: [`src-tauri/src/macos_undo.rs`](src-tauri/src/macos_undo.rs) registers each solo voxel edit with `NSUndoManager` so the system menu stays in sync with Rust stacks; collaboration does not use this path. If the webview has keyboard focus, `Cmd+Z` may still be handled by WebKit before AppKit—use the app menu or the in-viewport shortcut path if undo seems ignored.

## Tauri bundled resources

Files listed under [`src-tauri/tauri.conf.json`](src-tauri/tauri.conf.json) `bundle.resources` are copied into the app bundle with **the same relative path shape** under the resource directory (`$RESOURCE`). Leading `../` in those paths is encoded as `_up_` on disk (see [Embedding additional files](https://v2.tauri.app/develop/resources/)).

When resolving a bundled file in Rust with `app.path().resolve(path, BaseDirectory::Resource)`, use the **same `path` string** as in `bundle.resources` (e.g. `../public/Logo.voxelle`), not only the filename at the resource root. Dev may still work if code falls back to a filesystem path next to `CARGO_MANIFEST_DIR`; **release builds** only have the bundled layout, so a mismatch shows up there first.

## Rust → webview: what to emit and how

The webview does not automatically mirror Rust. If the user should see a change, or the UI’s React state should match native/session state, **something has to cross the boundary**—usually a Tauri event the frontend already `listen`s for, or the return value of an `invoke`.

**What deserves an emit:** Any authoritative change in Rust (or async work) that the UI is meant to reflect: session/collab state, errors, load/save and other long-running progress (see **Long waits** above), presence, file/scene metadata the sidebar shows, etc. If you only update structs, mutexes, or files and skip the event path, the UI can stay stale even though Rust is “correct.”

**How to do it:** Follow existing event names and helpers wired in [`src/App.tsx`](src/App.tsx) and the Rust emit sites—add or extend listeners and emits together; avoid one-off duplicate channels for the same concept.

**Collaboration:** Shared session state must reach **every** participant. The host’s webview and each guest’s client each need the update; guests get it over the WebSocket, not only via `app.emit` on the host process. Use the established helpers in [`src-tauri/src/collab.rs`](src-tauri/src/collab.rs) (e.g. [`broadcast_roster_to_guests`](src-tauri/src/collab.rs) for roster-shaped updates) that both emit to the local app and forward on `host_broadcast` where that pattern applies—emitting only to the host while mutating shared state is a common way to strand remote clients. See emits in [`src-tauri/src/collab.rs`](src-tauri/src/collab.rs) and [`src-tauri/src/lib.rs`](src-tauri/src/lib.rs).

## Format

Extend this file with project conventions as they solidify (e.g. build commands, test expectations). Keep entries short and scannable.
