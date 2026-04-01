# Main UI

| Name                          | Role                                                                                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tools sidebar** (left)      | Controls and inputs for making art (modes, brushes, palette-style tools as they land). Implemented in [`src/App.tsx`](src/App.tsx) as the left `aside`. |
| **Inspector sidebar** (right) | Project metadata and management: hierarchy / outliner-style tools, properties, and related panels. Implemented as the right `aside`.                    |
| **Status bar** (bottom)       | High-level feedback on what the app is doing (current file, load/collab state, optional FPS). The `footer.app-status-bar` row.                          |
| **Tool Panel** (bottom left)  | Tool options and selection. Implemented in [`src/App.tsx`](src/App.tsx) as the bottom left `aside`.                                                     |

**Long waits:** Whenever the user might wait more than a few seconds (load, save, mesh rebuild, visibility refresh, etc.), the status bar should explain **why**—not a spinner with no copy. Emit meaningful phases (and optional fraction) via the usual events (e.g. `voxelle-work-progress`, `voxelle-load-progress` from [`src-tauri/src/lib.rs`](src-tauri/src/lib.rs)); don't leave the UI silent during noticeable work.
