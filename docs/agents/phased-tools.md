# Phased viewport tools (multi-step gestures)

Some strokes are **not** "one pointer-down → one commit on up." They match the web Voxelle pattern: **phase 1** (e.g. drag a footprint in a face plane) → **phase 2** (adjust parameters such as depth) → **explicit commit** (Done / Enter), with **Escape** to cancel.

## `useStrokePhase<TData>` hook ([`src/useStrokePhase.ts`](src/useStrokePhase.ts))

All phased tools use the generic `useStrokePhase` hook. It manages:

- **Ordered phases** — define a `phases` array; `advance()` / `retreat()` move through them, `goTo()` jumps to any phase, `enter()` starts the machine.
- **State + ref pair** — `snapshot` (reactive, triggers re-renders) and `ref` (stable for closures / pointer handlers / `mergedStrokeAux`).
- **Terminal actions** — `cancel()` and `commit()` call user-supplied callbacks and reset state.
- **Keyboard** — built-in Escape → cancel, Enter → commit (opt-out via `keyboard: false`).

**Usage pattern:**

```ts
const myPhase = useStrokePhase<MyData>({
  phases: ["footprint", "depth"],
  onCancel: () => { invoke("voxel_stroke_preview_reset"); },
  onCommit: (snap) => { commitMyTool(snap.data); },
});

// pointer-up after drag:
myPhase.enter("footprint", { lineStart, endNorm });

// UI or effect advances to next phase:
myPhase.advance({ depth: 5 });

// query state:
myPhase.active        // boolean
myPhase.ref.current   // latest snapshot (for closures)
myPhase.snapshot      // reactive snapshot (for JSX / useEffect deps)
```

**In-tree examples:** `cuboidPhase`, `cylinderPhase` (both `useStrokePhase<DepthPhaseData>`), and `extrudePhase` (`useStrokePhase<Record<string, never>>`) in [`src/App.tsx`](src/App.tsx).

## Conventions for new phased tools

| Layer         | Convention                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Rust**      | Encode "phase 2+" parameters on the wire via [`stroke_modes::StrokeAux`](src-tauri/src/stroke_modes.rs) (e.g. optional `*_depth`, flags). Preview and edit paths should call the same geometry helpers so ghost and commit match. If phase 1 ends with a **stroke preview union** that must **not** become voxels on `voxel_stroke_end`, special-case in [`voxel_stroke_end`](src-tauri/src/lib.rs) (skip commit when only the partial preview is present) **or** clear preview without applying—same idea as cuboid + plane-only preview. |
| **React**     | Use `useStrokePhase` — it handles the state + ref mirror, keyboard shortcuts, and lifecycle. Read the phase data via `.ref.current` in `mergedStrokeAux` and pointer handlers. **Cancel** (`.cancel()`) resets state and invokes your `onCancel` callback (typically clears GPU preview). **Commit** (`.commit()`) invokes `onCommit` then resets.                                                                                                                                                                                         |
| **Gestures**  | On **new** `pointerdown` for a voxel stroke, cancel an active phase so the user can start over. On **interaction mode** or **stroke mode** change away from the tool, cancel. **Do not** commit partial geometry on pointer-up if the product should wait for Done—rely on Rust skip + frontend phase state.                                                                                                                                                                                                                               |
| **Overlays**  | Use a small floating control (viewport-anchored) with `onPointerDown` / `onPointerDownCapture` **stopPropagation** so clicks don't hit the viewport. Document shortcuts (Enter = commit, Escape = cancel) in code or help.                                                                                                                                                                                                                                                                                                                 |
| **Selection** | If the tool applies to both draw and selection, mirror the same aux + line-start semantics on `selection_stroke_at_screen`; avoid incremental drag merges during phase 1 if they would corrupt selection (see cuboid: skip selection drag while `drawStrokeMode === "cuboid"`).                                                                                                                                                                                                                                                            |

**Reference implementation:** search [`src/App.tsx`](src/App.tsx) for `cuboidPhase`, `commitCuboidSolidAtScreen`, and `mergedStrokeAux` cuboid fields. Rust: `DrawStrokeMode::Cuboid` + `StrokeAux::cuboid_depth` in [`stroke_modes.rs`](src-tauri/src/stroke_modes.rs), and the `voxel_stroke_end` cuboid plane-preview guard in [`lib.rs`](src-tauri/src/lib.rs).
