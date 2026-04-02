Add a new selection operation to Voxelle Desktop. Operation to add: $ARGUMENTS

Selection operations (translate, rotate, scale, mirror) all follow the same structure: an `_inner` function that does the work, a thin `#[tauri::command]` wrapper, undo registration, and a frontend binding. Use `selection_translate_inner` as the canonical reference.

## Step 1 — Write the inner function in `src-tauri/src/lib.rs`

Place it near the other `selection_*_inner` functions (search for `fn selection_translate_inner`).

```rust
fn selection_my_op_inner(
    state: &Arc<ViewerState>,
    app: &AppHandle,
    // operation-specific params (axis: u8, amount: i32, etc.)
) -> Result<bool, String> {
    let t_total = Instant::now();

    // 1. Snapshot selection before the edit (needed for undo)
    let before_sel = state.selection_cells.lock().clone();
    if before_sel.is_empty() {
        return Ok(false);
    }

    // 2. Lock file + voxel map together and call the voxel_edit function
    let deltas = {
        let mut fg = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
        let Some(file) = fg.as_mut() else { return Err("no model loaded".into()); };
        let Some(vmap) = vm.as_mut() else { return Err("voxel index not ready".into()); };
        voxel_edit::my_op(file, vmap, &before_sel, /* params */)
    };

    // 3. Update selection_cells to reflect the new positions (if voxels moved)
    // *state.selection_cells.lock() = new_sel;

    // 4. Flush GPU geometry
    if !deltas.is_empty() {
        finish_voxel_edit_gpu_deltas(
            state,
            &deltas,
            0.0,
            t_total,
            app,
            VoxelGpuRefreshReason::SoloEdit,
        )?;
    }

    // 5. Push undo entry
    push_selection_transform_undo(state, app, before_sel, deltas);

    // 6. Notify frontend that selection changed
    emit_selection_updated(app, state);
    Ok(true)
}
```

## Step 2 — Write the Tauri command wrapper

Directly after `_inner`, add the thin public wrapper:

```rust
#[tauri::command]
fn selection_my_op(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    // same params as _inner
) -> Result<bool, String> {
    selection_my_op_inner(state.inner(), &app, /* params */)
}
```

## Step 3 — Register the command

Follow `/tauri-command`:
- Add `selection_my_op` to `invoke_handler!(tauri::generate_handler![...])` in `lib.rs`
- Add `"selection_my_op"` to `src-tauri/permissions/voxelle.toml`

## Step 4 — Frontend binding in `src/App.tsx`

Add an `invoke` call where needed. Selection ops typically appear in toolbar buttons, keyboard shortcuts, or context menus:

```typescript
void invoke<boolean>("selection_my_op", { /* params */ })
  .then((changed) => { if (changed) { /* update UI */ } })
  .catch(() => {});
```

If the operation needs to be callable from the gizmo (like translate is via `gizmo_pointer_up`), it will already be wired — check `gizmo_pointer_up` in `lib.rs`.

## Step 5 — Check macOS undo menu integration (optional)

If this op should appear in the native Edit → Undo menu, look at how `push_selection_transform_undo` and `macos_undo::register_solo_edit_completed` are called in the existing ops and follow the same pattern.

## Checklist before finishing

- [ ] `selection_my_op_inner` written with correct lock order (current_file + voxel_map together)
- [ ] `finish_voxel_edit_gpu_deltas` called if deltas are non-empty
- [ ] `push_selection_transform_undo` called with the before-snapshot
- [ ] `emit_selection_updated` called at the end
- [ ] `#[tauri::command]` wrapper added
- [ ] Registered in `invoke_handler!` and `voxelle.toml`
- [ ] Frontend call added
- [ ] Build passes: `cargo build --manifest-path src-tauri/Cargo.toml`
