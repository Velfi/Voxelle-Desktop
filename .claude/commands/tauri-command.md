Add a new Tauri IPC command to Voxelle Desktop. The command to add is: $ARGUMENTS

This requires three steps in lockstep. Missing any one of them will cause a silent runtime failure ("Command not found" at the IPC layer, which App.tsx swallows).

## Step 1 — Write the Rust function in `src-tauri/src/lib.rs`

Add a `#[tauri::command]` function. Use the snake_case command name from $ARGUMENTS. Follow the existing patterns:

- Read-only commands: `fn my_command(state: State<'_, Arc<ViewerState>>) -> Result<ReturnType, String>`
- Mutating commands that need to emit events: also take `app: AppHandle`
- Fire-and-forget (no return value): return `Result<(), String>`

Place it near related commands. Do not add it at the very end of the file.

## Step 2 — Register in `invoke_handler!`

Find `tauri::generate_handler![` in `src-tauri/src/lib.rs` (there is exactly one, near the bottom of `run()`). Add the function name to the list. The name must exactly match the function name.

## Step 3 — Add to permissions (CRITICAL — do not skip)

Open `src-tauri/permissions/voxelle.toml` and add the snake_case command name to the `allow` list. This is a Tauri v2 capability allow-list enforced at the IPC layer. A command registered in the handler but absent here will be silently blocked with "Command not found" — the error is typically caught and discarded by frontend catch blocks, making it very hard to notice.

## Step 4 — Add the TypeScript caller (if needed)

In the relevant `.tsx` file, call:
```typescript
invoke<ReturnType>("command_name", { param1, param2 })
```

If the command is called from `SelectionGizmo.tsx`, it's already wired through the `SelectionGizmoRef` interface — add a method there and follow the existing `startDragIfHit` / `updateHover` pattern.

## Checklist before finishing

- [ ] `#[tauri::command]` function exists in `lib.rs`
- [ ] Name appears in `invoke_handler!(tauri::generate_handler![...])` in `lib.rs`
- [ ] Name appears in `src-tauri/permissions/voxelle.toml` allow list
- [ ] Frontend caller added (if applicable)
- [ ] Build passes: `cargo build --manifest-path src-tauri/Cargo.toml`