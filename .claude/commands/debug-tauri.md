Diagnose a Tauri IPC command that isn't working. Command or symptom to investigate: $ARGUMENTS

Silent failures are the norm here — App.tsx catch blocks swallow most errors, and the browser console is often the only signal. Work through this checklist in order; most issues are caught by step 3.

## Step 1 — Confirm the command is registered in `invoke_handler!`

Search `src-tauri/src/lib.rs` for `tauri::generate_handler![` and verify the function name appears in the list. The name must exactly match the Rust function name (snake_case).

```
grep -n "my_command" src-tauri/src/lib.rs
```

## Step 2 — Confirm the command is in `voxelle.toml`

Search `src-tauri/permissions/voxelle.toml` for the command name under the `allow` list. This is the most common silent-failure cause in this project — a command registered in the handler but missing here is blocked at the IPC layer with "Command not found", which catch blocks discard.

```
grep "my_command" src-tauri/permissions/voxelle.toml
```

If it's missing, add `"my_command",` to the allow list and restart the dev server.

## Step 3 — Temporarily surface the error in the frontend

Find the `invoke(...)` call and replace `.catch(() => {})` with `.catch((e) => console.error("[debug] my_command threw:", e))`. Run the app, trigger the command, and check the browser devtools console (View → Toggle Developer Tools in the Tauri window, or Cmd+Option+I).

Common error strings to look for:
- `"my_command not allowed. Command not found"` → missing from `voxelle.toml`
- `"Command not found"` (no "not allowed") → missing from `invoke_handler!`
- A Rust panic message → runtime error inside the command; check `RUST_BACKTRACE=1`
- `"invalid args"` or serde errors → TypeScript param names or types don't match Rust param names or types

Remove the debug logging once resolved.

## Step 4 — Verify parameter names match exactly

Tauri serialises frontend arguments by name. A TypeScript call like `invoke("cmd", { myParam: 1 })` must match a Rust parameter named `my_param: i32` — camelCase on the JS side maps to snake_case on the Rust side via serde. Double-check every parameter name.

## Step 5 — Verify the command is reachable at runtime

If the command guards on `state.viewer.lock().as_ref()` or similar and returns early silently, the issue may be timing — the viewer might not be initialised yet. Add a temporary `eprintln!` inside the Rust function to confirm it's being called at all.

## Step 6 — Check for a second registration path

This codebase has a secondary `invoke_handler!` for a mock/test path (search for the second `tauri::generate_handler!`). If the app is built with a feature flag that selects the other path, the command needs to be in both handler lists. Check both.

## Resolution checklist

- [ ] Function name in `invoke_handler!(tauri::generate_handler![...])`
- [ ] Command name in `src-tauri/permissions/voxelle.toml` allow list
- [ ] Frontend `.catch` temporarily logs the error (remove after diagnosis)
- [ ] Parameter names are snake_case on Rust side, camelCase on TS side, but otherwise identical
- [ ] Dev server restarted after editing `.toml` (changes don't hot-reload)
