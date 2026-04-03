Add a new user preference to Voxelle Desktop. The preference to add is: $ARGUMENTS

Preferences are persisted to `localStorage` under the key `voxelle-preferences`. They live in three files: `src/preferences.ts` (data model), `src/PreferencesModal.tsx` (UI), and optionally `src/App.tsx` (startup sync to Rust).

## Step 1 — Add to the type and defaults in `src/preferences.ts`

**Type** — add a field to `VoxelleDesktopPreferences`:
```typescript
/** One-line description of what this controls. */
myNewPref: boolean; // or number, string, etc.
```

**Default** — add to the `DEFAULTS` constant:
```typescript
myNewPref: true, // the value new users get
```

**Load** — add a line inside the `return { ... }` block of `loadPreferences()`, following the pattern of the field's type:
```typescript
// boolean:
myNewPref: typeof o.myNewPref === "boolean" ? o.myNewPref : DEFAULTS.myNewPref,
// number with clamp:
myNewPref: typeof o.myNewPref === "number" && Number.isFinite(o.myNewPref)
  ? clampInt(o.myNewPref, MIN, MAX)
  : DEFAULTS.myNewPref,
// string:
myNewPref: typeof o.myNewPref === "string" ? o.myNewPref.trim().slice(0, MAX_LEN) : DEFAULTS.myNewPref,
```

**Save** — add to `savePreferences()` before `localStorage.setItem`:
```typescript
merged.myNewPref = prefs.myNewPref;
```

> **Common mistake:** `loadPreferences` and `savePreferences` are separate code paths — a field added to `load` but forgotten in `save` will silently reset to its default on every restart. Always add both in the same commit and verify the checklist below.

## Step 2 — Add UI to `src/PreferencesModal.tsx`

**Handler** — add near the other `on*` handlers at the top of the component:
```typescript
const onMyNewPref = (checked: boolean) => {
  const next = { ...prefs, myNewPref: checked };
  setPrefs(next);
  savePreferences(next);
  // If Rust-backed, also: void invoke("set_my_new_pref", { enabled: checked }).catch(() => {});
};
```

**UI element** — add a checkbox in the appropriate section (`prefs-general`, `prefs-graphics`, etc.):
```tsx
<label className="prefs-checkbox-label">
  <input
    type="checkbox"
    checked={prefs.myNewPref}
    onChange={(e) => onMyNewPref(e.target.checked)}
  />
  Human-readable label
</label>
```

For a `<select>` or `<input type="number">`, follow the existing `toneMapping` or `autosaveIntervalSecs` patterns respectively.

## Step 3 — Rust-backed preferences only

If the preference controls a Rust-side behaviour (renderer setting, etc.), three more things are needed:

**a) Tauri command** — add `set_my_new_pref` following `/tauri-command`. The command typically does:
```rust
fn set_my_new_pref(state: State<'_, Arc<ViewerState>>, enabled: bool) -> Result<(), String> {
    if let Some(viewer) = state.viewer.lock().as_mut() {
        viewer.my_flag = enabled;
    }
    Ok(())
}
```

**b) Startup sync in `src/App.tsx`** — extend the existing `useEffect(() => { ... }, [])` that syncs other GPU preferences (search for `set_emission_lighting`), or add a new one:
```typescript
useEffect(() => {
  const p = loadPreferences();
  void invoke("set_my_new_pref", { enabled: p.myNewPref }).catch(() => {});
}, []);
```

**c) Register the command** — follow the `/tauri-command` skill: add to `invoke_handler!` in `lib.rs` and to `src-tauri/permissions/voxelle.toml`.

## Checklist before finishing

- [ ] Field added to `VoxelleDesktopPreferences` type in `preferences.ts`
- [ ] Default added to `DEFAULTS` in `preferences.ts`
- [ ] Load case added in `loadPreferences()` in `preferences.ts`
- [ ] Save line added in `savePreferences()` in `preferences.ts`
- [ ] Handler `on*` added in `PreferencesModal.tsx`
- [ ] UI control added in the right section of `PreferencesModal.tsx`
- [ ] (Rust-backed) Tauri command written, registered in handler + permissions
- [ ] (Rust-backed) Startup `useEffect` sync added in `App.tsx`