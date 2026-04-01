# Tauri bundled resources

Files listed under [`src-tauri/tauri.conf.json`](../../src-tauri/tauri.conf.json) `bundle.resources` are copied into the app bundle with **the same relative path shape** under the resource directory (`$RESOURCE`). Leading `../` in those paths is encoded as `_up_` on disk (see [Embedding additional files](https://v2.tauri.app/develop/resources/)).

When resolving a bundled file in Rust with `app.path().resolve(path, BaseDirectory::Resource)`, use the **same `path` string** as in `bundle.resources` (e.g. `../public/Logo.voxelle`), not only the filename at the resource root. Dev may still work if code falls back to a filesystem path next to `CARGO_MANIFEST_DIR`; **release builds** only have the bundled layout, so a mismatch shows up there first.
