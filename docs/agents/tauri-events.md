# Rust → webview: what to emit and how

The webview does not automatically mirror Rust. If the user should see a change, or the UI's React state should match native/session state, **something has to cross the boundary**—usually a Tauri event the frontend already `listen`s for, or the return value of an `invoke`.

**What deserves an emit:** Any authoritative change in Rust (or async work) that the UI is meant to reflect: session/collab state, errors, load/save and other long-running progress (see **Long waits** in [ui-layout.md](ui-layout.md)), presence, file/scene metadata the sidebar shows, etc. If you only update structs, mutexes, or files and skip the event path, the UI can stay stale even though Rust is "correct."

**How to do it:** Follow existing event names and helpers wired in [`src/App.tsx`](src/App.tsx) and the Rust emit sites—add or extend listeners and emits together; avoid one-off duplicate channels for the same concept.

**Collaboration:** Shared session state must reach **every** participant. The host's webview and each guest's client each need the update; guests get it over the WebSocket, not only via `app.emit` on the host process. Use the established helpers in [`src-tauri/src/collab.rs`](src-tauri/src/collab.rs) (e.g. [`broadcast_roster_to_guests`](src-tauri/src/collab.rs) for roster-shaped updates) that both emit to the local app and forward on `host_broadcast` where that pattern applies—emitting only to the host while mutating shared state is a common way to strand remote clients. See emits in [`src-tauri/src/collab.rs`](src-tauri/src/collab.rs) and [`src-tauri/src/lib.rs`](src-tauri/src/lib.rs).
