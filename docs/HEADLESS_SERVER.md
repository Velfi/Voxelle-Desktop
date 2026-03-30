# Headless server mode (integration testing)

Voxelle Desktop can start with the main window **hidden** and a small **HTTP server on localhost** so a test harness can wait until the app (including the GPU viewer) is ready before driving it.

Implementation: [`src-tauri/src/headless_server.rs`](../src-tauri/src/headless_server.rs), wired from [`src-tauri/src/lib.rs`](../src-tauri/src/lib.rs) `setup` (desktop builds only).

## Enabling

- **CLI:** pass `--headless-server`.
- **Environment:** set `VOXELLE_HEADLESS_SERVER` to `1`, `true`, or `yes` (case-insensitive).

CLI and env can be combined; the flag forces the mode on. Port can be set independently.

## Port

- **Default:** bind to `127.0.0.1:0` (ephemeral port chosen by the OS).
- **Environment:** `VOXELLE_HEADLESS_SERVER_PORT=<u16>` (use `0` for ephemeral).
- **CLI:** `--headless-server-port=<n>` or `--headless-server-port <n>`.

## Readiness protocol

1. The main window is **hidden** (`WebviewWindow::hide`) before `WgpuViewer` is created.
2. After `WgpuViewer::new` succeeds, the process binds the TCP listener, then prints **one line** to **stdout** (flushed):

   ```text
   VOXELLE_HEADLESS_READY	<port>
   ```

   The separator is a tab character (`\t`). `<port>` is the actual bound port (useful when the requested port was `0`).

3. **`GET http://127.0.0.1:<port>/health`** returns `200` with body:

   ```json
   {"ok":true,"mode":"headless-server"}
   ```

4. Any other path returns `404` with a small JSON `not_found` body.

## Example: dev build

Arguments after `--` are forwarded to the app binary:

```bash
npm run tauri dev -- --headless-server
npm run tauri dev -- --headless-server --headless-server-port=9342
```

With env only:

```bash
VOXELLE_HEADLESS_SERVER=1 npm run tauri dev
```

## Suggested test flow

1. Spawn the app in headless server mode.
2. Read **stdout** until a line starts with `VOXELLE_HEADLESS_READY`, then parse the port.
3. Poll **`GET /health`** until `200` (optional if you trust the ready line; health confirms the server task is accepting).
4. Drive the product as you normally would (e.g. `invoke` from the loaded webview, or Tauri WebDriver), keeping in mind the window is not visible.

## Limitations

- This is **not** a GPU-offscreen or displayless renderer: **wgpu** still uses the webview surface. Headless CI often needs a virtual display (e.g. **Xvfb** on Linux).
- The HTTP server is a **readiness / liveness** endpoint only; it does **not** expose Tauri `invoke` commands over HTTP.
