//! Local HTTP control plane for integration tests when the app is started with
//! `--headless-server` or `VOXELLE_HEADLESS_SERVER=1`.
//!
//! After the main window is hidden and the GPU viewer has initialized, the process prints a single
//! line to stdout: `VOXELLE_HEADLESS_READY\t<port>`. Tests can poll `GET http://127.0.0.1:<port>/health`.

use std::io::Write;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Returns `Some(port)` when headless server mode is enabled. Port `0` means bind to an ephemeral port.
pub fn parse_config() -> Option<u16> {
    let env_on = std::env::var("VOXELLE_HEADLESS_SERVER")
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false);
    let env_port: Option<u16> = std::env::var("VOXELLE_HEADLESS_SERVER_PORT")
        .ok()
        .and_then(|s| s.parse().ok());

    let mut enabled = env_on;
    let mut port = env_port.unwrap_or(0);

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--headless-server" {
            enabled = true;
        } else if let Some(p) = a.strip_prefix("--headless-server-port=") {
            if let Ok(n) = p.parse::<u16>() {
                port = n;
            }
        } else if a == "--headless-server-port" {
            if let Some(p) = args.next() {
                if let Ok(n) = p.parse::<u16>() {
                    port = n;
                }
            }
        }
    }

    enabled.then_some(port)
}

async fn handle_connection(mut stream: TcpStream) {
    let mut buf = [0u8; 2048];
    let n = match stream.read(&mut buf).await {
        Ok(n) if n > 0 => n,
        _ => return,
    };
    let req = String::from_utf8_lossy(&buf[..n]);
    let first = req.lines().next().unwrap_or("");
    let mut parts = first.split_whitespace();
    let _method = parts.next();
    let path = parts.next().unwrap_or("");
    let is_health = path == "/health" || path.starts_with("/health?");

    let (status, body) = if is_health {
        ("200 OK", r#"{"ok":true,"mode":"headless-server"}"#)
    } else {
        ("404 Not Found", r#"{"error":"not_found"}"#)
    };

    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes()).await;
}

pub async fn accept_loop(listener: TcpListener) {
    loop {
        if let Ok((stream, _)) = listener.accept().await {
            tokio::spawn(handle_connection(stream));
        }
    }
}

/// Binds the listener, prints `VOXELLE_HEADLESS_READY\t<port>` to stdout, and spawns the accept loop.
pub fn start(listener: TcpListener) -> Result<(), String> {
    let bound = listener
        .local_addr()
        .map_err(|e| format!("headless server local_addr: {e}"))?
        .port();
    println!("VOXELLE_HEADLESS_READY\t{bound}");
    std::io::stdout()
        .flush()
        .map_err(|e| format!("headless server stdout flush: {e}"))?;
    tauri::async_runtime::spawn(accept_loop(listener));
    Ok(())
}
