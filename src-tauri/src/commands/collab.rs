use crate::*;

// ── Host / Join / Leave ─────────────────────────────────────────────────────

#[tauri::command]
pub(crate) fn collab_host_start(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    port: u16,
    display_name: String,
    color_rgb: u32,
    enable_upnp: bool,
) -> Result<collab::CollabHostStartResponse, String> {
    let vs = Arc::clone(&*state);
    collab::start_host(
        app,
        vs.clone(),
        Arc::clone(&vs.collab),
        port,
        display_name,
        color_rgb,
        enable_upnp,
    )
}

#[tauri::command]
pub(crate) fn collab_join(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    url: String,
    display_name: String,
    color_rgb: u32,
) -> Result<(), String> {
    let vs = Arc::clone(&*state);
    let cm = Arc::clone(&vs.collab);
    let cancel = tokio_util::sync::CancellationToken::new();
    cm.lock().join_cancel = Some(cancel.clone());
    tauri::async_runtime::spawn(async move {
        let result = collab::client_connect_blocking(
            &url,
            app.clone(),
            vs,
            cm.clone(),
            display_name,
            color_rgb,
            cancel,
        )
        .await;
        cm.lock().join_cancel = None;
        if let Err(e) = result {
            let _ = app.emit("collab-error", e);
        }
    });
    Ok(())
}

#[tauri::command]
pub(crate) fn collab_cancel_join(state: State<'_, Arc<ViewerState>>) {
    let token = state.collab.lock().join_cancel.take();
    if let Some(t) = token {
        t.cancel();
    }
}

#[tauri::command]
pub(crate) fn collab_leave(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
) -> Result<(), String> {
    let (was_host, was_client, upnp_port) = {
        let mut c = state.collab.lock();
        let wh = c.is_host();
        let wc = c.is_client();
        let upnp = c.upnp_external_tcp_port;
        if wc {
            if let Some(tx) = &c.client_tx {
                let msg = serde_json::to_string(&collab::ClientToHost::Leave).unwrap();
                let _ = tx.try_send(collab::ClientOutgoing::Text(msg));
            }
        }
        c.leave();
        (wh, wc, upnp)
    };
    if let Some(p) = upnp_port {
        collab::schedule_remove_upnp_mapping(p);
    }
    *state.ping_flash.lock() = None;
    if was_host {
        let _ = app.emit(
            "collab-ended",
            "You stopped hosting. The session is no longer shared.",
        );
    } else if was_client {
        let _ = app.emit("collab-ended", "You left the collaboration session.");
    }
    Ok(())
}

// ── Peer management ─────────────────────────────────────────────────────────

#[tauri::command]
pub(crate) fn collab_local_peer_id(state: State<'_, Arc<ViewerState>>) -> u32 {
    state.collab.lock().local_peer_id
}

#[tauri::command]
pub(crate) fn collab_kick_peer(
    state: State<'_, Arc<ViewerState>>,
    target_peer: u32,
) -> Result<(), String> {
    collab::host_kick_peer(&state.collab, target_peer)
}

#[tauri::command]
pub(crate) fn collab_update_profile(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    display_name: String,
    color_rgb: u32,
) -> Result<(), String> {
    let mut c = state.collab.lock();
    if !c.is_active() {
        return Ok(());
    }
    let pid = c.local_peer_id;
    if c.is_client() {
        let msg = serde_json::to_string(&collab::ClientToHost::UpdateProfile {
            display_name,
            color_rgb,
        })
        .unwrap();
        if let Some(tx) = &c.client_tx {
            let _ = tx.try_send(collab::ClientOutgoing::Text(msg));
        }
        return Ok(());
    }
    if c.is_host() {
        for r in &mut c.roster {
            if r.peer_id == pid {
                r.display_name = display_name.clone();
                r.color_rgb = color_rgb;
            }
        }
        let roster = c.roster.clone();
        drop(c);
        collab::broadcast_roster_to_guests(&app, &state.collab, &roster);
    }
    Ok(())
}

#[tauri::command]
pub(crate) fn collab_set_can_edit(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    target_peer: u32,
    can_edit: bool,
) -> Result<(), String> {
    let msg = serde_json::to_string(&collab::ClientToHost::SetCanEdit {
        target_peer,
        can_edit,
    })
    .unwrap();
    let mut c = state.collab.lock();
    if c.is_client() {
        if let Some(tx) = &c.client_tx {
            let _ = tx.try_send(collab::ClientOutgoing::Text(msg));
        }
    } else if c.is_host() {
        for r in &mut c.roster {
            if r.peer_id == target_peer {
                r.can_edit = can_edit;
            }
        }
        let roster = c.roster.clone();
        drop(c);
        collab::broadcast_roster_to_guests(&app, &state.collab, &roster);
    }
    Ok(())
}

// ── Camera sharing ──────────────────────────────────────────────────────────

#[tauri::command]
pub(crate) fn collab_push_camera(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
) -> Result<(), String> {
    let vs = Arc::clone(&*state);
    let mut c = vs.collab.lock();
    if !c.is_active() {
        return Ok(());
    }
    let cam = state.camera.lock();
    let presence = collab::CameraPresence {
        target: [cam.target.x, cam.target.y, cam.target.z],
        radius: cam.smooth_spherical.radius,
        theta: cam.smooth_spherical.theta,
        phi: cam.smooth_spherical.phi,
        perspective: cam.perspective,
        fov_y: cam.fov_y,
        ortho_half_height: cam.ortho_half_height,
    };
    let pid = c.local_peer_id;
    c.presence.insert(pid, presence);
    if c.is_client() {
        let msg = serde_json::to_string(&collab::ClientToHost::Camera { presence }).unwrap();
        if let Some(tx) = &c.client_tx {
            let _ = tx.try_send(collab::ClientOutgoing::Text(msg));
        }
    } else if c.is_host() {
        // Guests only receive camera updates via WebSocket; without this broadcast the host's
        // peer id never appears in guests' `presence`, so "snap to host" always failed.
        let cam_ev = collab::HostToClient::Camera {
            peer_id: pid,
            presence,
        };
        let json = serde_json::to_string(&cam_ev).unwrap();
        let _ = app.emit("collab-camera", &json);
        if let Some(tx) = &c.host_broadcast {
            let _ = tx.send(tokio_tungstenite::tungstenite::Message::Text(json));
        }
    }
    Ok(())
}

#[tauri::command]
pub(crate) fn collab_snap_camera(
    state: State<'_, Arc<ViewerState>>,
    peer_id: u32,
) -> Result<(), String> {
    let pr = {
        let c = state.collab.lock();
        c.presence.get(&peer_id).copied()
    };
    let Some(pr) = pr else {
        return Err("no camera data for peer".into());
    };
    let mut cam = state.camera.lock();
    cam.target = glam::Vec3::new(pr.target[0], pr.target[1], pr.target[2]);
    cam.spherical.radius = pr.radius;
    cam.spherical.theta = pr.theta;
    cam.spherical.phi = pr.phi;
    // Leave smooth_target / smooth_spherical at their current values
    // so update_damping() interpolates smoothly to the new position.
    cam.perspective = pr.perspective;
    cam.fov_y = pr.fov_y;
    cam.ortho_half_height = pr.ortho_half_height;
    Ok(())
}

// ── Chat / Ping ─────────────────────────────────────────────────────────────

#[tauri::command]
pub(crate) fn collab_send_chat(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    text: String,
) -> Result<(), String> {
    let c = state.collab.lock();
    if c.is_host() {
        let host_name = c
            .roster
            .iter()
            .find(|r| r.peer_id == collab::HOST_PEER_ID)
            .map(|r| r.display_name.clone())
            .unwrap_or_else(|| "Host".into());
        let ev = collab::HostToClient::Chat {
            peer_id: collab::HOST_PEER_ID,
            display_name: host_name,
            text,
            ts_ms: chrono::Utc::now().timestamp_millis(),
        };
        let json = serde_json::to_string(&ev).unwrap();
        let _ = app.emit("collab-chat", &json);
        if let Some(tx) = &c.host_broadcast {
            let _ = tx.send(tokio_tungstenite::tungstenite::Message::Text(json));
        }
        return Ok(());
    }
    if let Some(tx) = &c.client_tx {
        let msg = serde_json::to_string(&collab::ClientToHost::Chat { text }).unwrap();
        let _ = tx.try_send(collab::ClientOutgoing::Text(msg));
    }
    Ok(())
}

#[tauri::command]
pub(crate) fn collab_send_ping(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    x: i32,
    y: i32,
    z: i32,
    #[allow(unused_variables)] emoji: Option<String>,
) -> Result<(), String> {
    let emoji = emoji.unwrap_or_default();
    let mut c = state.collab.lock();
    if c.is_host() {
        let host_color = c
            .roster
            .iter()
            .find(|r| r.peer_id == collab::HOST_PEER_ID)
            .map(|r| r.color_rgb)
            .unwrap_or(0xffff44);
        let host_name = c
            .roster
            .iter()
            .find(|r| r.peer_id == collab::HOST_PEER_ID)
            .map(|r| r.display_name.clone())
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "Host".to_string());
        drop(c);
        collab::record_ping_flash_colored(
            Arc::as_ref(&*state),
            x,
            y,
            z,
            host_color,
            host_name.clone(),
            emoji.clone(),
        );
        c = state.collab.lock();
        let ping = collab::HostToClient::Ping {
            peer_id: collab::HOST_PEER_ID,
            x,
            y,
            z,
            display_name: host_name,
            emoji,
        };
        let json = serde_json::to_string(&ping).unwrap();
        let _ = app.emit("collab-ping", &json);
        if let Some(tx) = &c.host_broadcast {
            let _ = tx.send(tokio_tungstenite::tungstenite::Message::Text(json));
        }
        return Ok(());
    }
    if let Some(tx) = &c.client_tx {
        let msg = serde_json::to_string(&collab::ClientToHost::Ping { x, y, z, emoji }).unwrap();
        let _ = tx.try_send(collab::ClientOutgoing::Text(msg));
    }
    Ok(())
}
