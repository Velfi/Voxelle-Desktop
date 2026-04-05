//! WebSocket setup/teardown, UPnP port mapping, peer connect/disconnect lifecycle.

use crate::voxelle::{empty_collab_placeholder, encode_payload_v4};
use crate::ViewerState;
use futures_util::{SinkExt, StreamExt};
use igd_next::aio::tokio::search_gateway;
use igd_next::{AddPortError, PortMappingProtocol, SearchOptions};
use parking_lot::Mutex;
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tauri::{AppHandle, Emitter, Runtime};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;
use tokio::sync::{broadcast, mpsc};
use tokio_tungstenite::{accept_async, connect_async, tungstenite::Message, WebSocketStream};
use tokio_util::sync::CancellationToken;

use super::{
    edits::{
        broadcast_edit_binary, broadcast_roster_to_guests, check_can_edit,
        decode_client_edit_binary, decode_host_edit_binary, host_remove_peer_from_session,
        touch_guest_activity,
    },
    presence::record_ping_flash,
    ClientOutgoing, ClientToHost, CollabHostStartResponse, CollabInboxItem, CollabNatResult,
    CollabPeerLeftKind, CollabRole, CollabRuntime, HostToClient, RosterEntry, BROADCAST_CAPACITY,
    CAMERA_BROADCAST_MIN_INTERVAL, CLIENT_HEARTBEAT_INTERVAL, CLIENT_HOST_SILENCE_TIMEOUT,
    CLIENT_JOIN_TIMEOUT, CLIENT_LATENCY_PROBE_INTERVAL, GUEST_ACTIVITY_CHECK_INTERVAL,
    GUEST_ACTIVITY_TIMEOUT, GUEST_TIMEOUT_KICK_REASON, HOST_KEEPALIVE_INTERVAL, HOST_PEER_ID,
    MAX_AVATAR_FILE_BYTES, SNAPSHOT_CHUNK_SIZE, SNAPSHOT_CHUNK_THRESHOLD, UPNP_LEASE_SECS,
    UPNP_RENEW_INTERVAL,
};

type WsTcp = WebSocketStream<TcpStream>;

/// Best-effort: delete a TCP mapping we added on the default gateway.
pub fn schedule_remove_upnp_mapping(external_tcp_port: u16) {
    tauri::async_runtime::spawn(async move {
        let Ok(gw) = search_gateway(SearchOptions::default()).await else {
            return;
        };
        let _ = gw
            .remove_port(PortMappingProtocol::TCP, external_tcp_port)
            .await;
    });
}

/// Read binary WebSocket frames carrying snapshot chunks and reassemble them.
/// Emits `voxelle-load-progress` events so the join modal shows download progress.
async fn receive_snapshot_chunks<R: Runtime, S>(
    read: &mut S,
    app: &AppHandle<R>,
    cancel: &CancellationToken,
    chunk_count: u32,
    snapshot_len: u64,
) -> Result<Vec<u8>, String>
where
    S: StreamExt<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    let mut buf = Vec::with_capacity(snapshot_len as usize);

    let _ = app.emit("voxelle-load-start", "Project from host");

    let mut received: u32 = 0;
    while received < chunk_count {
        let _ = app.emit(
            "voxelle-load-progress",
            serde_json::json!({
                "fraction": received as f64 / chunk_count as f64,
                "phase": format!("Downloading map… ({}/{})", received + 1, chunk_count),
            }),
        );

        let frame = tokio::select! {
            f = read.next() => f,
            _ = cancel.cancelled() => return Err("Join cancelled.".into()),
        };
        let msg = frame
            .ok_or_else(|| "Host closed the connection while sending map chunks.".to_string())?
            .map_err(|e| {
                format!(
                    "Error receiving map chunk {}/{chunk_count}: {e}",
                    received + 1
                )
            })?;
        match msg {
            Message::Binary(data) => {
                buf.extend_from_slice(&data);
                received += 1;
            }
            Message::Text(t) => {
                // A text frame during chunk transfer is an error or denial from the host.
                if let Ok(HostToClient::Deny { reason }) = serde_json::from_str::<HostToClient>(&t)
                {
                    return Err(format!("The host refused the connection:\n{reason}"));
                }
                return Err(format!(
                    "Expected binary map chunk {}/{chunk_count} but got a text message.",
                    received + 1
                ));
            }
            Message::Close(_) => {
                return Err("Host closed the connection while sending map chunks.".into());
            }
            // Ping/Pong — skip without advancing the chunk counter.
            _ => {}
        }
    }
    Ok(buf)
}

async fn try_upnp_internet_share<R: Runtime>(
    app: AppHandle<R>,
    collab_mtx: Arc<Mutex<CollabRuntime>>,
    lan_ip: IpAddr,
    port: u16,
) {
    let emit = |result: CollabNatResult| {
        let _ = app.emit("collab-nat-result", result);
    };
    if lan_ip.is_loopback() {
        emit(CollabNatResult {
            wan_url: None,
            error: Some(
                "Internet sharing needs a LAN address; this machine only reported loopback.".into(),
            ),
        });
        return;
    }
    if !matches!(lan_ip, IpAddr::V4(_)) {
        emit(CollabNatResult {
            wan_url: None,
            error: Some(
                "UPnP port mapping needs an IPv4 LAN address. Disable internet sharing or use manual port forwarding."
                    .into(),
            ),
        });
        return;
    }
    let gw = match search_gateway(SearchOptions::default()).await {
        Ok(g) => g,
        Err(e) => {
            emit(CollabNatResult {
                wan_url: None,
                error: Some(format!(
                    "No UPnP gateway found ({e}). Enable UPnP on your router or set up manual TCP port forwarding to this machine."
                )),
            });
            return;
        }
    };
    let local = SocketAddr::new(lan_ip, port);
    // Try a timed lease first (many consumer routers reject permanent leases).
    // Fall back to permanent (`0`) if the router only supports that.
    let needs_renewal = {
        let result = gw
            .add_port(
                PortMappingProtocol::TCP,
                port,
                local,
                UPNP_LEASE_SECS,
                "Voxelle collaboration",
            )
            .await;
        match result {
            Ok(()) => true,
            Err(AddPortError::OnlyPermanentLeasesSupported) => {
                if let Err(e) = gw
                    .add_port(
                        PortMappingProtocol::TCP,
                        port,
                        local,
                        0,
                        "Voxelle collaboration",
                    )
                    .await
                {
                    emit(CollabNatResult {
                        wan_url: None,
                        error: Some(format!(
                            "Router refused port mapping ({e}). The port may be in use or blocked by policy."
                        )),
                    });
                    return;
                }
                false
            }
            Err(e) => {
                emit(CollabNatResult {
                    wan_url: None,
                    error: Some(format!(
                        "Router refused port mapping ({e}). The port may be in use or blocked by policy."
                    )),
                });
                return;
            }
        }
    };
    // Record immediately so `collab_leave` can remove the mapping even if we fail below or the user leaves early.
    {
        let mut g = collab_mtx.lock();
        if g.is_host() {
            g.upnp_external_tcp_port = Some(port);
        }
    }
    // Keep the lease alive by re-requesting it before expiry.
    if needs_renewal {
        let cancel = CancellationToken::new();
        {
            let mut g = collab_mtx.lock();
            if g.is_host() {
                g.upnp_renew_cancel = Some(cancel.clone());
            }
        }
        let cm = Arc::clone(&collab_mtx);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(UPNP_RENEW_INTERVAL);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            interval.tick().await; // skip immediate first tick
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    _ = interval.tick() => {}
                }
                let Ok(gw) = search_gateway(SearchOptions::default()).await else {
                    log::warn!("UPnP renewal: could not find gateway");
                    continue;
                };
                if let Err(e) = gw
                    .add_port(
                        PortMappingProtocol::TCP,
                        port,
                        local,
                        UPNP_LEASE_SECS,
                        "Voxelle collaboration",
                    )
                    .await
                {
                    log::warn!("UPnP renewal failed: {e}");
                }
                if !cm.lock().is_host() {
                    break;
                }
            }
        });
    }
    let ext_ip = match gw.get_external_ip().await {
        Ok(ip) => ip,
        Err(e) => {
            let _ = gw.remove_port(PortMappingProtocol::TCP, port).await;
            {
                let mut g = collab_mtx.lock();
                if g.is_host() {
                    g.upnp_external_tcp_port = None;
                }
            }
            emit(CollabNatResult {
                wan_url: None,
                error: Some(format!(
                    "Mapped the port but could not read the public address ({e}). Try manual forwarding."
                )),
            });
            return;
        }
    };
    emit(CollabNatResult {
        wan_url: Some(format!("ws://{ext_ip}:{port}")),
        error: None,
    });
}

fn replace_file_on_main<R: Runtime>(
    app: &AppHandle<R>,
    state: &Arc<ViewerState>,
    bytes: &[u8],
) -> Result<(), String> {
    let file = crate::voxelle::decode_payload(bytes).map_err(|e| e.to_string())?;
    let mode = *state.gpu.rendering_mode.lock();
    let _ = app.emit("voxelle-load-start", "Project from host");
    let prepared = crate::prepare_load_scene_cpu(
        file.grid_size,
        &file.voxels,
        &file.objects,
        mode,
        Some(app),
    )?;
    let app = app.clone();
    let state_apply = Arc::clone(state);
    let state_emit = Arc::clone(state);
    let app_mesh = app.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    let _ = app.run_on_main_thread(move || {
        let r = crate::apply_mesh_and_camera(&state_apply, &app_mesh, file, prepared);
        let _ = tx.send(r);
    });
    let main_result = rx.recv().map_err(|_| "main thread closed".to_string())?;
    match main_result {
        Ok(()) => {
            crate::emit_voxelle_loaded(&app, "collab snapshot".to_string(), state_emit.as_ref());
            Ok(())
        }
        Err(e) => {
            let _ = app.emit("voxelle-load-error", e.clone());
            Err(e)
        }
    }
}

async fn handle_host_connection<R: Runtime>(
    mut ws: WsTcp,
    mut broadcast_rx: broadcast::Receiver<Message>,
    app: AppHandle<R>,
    state: Arc<ViewerState>,
    collab_mtx: Arc<Mutex<CollabRuntime>>,
    peer_id: u32,
    display_name: String,
    color_rgb: u32,
    mut kick_rx: watch::Receiver<Option<String>>,
) {
    // Update this peer's display name / color atomically under a single lock, then
    // read back the full roster for the broadcast. Using two separate lock
    // acquisitions (read-then-write) would create a TOCTOU window where concurrent
    // `handle_host_connection` tasks could overwrite each other's roster entries.
    let mut roster = {
        let mut g = collab_mtx.lock();
        if let Some(r) = g.roster.iter_mut().find(|r| r.peer_id == peer_id) {
            r.display_name = display_name;
            r.color_rgb = color_rgb;
        }
        g.roster.clone()
    };
    broadcast_roster_to_guests(&app, &collab_mtx, &roster);

    let snap_result: Result<Vec<u8>, String> = {
        let g = state.file.current_file.lock();
        let file = g.as_ref().cloned().unwrap_or_else(empty_collab_placeholder);
        encode_payload_v4(&file).map_err(|e| format!("snapshot encode failed: {e}"))
    };
    let snap = match snap_result {
        Ok(bytes) => bytes,
        Err(reason) => {
            let _ = ws
                .send(Message::Text(
                    serde_json::to_string(&HostToClient::Deny { reason }).unwrap(),
                ))
                .await;
            host_remove_peer_from_session(&app, &collab_mtx, peer_id, None);
            return;
        }
    };
    let (existing_avatars, existing_avatar_data) = {
        let g = collab_mtx.lock();
        (g.avatar_names.clone(), g.avatar_data.clone())
    };
    if snap.len() <= SNAPSHOT_CHUNK_THRESHOLD {
        // Small map – send inline Welcome (original path).
        let welcome = HostToClient::Welcome {
            peer_id,
            leader_id: HOST_PEER_ID,
            snapshot: snap,
            roster: roster.clone(),
            avatar_names: existing_avatars,
            avatar_data: existing_avatar_data,
        };
        let _ = ws
            .send(Message::Text(serde_json::to_string(&welcome).unwrap()))
            .await;
    } else {
        // Large map – send header then binary chunks (no base64 overhead).
        let chunk_count = snap.len().div_ceil(SNAPSHOT_CHUNK_SIZE).max(1) as u32;
        let header = HostToClient::WelcomeHeader {
            peer_id,
            leader_id: HOST_PEER_ID,
            roster: roster.clone(),
            snapshot_len: snap.len() as u64,
            chunk_count,
            avatar_names: existing_avatars,
            avatar_data: existing_avatar_data,
        };
        let _ = ws
            .send(Message::Text(serde_json::to_string(&header).unwrap()))
            .await;
        for chunk in snap.chunks(SNAPSHOT_CHUNK_SIZE) {
            let _ = ws.send(Message::Binary(chunk.to_vec())).await;
        }
    }

    let mut peer_already_removed = false;
    let mut kicked = false;
    // Tracks when this guest first started lagging. Cleared when they catch up.
    let mut lag_since: Option<Instant> = None;
    // Rate-limits camera-presence forwards for this guest: at most one per CAMERA_BROADCAST_MIN_INTERVAL.
    let mut last_camera_forward = Instant::now()
        .checked_sub(CAMERA_BROADCAST_MIN_INTERVAL)
        .unwrap_or_else(Instant::now);
    loop {
        tokio::select! {
            biased;
            r = kick_rx.changed() => {
                if r.is_err() {
                    break;
                }
                let reason_opt = kick_rx.borrow().clone();
                if let Some(reason) = reason_opt {
                    let _ = ws
                        .send(Message::Text(
                            serde_json::to_string(&HostToClient::Kicked { reason }).unwrap(),
                        ))
                        .await;
                    kicked = true;
                    break;
                }
            }
            bmsg = broadcast_rx.recv() => {
                match bmsg {
                    Ok(msg) => {
                        lag_since = None; // caught up — clear any lag window
                        let _ = ws.send(msg).await;
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        let now = Instant::now();
                        if let Some(since) = lag_since {
                            if now.duration_since(since) > super::GUEST_LAG_KICK_TIMEOUT {
                                // Too slow for too long — kick the guest cleanly.
                                log::warn!(
                                    "collab: kicking peer {peer_id} after lagging for > {}s",
                                    super::GUEST_LAG_KICK_TIMEOUT.as_secs()
                                );
                                let _ = ws.send(Message::Text(
                                    serde_json::to_string(&HostToClient::Kicked {
                                        reason: "connection too slow to keep up with the session".into(),
                                    }).unwrap(),
                                )).await;
                                kicked = true;
                                break;
                            }
                            // Still within the grace window — resync already sent, just wait.
                        } else {
                            // First lag for this peer: record time and push a fresh snapshot
                            // so the guest's state converges despite the missed messages.
                            lag_since = Some(now);
                            log::warn!(
                                "collab: peer {peer_id} lagged {n} broadcast messages — sending resync snapshot"
                            );
                            let snap_result: Result<Vec<u8>, String> = {
                                let g = state.file.current_file.lock();
                                let file = g.as_ref().cloned().unwrap_or_else(empty_collab_placeholder);
                                encode_payload_v4(&file).map_err(|e| e.to_string())
                            };
                            if let Ok(snap) = snap_result {
                                if snap.len() <= SNAPSHOT_CHUNK_THRESHOLD {
                                    if let Ok(json) = serde_json::to_string(&HostToClient::Snapshot { bytes: snap }) {
                                        let _ = ws.send(Message::Text(json)).await;
                                    }
                                } else {
                                    let chunk_count =
                                        snap.len().div_ceil(SNAPSHOT_CHUNK_SIZE).max(1) as u32;
                                    let header = HostToClient::WelcomeHeader {
                                        peer_id: 0,
                                        leader_id: HOST_PEER_ID,
                                        roster: Vec::new(),
                                        snapshot_len: snap.len() as u64,
                                        chunk_count,
                                        avatar_names: std::collections::HashMap::new(),
                                        avatar_data: std::collections::HashMap::new(),
                                    };
                                    if let Ok(json) = serde_json::to_string(&header) {
                                        let _ = ws.send(Message::Text(json)).await;
                                    }
                                    for chunk in snap.chunks(SNAPSHOT_CHUNK_SIZE) {
                                        let _ = ws.send(Message::Binary(chunk.to_vec())).await;
                                    }
                                }
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
            incoming = ws.next() => {
                let Some(msg_result) = incoming else { break; };
                let msg = match msg_result {
                    Ok(m) => m,
                    Err(_) => break,
                };
                let cmd = match msg {
                    Message::Text(t) => {
                        match serde_json::from_str::<ClientToHost>(&t) {
                            Ok(c) => c,
                            Err(_) => continue,
                        }
                    }
                    Message::Binary(data) => {
                        if let Some(deltas) = decode_client_edit_binary(&data) {
                            ClientToHost::Edit { deltas }
                        } else {
                            continue;
                        }
                    }
                    Message::Ping(_) | Message::Pong(_) => continue,
                    Message::Close(_) | Message::Frame(_) => break,
                };
                touch_guest_activity(&collab_mtx, peer_id);
                match cmd {
            ClientToHost::Join { .. } => {}
            ClientToHost::Heartbeat => {}
            ClientToHost::LatencyProbe { sent_ms } => {
                let ack = serde_json::to_string(&HostToClient::LatencyAck { sent_ms }).unwrap();
                let _ = ws.send(Message::Text(ack)).await;
            }
            ClientToHost::Leave => {
                host_remove_peer_from_session(
                    &app,
                    &collab_mtx,
                    peer_id,
                    Some(CollabPeerLeftKind::Left),
                );
                peer_already_removed = true;
                break;
            }
            ClientToHost::Edit { deltas } => {
                if !check_can_edit(&collab_mtx, peer_id) {
                    let _ = ws
                        .send(Message::Text(
                            serde_json::to_string(&HostToClient::Deny {
                                reason: "editing not allowed".into(),
                            })
                            .unwrap(),
                        ))
                        .await;
                    continue;
                }
                if deltas.is_empty() {
                    // No GPU work needed — handle inline instead of
                    // queueing for the render loop (also keeps tests
                    // working under mock_app which has no event loop).
                    let seq = {
                        let mut c = collab_mtx.lock();
                        c.next_seq += 1;
                        let undo = c.host_undo.entry(peer_id).or_default();
                        undo.push(deltas.clone());
                        if undo.len() > super::MAX_UNDO_PER_PEER {
                            let excess = undo.len() - super::MAX_UNDO_PER_PEER;
                            undo.drain(0..excess);
                        }
                        c.host_redo.remove(&peer_id);
                        c.next_seq
                    };
                    broadcast_edit_binary(&collab_mtx, seq, peer_id, &deltas);
                } else {
                    // Queue for the main-thread render loop — never block
                    // the tokio task on GPU work or hold collab_mtx across it.
                    state.file.collab_edit_inbox.lock().push_back(
                        CollabInboxItem::Edit { peer_id, deltas },
                    );
                }
            }
            ClientToHost::Undo => {
                if !check_can_edit(&collab_mtx, peer_id) {
                    let _ = ws
                        .send(Message::Text(
                            serde_json::to_string(&HostToClient::Deny {
                                reason: "editing not allowed".into(),
                            })
                            .unwrap(),
                        ))
                        .await;
                    continue;
                }
                state.file.collab_edit_inbox.lock().push_back(
                    CollabInboxItem::Undo { peer_id },
                );
            }
            ClientToHost::Redo => {
                if !check_can_edit(&collab_mtx, peer_id) {
                    let _ = ws
                        .send(Message::Text(
                            serde_json::to_string(&HostToClient::Deny {
                                reason: "editing not allowed".into(),
                            })
                            .unwrap(),
                        ))
                        .await;
                    continue;
                }
                state.file.collab_edit_inbox.lock().push_back(
                    CollabInboxItem::Redo { peer_id },
                );
            }
            ClientToHost::Chat { text } => {
                let name = roster
                    .iter()
                    .find(|r| r.peer_id == peer_id)
                    .map(|r| r.display_name.clone())
                    .unwrap_or_else(|| "peer".into());
                let ev = HostToClient::Chat {
                    peer_id,
                    display_name: name,
                    text,
                    ts_ms: chrono::Utc::now().timestamp_millis(),
                };
                let json = serde_json::to_string(&ev).unwrap();
                let _ = app.emit("collab-chat", &json);
                {
                    let g = collab_mtx.lock();
                    if let Some(tx) = &g.host_broadcast {
                        let _ = tx.send(Message::Text(json));
                    }
                }
            }
            ClientToHost::UpdateProfile {
                display_name,
                color_rgb,
            } => {
                for r in &mut roster {
                    if r.peer_id == peer_id {
                        r.display_name = display_name.clone();
                        r.color_rgb = color_rgb;
                    }
                }
                collab_mtx.lock().roster = roster.clone();
                broadcast_roster_to_guests(&app, &collab_mtx, &roster);
            }
            ClientToHost::Ping { x, y, z, emoji } => {
                record_ping_flash(state.as_ref(), peer_id, x, y, z, emoji.clone());
                let display_name = roster
                    .iter()
                    .find(|r| r.peer_id == peer_id)
                    .map(|r| r.display_name.clone())
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| "Guest".to_string());
                let ping = HostToClient::Ping {
                    peer_id,
                    x,
                    y,
                    z,
                    display_name,
                    emoji,
                };
                let json = serde_json::to_string(&ping).unwrap();
                let _ = app.emit("collab-ping", &json);
                {
                    let g = collab_mtx.lock();
                    if let Some(tx) = &g.host_broadcast {
                        let _ = tx.send(Message::Text(json));
                    }
                }
            }
            ClientToHost::Camera { presence } => {
                // Always update the presence record so snap-to-peer is current.
                collab_mtx.lock().presence.insert(peer_id, presence);
                // Rate-limit broadcasts: each guest gets at most one camera forward
                // per CAMERA_BROADCAST_MIN_INTERVAL to cap broadcast traffic at 40+ peers.
                let now = Instant::now();
                if now.duration_since(last_camera_forward) >= CAMERA_BROADCAST_MIN_INTERVAL {
                    last_camera_forward = now;
                    let cam_ev = HostToClient::Camera { peer_id, presence };
                    let json = serde_json::to_string(&cam_ev).unwrap();
                    let _ = app.emit("collab-camera", &json);
                    {
                        let g = collab_mtx.lock();
                        if let Some(tx) = &g.host_broadcast {
                            let _ = tx.send(Message::Text(json));
                        }
                    }
                }
            }
            ClientToHost::AvatarChoice { avatar_name } => {
                collab_mtx
                    .lock()
                    .avatar_names
                    .insert(peer_id, avatar_name.clone());
                let ev = HostToClient::AvatarChoice {
                    peer_id,
                    avatar_name,
                };
                let json = serde_json::to_string(&ev).unwrap();
                let _ = app.emit("collab-avatar-choice", &json);
                {
                    let g = collab_mtx.lock();
                    if let Some(tx) = &g.host_broadcast {
                        let _ = tx.send(Message::Text(json));
                    }
                }
            }
            ClientToHost::AvatarData { name, bytes } => {
                if bytes.len() > MAX_AVATAR_FILE_BYTES {
                    continue;
                }
                collab_mtx
                    .lock()
                    .avatar_data
                    .insert(name.clone(), bytes.clone());
                let ev = HostToClient::AvatarData {
                    peer_id,
                    name: name.clone(),
                    bytes: bytes.clone(),
                };
                let json = serde_json::to_string(&ev).unwrap();
                // Notify the local render loop so it can decode and cache the mesh.
                let _ = app.emit("collab-avatar-data", &json);
                {
                    let g = collab_mtx.lock();
                    if let Some(tx) = &g.host_broadcast {
                        let _ = tx.send(Message::Text(json));
                    }
                }
            }
            ClientToHost::SetCanEdit {
                target_peer,
                can_edit: ce,
            } => {
                if peer_id != HOST_PEER_ID {
                    continue;
                }
                for r in &mut roster {
                    if r.peer_id == target_peer {
                        r.can_edit = ce;
                    }
                }
                collab_mtx.lock().roster = roster.clone();
                broadcast_roster_to_guests(&app, &collab_mtx, &roster);
            }
                }
            }
        }
    }
    if !peer_already_removed {
        host_remove_peer_from_session(
            &app,
            &collab_mtx,
            peer_id,
            if kicked {
                None
            } else {
                Some(CollabPeerLeftKind::Disconnected)
            },
        );
    }
}

pub fn start_host<R: Runtime>(
    app: AppHandle<R>,
    state: Arc<ViewerState>,
    collab_mtx: Arc<Mutex<CollabRuntime>>,
    port: u16,
    display_name: String,
    color_rgb: u32,
    enable_upnp: bool,
) -> Result<CollabHostStartResponse, String> {
    let mut c = collab_mtx.lock();
    if c.is_active() {
        return Err("already in a session".into());
    }
    let shutdown = Arc::new(AtomicBool::new(false));
    let sd = Arc::clone(&shutdown);
    c.shutdown = Some(shutdown);
    c.role = CollabRole::Hosting;
    c.local_peer_id = HOST_PEER_ID;
    c.leader_id = HOST_PEER_ID;
    c.roster = vec![RosterEntry {
        peer_id: HOST_PEER_ID,
        display_name,
        color_rgb,
        is_leader: true,
        can_edit: true,
    }];
    let (btx, _) = broadcast::channel::<Message>(BROADCAST_CAPACITY);
    c.host_broadcast = Some(btx.clone());
    let roster_json = serde_json::to_string(&c.roster).unwrap();
    drop(c);
    let _ = app.emit("collab-roster", roster_json);
    let _ = app.emit("collab-local-peer", HOST_PEER_ID);

    let cm_watch = Arc::clone(&collab_mtx);
    let sd_watch = Arc::clone(&sd);
    tauri::async_runtime::spawn(async move {
        let mut interval = tokio::time::interval(GUEST_ACTIVITY_CHECK_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            interval.tick().await;
            if sd_watch.load(Ordering::SeqCst) {
                break;
            }
            let stale: Vec<u32> = {
                let g = cm_watch.lock();
                if !g.is_host() {
                    continue;
                }
                let now = Instant::now();
                g.guest_last_activity
                    .iter()
                    .filter(|(_, t)| now.duration_since(**t) > GUEST_ACTIVITY_TIMEOUT)
                    .map(|(pid, _)| *pid)
                    .collect()
            };
            for pid in stale {
                let tx_opt = cm_watch.lock().host_peer_kick_tx.get(&pid).cloned();
                if let Some(tx) = tx_opt {
                    let _ = tx.send(Some(GUEST_TIMEOUT_KICK_REASON.to_string()));
                }
            }
        }
    });

    let cm_keep = Arc::clone(&collab_mtx);
    let sd_keep = Arc::clone(&sd);
    tauri::async_runtime::spawn(async move {
        let mut interval = tokio::time::interval(HOST_KEEPALIVE_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let keepalive_json = serde_json::to_string(&HostToClient::Keepalive).unwrap_or_default();
        loop {
            interval.tick().await;
            if sd_keep.load(Ordering::SeqCst) {
                break;
            }
            let send = {
                let g = cm_keep.lock();
                if !g.is_host() {
                    break;
                }
                g.host_broadcast
                    .as_ref()
                    .map(|tx| tx.send(Message::Text(keepalive_json.clone())))
            };
            if send.is_none() {
                break;
            }
        }
    });

    let lan_ip = local_ip_address::local_ip().unwrap_or(IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));

    let app2 = app.clone();
    let state2 = Arc::clone(&state);
    let cm2 = Arc::clone(&collab_mtx);
    let enable_upnp_task = enable_upnp;
    tauri::async_runtime::spawn(async move {
        let listener = match TcpListener::bind(("0.0.0.0", port)).await {
            Ok(l) => l,
            Err(e) => {
                cm2.lock().leave();
                let msg = format!(
                    "Could not listen for collaboration connections on port {port}.\n\n{e}\n\nAnother program may be using this port, or another Voxelle window may already be hosting on it. Try a different port or close the other session."
                );
                let _ = app2.emit("collab-error", msg);
                let _ = app2.emit("collab-ended", "");
                return;
            }
        };
        if enable_upnp_task {
            let app_u = app2.clone();
            let cm_u = Arc::clone(&cm2);
            let lip = lan_ip;
            let p = port;
            tokio::spawn(async move {
                try_upnp_internet_share(app_u, cm_u, lip, p).await;
            });
        }
        let mut next_peer: u32 = 2;
        while !sd.load(Ordering::SeqCst) {
            let acc = tokio::time::timeout(Duration::from_millis(400), listener.accept()).await;
            let Ok(Ok((stream, _))) = acc else {
                continue;
            };
            let mut ws = match accept_async(stream).await {
                Ok(w) => w,
                Err(_) => continue,
            };
            let pid = next_peer;
            next_peer += 1;
            let first = match ws.next().await {
                Some(Ok(Message::Text(t))) => t,
                _ => continue,
            };
            let (dname, col) = match serde_json::from_str::<ClientToHost>(&first) {
                Ok(ClientToHost::Join {
                    display_name,
                    color_rgb,
                }) => (display_name, color_rgb),
                _ => continue,
            };
            let app3 = app2.clone();
            let st3 = Arc::clone(&state2);
            let cm3 = Arc::clone(&cm2);
            let (kick_tx, kick_rx) = watch::channel(None);
            {
                let mut g = cm3.lock();
                g.roster.push(RosterEntry {
                    peer_id: pid,
                    display_name: dname.clone(),
                    color_rgb: col,
                    is_leader: false,
                    can_edit: false,
                });
                g.guest_last_activity.insert(pid, Instant::now());
                g.host_peer_kick_tx.insert(pid, kick_tx);
            }
            let sub = cm3.lock().host_broadcast.as_ref().unwrap().subscribe();
            tokio::spawn(handle_host_connection(
                ws, sub, app3, st3, cm3, pid, dname, col, kick_rx,
            ));
        }
    });

    Ok(CollabHostStartResponse {
        lan_url: format!("ws://{}:{port}", lan_ip),
        nat: if enable_upnp {
            "pending".into()
        } else {
            "none".into()
        },
    })
}

fn hint_for_ws_connect_error(detail_lower: &str) -> &'static str {
    if detail_lower.contains("connection refused") {
        "Nothing is listening on that address and port, or the host app is not running."
    } else if detail_lower.contains("timed out") || detail_lower.contains("timeout") {
        "The connection timed out. Check the IP address, VPN, and firewall settings."
    } else if detail_lower.contains("no route to host")
        || detail_lower.contains("host unreachable")
        || detail_lower.contains("network is unreachable")
    {
        "The network could not reach that host."
    } else if detail_lower.contains("dns")
        || detail_lower.contains("failed to resolve")
        || detail_lower.contains("name or service not known")
    {
        "Could not resolve the hostname. Check the spelling."
    } else if detail_lower.contains("invalid") && detail_lower.contains("url") {
        "Use a URL that starts with ws:// and includes the host and port (for example ws://192.168.1.5:7733)."
    } else {
        "Verify the WebSocket URL (ws://host:port), that the host is reachable on your network, and that firewalls allow this traffic."
    }
}

fn format_ws_connect_failure(url: &str, err: tokio_tungstenite::tungstenite::Error) -> String {
    let detail = err.to_string();
    let detail_lower = detail.to_lowercase();
    let hint = hint_for_ws_connect_error(&detail_lower);
    format!("Could not open a WebSocket to:\n{url}\n\n{detail}\n\n{hint}")
}

pub async fn client_connect_blocking<R: Runtime>(
    url: &str,
    app: AppHandle<R>,
    state: Arc<ViewerState>,
    collab_mtx: Arc<Mutex<CollabRuntime>>,
    display_name: String,
    color_rgb: u32,
    cancel: CancellationToken,
) -> Result<(), String> {
    let deadline = tokio::time::Instant::now() + CLIENT_JOIN_TIMEOUT;
    let timeout_msg = || {
        format!(
        "Could not reach the host at\n{url}\n\nThe connection timed out after {} seconds.\n\nCheck the address and make sure the host is reachable on your network.",
        CLIENT_JOIN_TIMEOUT.as_secs()
    )
    };

    let (ws, _) = tokio::select! {
        res = connect_async(url) => res.map_err(|e| format_ws_connect_failure(url, e))?,
        _ = cancel.cancelled() => return Err("Join cancelled.".into()),
        _ = tokio::time::sleep_until(deadline) => return Err(timeout_msg()),
    };
    let (mut write, mut read) = ws.split();
    let join = serde_json::to_string(&ClientToHost::Join {
        display_name: display_name.clone(),
        color_rgb,
    })
    .unwrap();
    tokio::select! {
        res = write.send(Message::Text(join)) => {
            res.map_err(|e| {
                format!(
                    "Could not send the join request to:\n{url}\n\n{e}\n\nThe connection may have dropped; try again."
                )
            })?;
        }
        _ = cancel.cancelled() => return Err("Join cancelled.".into()),
        _ = tokio::time::sleep_until(deadline) => return Err(timeout_msg()),
    }

    let first = tokio::select! {
        res = read.next() => {
            res.ok_or_else(|| {
                format!(
                    "The host at\n{url}\nclosed the connection before replying.\n\nCheck that the URL is a Voxelle collaboration link (ws://…) and that the host is still running."
                )
            })?
            .map_err(|e| {
                format!(
                    "Error while waiting for the host at\n{url}\n\n{e}\n\nThe connection may have been reset."
                )
            })?
        }
        _ = cancel.cancelled() => return Err("Join cancelled.".into()),
        _ = tokio::time::sleep_until(deadline) => return Err(timeout_msg()),
    };
    let Message::Text(t) = first else {
        return Err(format!(
            "The server at\n{url}\nsent a non-text WebSocket frame.\n\nUse a Voxelle collaboration URL (ws://host:port)."
        ));
    };
    let welcome: HostToClient = serde_json::from_str(&t).map_err(|e| {
        format!(
            "Could not parse the host's first message (expected JSON welcome).\n{url}\n\n{e}\n\nCheck that you are connecting to a Voxelle host, not another WebSocket server."
        )
    })?;
    match welcome {
        HostToClient::Deny { reason } => {
            return Err(format!("The host refused the connection:\n{reason}"));
        }
        HostToClient::Welcome {
            peer_id,
            leader_id,
            snapshot,
            roster,
            avatar_names,
            avatar_data,
        } => {
            {
                let mut c = collab_mtx.lock();
                c.role = CollabRole::Client;
                c.local_peer_id = peer_id;
                c.leader_id = leader_id;
                c.roster = roster;
                c.avatar_names = avatar_names;
                c.avatar_data = avatar_data;
            }
            if let Err(e) = replace_file_on_main(&app, &state, &snapshot) {
                collab_mtx.lock().leave();
                return Err(format!(
                    "Connected to the host, but your scene could not be replaced with theirs:\n{e}"
                ));
            }
            let _ = app.emit("collab-joined", ());
            let _ = app.emit("collab-local-peer", peer_id);
            let _ = app.emit(
                "collab-roster",
                serde_json::to_string(&collab_mtx.lock().roster).unwrap(),
            );
        }
        HostToClient::WelcomeHeader {
            peer_id,
            leader_id,
            roster,
            snapshot_len,
            chunk_count,
            avatar_names,
            avatar_data,
        } => {
            {
                let mut c = collab_mtx.lock();
                c.role = CollabRole::Client;
                c.local_peer_id = peer_id;
                c.leader_id = leader_id;
                c.roster = roster;
                c.avatar_names = avatar_names;
                c.avatar_data = avatar_data;
            }
            let _ = app.emit("collab-joined", ());
            let _ = app.emit("collab-local-peer", peer_id);
            let _ = app.emit(
                "collab-roster",
                serde_json::to_string(&collab_mtx.lock().roster).unwrap(),
            );

            // Receive chunked snapshot.
            let snapshot =
                receive_snapshot_chunks(&mut read, &app, &cancel, chunk_count, snapshot_len)
                    .await?;

            if let Err(e) = replace_file_on_main(&app, &state, &snapshot) {
                collab_mtx.lock().leave();
                return Err(format!(
                    "Connected to the host, but your scene could not be replaced with theirs:\n{e}"
                ));
            }
        }
        _ => {
            return Err(
                "The server sent an unexpected message instead of a welcome.\n\nCheck that this URL is a Voxelle collaboration host (ws://…)."
                    .into(),
            );
        }
    }

    let (tx, mut rx) = mpsc::channel::<ClientOutgoing>(512);
    let tx_hb = tx.clone();
    let tx_probe = tx.clone();
    collab_mtx.lock().client_tx = Some(tx);
    let heartbeat_msg =
        ClientOutgoing::Text(serde_json::to_string(&ClientToHost::Heartbeat).unwrap());
    tauri::async_runtime::spawn(async move {
        let mut interval = tokio::time::interval(CLIENT_HEARTBEAT_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            interval.tick().await;
            if tx_hb.try_send(heartbeat_msg.clone()).is_err() {
                break;
            }
        }
    });
    tauri::async_runtime::spawn(async move {
        let mut interval = tokio::time::interval(CLIENT_LATENCY_PROBE_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            interval.tick().await;
            let sent_ms = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            let msg = ClientOutgoing::Text(
                serde_json::to_string(&ClientToHost::LatencyProbe { sent_ms }).unwrap(),
            );
            if tx_probe.try_send(msg).is_err() {
                break;
            }
        }
    });

    let mut write_w = write;
    tauri::async_runtime::spawn(async move {
        while let Some(m) = rx.recv().await {
            let ws_msg = match m {
                ClientOutgoing::Text(t) => Message::Text(t),
                ClientOutgoing::Binary(b) => Message::Binary(b),
            };
            if write_w.send(ws_msg).await.is_err() {
                break;
            }
        }
    });

    let app4 = app.clone();
    let st4 = Arc::clone(&state);
    let cm4 = Arc::clone(&collab_mtx);
    tauri::async_runtime::spawn(async move {
        let mut host_timed_out = false;
        // Buffer for reassembling chunked mid-session snapshots (binary frames) from the host.
        let mut pending_snapshot: Option<(u32, u32, Vec<u8>)> = None; // (chunks_expected, chunks_received, bytes)
        loop {
            let frame = tokio::time::timeout(CLIENT_HOST_SILENCE_TIMEOUT, read.next()).await;
            let msg = match frame {
                Err(_) => {
                    host_timed_out = true;
                    break;
                }
                Ok(None) => break,
                Ok(Some(Err(_))) => break,
                Ok(Some(Ok(m))) => m,
            };
            match msg {
                Message::Close(_) => break,
                Message::Binary(data) => {
                    // Tagged binary edit frame from host?
                    if let Some((seq, peer_id, deltas)) = decode_host_edit_binary(&data) {
                        let _ = seq;
                        let local = cm4.lock().local_peer_id;
                        if peer_id != local {
                            st4.file
                                .collab_edit_inbox
                                .lock()
                                .push_back(CollabInboxItem::Edit { peer_id, deltas });
                        }
                    } else if let Some((expected, ref mut received, ref mut buf)) = pending_snapshot
                    {
                        // Snapshot chunks: accumulate them.
                        buf.extend_from_slice(&data);
                        *received += 1;
                        if *received >= expected {
                            let bytes = std::mem::take(buf);
                            pending_snapshot = None;
                            let _ = replace_file_on_main(&app4, &st4, &bytes);
                        }
                    }
                }
                Message::Ping(_) | Message::Pong(_) => {}
                Message::Text(t) => {
                    if let Ok(ev) = serde_json::from_str::<HostToClient>(&t) {
                        match ev {
                            HostToClient::Keepalive => {}
                            HostToClient::LatencyAck { sent_ms } => {
                                let now_ms = SystemTime::now()
                                    .duration_since(SystemTime::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_millis()
                                    as u64;
                                let rtt_ms = now_ms.saturating_sub(sent_ms);
                                let _ = app4.emit("collab-latency-ms", rtt_ms as u32);
                            }
                            HostToClient::Edit {
                                deltas, peer_id, ..
                            } => {
                                let local = cm4.lock().local_peer_id;
                                if peer_id == local {
                                    continue;
                                }
                                st4.file
                                    .collab_edit_inbox
                                    .lock()
                                    .push_back(CollabInboxItem::Edit { peer_id, deltas });
                            }
                            HostToClient::Roster { roster } => {
                                cm4.lock().roster = roster.clone();
                                let _ = app4
                                    .emit("collab-roster", serde_json::to_string(&roster).unwrap());
                            }
                            HostToClient::Snapshot { bytes } => {
                                let _ = replace_file_on_main(&app4, &st4, &bytes);
                            }
                            HostToClient::WelcomeHeader {
                                snapshot_len,
                                chunk_count,
                                ..
                            } => {
                                // Mid-session chunked snapshot broadcast; start collecting binary frames.
                                pending_snapshot = Some((
                                    chunk_count,
                                    0,
                                    Vec::with_capacity(snapshot_len as usize),
                                ));
                            }
                            HostToClient::Kicked { reason } => {
                                {
                                    let mut c = cm4.lock();
                                    c.leave();
                                }
                                *st4.ping_flash.lock() = None;
                                let _ = app4.emit("collab-kicked", reason);
                                break;
                            }
                            HostToClient::Chat { .. } => {
                                let _ = app4.emit("collab-chat", t);
                            }
                            HostToClient::Ping {
                                peer_id: pid,
                                x,
                                y,
                                z,
                                display_name: _,
                                emoji,
                            } => {
                                record_ping_flash(st4.as_ref(), pid, x, y, z, emoji);
                                let _ = app4.emit("collab-ping", t);
                            }
                            HostToClient::Camera { peer_id, presence } => {
                                cm4.lock().presence.insert(peer_id, presence);
                                let _ = app4.emit("collab-camera", t);
                            }
                            HostToClient::AvatarChoice {
                                peer_id,
                                ref avatar_name,
                            } => {
                                cm4.lock().avatar_names.insert(peer_id, avatar_name.clone());
                                let _ = app4.emit("collab-avatar-choice", t);
                            }
                            HostToClient::AvatarData {
                                ref name,
                                ref bytes,
                                ..
                            } => {
                                if bytes.len() <= MAX_AVATAR_FILE_BYTES {
                                    cm4.lock().avatar_data.insert(name.clone(), bytes.clone());
                                }
                                let _ = app4.emit("collab-avatar-data", t);
                            }
                            _ => {
                                let _ = app4.emit("collab-msg", t);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        let still_client = cm4.lock().is_client();
        if still_client {
            cm4.lock().leave();
            *st4.ping_flash.lock() = None;
            let reason = if host_timed_out {
                "The host stopped responding. Your connection was closed."
            } else {
                "Disconnected from the host. The connection was closed."
            };
            let _ = app4.emit("collab-ended", reason);
        }
    });

    Ok(())
}
