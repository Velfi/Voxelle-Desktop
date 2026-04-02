//! Multi-user editing: WebSocket host/client, host-authoritative voxel ops, per-peer undo on host.

use crate::camera::Spherical;
use crate::voxel_edit;
use crate::voxelle::{empty_collab_placeholder, encode_payload_v4};
use crate::ViewerState;
use crate::VoxelGpuRefreshReason;
use futures_util::{SinkExt, StreamExt};
use glam::Vec3;
use igd_next::aio::tokio::search_gateway;
use igd_next::{AddPortError, PortMappingProtocol, SearchOptions};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tauri::{AppHandle, Emitter, Runtime};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::broadcast;
use tokio::sync::watch;
use tokio_tungstenite::{accept_async, connect_async, tungstenite::Message, WebSocketStream};
use tokio_util::sync::CancellationToken;

type WsTcp = WebSocketStream<TcpStream>;

/// If a guest sends no message (including [`ClientToHost::Heartbeat`]) for this long, the host drops them from the roster.
const GUEST_ACTIVITY_TIMEOUT: Duration = Duration::from_secs(45);
const GUEST_ACTIVITY_CHECK_INTERVAL: Duration = Duration::from_secs(2);
/// Clients send this periodically so idle tabs still count as alive.
const CLIENT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(20);
/// Maximum time allowed for the entire client join handshake (connect + send join + receive welcome).
const CLIENT_JOIN_TIMEOUT: Duration = Duration::from_secs(10);
/// UPnP lease duration in seconds.  Many consumer routers reject permanent
/// leases (`0`), so we request a 1-hour lease and renew periodically.
const UPNP_LEASE_SECS: u32 = 3600;
/// How often to re-call `add_port` to keep the lease alive (75 % of lease).
const UPNP_RENEW_INTERVAL: Duration = Duration::from_secs(45 * 60);
/// Host → guests: lightweight message so clients can tell the host is still alive (see [`CLIENT_HOST_SILENCE_TIMEOUT`]).
const HOST_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(15);
/// Guest: if no inbound WebSocket frame for this long, treat the host as unresponsive.
const CLIENT_HOST_SILENCE_TIMEOUT: Duration = Duration::from_secs(45);
/// How often a guest sends a latency probe to measure round-trip time.
const CLIENT_LATENCY_PROBE_INTERVAL: Duration = Duration::from_secs(5);

const GUEST_TIMEOUT_KICK_REASON: &str = "timed out (no activity)";

/// Raw snapshot bytes below this threshold are sent inline in the Welcome message;
/// above it the host sends a [`HostToClient::WelcomeHeader`] followed by binary
/// WebSocket frames carrying the raw snapshot data in chunks.
const SNAPSHOT_CHUNK_THRESHOLD: usize = 2 * 1024 * 1024; // 2 MB
/// Each binary chunk carries at most this many raw bytes.
const SNAPSHOT_CHUNK_SIZE: usize = 4 * 1024 * 1024; // 4 MB

/// Response from [`start_host`]: LAN link is always available; UPnP completes asynchronously via [`CollabNatResult`] (`collab-nat-result`).
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CollabHostStartResponse {
    pub lan_url: String,
    /// `"none"` if UPnP was not requested, or `"pending"` while the router is contacted.
    pub nat: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CollabNatResult {
    pub wan_url: Option<String>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RosterEntry {
    pub peer_id: u32,
    pub display_name: String,
    pub color_rgb: u32,
    pub is_leader: bool,
    pub can_edit: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct CameraPresence {
    pub target: [f32; 3],
    pub radius: f32,
    pub theta: f32,
    pub phi: f32,
    pub perspective: bool,
    pub fov_y: f32,
    pub ortho_half_height: f32,
}

/// World-space eye position (orbit target + spherical offset), matching [`crate::camera::OrbitCamera::smooth_eye`].
pub fn presence_eye(p: &CameraPresence) -> Vec3 {
    let target = Vec3::new(p.target[0], p.target[1], p.target[2]);
    let s = Spherical {
        radius: p.radius,
        theta: p.theta,
        phi: p.phi,
    };
    target + s.to_offset()
}

/// Ephemeral world highlight when a peer pings a voxel cell (`collab-ping`).
#[derive(Clone)]
pub struct PingFlash {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub color_rgb: u32,
    pub until: std::time::Instant,
    pub started: std::time::Instant,
    pub display_name: String,
}

/// An edit, undo, or redo request from a guest, queued for the host's main-thread
/// render loop to process.  Pushed from tokio WebSocket handlers, drained each frame.
pub enum CollabInboxItem {
    Edit {
        peer_id: u32,
        deltas: Vec<voxel_edit::VoxelEditDelta>,
    },
    Undo {
        peer_id: u32,
    },
    Redo {
        peer_id: u32,
    },
}

/// Brief permission check — does `peer_id` have `can_edit` in the roster?
fn check_can_edit(collab_mtx: &Mutex<CollabRuntime>, peer_id: u32) -> bool {
    collab_mtx
        .lock()
        .roster
        .iter()
        .find(|r| r.peer_id == peer_id)
        .map(|r| r.can_edit)
        .unwrap_or(false)
}

pub fn record_ping_flash_colored(
    state: &ViewerState,
    x: i32,
    y: i32,
    z: i32,
    color_rgb: u32,
    display_name: String,
) {
    let now = std::time::Instant::now();
    {
        let mut g = state.ping_flash.lock();
        *g = Some(PingFlash {
            x,
            y,
            z,
            color_rgb,
            until: now + std::time::Duration::from_secs_f32(2.8),
            started: now,
            display_name,
        });
    }
}

/// Resolves accent color and display name from the roster. Do **not** call while holding [`ViewerState::collab`].
pub fn record_ping_flash(state: &ViewerState, peer_id: u32, x: i32, y: i32, z: i32) {
    let (color_rgb, display_name) = {
        let c = state.collab.lock();
        c.roster.iter().find(|r| r.peer_id == peer_id).map(|r| {
            (
                r.color_rgb,
                if r.display_name.is_empty() {
                    "Guest".to_string()
                } else {
                    r.display_name.clone()
                },
            )
        })
    }
    .unwrap_or((0xffff44, "Guest".to_string()));
    record_ping_flash_colored(state, x, y, z, color_rgb, display_name);
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ClientToHost {
    Join {
        display_name: String,
        color_rgb: u32,
    },
    Edit {
        deltas: Vec<voxel_edit::VoxelEditDelta>,
    },
    Undo,
    Redo,
    Chat {
        text: String,
    },
    Ping {
        x: i32,
        y: i32,
        z: i32,
    },
    Camera {
        presence: CameraPresence,
    },
    SetCanEdit {
        target_peer: u32,
        can_edit: bool,
    },
    /// Rename / recolor this connection (host applies and broadcasts [`HostToClient::Roster`]).
    UpdateProfile {
        display_name: String,
        color_rgb: u32,
    },
    /// Periodic liveness; host also treats any other inbound message as activity.
    Heartbeat,
    /// Round-trip latency probe; host echoes `sent_ms` back in a [`HostToClient::LatencyAck`].
    LatencyProbe { sent_ms: u64 },
    /// Guest is leaving the session (best-effort before the socket closes).
    Leave,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum HostToClient {
    Welcome {
        peer_id: u32,
        leader_id: u32,
        snapshot: Vec<u8>,
        roster: Vec<RosterEntry>,
    },
    Roster {
        roster: Vec<RosterEntry>,
    },
    Edit {
        seq: u64,
        peer_id: u32,
        deltas: Vec<voxel_edit::VoxelEditDelta>,
    },
    Chat {
        peer_id: u32,
        display_name: String,
        text: String,
        ts_ms: i64,
    },
    Ping {
        peer_id: u32,
        x: i32,
        y: i32,
        z: i32,
        #[serde(default)]
        display_name: String,
    },
    Camera {
        peer_id: u32,
        presence: CameraPresence,
    },
    Deny {
        reason: String,
    },
    Kicked {
        reason: String,
    },
    Snapshot {
        bytes: Vec<u8>,
    },
    /// Like [`Welcome`] but without inline snapshot data; followed by binary WebSocket frames.
    WelcomeHeader {
        peer_id: u32,
        leader_id: u32,
        roster: Vec<RosterEntry>,
        snapshot_len: u64,
        chunk_count: u32,
    },
    /// Broadcast periodically while hosting so guests reset their read timeout during idle sessions.
    Keepalive,
    /// Echo of a guest's [`ClientToHost::LatencyProbe`]; guest computes RTT as `now_ms - sent_ms`.
    LatencyAck { sent_ms: u64 },
}

pub struct CollabRuntime {
    pub role: CollabRole,
    pub local_peer_id: u32,
    pub leader_id: u32,
    pub roster: Vec<RosterEntry>,
    pub presence: HashMap<u32, CameraPresence>,
    pub next_seq: u64,
    shutdown: Option<Arc<AtomicBool>>,
    /// Each vec is one logical edit (stroke or click).
    pub host_undo: HashMap<u32, Vec<Vec<voxel_edit::VoxelEditDelta>>>,
    pub host_redo: HashMap<u32, Vec<Vec<voxel_edit::VoxelEditDelta>>>,
    /// Host → all connected guest websockets.  Carries [`Message`] so we can send both
    /// JSON text frames and raw binary snapshot chunks.
    pub host_broadcast: Option<broadcast::Sender<Message>>,
    pub client_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
    /// Per connected guest (peer id ≥ 2): send [`Some`] with kick reason to close the socket.
    pub host_peer_kick_tx: HashMap<u32, watch::Sender<Option<String>>>,
    /// Last inbound activity from each guest (any message or heartbeat). Host only.
    pub guest_last_activity: HashMap<u32, Instant>,
    /// TCP port we opened on the IGD via UPnP (usually same as listen port); cleared when mapping is removed.
    pub upnp_external_tcp_port: Option<u16>,
    /// Token to cancel the background UPnP lease renewal loop (host only).
    pub upnp_renew_cancel: Option<CancellationToken>,
    /// Token to cancel an in-flight client join attempt.
    pub join_cancel: Option<CancellationToken>,
}

impl Default for CollabRuntime {
    fn default() -> Self {
        Self {
            role: CollabRole::None,
            local_peer_id: 0,
            leader_id: 0,
            roster: Vec::new(),
            presence: HashMap::new(),
            next_seq: 0,
            shutdown: None,
            host_undo: HashMap::new(),
            host_redo: HashMap::new(),
            host_broadcast: None,
            client_tx: None,
            host_peer_kick_tx: HashMap::new(),
            guest_last_activity: HashMap::new(),
            upnp_external_tcp_port: None,
            upnp_renew_cancel: None,
            join_cancel: None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum CollabRole {
    #[default]
    None,
    Hosting,
    Client,
}

impl CollabRuntime {
    pub fn is_active(&self) -> bool {
        !matches!(self.role, CollabRole::None)
    }

    pub fn is_host(&self) -> bool {
        matches!(self.role, CollabRole::Hosting)
    }

    pub fn is_client(&self) -> bool {
        matches!(self.role, CollabRole::Client)
    }

    pub fn leave(&mut self) {
        if let Some(f) = &self.shutdown {
            f.store(true, Ordering::SeqCst);
        }
        self.shutdown = None;
        self.role = CollabRole::None;
        self.roster.clear();
        self.presence.clear();
        self.host_undo.clear();
        self.host_redo.clear();
        self.local_peer_id = 0;
        self.host_broadcast = None;
        self.client_tx = None;
        self.host_peer_kick_tx.clear();
        self.guest_last_activity.clear();
        if let Some(t) = self.upnp_renew_cancel.take() {
            t.cancel();
        }
        self.upnp_external_tcp_port = None;
        self.join_cancel = None;
    }
}

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

pub const HOST_PEER_ID: u32 = 1;

#[derive(Clone, Copy)]
pub(crate) enum CollabPeerLeftKind {
    /// Client sent [`ClientToHost::Leave`] before disconnecting.
    Left,
    /// Socket closed or failed without an explicit leave message.
    Disconnected,
}

fn replace_file_on_main<R: Runtime>(
    app: &AppHandle<R>,
    state: &Arc<ViewerState>,
    bytes: &[u8],
) -> Result<(), String> {
    let file = crate::voxelle::decode_payload(bytes).map_err(|e| e.to_string())?;
    let mode = *state.rendering_mode.lock();
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
        let r = crate::apply_mesh_and_camera(&state_apply, &app_mesh, file, prepared, false);
        let _ = tx.send(r);
    });
    let main_result = rx.recv().map_err(|_| "main thread closed".to_string())?;
    match main_result {
        Ok(()) => {
            crate::emit_voxelle_loaded(
                &app,
                "collab snapshot".to_string(),
                state_emit.as_ref(),
                false,
            );
            Ok(())
        }
        Err(e) => {
            let _ = app.emit("voxelle-load-error", e.clone());
            Err(e)
        }
    }
}

fn emit_and_broadcast<R: Runtime>(collab: &Mutex<CollabRuntime>, _app: &AppHandle<R>, json: &str) {
    let g = collab.lock();
    if let Some(tx) = &g.host_broadcast {
        let _ = tx.send(Message::Text(json.to_string()));
    }
}

/// Host local edit after GPU sync: notify guests + UI.
pub fn host_emit_edit_batch<R: Runtime>(
    collab_mtx: &Mutex<CollabRuntime>,
    app: &AppHandle<R>,
    seq: u64,
    peer_id: u32,
    deltas: &[voxel_edit::VoxelEditDelta],
) {
    let br = HostToClient::Edit {
        seq,
        peer_id,
        deltas: deltas.to_vec(),
    };
    if let Ok(json) = serde_json::to_string(&br) {
        emit_and_broadcast(collab_mtx, app, &json);
    }
}

/// Process one queued guest edit/undo/redo on the main thread.
///
/// Called from the render-loop drain — `collab_mtx` is **not** held when this runs,
/// and we only acquire it briefly for metadata after GPU work is finished.
pub fn process_inbox_item<R: Runtime>(
    app: &AppHandle<R>,
    state: &Arc<ViewerState>,
    collab_mtx: &Arc<Mutex<CollabRuntime>>,
    item: CollabInboxItem,
) {
    match item {
        CollabInboxItem::Edit { peer_id, deltas } => {
            // Apply voxel deltas to the file.
            if !deltas.is_empty() {
                let r: Result<(), String> = (|| {
                    let mut fg = state.current_file.lock();
                    let mut vm = state.voxel_map.lock();
                    let file = fg.as_mut().ok_or("no model loaded")?;
                    let vmap = vm.as_mut().ok_or("voxel index not ready")?;
                    for d in &deltas {
                        voxel_edit::apply_forward_delta(file, vmap, d)?;
                    }
                    Ok(())
                })();
                if let Err(e) = r {
                    let _ = app.emit("collab-error", e);
                    return;
                }
                let t = std::time::Instant::now();
                if let Err(e) = crate::finish_voxel_edit_gpu_deltas(
                    state,
                    &deltas,
                    0.0,
                    t,
                    app,
                    VoxelGpuRefreshReason::CollabApply,
                ) {
                    let _ = app.emit("collab-error", e);
                    return;
                }
            }
            // Brief lock: update seq / undo / redo, then broadcast.
            let seq = {
                let mut c = collab_mtx.lock();
                if !c.roster.iter().any(|r| r.peer_id == peer_id) {
                    return; // peer left while queued
                }
                c.next_seq += 1;
                c.host_undo.entry(peer_id).or_default().push(deltas.clone());
                c.host_redo.remove(&peer_id);
                c.next_seq
            };
            let br = HostToClient::Edit {
                seq,
                peer_id,
                deltas,
            };
            if let Ok(json) = serde_json::to_string(&br) {
                emit_and_broadcast(collab_mtx, app, &json);
            }
        }
        CollabInboxItem::Undo { peer_id } => {
            // Pop from the undo stack (brief lock).
            let original = collab_mtx
                .lock()
                .host_undo
                .entry(peer_id)
                .or_default()
                .pop();
            let Some(original) = original else {
                return;
            };
            // Apply inverse deltas + GPU (no collab lock held).
            let mesh_result: Result<Vec<voxel_edit::VoxelEditDelta>, String> = (|| {
                let mut fg = state.current_file.lock();
                let mut vm = state.voxel_map.lock();
                let file = fg.as_mut().ok_or("no model loaded")?;
                let vmap = vm.as_mut().ok_or("voxel index not ready")?;
                let mut mesh = Vec::with_capacity(original.len());
                for d in original.iter().rev() {
                    voxel_edit::apply_inverse_delta(file, vmap, d)?;
                    mesh.push(voxel_edit::mesh_delta_after_inverse_of(d));
                }
                Ok(mesh)
            })();
            let Ok(mesh_refresh) = mesh_result else {
                return;
            };
            let t = std::time::Instant::now();
            if crate::finish_voxel_edit_gpu_deltas(
                state,
                &mesh_refresh,
                0.0,
                t,
                app,
                VoxelGpuRefreshReason::Undo,
            )
            .is_err()
            {
                return;
            }
            // Brief lock: push to redo, broadcast.
            let seq = {
                let mut c = collab_mtx.lock();
                if !c.roster.iter().any(|r| r.peer_id == peer_id) {
                    return;
                }
                c.host_redo.entry(peer_id).or_default().push(original);
                c.next_seq
            };
            let br = HostToClient::Edit {
                seq,
                peer_id,
                deltas: mesh_refresh,
            };
            if let Ok(json) = serde_json::to_string(&br) {
                emit_and_broadcast(collab_mtx, app, &json);
            }
        }
        CollabInboxItem::Redo { peer_id } => {
            // Pop from the redo stack (brief lock).
            let forward = collab_mtx
                .lock()
                .host_redo
                .entry(peer_id)
                .or_default()
                .pop();
            let Some(forward) = forward else {
                return;
            };
            // Apply forward deltas + GPU (no collab lock held).
            let apply_result: Result<(), String> = (|| {
                let mut fg = state.current_file.lock();
                let mut vm = state.voxel_map.lock();
                let file = fg.as_mut().ok_or("no model loaded")?;
                let vmap = vm.as_mut().ok_or("voxel index not ready")?;
                for d in &forward {
                    voxel_edit::apply_forward_delta(file, vmap, d)?;
                }
                Ok(())
            })();
            if apply_result.is_err() {
                return;
            }
            let t = std::time::Instant::now();
            if crate::finish_voxel_edit_gpu_deltas(
                state,
                &forward,
                0.0,
                t,
                app,
                VoxelGpuRefreshReason::Redo,
            )
            .is_err()
            {
                return;
            }
            // Brief lock: push to undo, broadcast.
            let seq = {
                let mut c = collab_mtx.lock();
                if !c.roster.iter().any(|r| r.peer_id == peer_id) {
                    return;
                }
                c.host_undo
                    .entry(peer_id)
                    .or_default()
                    .push(forward.clone());
                c.next_seq
            };
            let br = HostToClient::Edit {
                seq,
                peer_id,
                deltas: forward,
            };
            if let Ok(json) = serde_json::to_string(&br) {
                emit_and_broadcast(collab_mtx, app, &json);
            }
        }
    }
}

fn flush_edit_batch<R: Runtime>(
    app: &AppHandle<R>,
    state: &Arc<ViewerState>,
    collab_mtx: &Arc<Mutex<CollabRuntime>>,
    batch: &mut Vec<(u32, Vec<voxel_edit::VoxelEditDelta>)>,
) {
    if batch.is_empty() {
        return;
    }
    let all_deltas: Vec<voxel_edit::VoxelEditDelta> =
        batch.iter().flat_map(|(_, d)| d.iter().copied()).collect();

    let r: Result<(), String> = (|| {
        let mut fg = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
        let file = fg.as_mut().ok_or("no model loaded")?;
        let vmap = vm.as_mut().ok_or("voxel index not ready")?;
        for d in &all_deltas {
            voxel_edit::apply_forward_delta(file, vmap, d)?;
        }
        Ok(())
    })();
    if let Err(e) = r {
        let _ = app.emit("collab-error", e);
        batch.clear();
        return;
    }

    let t = std::time::Instant::now();
    if let Err(e) = crate::finish_voxel_edit_gpu_deltas(
        state,
        &all_deltas,
        0.0,
        t,
        app,
        VoxelGpuRefreshReason::CollabApply,
    ) {
        let _ = app.emit("collab-error", e);
        batch.clear();
        return;
    }

    for (peer_id, deltas) in batch.drain(..) {
        let seq = {
            let mut c = collab_mtx.lock();
            if !c.roster.iter().any(|r| r.peer_id == peer_id) {
                continue;
            }
            c.next_seq += 1;
            c.host_undo.entry(peer_id).or_default().push(deltas.clone());
            c.host_redo.remove(&peer_id);
            c.next_seq
        };
        let br = HostToClient::Edit { seq, peer_id, deltas };
        if let Ok(json) = serde_json::to_string(&br) {
            emit_and_broadcast(collab_mtx, app, &json);
        }
    }
}

/// Drains a batch of [`CollabInboxItem`]s, coalescing consecutive `Edit` items into a single
/// GPU refresh to avoid N mesh rebuilds per frame when many players edit simultaneously.
pub fn process_inbox_items_batched<R: Runtime>(
    app: &AppHandle<R>,
    state: &Arc<ViewerState>,
    collab_mtx: &Arc<Mutex<CollabRuntime>>,
    items: Vec<CollabInboxItem>,
) {
    let mut batch: Vec<(u32, Vec<voxel_edit::VoxelEditDelta>)> = Vec::new();
    for item in items {
        match item {
            CollabInboxItem::Edit { peer_id, deltas } if !deltas.is_empty() => {
                batch.push((peer_id, deltas));
            }
            other => {
                flush_edit_batch(app, state, collab_mtx, &mut batch);
                process_inbox_item(app, state, collab_mtx, other);
            }
        }
    }
    flush_edit_batch(app, state, collab_mtx, &mut batch);
}

/// Pushes the current host scene to all guests (after load, new project, or open file).
pub fn broadcast_snapshot_to_guests(state: &Arc<ViewerState>) {
    let bytes: Result<Vec<u8>, String> = (|| {
        let g = state.current_file.lock();
        let file = g.as_ref().cloned().unwrap_or_else(empty_collab_placeholder);
        encode_payload_v4(&file).map_err(|e| e.to_string())
    })();
    let Ok(bytes) = bytes else {
        return;
    };
    if bytes.len() <= SNAPSHOT_CHUNK_THRESHOLD {
        let json = match serde_json::to_string(&HostToClient::Snapshot { bytes }) {
            Ok(j) => j,
            Err(_) => return,
        };
        let g = state.collab.lock();
        if !g.is_host() {
            return;
        }
        if let Some(tx) = &g.host_broadcast {
            let _ = tx.send(Message::Text(json));
        }
    } else {
        let chunk_count = bytes.len().div_ceil(SNAPSHOT_CHUNK_SIZE).max(1) as u32;
        let header = serde_json::to_string(&HostToClient::WelcomeHeader {
            peer_id: 0,
            leader_id: HOST_PEER_ID,
            roster: Vec::new(),
            snapshot_len: bytes.len() as u64,
            chunk_count,
        })
        .unwrap();
        let g = state.collab.lock();
        if !g.is_host() {
            return;
        }
        if let Some(tx) = &g.host_broadcast {
            let _ = tx.send(Message::Text(header));
            for chunk in bytes.chunks(SNAPSHOT_CHUNK_SIZE) {
                let _ = tx.send(Message::Binary(chunk.to_vec()));
            }
        }
    }
}

/// Host UI: disconnect a guest by peer id (sends [`HostToClient::Kicked`]).
pub fn host_kick_peer(collab_mtx: &Mutex<CollabRuntime>, target_peer: u32) -> Result<(), String> {
    let kick_tx = {
        let c = collab_mtx.lock();
        if !c.is_host() {
            return Err("only the host can remove peers".into());
        }
        if target_peer == HOST_PEER_ID {
            return Err("cannot remove the host".into());
        }
        c.host_peer_kick_tx.get(&target_peer).cloned()
    };
    let Some(kick_tx) = kick_tx else {
        return Err("peer is not connected".into());
    };
    let _ = kick_tx.send(Some("removed by host".into()));
    Ok(())
}

pub(crate) fn broadcast_roster_to_guests<R: Runtime>(
    app: &AppHandle<R>,
    collab_mtx: &Arc<Mutex<CollabRuntime>>,
    roster: &[RosterEntry],
) {
    let _ = app.emit(
        "collab-roster",
        serde_json::to_string(roster).unwrap_or_default(),
    );
    let br = HostToClient::Roster {
        roster: roster.to_vec(),
    };
    let g = collab_mtx.lock();
    if let Some(tx) = &g.host_broadcast {
        if let Ok(json) = serde_json::to_string(&br) {
            let _ = tx.send(Message::Text(json));
        }
    }
}

fn host_remove_peer_from_session<R: Runtime>(
    app: &AppHandle<R>,
    collab_mtx: &Arc<Mutex<CollabRuntime>>,
    peer_id: u32,
    notify: Option<CollabPeerLeftKind>,
) {
    let roster_vec = {
        let mut g = collab_mtx.lock();
        let Some(display_name) = g
            .roster
            .iter()
            .find(|r| r.peer_id == peer_id)
            .map(|r| r.display_name.clone())
        else {
            return;
        };
        g.roster.retain(|r| r.peer_id != peer_id);
        g.host_peer_kick_tx.remove(&peer_id);
        g.guest_last_activity.remove(&peer_id);
        g.presence.remove(&peer_id);
        g.host_undo.remove(&peer_id);
        g.host_redo.remove(&peer_id);
        (g.roster.clone(), display_name)
    };
    if let Some(kind) = notify {
        let reason = match kind {
            CollabPeerLeftKind::Left => "left",
            CollabPeerLeftKind::Disconnected => "disconnected",
        };
        let payload = serde_json::json!({
            "peerId": peer_id,
            "displayName": roster_vec.1,
            "reason": reason,
        })
        .to_string();
        let _ = app.emit("collab-peer-left", payload);
    }
    broadcast_roster_to_guests(app, collab_mtx, &roster_vec.0);
}

fn touch_guest_activity(collab_mtx: &Mutex<CollabRuntime>, peer_id: u32) {
    if peer_id == HOST_PEER_ID {
        return;
    }
    collab_mtx
        .lock()
        .guest_last_activity
        .insert(peer_id, Instant::now());
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
    let mut roster = collab_mtx.lock().roster.clone();
    if let Some(r) = roster.iter_mut().find(|r| r.peer_id == peer_id) {
        r.display_name = display_name;
        r.color_rgb = color_rgb;
    }
    collab_mtx.lock().roster = roster.clone();
    broadcast_roster_to_guests(&app, &collab_mtx, &roster);

    let snap_result: Result<Vec<u8>, String> = (|| {
        let g = state.current_file.lock();
        let file = g.as_ref().cloned().unwrap_or_else(empty_collab_placeholder);
        encode_payload_v4(&file).map_err(|e| format!("snapshot encode failed: {e}"))
    })();
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
    if snap.len() <= SNAPSHOT_CHUNK_THRESHOLD {
        // Small map – send inline Welcome (original path).
        let welcome = HostToClient::Welcome {
            peer_id,
            leader_id: HOST_PEER_ID,
            snapshot: snap,
            roster: roster.clone(),
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
                if let Ok(msg) = bmsg {
                    let _ = ws.send(msg).await;
                }
            }
            incoming = ws.next() => {
                let Some(msg_result) = incoming else { break; };
                let msg = match msg_result {
                    Ok(m) => m,
                    Err(_) => break,
                };
                let t = match msg {
                    Message::Text(t) => t,
                    Message::Ping(_) | Message::Pong(_) => continue,
                    Message::Close(_) | Message::Binary(_) | Message::Frame(_) => break,
                };
                let Ok(cmd) = serde_json::from_str::<ClientToHost>(&t) else {
                    continue;
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
                        c.host_undo
                            .entry(peer_id)
                            .or_default()
                            .push(deltas.clone());
                        c.host_redo.remove(&peer_id);
                        c.next_seq
                    };
                    let br = HostToClient::Edit {
                        seq,
                        peer_id,
                        deltas,
                    };
                    if let Ok(json) = serde_json::to_string(&br) {
                        emit_and_broadcast(&collab_mtx, &app, &json);
                    }
                } else {
                    // Queue for the main-thread render loop — never block
                    // the tokio task on GPU work or hold collab_mtx across it.
                    state.collab_edit_inbox.lock().push_back(
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
                state.collab_edit_inbox.lock().push_back(
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
                state.collab_edit_inbox.lock().push_back(
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
            ClientToHost::Ping { x, y, z } => {
                record_ping_flash(state.as_ref(), peer_id, x, y, z);
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
                collab_mtx.lock().presence.insert(peer_id, presence);
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
    let (btx, _) = broadcast::channel::<Message>(128);
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
            let acc =
                tokio::time::timeout(std::time::Duration::from_millis(400), listener.accept())
                    .await;
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
        } => {
            {
                let mut c = collab_mtx.lock();
                c.role = CollabRole::Client;
                c.local_peer_id = peer_id;
                c.leader_id = leader_id;
                c.roster = roster;
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
        } => {
            {
                let mut c = collab_mtx.lock();
                c.role = CollabRole::Client;
                c.local_peer_id = peer_id;
                c.leader_id = leader_id;
                c.roster = roster;
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

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let tx_hb = tx.clone();
    let tx_probe = tx.clone();
    collab_mtx.lock().client_tx = Some(tx);
    let heartbeat_msg = serde_json::to_string(&ClientToHost::Heartbeat).unwrap();
    tauri::async_runtime::spawn(async move {
        let mut interval = tokio::time::interval(CLIENT_HEARTBEAT_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            interval.tick().await;
            if tx_hb.send(heartbeat_msg.clone()).is_err() {
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
            let msg = serde_json::to_string(&ClientToHost::LatencyProbe { sent_ms }).unwrap();
            if tx_probe.send(msg).is_err() {
                break;
            }
        }
    });

    let mut write_w = write;
    tauri::async_runtime::spawn(async move {
        while let Some(m) = rx.recv().await {
            if write_w.send(Message::Text(m)).await.is_err() {
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
                    // Binary frames are snapshot chunks; accumulate them.
                    if let Some((expected, ref mut received, ref mut buf)) = pending_snapshot {
                        buf.extend_from_slice(&data);
                        *received += 1;
                        if *received >= expected {
                            let bytes = std::mem::take(buf);
                            pending_snapshot = None;
                            let _ = replace_file_on_main(&app4, &st4, &bytes);
                        }
                    }
                    // Binary frames outside an active chunk transfer are ignored.
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
                                    .as_millis() as u64;
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
                                st4.collab_edit_inbox
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
                            } => {
                                record_ping_flash(st4.as_ref(), pid, x, y, z);
                                let _ = app4.emit("collab-ping", t);
                            }
                            HostToClient::Camera { peer_id, presence } => {
                                cm4.lock().presence.insert(peer_id, presence);
                                let _ = app4.emit("collab-camera", t);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minimal_viewer_state_for_collab_tests;
    use crate::voxel_edit::VoxelEditDelta;
    use crate::voxelle::{MaterialId, Voxel};
    use futures_util::{SinkExt, StreamExt};
    use std::net::TcpListener;
    use tokio_tungstenite::{connect_async, tungstenite::Message, MaybeTlsStream};

    type TestWs = tokio_tungstenite::WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

    fn pick_listen_port() -> u16 {
        TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }

    fn stop_host(cm: &Arc<Mutex<CollabRuntime>>) {
        cm.lock().leave();
    }

    async fn ws_join(
        port: u16,
    ) -> Result<
        (
            futures_util::stream::SplitSink<TestWs, Message>,
            futures_util::stream::SplitStream<TestWs>,
            u32,
        ),
        String,
    > {
        let url = format!("ws://127.0.0.1:{port}");
        let (ws, _) = connect_async(&url).await.map_err(|e| e.to_string())?;
        let (mut write, mut read) = ws.split();
        let join = serde_json::to_string(&ClientToHost::Join {
            display_name: "Guest".into(),
            color_rgb: 0xff_00_ff,
        })
        .unwrap();
        write
            .send(Message::Text(join))
            .await
            .map_err(|e| e.to_string())?;
        let first = read
            .next()
            .await
            .ok_or_else(|| "websocket closed".to_string())?
            .map_err(|e| e.to_string())?;
        let Message::Text(t) = first else {
            return Err("expected text frame".into());
        };
        let welcome: HostToClient = serde_json::from_str(&t).map_err(|e| e.to_string())?;
        let peer_id = match welcome {
            HostToClient::Welcome { peer_id, .. } => peer_id,
            _ => return Err("expected Welcome".into()),
        };
        Ok((write, read, peer_id))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn host_kick_sends_kicked_message() {
        let app = tauri::test::mock_app();
        let handle = app.handle().clone();
        let state = minimal_viewer_state_for_collab_tests();
        let cm = Arc::clone(&state.collab);
        let port = pick_listen_port();
        start_host(
            handle,
            Arc::clone(&state),
            Arc::clone(&cm),
            port,
            "Host".into(),
            0x00_ff_00,
            false,
        )
        .expect("start_host");
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;

        let (_w, mut read, peer_id) = ws_join(port).await.expect("join");
        assert_eq!(peer_id, 2);
        host_kick_peer(&cm, peer_id).expect("kick");

        let mut kicked_reason: Option<String> = None;
        for _ in 0..32 {
            let next = read.next().await.expect("frame").expect("ws");
            let Message::Text(t) = next else {
                continue;
            };
            let ev: HostToClient = serde_json::from_str(&t).unwrap();
            if let HostToClient::Kicked { reason } = ev {
                kicked_reason = Some(reason);
                break;
            }
        }
        let reason = kicked_reason.expect("expected a Kicked message after host kick");
        assert!(reason.contains("removed by host"), "{}", reason);
        stop_host(&cm);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn reconnect_gets_new_peer_id_and_no_ghost() {
        let app = tauri::test::mock_app();
        let handle = app.handle().clone();
        let state = minimal_viewer_state_for_collab_tests();
        let cm = Arc::clone(&state.collab);
        let port = pick_listen_port();
        start_host(
            handle,
            Arc::clone(&state),
            Arc::clone(&cm),
            port,
            "Host".into(),
            0x00_ff_00,
            false,
        )
        .expect("start_host");
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;

        {
            let (_w1, _r1, id_a) = ws_join(port).await.expect("join a");
            assert_eq!(id_a, 2);
        }

        for _ in 0..50 {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            let g = cm.lock();
            if g.roster.iter().all(|r| r.peer_id != 2) {
                break;
            }
            drop(g);
        }
        {
            let g = cm.lock();
            assert!(
                g.roster.iter().all(|r| r.peer_id != 2),
                "roster should not retain disconnected guest: {:?}",
                g.roster
            );
        }

        let (_w2, _r2, id_b) = ws_join(port).await.expect("join b");
        assert_eq!(id_b, 3);
        {
            let g = cm.lock();
            let ids: Vec<u32> = g.roster.iter().map(|r| r.peer_id).collect();
            assert_eq!(ids, vec![HOST_PEER_ID, 3]);
        }
        stop_host(&cm);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn edit_denied_when_can_edit_false() {
        let app = tauri::test::mock_app();
        let handle = app.handle().clone();
        let state = minimal_viewer_state_for_collab_tests();
        let cm = Arc::clone(&state.collab);
        let port = pick_listen_port();
        start_host(
            handle,
            Arc::clone(&state),
            Arc::clone(&cm),
            port,
            "Host".into(),
            0x00_ff_00,
            false,
        )
        .expect("start_host");
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;

        let (mut write, mut read, peer_id) = ws_join(port).await.expect("join");
        assert_eq!(peer_id, 2);
        let dummy = Voxel {
            x: 0,
            y: 0,
            z: 0,
            color: 1,
            material: MaterialId::Plastic,
            object_id: 0,
        };
        let edit = serde_json::to_string(&ClientToHost::Edit {
            deltas: vec![VoxelEditDelta::Added(dummy)],
        })
        .unwrap();
        write.send(Message::Text(edit)).await.expect("send edit");

        let mut deny_reason: Option<String> = None;
        for _ in 0..32 {
            let next = read.next().await.expect("frame").expect("ws");
            let Message::Text(t) = next else {
                continue;
            };
            let ev: HostToClient = serde_json::from_str(&t).unwrap();
            if let HostToClient::Deny { reason } = ev {
                deny_reason = Some(reason);
                break;
            }
        }
        let reason = deny_reason.expect("expected Deny after edit when can_edit is false");
        assert!(reason.contains("editing not allowed"), "{}", reason);
        stop_host(&cm);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn edit_empty_allowed_when_can_edit_true() {
        let app = tauri::test::mock_app();
        let handle = app.handle().clone();
        let state = minimal_viewer_state_for_collab_tests();
        let cm = Arc::clone(&state.collab);
        let port = pick_listen_port();
        start_host(
            handle.clone(),
            Arc::clone(&state),
            Arc::clone(&cm),
            port,
            "Host".into(),
            0x00_ff_00,
            false,
        )
        .expect("start_host");
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;

        let (mut write, mut read, peer_id) = ws_join(port).await.expect("join");
        assert_eq!(peer_id, 2);
        {
            let mut g = cm.lock();
            if let Some(r) = g.roster.iter_mut().find(|r| r.peer_id == 2) {
                r.can_edit = true;
            }
            let roster = g.roster.clone();
            drop(g);
            broadcast_roster_to_guests(&handle, &cm, &roster);
        }

        let edit = serde_json::to_string(&ClientToHost::Edit { deltas: vec![] }).unwrap();
        write.send(Message::Text(edit)).await.expect("send edit");

        loop {
            let next = read.next().await.expect("frame").expect("ws");
            let Message::Text(t) = next else {
                continue;
            };
            let ev: HostToClient = serde_json::from_str(&t).unwrap();
            match ev {
                HostToClient::Roster { .. } | HostToClient::Keepalive => continue,
                HostToClient::Deny { .. } => panic!("unexpected Deny when can_edit"),
                HostToClient::Edit { .. } => break,
                other => panic!("expected Edit broadcast: {other:?}"),
            }
        }
        stop_host(&cm);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn guest_set_can_edit_via_wire_is_ignored() {
        let app = tauri::test::mock_app();
        let handle = app.handle().clone();
        let state = minimal_viewer_state_for_collab_tests();
        let cm = Arc::clone(&state.collab);
        let port = pick_listen_port();
        start_host(
            handle,
            Arc::clone(&state),
            Arc::clone(&cm),
            port,
            "Host".into(),
            0x00_ff_00,
            false,
        )
        .expect("start_host");
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;

        let (mut write, _read, peer_id) = ws_join(port).await.expect("join");
        assert_eq!(peer_id, 2);
        let msg = serde_json::to_string(&ClientToHost::SetCanEdit {
            target_peer: 2,
            can_edit: true,
        })
        .unwrap();
        write.send(Message::Text(msg)).await.expect("send");

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        {
            let g = cm.lock();
            let guest = g.roster.iter().find(|r| r.peer_id == 2).expect("guest");
            assert!(!guest.can_edit);
        }
        stop_host(&cm);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn guest_leave_removes_from_roster() {
        let app = tauri::test::mock_app();
        let handle = app.handle().clone();
        let state = minimal_viewer_state_for_collab_tests();
        let cm = Arc::clone(&state.collab);
        let port = pick_listen_port();
        start_host(
            handle,
            Arc::clone(&state),
            Arc::clone(&cm),
            port,
            "Host".into(),
            0x00_ff_00,
            false,
        )
        .expect("start_host");
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;

        let (mut write, mut read, peer_id) = ws_join(port).await.expect("join");
        assert_eq!(peer_id, 2);
        let leave = serde_json::to_string(&ClientToHost::Leave).unwrap();
        write.send(Message::Text(leave)).await.expect("leave");

        let _ = read.next().await;
        tokio::time::sleep(std::time::Duration::from_millis(120)).await;
        {
            let g = cm.lock();
            assert_eq!(g.roster.len(), 1);
            assert_eq!(g.roster[0].peer_id, HOST_PEER_ID);
        }
        stop_host(&cm);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn two_guests_distinct_peers() {
        let app = tauri::test::mock_app();
        let handle = app.handle().clone();
        let state = minimal_viewer_state_for_collab_tests();
        let cm = Arc::clone(&state.collab);
        let port = pick_listen_port();
        start_host(
            handle,
            Arc::clone(&state),
            Arc::clone(&cm),
            port,
            "Host".into(),
            0x00_ff_00,
            false,
        )
        .expect("start_host");
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;

        let (_w1, _r1, a) = ws_join(port).await.expect("g1");
        let (_w2, _r2, b) = ws_join(port).await.expect("g2");
        assert_eq!(a, 2);
        assert_eq!(b, 3);
        {
            let g = cm.lock();
            assert_eq!(g.roster.len(), 3);
        }
        stop_host(&cm);
    }

    // ---------------------------------------------------------------
    //  Helper: set up a hosting collab state with one peer in roster
    // ---------------------------------------------------------------
    fn setup_host_collab(state: &Arc<crate::ViewerState>, peer_id: u32) {
        let mut c = state.collab.lock();
        c.role = CollabRole::Hosting;
        c.local_peer_id = HOST_PEER_ID;
        c.leader_id = HOST_PEER_ID;
        c.roster = vec![
            RosterEntry {
                peer_id: HOST_PEER_ID,
                display_name: "Host".into(),
                color_rgb: 0x00_ff_00,
                is_leader: true,
                can_edit: true,
            },
            RosterEntry {
                peer_id,
                display_name: "Guest".into(),
                color_rgb: 0xff_00_ff,
                is_leader: false,
                can_edit: true,
            },
        ];
        let (btx, _) = tokio::sync::broadcast::channel::<Message>(64);
        c.host_broadcast = Some(btx);
    }

    fn dummy_voxel(x: i32, y: i32, z: i32) -> Voxel {
        Voxel {
            x,
            y,
            z,
            color: 1,
            material: MaterialId::Plastic,
            object_id: 0,
        }
    }

    /// Populate current_file + voxel_map so voxel edits can be applied.
    fn seed_file(state: &Arc<crate::ViewerState>) {
        use crate::voxelle::empty_collab_placeholder;
        *state.current_file.lock() = Some(empty_collab_placeholder());
        *state.voxel_map.lock() = Some(ahash::AHashMap::new());
    }

    // ---------------------------------------------------------------
    //  process_inbox_item: Edit applies voxel to file
    // ---------------------------------------------------------------
    #[test]
    fn inbox_edit_applies_voxel_to_file() {
        let app = tauri::test::mock_app();
        let handle = app.handle().clone();
        let state = minimal_viewer_state_for_collab_tests();
        setup_host_collab(&state, 2);
        seed_file(&state);

        let v = dummy_voxel(10, 20, 30);
        process_inbox_item(
            &handle,
            &state,
            &state.collab,
            CollabInboxItem::Edit {
                peer_id: 2,
                deltas: vec![VoxelEditDelta::Added(v)],
            },
        );

        // The voxel should be in the file even though GPU update fails.
        let fg = state.current_file.lock();
        let file = fg.as_ref().expect("file");
        assert!(
            file.voxels
                .iter()
                .any(|vx| vx.x == 10 && vx.y == 20 && vx.z == 30),
            "voxel should have been added to current_file"
        );
    }

    // ---------------------------------------------------------------
    //  process_inbox_item: empty Edit updates collab state
    // ---------------------------------------------------------------
    #[test]
    fn inbox_edit_empty_updates_collab_state() {
        let app = tauri::test::mock_app();
        let handle = app.handle().clone();
        let state = minimal_viewer_state_for_collab_tests();
        setup_host_collab(&state, 2);
        seed_file(&state);

        let prev_seq = state.collab.lock().next_seq;

        process_inbox_item(
            &handle,
            &state,
            &state.collab,
            CollabInboxItem::Edit {
                peer_id: 2,
                deltas: vec![],
            },
        );

        let c = state.collab.lock();
        assert_eq!(c.next_seq, prev_seq + 1, "seq should have incremented");
        let undo_stack = c.host_undo.get(&2).expect("undo stack for peer 2");
        assert_eq!(undo_stack.len(), 1, "one undo entry should exist");
        assert!(
            undo_stack[0].is_empty(),
            "undo entry should be empty deltas"
        );
    }

    // ---------------------------------------------------------------
    //  process_inbox_item: Undo applies inverse to file
    // ---------------------------------------------------------------
    #[test]
    fn inbox_undo_applies_inverse_to_file() {
        let app = tauri::test::mock_app();
        let handle = app.handle().clone();
        let state = minimal_viewer_state_for_collab_tests();
        setup_host_collab(&state, 2);
        seed_file(&state);

        // Add a voxel to the file and undo stack.
        let v = dummy_voxel(5, 6, 7);
        {
            let mut fg = state.current_file.lock();
            let mut vm = state.voxel_map.lock();
            let file = fg.as_mut().unwrap();
            let vmap = vm.as_mut().unwrap();
            crate::voxel_edit::apply_forward_delta(file, vmap, &VoxelEditDelta::Added(v))
                .expect("seed add");
        }
        state
            .collab
            .lock()
            .host_undo
            .entry(2)
            .or_default()
            .push(vec![VoxelEditDelta::Added(v)]);

        // Process Undo — should remove the voxel.
        process_inbox_item(
            &handle,
            &state,
            &state.collab,
            CollabInboxItem::Undo { peer_id: 2 },
        );

        let fg = state.current_file.lock();
        let file = fg.as_ref().expect("file");
        assert!(
            !file
                .voxels
                .iter()
                .any(|vx| vx.x == 5 && vx.y == 6 && vx.z == 7),
            "voxel should have been removed by undo"
        );
    }

    // ---------------------------------------------------------------
    //  process_inbox_item: Redo applies forward to file
    // ---------------------------------------------------------------
    #[test]
    fn inbox_redo_applies_forward_to_file() {
        let app = tauri::test::mock_app();
        let handle = app.handle().clone();
        let state = minimal_viewer_state_for_collab_tests();
        setup_host_collab(&state, 2);
        seed_file(&state);

        let v = dummy_voxel(8, 9, 10);
        // Put a forward delta on the redo stack.
        state
            .collab
            .lock()
            .host_redo
            .entry(2)
            .or_default()
            .push(vec![VoxelEditDelta::Added(v)]);

        process_inbox_item(
            &handle,
            &state,
            &state.collab,
            CollabInboxItem::Redo { peer_id: 2 },
        );

        let fg = state.current_file.lock();
        let file = fg.as_ref().expect("file");
        assert!(
            file.voxels
                .iter()
                .any(|vx| vx.x == 8 && vx.y == 9 && vx.z == 10),
            "voxel should have been added by redo"
        );
    }

    // ---------------------------------------------------------------
    //  process_inbox_item: edit for removed peer is silently dropped
    // ---------------------------------------------------------------
    #[test]
    fn inbox_edit_skips_removed_peer() {
        let app = tauri::test::mock_app();
        let handle = app.handle().clone();
        let state = minimal_viewer_state_for_collab_tests();
        setup_host_collab(&state, 2);
        seed_file(&state);

        let prev_seq = state.collab.lock().next_seq;

        // Empty edit for peer 99 who is NOT in the roster.
        process_inbox_item(
            &handle,
            &state,
            &state.collab,
            CollabInboxItem::Edit {
                peer_id: 99,
                deltas: vec![],
            },
        );

        let c = state.collab.lock();
        assert_eq!(
            c.next_seq, prev_seq,
            "seq should NOT have incremented for missing peer"
        );
        assert!(
            !c.host_undo.contains_key(&99),
            "no undo entry should exist for missing peer"
        );
    }

    // ---------------------------------------------------------------
    //  Integration: guest undo reaches the inbox
    // ---------------------------------------------------------------
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn guest_undo_pushes_to_inbox() {
        let app = tauri::test::mock_app();
        let handle = app.handle().clone();
        let state = minimal_viewer_state_for_collab_tests();
        let cm = Arc::clone(&state.collab);
        let port = pick_listen_port();
        start_host(
            handle.clone(),
            Arc::clone(&state),
            Arc::clone(&cm),
            port,
            "Host".into(),
            0x00_ff_00,
            false,
        )
        .expect("start_host");
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;

        let (mut write, _read, peer_id) = ws_join(port).await.expect("join");
        assert_eq!(peer_id, 2);
        // Grant edit permission.
        {
            let mut g = cm.lock();
            if let Some(r) = g.roster.iter_mut().find(|r| r.peer_id == 2) {
                r.can_edit = true;
            }
        }

        let undo = serde_json::to_string(&ClientToHost::Undo).unwrap();
        write.send(Message::Text(undo)).await.expect("send undo");
        // Give the handler time to push to inbox.
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;

        let inbox = state.collab_edit_inbox.lock();
        assert!(
            inbox
                .iter()
                .any(|item| matches!(item, CollabInboxItem::Undo { peer_id: 2 })),
            "inbox should contain an Undo for peer 2, got {:?} items",
            inbox.len()
        );
        stop_host(&cm);
    }

    // ---------------------------------------------------------------
    //  Integration: guest redo reaches the inbox
    // ---------------------------------------------------------------
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn guest_redo_pushes_to_inbox() {
        let app = tauri::test::mock_app();
        let handle = app.handle().clone();
        let state = minimal_viewer_state_for_collab_tests();
        let cm = Arc::clone(&state.collab);
        let port = pick_listen_port();
        start_host(
            handle.clone(),
            Arc::clone(&state),
            Arc::clone(&cm),
            port,
            "Host".into(),
            0x00_ff_00,
            false,
        )
        .expect("start_host");
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;

        let (mut write, _read, peer_id) = ws_join(port).await.expect("join");
        assert_eq!(peer_id, 2);
        {
            let mut g = cm.lock();
            if let Some(r) = g.roster.iter_mut().find(|r| r.peer_id == 2) {
                r.can_edit = true;
            }
        }

        let redo = serde_json::to_string(&ClientToHost::Redo).unwrap();
        write.send(Message::Text(redo)).await.expect("send redo");
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;

        let inbox = state.collab_edit_inbox.lock();
        assert!(
            inbox
                .iter()
                .any(|item| matches!(item, CollabInboxItem::Redo { peer_id: 2 })),
            "inbox should contain a Redo for peer 2, got {:?} items",
            inbox.len()
        );
        stop_host(&cm);
    }

    // ---------------------------------------------------------------
    //  Integration: denied edit does not reach inbox
    // ---------------------------------------------------------------
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn edit_denied_does_not_reach_inbox() {
        let app = tauri::test::mock_app();
        let handle = app.handle().clone();
        let state = minimal_viewer_state_for_collab_tests();
        let cm = Arc::clone(&state.collab);
        let port = pick_listen_port();
        start_host(
            handle,
            Arc::clone(&state),
            Arc::clone(&cm),
            port,
            "Host".into(),
            0x00_ff_00,
            false,
        )
        .expect("start_host");
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;

        let (mut write, mut read, peer_id) = ws_join(port).await.expect("join");
        assert_eq!(peer_id, 2);
        // can_edit is false by default — do NOT grant permission.

        let v = dummy_voxel(1, 2, 3);
        let edit = serde_json::to_string(&ClientToHost::Edit {
            deltas: vec![VoxelEditDelta::Added(v)],
        })
        .unwrap();
        write.send(Message::Text(edit)).await.expect("send edit");

        // Wait for Deny.
        let mut got_deny = false;
        for _ in 0..32 {
            let next = read.next().await.expect("frame").expect("ws");
            let Message::Text(t) = next else { continue };
            if let Ok(HostToClient::Deny { .. }) = serde_json::from_str(&t) {
                got_deny = true;
                break;
            }
        }
        assert!(got_deny, "should have received Deny");

        // Inbox should be empty — the edit was denied before reaching it.
        let inbox = state.collab_edit_inbox.lock();
        assert!(inbox.is_empty(), "inbox should be empty after denied edit");
        stop_host(&cm);
    }

    // ---------------------------------------------------------------
    //  Integration: non-empty edit reaches inbox when allowed
    // ---------------------------------------------------------------
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn edit_allowed_pushes_to_inbox() {
        let app = tauri::test::mock_app();
        let handle = app.handle().clone();
        let state = minimal_viewer_state_for_collab_tests();
        let cm = Arc::clone(&state.collab);
        let port = pick_listen_port();
        start_host(
            handle.clone(),
            Arc::clone(&state),
            Arc::clone(&cm),
            port,
            "Host".into(),
            0x00_ff_00,
            false,
        )
        .expect("start_host");
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;

        let (mut write, _read, peer_id) = ws_join(port).await.expect("join");
        assert_eq!(peer_id, 2);
        {
            let mut g = cm.lock();
            if let Some(r) = g.roster.iter_mut().find(|r| r.peer_id == 2) {
                r.can_edit = true;
            }
        }

        let v = dummy_voxel(3, 4, 5);
        let edit = serde_json::to_string(&ClientToHost::Edit {
            deltas: vec![VoxelEditDelta::Added(v)],
        })
        .unwrap();
        write.send(Message::Text(edit)).await.expect("send edit");
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;

        let inbox = state.collab_edit_inbox.lock();
        assert!(
            inbox
                .iter()
                .any(|item| matches!(item, CollabInboxItem::Edit { peer_id: 2, .. })),
            "inbox should contain an Edit for peer 2, got {:?} items",
            inbox.len()
        );
        stop_host(&cm);
    }
}
