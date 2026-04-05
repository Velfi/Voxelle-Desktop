//! Multi-user editing: WebSocket host/client, host-authoritative voxel ops, per-peer undo on host.

pub mod edits;
pub mod network;
pub mod presence;

use crate::voxel_edit;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tokio::sync::{broadcast, mpsc};
use tokio_tungstenite::tungstenite::Message;
use tokio_util::sync::CancellationToken;

// ── Re-exports for callers that import directly from `crate::collab` ────────

pub(crate) use edits::broadcast_roster_to_guests;
pub use edits::{
    broadcast_snapshot_to_guests, host_emit_edit_batch, host_kick_peer, process_inbox_item,
    process_inbox_items_batched,
};
pub use network::{client_connect_blocking, schedule_remove_upnp_mapping, start_host};
pub use presence::{presence_eye, record_ping_flash, record_ping_flash_colored};

// ── Constants ────────────────────────────────────────────────────────────────

/// Broadcast channel capacity.  Large enough that a slow guest has time to drain
/// before messages are overwritten; we handle [`broadcast::error::RecvError::Lagged`]
/// explicitly rather than silently dropping the guest.
pub(crate) const BROADCAST_CAPACITY: usize = 2048;

/// How long a guest may continuously lag the broadcast channel before being kicked.
/// Within this window we send them a resync snapshot and keep them in the session.
pub(crate) const GUEST_LAG_KICK_TIMEOUT: Duration = Duration::from_secs(10);

/// Minimum interval between camera-presence forwards for a single peer.
/// Limits broadcast traffic to ≤ 10 Hz per guest regardless of push rate.
pub(crate) const CAMERA_BROADCAST_MIN_INTERVAL: Duration = Duration::from_millis(100);

/// Maximum undo-stack depth per peer.  Oldest strokes are evicted when the cap is
/// reached so memory is bounded even in long sessions with many editors.
pub(crate) const MAX_UNDO_PER_PEER: usize = 100;

/// If a guest sends no message (including [`ClientToHost::Heartbeat`]) for this long, the host drops them from the roster.
pub(crate) const GUEST_ACTIVITY_TIMEOUT: Duration = Duration::from_secs(45);
pub(crate) const GUEST_ACTIVITY_CHECK_INTERVAL: Duration = Duration::from_secs(2);
/// Clients send this periodically so idle tabs still count as alive.
pub(crate) const CLIENT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(20);
/// Maximum time allowed for the entire client join handshake (connect + send join + receive welcome).
pub(crate) const CLIENT_JOIN_TIMEOUT: Duration = Duration::from_secs(10);
/// UPnP lease duration in seconds.  Many consumer routers reject permanent
/// leases (`0`), so we request a 1-hour lease and renew periodically.
pub(crate) const UPNP_LEASE_SECS: u32 = 3600;
/// How often to re-call `add_port` to keep the lease alive (75 % of lease).
pub(crate) const UPNP_RENEW_INTERVAL: Duration = Duration::from_secs(45 * 60);
/// Host → guests: lightweight message so clients can tell the host is still alive (see [`CLIENT_HOST_SILENCE_TIMEOUT`]).
pub(crate) const HOST_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(15);
/// Guest: if no inbound WebSocket frame for this long, treat the host as unresponsive.
pub(crate) const CLIENT_HOST_SILENCE_TIMEOUT: Duration = Duration::from_secs(45);
/// How often a guest sends a latency probe to measure round-trip time.
pub(crate) const CLIENT_LATENCY_PROBE_INTERVAL: Duration = Duration::from_secs(5);

pub(crate) const GUEST_TIMEOUT_KICK_REASON: &str = "timed out (no activity)";

/// Maximum raw byte size accepted for a peer-supplied custom avatar file.
/// Peers that send larger payloads are silently ignored.
pub const MAX_AVATAR_FILE_BYTES: usize = 64 * 1024; // 64 KB

/// Raw snapshot bytes below this threshold are sent inline in the Welcome message;
/// above it the host sends a [`HostToClient::WelcomeHeader`] followed by binary
/// WebSocket frames carrying the raw snapshot data in chunks.
pub(crate) const SNAPSHOT_CHUNK_THRESHOLD: usize = 2 * 1024 * 1024; // 2 MB
/// Each binary chunk carries at most this many raw bytes.
pub(crate) const SNAPSHOT_CHUNK_SIZE: usize = 4 * 1024 * 1024; // 4 MB

pub const HOST_PEER_ID: u32 = 1;

// ── Response / result types ──────────────────────────────────────────────────

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

// ── Roster / presence types ──────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RosterEntry {
    pub peer_id: u32,
    pub display_name: String,
    pub color_rgb: u32,
    pub is_leader: bool,
    pub can_edit: bool,
}

// Re-export presence types so callers use `collab::CameraPresence` etc.
pub use presence::{CameraPresence, PingFlash};

// ── Wire message enums ───────────────────────────────────────────────────────

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
        #[serde(default)]
        emoji: String,
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
    LatencyProbe {
        sent_ms: u64,
    },
    /// Guest is leaving the session (best-effort before the socket closes).
    Leave,
    /// Guest is changing their avatar model.
    AvatarChoice {
        avatar_name: String,
    },
    /// Guest is uploading the raw bytes of a custom avatar file so the host can
    /// redistribute them to all other peers.  Only sent for non-embedded avatars.
    AvatarData {
        name: String,
        bytes: Vec<u8>,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum HostToClient {
    Welcome {
        peer_id: u32,
        leader_id: u32,
        snapshot: Vec<u8>,
        roster: Vec<RosterEntry>,
        /// Avatar name chosen by each existing peer so the new joiner sees them immediately.
        #[serde(default)]
        avatar_names: HashMap<u32, String>,
        /// Raw `.voxelle` bytes for any custom (non-embedded) avatars currently in use,
        /// keyed by name, so the new joiner can decode them without a separate round-trip.
        #[serde(default)]
        avatar_data: HashMap<String, Vec<u8>>,
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
        #[serde(default)]
        emoji: String,
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
        /// Avatar name chosen by each existing peer so the new joiner sees them immediately.
        #[serde(default)]
        avatar_names: HashMap<u32, String>,
        /// Raw `.voxelle` bytes for any custom (non-embedded) avatars currently in use.
        #[serde(default)]
        avatar_data: HashMap<String, Vec<u8>>,
    },
    /// Broadcast periodically while hosting so guests reset their read timeout during idle sessions.
    Keepalive,
    /// Echo of a guest's [`ClientToHost::LatencyProbe`]; guest computes RTT as `now_ms - sent_ms`.
    LatencyAck {
        sent_ms: u64,
    },
    /// A peer changed their avatar model; broadcast to all other peers.
    AvatarChoice {
        peer_id: u32,
        avatar_name: String,
    },
    /// Raw bytes of a peer's custom avatar file; broadcast to all other peers so
    /// they can decode and render it without having the file locally.
    AvatarData {
        peer_id: u32,
        name: String,
        bytes: Vec<u8>,
    },
}

// ── Channel / inbox types ────────────────────────────────────────────────────

/// Payload sent through the client → host outbound channel.
#[derive(Clone)]
pub enum ClientOutgoing {
    /// JSON text frame (non-edit messages).
    Text(String),
    /// Binary frame (edit deltas, bincode-encoded with tag prefix).
    Binary(Vec<u8>),
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

// ── Peer left kind ───────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub(crate) enum CollabPeerLeftKind {
    /// Client sent [`ClientToHost::Leave`] before disconnecting.
    Left,
    /// Socket closed or failed without an explicit leave message.
    Disconnected,
}

// ── Runtime struct ───────────────────────────────────────────────────────────

pub struct CollabRuntime {
    pub role: CollabRole,
    pub local_peer_id: u32,
    pub leader_id: u32,
    pub roster: Vec<RosterEntry>,
    pub presence: HashMap<u32, CameraPresence>,
    /// Avatar name chosen by each peer (`""` = default glow dot).
    pub avatar_names: HashMap<u32, String>,
    /// Raw `.voxelle` bytes for custom (non-embedded) avatars, keyed by avatar name.
    /// Populated when any peer sends `AvatarData`; used to decode mesh for rendering
    /// and to forward to new joiners in the `Welcome` message.
    pub avatar_data: HashMap<String, Vec<u8>>,
    pub next_seq: u64,
    shutdown: Option<Arc<AtomicBool>>,
    /// Each vec is one logical edit (stroke or click).
    pub host_undo: HashMap<u32, Vec<Vec<voxel_edit::VoxelEditDelta>>>,
    pub host_redo: HashMap<u32, Vec<Vec<voxel_edit::VoxelEditDelta>>>,
    /// Host → all connected guest websockets.  Carries [`Message`] so we can send both
    /// JSON text frames and raw binary snapshot chunks.
    pub host_broadcast: Option<broadcast::Sender<Message>>,
    pub client_tx: Option<mpsc::Sender<ClientOutgoing>>,
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
            avatar_names: HashMap::new(),
            avatar_data: HashMap::new(),
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

    /// True when we are a guest with `can_edit` granted by the host.
    /// Returns `false` when not connected as a client (solo / host).
    pub fn client_can_edit(&self) -> bool {
        self.is_client()
            && self
                .roster
                .iter()
                .find(|r| r.peer_id == self.local_peer_id)
                .map(|r| r.can_edit)
                .unwrap_or(false)
    }

    pub fn leave(&mut self) {
        if let Some(f) = &self.shutdown {
            f.store(true, Ordering::SeqCst);
        }
        self.shutdown = None;
        self.role = CollabRole::None;
        self.roster.clear();
        self.presence.clear();
        self.avatar_names.clear();
        self.avatar_data.clear();
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

// Re-export encode_client_edit_binary for external callers (commands/collab.rs etc.)
pub use edits::encode_client_edit_binary;

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minimal_viewer_state_for_collab_tests;
    use crate::voxel_edit::VoxelEditDelta;
    use crate::voxelle::{MaterialId, Voxel};
    use edits::{broadcast_roster_to_guests, decode_host_edit_binary, encode_client_edit_binary};
    use futures_util::{SinkExt, StreamExt};
    use network::start_host;
    use parking_lot::Mutex;
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
            match next {
                Message::Binary(data) => {
                    if decode_host_edit_binary(&data).is_some() {
                        break;
                    }
                }
                Message::Text(t) => {
                    let ev: HostToClient = serde_json::from_str(&t).unwrap();
                    match ev {
                        HostToClient::Roster { .. } | HostToClient::Keepalive => continue,
                        HostToClient::Deny { .. } => panic!("unexpected Deny when can_edit"),
                        HostToClient::Edit { .. } => break,
                        other => panic!("expected Edit broadcast: {other:?}"),
                    }
                }
                _ => continue,
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
        *state.file.current_file.lock() = Some(empty_collab_placeholder());
        *state.file.voxel_map.lock() = Some(ahash::AHashMap::new());
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
        let fg = state.file.current_file.lock();
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
            let mut fg = state.file.current_file.lock();
            let mut vm = state.file.voxel_map.lock();
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

        let fg = state.file.current_file.lock();
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

        let fg = state.file.current_file.lock();
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

        let inbox = state.file.collab_edit_inbox.lock();
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

        let inbox = state.file.collab_edit_inbox.lock();
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
        let inbox = state.file.collab_edit_inbox.lock();
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

        let inbox = state.file.collab_edit_inbox.lock();
        assert!(
            inbox
                .iter()
                .any(|item| matches!(item, CollabInboxItem::Edit { peer_id: 2, .. })),
            "inbox should contain an Edit for peer 2, got {:?} items",
            inbox.len()
        );
        stop_host(&cm);
    }

    // ---------------------------------------------------------------
    //  Stress: 50 concurrent clients join, each floods the host with
    //  empty edits, and the roster must not shrink mid-flood.
    //
    //  Empty edits are processed inline in handle_host_connection (no
    //  render loop required), so next_seq must advance by exactly
    //  N × EDITS_PER_CLIENT when all tasks finish.
    // ---------------------------------------------------------------
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn stress_50_clients_concurrent_join_and_flood() {
        const N: usize = 50;
        const EDITS_PER_CLIENT: usize = 20;

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
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let (go_tx, _) = tokio::sync::broadcast::channel::<()>(1);
        let (joined_tx, mut joined_rx) = tokio::sync::mpsc::channel::<u32>(N);
        let (done_tx, mut done_rx) = tokio::sync::mpsc::channel::<()>(N);
        let empty_edit = serde_json::to_string(&ClientToHost::Edit { deltas: vec![] }).unwrap();

        let mut handles = Vec::with_capacity(N);
        for _ in 0..N {
            let mut go_rx = go_tx.subscribe();
            let joined = joined_tx.clone();
            let done = done_tx.clone();
            let edit_msg = empty_edit.clone();
            handles.push(tokio::spawn(async move {
                let (mut write, _read, peer_id) = ws_join(port).await.unwrap();
                let _ = joined.send(peer_id).await;
                let _ = go_rx.recv().await; // wait for flood signal
                for _ in 0..EDITS_PER_CLIENT {
                    if write.send(Message::Text(edit_msg.clone())).await.is_err() {
                        break;
                    }
                }
                let _ = done.send(()).await;
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }));
        }
        drop(joined_tx);
        drop(done_tx);

        // Wait for all N guests to join
        for _ in 0..N {
            joined_rx.recv().await.expect("guest join timed out");
        }
        assert_eq!(
            cm.lock().roster.len(),
            N + 1,
            "roster should have {} entries after join",
            N + 1
        );

        // Grant edit permission so empty edits are processed inline
        {
            let mut g = cm.lock();
            for r in g.roster.iter_mut() {
                r.can_edit = true;
            }
        }

        // Release all clients simultaneously
        let _ = go_tx.send(());

        // Wait for every client's flood loop to finish sending
        for _ in 0..N {
            done_rx.recv().await.expect("done signal timed out");
        }

        // Poll next_seq until it reaches the expected value (host may still be draining
        // frames from the TCP receive buffer when the client write loop finishes).
        let expected_seq = (N * EDITS_PER_CLIENT) as u64;
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if cm.lock().next_seq == expected_seq {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "timed out waiting for next_seq={expected_seq}; got {}",
                cm.lock().next_seq
            );
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        // No guest should have been dropped during the flood
        assert_eq!(
            cm.lock().roster.len(),
            N + 1,
            "roster shrank during flood — a guest was spuriously disconnected"
        );

        stop_host(&cm);
        for h in handles {
            let _ = h.await;
        }
    }

    // ---------------------------------------------------------------
    //  Verify that a broadcast channel whose ring buffer overflows
    //  returns RecvError::Lagged rather than panicking or silently
    //  losing the receiver — this is the precondition our resync
    //  logic in handle_host_connection depends on.
    // ---------------------------------------------------------------
    #[tokio::test]
    async fn broadcast_overflow_returns_lagged_not_panic() {
        // Capacity 2: sending a third message must evict the oldest.
        let (tx, mut rx) = tokio::sync::broadcast::channel::<Message>(2);
        for i in 0u32..3 {
            tx.send(Message::Text(format!("msg{i}"))).unwrap();
        }
        // First recv must be Lagged (the ring wrapped)
        match rx.recv().await {
            Err(broadcast::error::RecvError::Lagged(n)) => {
                assert!(n >= 1, "lagged count must be ≥ 1, got {n}");
            }
            Ok(_) => panic!("expected RecvError::Lagged, got a message"),
            Err(broadcast::error::RecvError::Closed) => {
                panic!("expected RecvError::Lagged, got Closed")
            }
        }
        // After recovering the receiver is positioned at the oldest surviving message
        // (capacity 2, 3 sent → msg0 evicted, ring holds msg1 and msg2; lag repositions to msg1)
        match rx.recv().await {
            Ok(Message::Text(t)) => assert_eq!(
                t, "msg1",
                "should receive oldest surviving message after lag"
            ),
            other => panic!("unexpected result after lag recovery: {other:?}"),
        }
    }

    #[test]
    fn binary_edit_encoding_is_smaller_than_json() {
        let deltas: Vec<VoxelEditDelta> = (0..50)
            .map(|i| {
                VoxelEditDelta::Added(Voxel {
                    x: i,
                    y: i * 2,
                    z: -i,
                    color: 0xFF8800,
                    material: MaterialId::Plastic,
                    object_id: 0,
                })
            })
            .collect();

        // JSON encoding (old path)
        let json = serde_json::to_string(&ClientToHost::Edit {
            deltas: deltas.clone(),
        })
        .unwrap();
        let json_bytes = json.len();

        // Bincode encoding (new path)
        let bin = encode_client_edit_binary(&deltas);
        let bin_bytes = bin.len();

        assert!(
            bin_bytes < json_bytes,
            "binary ({bin_bytes} B) should be smaller than JSON ({json_bytes} B)"
        );

        // Verify round-trip
        let decoded = edits::decode_client_edit_binary(&bin).expect("should decode");
        assert_eq!(decoded.len(), deltas.len());

        // Also check host edit encoding
        let host_json = serde_json::to_string(&HostToClient::Edit {
            seq: 42,
            peer_id: 3,
            deltas: deltas.clone(),
        })
        .unwrap();
        let host_bin = edits::encode_host_edit_binary(42, 3, &deltas);

        assert!(
            host_bin.len() < host_json.len(),
            "host binary ({} B) should be smaller than JSON ({} B)",
            host_bin.len(),
            host_json.len()
        );

        let (seq, peer_id, host_decoded) =
            decode_host_edit_binary(&host_bin).expect("should decode");
        assert_eq!(seq, 42);
        assert_eq!(peer_id, 3);
        assert_eq!(host_decoded.len(), deltas.len());

        // Print the savings for visibility in test output
        eprintln!(
            "client edit: JSON {json_bytes} B → bincode {bin_bytes} B ({:.0}% smaller)",
            (1.0 - bin_bytes as f64 / json_bytes as f64) * 100.0
        );
        eprintln!(
            "host edit:   JSON {} B → bincode {} B ({:.0}% smaller)",
            host_json.len(),
            host_bin.len(),
            (1.0 - host_bin.len() as f64 / host_json.len() as f64) * 100.0
        );
    }
}
