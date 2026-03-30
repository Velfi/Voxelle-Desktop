//! Multi-user editing: WebSocket host/client, host-authoritative voxel ops, per-peer undo on host.

use crate::camera::Spherical;
use crate::finish_voxel_edit_gpu;
use crate::voxel_edit;
use crate::voxelle::encode_payload_v4;
use crate::ViewerState;
use futures_util::{SinkExt, StreamExt};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tauri::{AppHandle, Emitter};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::broadcast;
use tokio_tungstenite::{accept_async, connect_async, tungstenite::Message, WebSocketStream};

type WsTcp = WebSocketStream<TcpStream>;

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
#[derive(Clone, Copy)]
pub struct PingFlash {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub color_rgb: u32,
    pub until: std::time::Instant,
}

pub fn record_ping_flash_colored(state: &ViewerState, x: i32, y: i32, z: i32, color_rgb: u32) {
    if let Ok(mut g) = state.ping_flash.lock() {
        *g = Some(PingFlash {
            x,
            y,
            z,
            color_rgb,
            until: std::time::Instant::now() + std::time::Duration::from_secs_f32(2.8),
        });
    }
}

/// Resolves accent color from the roster. Do **not** call while holding [`ViewerState::collab`].
pub fn record_ping_flash(state: &ViewerState, peer_id: u32, x: i32, y: i32, z: i32) {
    let color_rgb = state
        .collab
        .lock()
        .ok()
        .and_then(|c| {
            c.roster
                .iter()
                .find(|r| r.peer_id == peer_id)
                .map(|r| r.color_rgb)
        })
        .unwrap_or(0xffff44);
    record_ping_flash_colored(state, x, y, z, color_rgb);
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ClientToHost {
    Join {
        display_name: String,
        color_rgb: u32,
    },
    Edit {
        delta: voxel_edit::VoxelEditDelta,
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
        delta: voxel_edit::VoxelEditDelta,
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
    },
    Camera {
        peer_id: u32,
        presence: CameraPresence,
    },
    Deny {
        reason: String,
    },
    Snapshot {
        bytes: Vec<u8>,
    },
}

pub struct CollabRuntime {
    pub role: CollabRole,
    pub local_peer_id: u32,
    pub leader_id: u32,
    pub roster: Vec<RosterEntry>,
    pub presence: HashMap<u32, CameraPresence>,
    pub next_seq: u64,
    shutdown: Option<Arc<AtomicBool>>,
    pub host_undo: HashMap<u32, Vec<voxel_edit::VoxelEditDelta>>,
    pub host_redo: HashMap<u32, Vec<voxel_edit::VoxelEditDelta>>,
    /// Host → all connected guest websockets.
    pub host_broadcast: Option<broadcast::Sender<String>>,
    pub client_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
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
    }
}

pub const HOST_PEER_ID: u32 = 1;

fn apply_delta_on_main(
    app: &AppHandle,
    state: &Arc<ViewerState>,
    delta: &voxel_edit::VoxelEditDelta,
) -> Result<(), String> {
    let delta = *delta;
    let app = app.clone();
    let state = Arc::clone(state);
    let (tx, rx) = std::sync::mpsc::channel();
    let _ = app.run_on_main_thread(move || {
        let r = (|| {
            let mut fg = state.current_file.lock().map_err(|e| e.to_string())?;
            let mut vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
            let Some(file) = fg.as_mut() else {
                return Err("no model loaded".into());
            };
            let Some(vmap) = vm.as_mut() else {
                return Err("voxel index not ready".into());
            };
            match delta {
                voxel_edit::VoxelEditDelta::Added(v) => {
                    voxel_edit::push_voxel_known(file, vmap, v);
                }
                voxel_edit::VoxelEditDelta::Removed { voxel } => {
                    voxel_edit::remove_voxel_at(file, vmap, (voxel.x, voxel.y, voxel.z))
                        .ok_or_else(|| "remote remove".to_string())?;
                }
            }
            Ok::<(), String>(())
        })();
        let r2 = r.and_then(|_| {
            let t = std::time::Instant::now();
            finish_voxel_edit_gpu(&state, &delta, 0.0, t)
        });
        let _ = tx.send(r2);
    });
    rx.recv().map_err(|_| "main thread closed".to_string())?
}

fn replace_file_on_main(
    app: &AppHandle,
    state: &Arc<ViewerState>,
    bytes: &[u8],
) -> Result<(), String> {
    let file = crate::voxelle::decode_payload(bytes).map_err(|e| e.to_string())?;
    let app = app.clone();
    let state = Arc::clone(state);
    let app_mesh = app.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    let _ = app.run_on_main_thread(move || {
        let r = crate::apply_mesh_and_camera(&state, &app_mesh, file);
        let _ = tx.send(r);
    });
    rx.recv().map_err(|_| "main thread closed".to_string())??;
    let _ = app.emit("voxelle-loaded", "collab snapshot");
    Ok(())
}

pub fn host_apply_remote_edit(
    _app: &AppHandle,
    state: &Arc<ViewerState>,
    collab: &mut CollabRuntime,
    peer_id: u32,
    delta: voxel_edit::VoxelEditDelta,
) -> Result<(), String> {
    apply_delta_on_main(_app, state, &delta)?;
    collab.next_seq += 1;
    collab.host_undo.entry(peer_id).or_default().push(delta);
    collab.host_redo.remove(&peer_id);
    Ok(())
}

pub fn host_undo_peer(
    app: &AppHandle,
    state: &Arc<ViewerState>,
    collab: &mut CollabRuntime,
    peer_id: u32,
) -> Result<Option<voxel_edit::VoxelEditDelta>, String> {
    let original = collab.host_undo.entry(peer_id).or_default().pop();
    let Some(original) = original else {
        return Ok(None);
    };
    let mesh_delta = {
        let mut fg = state.current_file.lock().map_err(|e| e.to_string())?;
        let mut vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        match original {
            voxel_edit::VoxelEditDelta::Added(v) => {
                voxel_edit::remove_voxel_at(file, vmap, (v.x, v.y, v.z))
                    .ok_or_else(|| "host undo".to_string())?;
                voxel_edit::VoxelEditDelta::Removed { voxel: v }
            }
            voxel_edit::VoxelEditDelta::Removed { voxel } => {
                voxel_edit::push_voxel_known(file, vmap, voxel);
                voxel_edit::VoxelEditDelta::Added(voxel)
            }
        }
    };
    let t = std::time::Instant::now();
    finish_voxel_edit_gpu(state, &mesh_delta, 0.0, t)?;
    collab.host_redo.entry(peer_id).or_default().push(original);
    let _ = app;
    Ok(Some(mesh_delta))
}

pub fn host_redo_peer(
    _app: &AppHandle,
    state: &Arc<ViewerState>,
    collab: &mut CollabRuntime,
    peer_id: u32,
) -> Result<Option<voxel_edit::VoxelEditDelta>, String> {
    let forward = collab.host_redo.entry(peer_id).or_default().pop();
    let Some(forward) = forward else {
        return Ok(None);
    };
    let mesh_delta = {
        let mut fg = state.current_file.lock().map_err(|e| e.to_string())?;
        let mut vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        match forward {
            voxel_edit::VoxelEditDelta::Added(v) => {
                voxel_edit::push_voxel_known(file, vmap, v);
                voxel_edit::VoxelEditDelta::Added(v)
            }
            voxel_edit::VoxelEditDelta::Removed { voxel } => {
                voxel_edit::remove_voxel_at(file, vmap, (voxel.x, voxel.y, voxel.z))
                    .ok_or_else(|| "host redo".to_string())?;
                voxel_edit::VoxelEditDelta::Removed { voxel }
            }
        }
    };
    let t = std::time::Instant::now();
    finish_voxel_edit_gpu(state, &mesh_delta, 0.0, t)?;
    collab.host_undo.entry(peer_id).or_default().push(forward);
    Ok(Some(mesh_delta))
}

fn emit_and_broadcast(collab: &std::sync::Mutex<CollabRuntime>, _app: &AppHandle, json: &str) {
    if let Ok(g) = collab.lock() {
        if let Some(tx) = &g.host_broadcast {
            let _ = tx.send(json.to_string());
        }
    }
}

/// Host local edit after GPU sync: notify guests + UI.
pub fn host_emit_edit(
    collab_mtx: &std::sync::Mutex<CollabRuntime>,
    app: &AppHandle,
    seq: u64,
    peer_id: u32,
    delta: voxel_edit::VoxelEditDelta,
) {
    let br = HostToClient::Edit {
        seq,
        peer_id,
        delta,
    };
    if let Ok(json) = serde_json::to_string(&br) {
        emit_and_broadcast(collab_mtx, app, &json);
    }
}

async fn handle_host_connection(
    mut ws: WsTcp,
    mut broadcast_rx: broadcast::Receiver<String>,
    app: AppHandle,
    state: Arc<ViewerState>,
    collab_mtx: Arc<std::sync::Mutex<CollabRuntime>>,
    peer_id: u32,
    display_name: String,
    color_rgb: u32,
) {
    let mut roster = collab_mtx.lock().unwrap().roster.clone();
    if let Some(r) = roster.iter_mut().find(|r| r.peer_id == peer_id) {
        r.display_name = display_name;
        r.color_rgb = color_rgb;
    }
    collab_mtx.lock().unwrap().roster = roster.clone();
    let _ = app.emit("collab-roster", serde_json::to_string(&roster).unwrap());

    let snap = {
        let g = state.current_file.lock().unwrap();
        encode_payload_v4(g.as_ref().unwrap()).unwrap()
    };
    let welcome = HostToClient::Welcome {
        peer_id,
        leader_id: HOST_PEER_ID,
        snapshot: snap,
        roster: roster.clone(),
    };
    let _ = ws
        .send(Message::Text(serde_json::to_string(&welcome).unwrap()))
        .await;

    loop {
        tokio::select! {
            biased;
            bmsg = broadcast_rx.recv() => {
                if let Ok(text) = bmsg {
                    let _ = ws.send(Message::Text(text)).await;
                }
            }
            incoming = ws.next() => {
                let Some(msg) = incoming else { break; };
                let Ok(Message::Text(t)) = msg else {
                    continue;
                };
                let Ok(cmd) = serde_json::from_str::<ClientToHost>(&t) else {
                    continue;
                };
                match cmd {
            ClientToHost::Join { .. } => {}
            ClientToHost::Edit { delta } => {
                let allowed = roster
                    .iter()
                    .find(|r| r.peer_id == peer_id)
                    .map(|r| r.can_edit)
                    .unwrap_or(false);
                if !allowed {
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
                let seq = {
                    let mut c = collab_mtx.lock().unwrap();
                    if let Err(e) =
                        host_apply_remote_edit(&app, &state, &mut c, peer_id, delta)
                    {
                        let _ = app.emit("collab-error", e);
                        continue;
                    }
                    c.next_seq
                };
                let br = HostToClient::Edit {
                    seq,
                    peer_id,
                    delta,
                };
                let json = serde_json::to_string(&br).unwrap();
                emit_and_broadcast(&collab_mtx, &app, &json);
            }
            ClientToHost::Undo => {
                let mesh = {
                    let mut c = collab_mtx.lock().unwrap();
                    host_undo_peer(&app, &state, &mut c, peer_id)
                };
                let Ok(Some(d)) = mesh else {
                    continue;
                };
                let seq = collab_mtx.lock().unwrap().next_seq;
                let json = serde_json::to_string(&HostToClient::Edit {
                    seq,
                    peer_id,
                    delta: d,
                })
                .unwrap();
                emit_and_broadcast(&collab_mtx, &app, &json);
            }
            ClientToHost::Redo => {
                let mesh = {
                    let mut c = collab_mtx.lock().unwrap();
                    host_redo_peer(&app, &state, &mut c, peer_id)
                };
                let Ok(Some(d)) = mesh else {
                    continue;
                };
                let seq = collab_mtx.lock().unwrap().next_seq;
                let json = serde_json::to_string(&HostToClient::Edit {
                    seq,
                    peer_id,
                    delta: d,
                })
                .unwrap();
                emit_and_broadcast(&collab_mtx, &app, &json);
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
                if let Ok(g) = collab_mtx.lock() {
                    if let Some(tx) = &g.host_broadcast {
                        let _ = tx.send(json);
                    }
                }
            }
            ClientToHost::Ping { x, y, z } => {
                record_ping_flash(state.as_ref(), peer_id, x, y, z);
                let ping = HostToClient::Ping { peer_id, x, y, z };
                let json = serde_json::to_string(&ping).unwrap();
                let _ = app.emit("collab-ping", &json);
                if let Ok(g) = collab_mtx.lock() {
                    if let Some(tx) = &g.host_broadcast {
                        let _ = tx.send(json);
                    }
                }
            }
            ClientToHost::Camera { presence } => {
                collab_mtx
                    .lock()
                    .unwrap()
                    .presence
                    .insert(peer_id, presence);
                let cam_ev = HostToClient::Camera { peer_id, presence };
                let json = serde_json::to_string(&cam_ev).unwrap();
                let _ = app.emit("collab-camera", &json);
                if let Ok(g) = collab_mtx.lock() {
                    if let Some(tx) = &g.host_broadcast {
                        let _ = tx.send(json);
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
                collab_mtx.lock().unwrap().roster = roster.clone();
                let _ = app.emit("collab-roster", serde_json::to_string(&roster).unwrap());
            }
                }
            }
        }
    }
}

pub fn start_host(
    app: AppHandle,
    state: Arc<ViewerState>,
    collab_mtx: Arc<std::sync::Mutex<CollabRuntime>>,
    port: u16,
) -> Result<String, String> {
    let mut c = collab_mtx.lock().map_err(|e| e.to_string())?;
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
        display_name: "Host".into(),
        color_rgb: 0x4488ff,
        is_leader: true,
        can_edit: true,
    }];
    let (btx, _) = broadcast::channel::<String>(128);
    c.host_broadcast = Some(btx.clone());
    let roster_json = serde_json::to_string(&c.roster).unwrap();
    drop(c);
    let _ = app.emit("collab-roster", roster_json);
    let _ = app.emit("collab-local-peer", HOST_PEER_ID);

    let app2 = app.clone();
    let state2 = Arc::clone(&state);
    let cm2 = Arc::clone(&collab_mtx);
    tauri::async_runtime::spawn(async move {
        let listener = match TcpListener::bind(("0.0.0.0", port)).await {
            Ok(l) => l,
            Err(e) => {
                let _ = app2.emit("collab-error", format!("bind: {e}"));
                return;
            }
        };
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
            {
                let mut g = cm3.lock().unwrap();
                g.roster.push(RosterEntry {
                    peer_id: pid,
                    display_name: dname.clone(),
                    color_rgb: col,
                    is_leader: false,
                    can_edit: false,
                });
            }
            let sub = cm3
                .lock()
                .unwrap()
                .host_broadcast
                .as_ref()
                .unwrap()
                .subscribe();
            tokio::spawn(handle_host_connection(
                ws, sub, app3, st3, cm3, pid, dname, col,
            ));
        }
    });

    let ip = local_ip_address::local_ip()
        .map(|i| i.to_string())
        .unwrap_or_else(|_| "127.0.0.1".into());
    Ok(format!("ws://{ip}:{port}"))
}

pub async fn client_connect_blocking(
    url: &str,
    app: AppHandle,
    state: Arc<ViewerState>,
    collab_mtx: Arc<std::sync::Mutex<CollabRuntime>>,
    display_name: String,
    color_rgb: u32,
) -> Result<(), String> {
    let (ws, _) = connect_async(url)
        .await
        .map_err(|e| format!("connect: {e}"))?;
    let (mut write, mut read) = ws.split();
    let join = serde_json::to_string(&ClientToHost::Join {
        display_name: display_name.clone(),
        color_rgb,
    })
    .unwrap();
    write
        .send(Message::Text(join))
        .await
        .map_err(|e| e.to_string())?;

    let first = read
        .next()
        .await
        .ok_or_else(|| "closed".to_string())?
        .map_err(|e| e.to_string())?;
    let Message::Text(t) = first else {
        return Err("unexpected frame".into());
    };
    let welcome: HostToClient = serde_json::from_str(&t).map_err(|e| e.to_string())?;
    match welcome {
        HostToClient::Welcome {
            peer_id,
            leader_id,
            snapshot,
            roster,
        } => {
            {
                let mut c = collab_mtx.lock().map_err(|e| e.to_string())?;
                c.role = CollabRole::Client;
                c.local_peer_id = peer_id;
                c.leader_id = leader_id;
                c.roster = roster;
            }
            replace_file_on_main(&app, &state, &snapshot)?;
            let _ = app.emit("collab-joined", ());
            let _ = app.emit("collab-local-peer", peer_id);
            let _ = app.emit(
                "collab-roster",
                serde_json::to_string(&collab_mtx.lock().unwrap().roster).unwrap(),
            );
        }
        _ => return Err("expected welcome".into()),
    }

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    collab_mtx.lock().map_err(|e| e.to_string())?.client_tx = Some(tx);

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
        while let Some(msg) = read.next().await {
            let Ok(Message::Text(t)) = msg else {
                continue;
            };
            if let Ok(ev) = serde_json::from_str::<HostToClient>(&t) {
                match ev {
                    HostToClient::Edit { delta, peer_id, .. } => {
                        let local = cm4.lock().unwrap().local_peer_id;
                        if peer_id == local {
                            continue;
                        }
                        let _ = apply_delta_on_main(&app4, &st4, &delta);
                        let _ = app4.emit("collab-edit", t);
                    }
                    HostToClient::Roster { roster } => {
                        cm4.lock().unwrap().roster = roster.clone();
                        let _ = app4.emit("collab-roster", serde_json::to_string(&roster).unwrap());
                    }
                    HostToClient::Snapshot { bytes } => {
                        let _ = replace_file_on_main(&app4, &st4, &bytes);
                    }
                    HostToClient::Chat { .. } => {
                        let _ = app4.emit("collab-chat", t);
                    }
                    HostToClient::Ping {
                        peer_id: pid,
                        x,
                        y,
                        z,
                    } => {
                        record_ping_flash(st4.as_ref(), pid, x, y, z);
                        let _ = app4.emit("collab-ping", t);
                    }
                    HostToClient::Camera { peer_id, presence } => {
                        cm4.lock().unwrap().presence.insert(peer_id, presence);
                        let _ = app4.emit("collab-camera", t);
                    }
                    _ => {
                        let _ = app4.emit("collab-msg", t);
                    }
                }
            }
        }
    });

    Ok(())
}
