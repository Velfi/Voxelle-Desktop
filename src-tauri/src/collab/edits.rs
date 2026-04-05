//! Edit encoding/decoding, per-peer undo stacks, edit broadcast, edit acknowledgment protocol.

use crate::voxel_edit;
use crate::voxelle::{empty_collab_placeholder, encode_payload_v4};
use crate::ViewerState;
use crate::VoxelGpuRefreshReason;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Runtime};
use tokio_tungstenite::tungstenite::Message;

use super::{
    CollabInboxItem, CollabRuntime, HostToClient, RosterEntry, SNAPSHOT_CHUNK_SIZE,
    SNAPSHOT_CHUNK_THRESHOLD, HOST_PEER_ID, MAX_UNDO_PER_PEER,
};

// ── Binary edit framing ─────────────────────────────────────────────────────

/// Binary frame: client → host edit deltas (bincode `Vec<VoxelEditDelta>`).
const BIN_TAG_CLIENT_EDIT: u8 = 0x45; // 'E'
/// Binary frame: host → client edit broadcast (bincode `(u64, u32, Vec<VoxelEditDelta>)`).
const BIN_TAG_HOST_EDIT: u8 = 0x48; // 'H'

pub fn encode_client_edit_binary(deltas: &[voxel_edit::VoxelEditDelta]) -> Vec<u8> {
    let payload = bincode::serialize(deltas).expect("bincode serialize");
    let mut buf = Vec::with_capacity(1 + payload.len());
    buf.push(BIN_TAG_CLIENT_EDIT);
    buf.extend_from_slice(&payload);
    buf
}

pub(super) fn decode_client_edit_binary(data: &[u8]) -> Option<Vec<voxel_edit::VoxelEditDelta>> {
    if data.first() != Some(&BIN_TAG_CLIENT_EDIT) {
        return None;
    }
    bincode::deserialize(&data[1..]).ok()
}

pub(super) fn encode_host_edit_binary(
    seq: u64,
    peer_id: u32,
    deltas: &[voxel_edit::VoxelEditDelta],
) -> Vec<u8> {
    let payload = bincode::serialize(&(seq, peer_id, deltas)).expect("bincode serialize");
    let mut buf = Vec::with_capacity(1 + payload.len());
    buf.push(BIN_TAG_HOST_EDIT);
    buf.extend_from_slice(&payload);
    buf
}

pub(super) fn decode_host_edit_binary(
    data: &[u8],
) -> Option<(u64, u32, Vec<voxel_edit::VoxelEditDelta>)> {
    if data.first() != Some(&BIN_TAG_HOST_EDIT) {
        return None;
    }
    bincode::deserialize(&data[1..]).ok()
}

pub(super) fn broadcast_edit_binary(
    collab: &Mutex<CollabRuntime>,
    seq: u64,
    peer_id: u32,
    deltas: &[voxel_edit::VoxelEditDelta],
) {
    let bin = encode_host_edit_binary(seq, peer_id, deltas);
    let g = collab.lock();
    if let Some(tx) = &g.host_broadcast {
        let _ = tx.send(Message::Binary(bin));
    }
}

/// Host local edit after GPU sync: notify guests + UI.
pub fn host_emit_edit_batch<R: Runtime>(
    collab_mtx: &Mutex<CollabRuntime>,
    _app: &AppHandle<R>,
    seq: u64,
    peer_id: u32,
    deltas: &[voxel_edit::VoxelEditDelta],
) {
    broadcast_edit_binary(collab_mtx, seq, peer_id, deltas);
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
                    let mut fg = state.file.current_file.lock();
                    let mut vm = state.file.voxel_map.lock();
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
                let undo = c.host_undo.entry(peer_id).or_default();
                undo.push(deltas.clone());
                if undo.len() > MAX_UNDO_PER_PEER {
                    let excess = undo.len() - MAX_UNDO_PER_PEER;
                    undo.drain(0..excess);
                }
                c.host_redo.remove(&peer_id);
                c.next_seq
            };
            broadcast_edit_binary(collab_mtx, seq, peer_id, &deltas);
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
                let mut fg = state.file.current_file.lock();
                let mut vm = state.file.voxel_map.lock();
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
            broadcast_edit_binary(collab_mtx, seq, peer_id, &mesh_refresh);
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
                let mut fg = state.file.current_file.lock();
                let mut vm = state.file.voxel_map.lock();
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
            broadcast_edit_binary(collab_mtx, seq, peer_id, &forward);
        }
    }
}

pub(super) fn flush_edit_batch<R: Runtime>(
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
        let mut fg = state.file.current_file.lock();
        let mut vm = state.file.voxel_map.lock();
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
            let undo = c.host_undo.entry(peer_id).or_default();
            undo.push(deltas.clone());
            if undo.len() > MAX_UNDO_PER_PEER {
                let excess = undo.len() - MAX_UNDO_PER_PEER;
                undo.drain(0..excess);
            }
            c.host_redo.remove(&peer_id);
            c.next_seq
        };
        broadcast_edit_binary(collab_mtx, seq, peer_id, &deltas);
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
    let bytes: Result<Vec<u8>, String> = {
        let g = state.file.current_file.lock();
        let file = g.as_ref().cloned().unwrap_or_else(empty_collab_placeholder);
        encode_payload_v4(&file).map_err(|e| e.to_string())
    };
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
            avatar_names: HashMap::new(),
            avatar_data: HashMap::new(),
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

pub(super) fn host_remove_peer_from_session<R: Runtime>(
    app: &AppHandle<R>,
    collab_mtx: &Arc<Mutex<CollabRuntime>>,
    peer_id: u32,
    notify: Option<super::CollabPeerLeftKind>,
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
        g.avatar_names.remove(&peer_id);
        g.host_undo.remove(&peer_id);
        g.host_redo.remove(&peer_id);
        (g.roster.clone(), display_name)
    };
    if let Some(kind) = notify {
        let reason = match kind {
            super::CollabPeerLeftKind::Left => "left",
            super::CollabPeerLeftKind::Disconnected => "disconnected",
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

pub(super) fn touch_guest_activity(collab_mtx: &Mutex<CollabRuntime>, peer_id: u32) {
    if peer_id == HOST_PEER_ID {
        return;
    }
    collab_mtx
        .lock()
        .guest_last_activity
        .insert(peer_id, std::time::Instant::now());
}

/// Brief permission check — does `peer_id` have `can_edit` in the roster?
pub(super) fn check_can_edit(collab_mtx: &Mutex<CollabRuntime>, peer_id: u32) -> bool {
    collab_mtx
        .lock()
        .roster
        .iter()
        .find(|r| r.peer_id == peer_id)
        .map(|r| r.can_edit)
        .unwrap_or(false)
}
