//! Helpers called from the frame loop (`RunEvent::MainEventsCleared`) and from
//! various Tauri commands.  Extracted from `lib.rs` for readability.

use crate::camera::OrbitCamera;
use crate::collab;
use crate::greedy_mesh;
use crate::load_pipeline;
use crate::render::{GpuPeerLabel, WgpuViewer};
use crate::state::*;
use crate::voxel_edit;
use crate::voxelle;

use ahash::{AHashMap, AHashSet, AHasher};
use std::hash::{Hash, Hasher};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Runtime};

#[cfg(desktop)]
use crate::native_menu::SelectionMenuState;
#[cfg(desktop)]
use tauri::Manager;

// ---------------------------------------------------------------------------
// Work-progress helpers
// ---------------------------------------------------------------------------

pub(crate) fn emit_work_progress<R: Runtime>(
    app: &AppHandle<R>,
    fraction: f32,
    phase: impl Into<String>,
) {
    let _ = app.emit(
        "voxelle-work-progress",
        load_pipeline::LoadProgressPayload {
            fraction: fraction.clamp(0.0, 1.0),
            phase: phase.into(),
        },
    );
}

/// When armed, [`Drop`] emits 100% work progress so the status bar clears after `?` early returns too.
pub(crate) struct WorkProgressGuard<'a, R: Runtime> {
    app: &'a AppHandle<R>,
    armed: bool,
}

impl<'a, R: Runtime> WorkProgressGuard<'a, R> {
    pub fn new(app: &'a AppHandle<R>) -> Self {
        Self { app, armed: false }
    }

    pub fn arm(&mut self) {
        self.armed = true;
    }
}

impl<R: Runtime> Drop for WorkProgressGuard<'_, R> {
    fn drop(&mut self) {
        if self.armed {
            emit_work_progress(self.app, 1.0, "");
        }
    }
}

// ---------------------------------------------------------------------------
// VoxelGpuRefreshReason
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub(crate) enum VoxelGpuRefreshReason {
    SoloEdit,
    Undo,
    Redo,
    CollabApply,
}

impl VoxelGpuRefreshReason {
    pub(crate) fn label(self) -> &'static str {
        match self {
            VoxelGpuRefreshReason::SoloEdit => "Applying edit…",
            VoxelGpuRefreshReason::Undo => "Undo…",
            VoxelGpuRefreshReason::Redo => "Redo…",
            VoxelGpuRefreshReason::CollabApply => "Applying remote edit…",
        }
    }
}

// ---------------------------------------------------------------------------
// Menu helpers
// ---------------------------------------------------------------------------

pub(crate) fn scene_menu_flags(state: &ViewerState) -> (bool, bool, bool) {
    let file = state.file.current_file.lock();
    let has_project = file.is_some();
    let has_voxels = file.as_ref().map(|f| !f.voxels.is_empty()).unwrap_or(false);
    drop(file);
    let has_selection = !state.selection.selection_cells.lock().is_empty();
    (has_project, has_voxels, has_selection)
}

/// Disables Selection menu entries when there are no voxels and/or no active selection (same rules as web).
/// Does not lock [`ViewerState`]: pass [`scene_menu_flags`] (or explicit booleans) so callers never
/// nest this under `viewer` / `current_file` guards.
#[cfg(desktop)]
pub(crate) fn selection_menu_sync_enabled_for_scene<R: Runtime>(
    app: &AppHandle<R>,
    has_project: bool,
    has_voxels: bool,
    has_selection: bool,
) {
    let Some(menu) = app.try_state::<SelectionMenuState>() else {
        return;
    };

    let apply = |item: &tauri::menu::MenuItem<tauri::Wry>, enabled: bool| {
        let _ = item.set_enabled(enabled);
    };

    apply(&menu.save, has_project);
    apply(&menu.save_as, has_project);
    apply(&menu.close_project, has_project);
    apply(&menu.sel_all, has_voxels);
    apply(&menu.sel_by_color, has_voxels);
    apply(&menu.sel_connected, has_selection);
    apply(&menu.sel_coplanar, has_voxels);
    apply(&menu.sel_coplanar_empty, has_voxels);
    apply(&menu.sel_grow, has_selection);
    apply(&menu.sel_shrink, has_selection);
    apply(&menu.sel_invert, has_voxels);
    apply(&menu.sel_deselect_all, has_selection);
    apply(&menu.sel_deselect_inner, has_selection);
    apply(&menu.sel_deselect_voxels, has_selection);
    apply(&menu.sel_deselect_empty, has_selection);
}

// ---------------------------------------------------------------------------
// Gizmo helpers
// ---------------------------------------------------------------------------

/// Returns the pending visual offset for the selection during a move drag, or `(0,0,0)`.
pub(crate) fn pending_gizmo_translate(state: &ViewerState) -> (i32, i32, i32) {
    match &*state.gizmos.selection_gizmo_drag.lock() {
        SelectionGizmoDrag::Move {
            pending_dx,
            pending_dy,
            pending_dz,
            ..
        } => (*pending_dx, *pending_dy, *pending_dz),
        _ => (0, 0, 0),
    }
}

/// Returns the axis index (0=X, 1=Y, 2=Z) to highlight, or 255 for none.
/// During an active drag the dragged axis stays highlighted; otherwise falls back to hover state.
pub(crate) fn gizmo_highlighted_axis(state: &ViewerState) -> u8 {
    match &*state.gizmos.selection_gizmo_drag.lock() {
        SelectionGizmoDrag::Move { world_axis, .. } => *world_axis,
        SelectionGizmoDrag::Rotate { ring, .. } => *ring,
        SelectionGizmoDrag::Scale { .. } => 255, // no axis highlight for scale ring
        SelectionGizmoDrag::None => state.gizmos.hovered_gizmo_axis.load(Ordering::Relaxed),
    }
}

pub(crate) fn sync_gizmo_gpu(viewer: &mut WgpuViewer, state: &ViewerState, cam: &OrbitCamera) {
    let mode = *state.preview.preview_mode.lock();
    let gen_center = *state.gizmos.generator_gizmo_center.lock();
    let sel = state.selection.selection_cells.lock();
    if gen_center.is_none()
        && (sel.is_empty() || matches!(mode, PreviewMode::Stamp | PreviewMode::Punch))
    {
        drop(sel);
        viewer.upload_gizmo_lines(&[]);
        viewer.upload_gizmo_tris(&[]);
        viewer.upload_gizmo_delta_label(None);
        return;
    }

    let pending = pending_gizmo_translate(state);
    let pivot = if let Some([gx, gy, gz]) = gen_center {
        drop(sel);
        glam::Vec3::new(
            gx + pending.0 as f32,
            gy + pending.1 as f32,
            gz + pending.2 as f32,
        )
    } else {
        let mut min_x = i32::MAX;
        let mut max_x = i32::MIN;
        let mut min_y = i32::MAX;
        let mut max_y = i32::MIN;
        let mut min_z = i32::MAX;
        let mut max_z = i32::MIN;
        for &(x, y, z) in sel.iter() {
            if x < min_x {
                min_x = x;
            }
            if x > max_x {
                max_x = x;
            }
            if y < min_y {
                min_y = y;
            }
            if y > max_y {
                max_y = y;
            }
            if z < min_z {
                min_z = z;
            }
            if z > max_z {
                max_z = z;
            }
        }
        drop(sel);
        glam::Vec3::new(
            (min_x + max_x) as f32 * 0.5 + pending.0 as f32,
            (min_y + max_y) as f32 * 0.5 + pending.1 as f32,
            (min_z + max_z) as f32 * 0.5 + pending.2 as f32,
        )
    };
    let inv_view = cam.view_matrix().inverse();
    let cam_eye = glam::Vec3::new(inv_view.w_axis.x, inv_view.w_axis.y, inv_view.w_axis.z);
    let dist = (cam_eye - pivot).length().max(1.0);
    let arm = (dist * 0.13_f32).clamp(1.5, 20.0);

    let is_extrude = matches!(mode, PreviewMode::SelectExtrude);

    // Axis colors in linear space (HDR target): X=red, Y=green, Z=blue
    let highlight_axis = if is_extrude {
        state.gizmos.hovered_extrude_axis.load(Ordering::Relaxed)
    } else {
        gizmo_highlighted_axis(state)
    };
    let mut cols: [[f32; 3]; 3] = [[1.00, 0.22, 0.22], [0.18, 0.88, 0.18], [0.22, 0.45, 1.00]];
    if highlight_axis < 3 {
        let c = &mut cols[highlight_axis as usize];
        c[0] *= 1.6;
        c[1] *= 1.6;
        c[2] *= 1.6;
    }
    let dirs = [
        glam::Vec3::X,
        -glam::Vec3::X,
        glam::Vec3::Y,
        -glam::Vec3::Y,
        glam::Vec3::Z,
        -glam::Vec3::Z,
    ];
    // Perpendicular basis vectors for each axis's arrowhead cone and ring
    let perps: [(glam::Vec3, glam::Vec3); 3] = [
        (glam::Vec3::Y, glam::Vec3::Z),
        (glam::Vec3::X, glam::Vec3::Z),
        (glam::Vec3::X, glam::Vec3::Y),
    ];
    let cone_h = arm * 0.14;
    let cone_r = arm * 0.055;
    // World-space half-widths for billboard quads
    let shaft_hw = arm * 0.018;
    let ring_hw = arm * 0.014;

    // lv is now TriangleList (camera-facing quads) — same buffer, topology changed in pipeline
    let mut lv: Vec<f32> = Vec::with_capacity(256 * 6);
    let mut tv: Vec<f32> = Vec::with_capacity(72 * 6);

    let pv = |buf: &mut Vec<f32>, p: glam::Vec3, c: [f32; 3]| {
        buf.extend_from_slice(&[p.x, p.y, p.z, c[0], c[1], c[2]]);
    };

    // Emit a camera-facing ribbon quad for the segment p0→p1.
    let quad = |buf: &mut Vec<f32>, p0: glam::Vec3, p1: glam::Vec3, hw: f32, c: [f32; 3]| {
        let seg = (p1 - p0).normalize_or_zero();
        let to_cam = (cam_eye - (p0 + p1) * 0.5).normalize_or_zero();
        let right = seg.cross(to_cam).normalize_or_zero() * hw;
        // Two triangles forming a quad
        buf.extend_from_slice(&[
            (p0 - right).x,
            (p0 - right).y,
            (p0 - right).z,
            c[0],
            c[1],
            c[2],
            (p0 + right).x,
            (p0 + right).y,
            (p0 + right).z,
            c[0],
            c[1],
            c[2],
            (p1 + right).x,
            (p1 + right).y,
            (p1 + right).z,
            c[0],
            c[1],
            c[2],
            (p0 - right).x,
            (p0 - right).y,
            (p0 - right).z,
            c[0],
            c[1],
            c[2],
            (p1 + right).x,
            (p1 + right).y,
            (p1 + right).z,
            c[0],
            c[1],
            c[2],
            (p1 - right).x,
            (p1 - right).y,
            (p1 - right).z,
            c[0],
            c[1],
            c[2],
        ]);
    };

    for (i, &dir) in dirs.iter().enumerate() {
        let axis = i / 2;
        let col = cols[axis];
        let tip = pivot + dir * arm;
        let (u, v_ax) = perps[axis];

        if is_extrude {
            // Extrude style: thicker shaft + sphere ball tip
            let ball_r = cone_r * 1.6;
            let ball_center = tip;
            quad(
                &mut lv,
                pivot,
                ball_center - dir * ball_r,
                shaft_hw * 1.6,
                col,
            );
            const N_LON: usize = 8;
            const N_LAT: usize = 6;
            for lat in 0..N_LAT {
                let t0 = lat as f32 * std::f32::consts::PI / N_LAT as f32;
                let t1 = (lat + 1) as f32 * std::f32::consts::PI / N_LAT as f32;
                for lon in 0..N_LON {
                    let p0 = lon as f32 * 2.0 * std::f32::consts::PI / N_LON as f32;
                    let p1 = (lon + 1) as f32 * 2.0 * std::f32::consts::PI / N_LON as f32;
                    let sph = |t: f32, p: f32| -> glam::Vec3 {
                        ball_center
                            + (u * (t.sin() * p.cos()) + v_ax * (t.sin() * p.sin()) + dir * t.cos())
                                * ball_r
                    };
                    let (a, b, c, d) = (sph(t0, p0), sph(t0, p1), sph(t1, p0), sph(t1, p1));
                    pv(&mut tv, a, col);
                    pv(&mut tv, c, col);
                    pv(&mut tv, d, col);
                    pv(&mut tv, a, col);
                    pv(&mut tv, d, col);
                    pv(&mut tv, b, col);
                }
            }
        } else {
            // Translate style: thin shaft + pyramid arrowhead
            quad(&mut lv, pivot, tip, shaft_hw, col);
            // Pyramid arrowhead (4 triangles + 2-triangle base cap)
            let base_c = tip - dir * cone_h;
            let base = [
                base_c + u * cone_r,
                base_c + v_ax * cone_r,
                base_c - u * cone_r,
                base_c - v_ax * cone_r,
            ];
            for j in 0..4usize {
                pv(&mut tv, tip, col);
                pv(&mut tv, base[j], col);
                pv(&mut tv, base[(j + 1) % 4], col);
            }
            pv(&mut tv, base[0], col);
            pv(&mut tv, base[1], col);
            pv(&mut tv, base[2], col);
            pv(&mut tv, base[0], col);
            pv(&mut tv, base[2], col);
            pv(&mut tv, base[3], col);
        }
    }

    // Rotation rings — only for the selection translate gizmo, not extrude or generator-override gizmos
    if !is_extrude && gen_center.is_none() {
        const RING_N: usize = 24;
        let ring_r = arm * 0.72;
        for ring in 0..3usize {
            let col = cols[ring];
            let (u, v_ax) = perps[ring];
            for i in 0..RING_N {
                let a0 = i as f32 * 2.0 * std::f32::consts::PI / RING_N as f32;
                let a1 = (i + 1) as f32 * 2.0 * std::f32::consts::PI / RING_N as f32;
                let p0 = pivot + (u * a0.cos() + v_ax * a0.sin()) * ring_r;
                let p1 = pivot + (u * a1.cos() + v_ax * a1.sin()) * ring_r;
                quad(&mut lv, p0, p1, ring_hw, col);
            }
        }
    }

    // Scale ring — camera-facing circle at the joint radius (bone tool).
    if let Some(radius) = *state.gizmos.generator_gizmo_ring_radius.lock() {
        const RING_N: usize = 32;
        let ring_col = [1.0_f32, 0.7, 0.2]; // orange/gold
                                            // Camera-facing ring: use view-space right/up as the ring plane.
        let cam_right = inv_view.x_axis.truncate().normalize();
        let cam_up = inv_view.y_axis.truncate().normalize();
        let r = radius.max(dist * 0.015); // minimum screen size
        for i in 0..RING_N {
            let a0 = i as f32 * 2.0 * std::f32::consts::PI / RING_N as f32;
            let a1 = (i + 1) as f32 * 2.0 * std::f32::consts::PI / RING_N as f32;
            let p0 = pivot + (cam_right * a0.cos() + cam_up * a0.sin()) * r;
            let p1 = pivot + (cam_right * a1.cos() + cam_up * a1.sin()) * r;
            quad(&mut lv, p0, p1, ring_hw * 1.5, ring_col);
        }
    }

    viewer.upload_gizmo_lines(&lv);
    viewer.upload_gizmo_tris(&tv);

    if is_extrude {
        // Show extrude depth label at the tip of the active axis handle.
        let drag_info = match &*state.gizmos.extrude_gizmo_drag.lock() {
            ExtrudeGizmoDrag::Drag {
                depth,
                world_axis,
                positive,
                ..
            } if *depth != 0 => Some((*depth, *world_axis, *positive)),
            _ => None,
        };
        if let Some((d, axis, pos)) = drag_info {
            let text = format!("{:+}", d);
            let dir = match (axis, pos) {
                (0, true) => glam::Vec3::X,
                (0, false) => -glam::Vec3::X,
                (1, true) => glam::Vec3::Y,
                (1, false) => -glam::Vec3::Y,
                (2, true) => glam::Vec3::Z,
                (2, false) => -glam::Vec3::Z,
                _ => glam::Vec3::X,
            };
            let tip = pivot + dir * arm;
            let (vw, vh) = viewer.viewport_size();
            if let Some((sx, sy)) =
                voxel_edit::world_to_viewport_pixels(cam, vw as f32, vh as f32, tip.x, tip.y, tip.z)
            {
                viewer.upload_gizmo_delta_label(Some(GpuPeerLabel {
                    name: text,
                    color_rgb: 0x9FD8FF,
                    x: sx,
                    y: sy,
                }));
            } else {
                viewer.upload_gizmo_delta_label(None);
            }
        } else {
            viewer.upload_gizmo_delta_label(None);
        }
    } else if pending != (0, 0, 0) {
        let text = format!("{:+}, {:+}, {:+}", pending.0, pending.1, pending.2);
        let (vw, vh) = viewer.viewport_size();
        if let Some((sx, sy)) = voxel_edit::world_to_viewport_pixels(
            cam, vw as f32, vh as f32, pivot.x, pivot.y, pivot.z,
        ) {
            viewer.upload_gizmo_delta_label(Some(GpuPeerLabel {
                name: text,
                color_rgb: 0x9FD8FF,
                x: sx,
                y: sy,
            }));
        } else {
            viewer.upload_gizmo_delta_label(None);
        }
    } else {
        viewer.upload_gizmo_delta_label(None);
    }
}

// ---------------------------------------------------------------------------
// Ping flash
// ---------------------------------------------------------------------------

pub(crate) fn sync_ping_flash(viewer: &mut WgpuViewer, state: &ViewerState, cam: &OrbitCamera) {
    let snap = state.ping_flash.lock().clone();
    let Some(f) = snap else {
        viewer.clear_ping_mesh();
        viewer.clear_ping_label();
        return;
    };
    if std::time::Instant::now() > f.until {
        *state.ping_flash.lock() = None;
        viewer.clear_ping_mesh();
        viewer.clear_ping_label();
        return;
    }
    let r = ((f.color_rgb >> 16) & 0xff) as f32 / 255.0;
    let g = ((f.color_rgb >> 8) & 0xff) as f32 / 255.0;
    let b = (f.color_rgb & 0xff) as f32 / 255.0;
    let cx = f.x as f32 + 0.5;
    let cy = f.y as f32 + 0.5;
    let cz = f.z as f32 + 0.5;
    let solid = greedy_mesh::preview_cube_mesh(cx, cy, cz, 0.52, [r, g, b], 1.0);
    let wire = greedy_mesh::preview_cube_wireframe_mesh(
        cx,
        cy,
        cz,
        0.52,
        [r * 0.25, g * 0.25, b * 0.25],
        2.0,
    );
    viewer.upload_ping_mesh(&solid, &wire);
    let elapsed = f.started.elapsed().as_secs_f32();
    let wave_verts = greedy_mesh::ping_ripple_line_vertices(f.x, f.y, f.z, elapsed, [r, g, b]);
    viewer.upload_ping_wave_lines(&wave_verts);

    // Project ping world position to screen for GPU text label
    let (w, h) = {
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let label_text = if f.emoji.is_empty() {
        f.display_name.clone()
    } else if f.display_name.is_empty() {
        f.emoji.clone()
    } else {
        format!("{} {}", f.emoji, f.display_name)
    };
    if w > 0.0 && h > 0.0 && !label_text.is_empty() {
        if let Some((sx, sy)) = voxel_edit::world_to_viewport_pixels(cam, w, h, cx, cy, cz) {
            viewer.upload_ping_label(GpuPeerLabel {
                name: label_text,
                color_rgb: f.color_rgb,
                x: sx,
                y: sy,
            });
        } else {
            viewer.clear_ping_label();
        }
    } else {
        viewer.clear_ping_label();
    }
}

// ---------------------------------------------------------------------------
// Collab presence
// ---------------------------------------------------------------------------

pub(crate) fn lerp_presence(
    smooth: &collab::CameraPresence,
    target: &collab::CameraPresence,
    t: f32,
) -> collab::CameraPresence {
    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }
    collab::CameraPresence {
        target: [
            lerp(smooth.target[0], target.target[0], t),
            lerp(smooth.target[1], target.target[1], t),
            lerp(smooth.target[2], target.target[2], t),
        ],
        radius: lerp(smooth.radius, target.radius, t),
        theta: lerp(smooth.theta, target.theta, t),
        phi: lerp(smooth.phi, target.phi, t),
        perspective: target.perspective,
        fov_y: lerp(smooth.fov_y, target.fov_y, t),
        ortho_half_height: lerp(smooth.ortho_half_height, target.ortho_half_height, t),
    }
}

/// Build screen-space peer labels and upload to the GPU renderer (replaces IPC polling).
pub(crate) fn sync_collab_peer_labels(
    viewer: &mut WgpuViewer,
    state: &ViewerState,
    cam: &OrbitCamera,
) {
    let (w, h) = {
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    if w <= 0.0 || h <= 0.0 {
        viewer.clear_peer_labels();
        return;
    }
    let c = state.collab.lock();
    if !c.is_active() {
        viewer.clear_peer_labels();
        return;
    }
    let local_id = c.local_peer_id;
    let roster = c.roster.clone();
    drop(c);

    let smooth = state.smooth_presence.lock();
    let mut labels = Vec::new();
    for (pid, pr) in smooth.iter() {
        if *pid == local_id {
            continue;
        }
        let eye = collab::presence_eye(pr);
        let Some((sx, sy)) = voxel_edit::world_to_viewport_pixels(cam, w, h, eye.x, eye.y, eye.z)
        else {
            continue;
        };
        let entry = roster.iter().find(|r| r.peer_id == *pid);
        let name = entry.map(|r| r.display_name.clone()).unwrap_or_default();
        let color_rgb = entry.map(|r| r.color_rgb).unwrap_or(0x888888);
        labels.push(GpuPeerLabel {
            name,
            color_rgb,
            x: sx,
            y: sy,
        });
    }
    viewer.upload_peer_labels(labels);
}

pub(crate) fn sync_collab_peer_avatars(
    viewer: &mut WgpuViewer,
    state: &Arc<ViewerState>,
    cam: &crate::camera::OrbitCamera,
) {
    const SMOOTH_T: f32 = 0.12;

    let (local_id, roster, presence, avatar_names, avatar_data) = {
        let c = state.collab.lock();
        if !c.is_active() {
            viewer.clear_avatar_peers();
            state.smooth_presence.lock().clear();
            return;
        }
        (
            c.local_peer_id,
            c.roster.clone(),
            c.presence.clone(),
            c.avatar_names.clone(),
            c.avatar_data.clone(),
        )
    };

    // Lerp smooth presence toward raw presence each frame.
    let mut smooth = state.smooth_presence.lock();
    smooth.retain(|pid, _| presence.contains_key(pid));
    for (&pid, raw) in &presence {
        let entry = smooth.entry(pid).or_insert(*raw);
        *entry = lerp_presence(entry, raw, SMOOTH_T);
    }

    // Remove GPU entries for peers that left.
    let present_ids: Vec<u32> = smooth.keys().copied().collect();
    for id in viewer
        .avatar_peers
        .iter()
        .map(|p| p.peer_id)
        .collect::<Vec<_>>()
    {
        if !present_ids.contains(&id) {
            viewer.remove_avatar_peer(id);
        }
    }

    // Build view-proj from the local camera.
    let (vw, vh) = viewer.viewport_size();
    let vp = cam.proj_matrix(vw.max(1) as f32, vh.max(1) as f32) * cam.view_matrix();

    for (&pid, pr) in smooth.iter() {
        if pid == local_id {
            continue;
        }

        let color = roster
            .iter()
            .find(|r| r.peer_id == pid)
            .map(|r| r.color_rgb)
            .unwrap_or(0x6688cc);
        let rf = ((color >> 16) & 0xff) as f32 / 255.0;
        let gf = ((color >> 8) & 0xff) as f32 / 255.0;
        let bf = (color & 0xff) as f32 / 255.0;

        let eye = collab::presence_eye(pr);
        let target = glam::Vec3::new(pr.target[0], pr.target[1], pr.target[2]);
        if (target - eye).length_squared() < 1e-8 {
            continue;
        }

        let avatar_name = avatar_names.get(&pid).cloned().unwrap_or_default();

        // Tint: peer accent color for default glow dot; neutral white for named avatars.
        let tint = if avatar_name.is_empty() {
            [rf, gf, bf]
        } else {
            [1.0_f32, 1.0, 1.0]
        };

        // Build orientation: avatar faces the viewer (away from peer's look target).
        // forward = peer's look direction; we store -forward as the Z column so the
        // rotation matrix has det=+1 (proper rotation, no winding flip).  The greedy
        // mesh has outward CCW faces in right-handed local space, so det=+1 is required
        // to preserve winding and keep normals pointing outward.
        let forward = (target - eye).normalize();
        let ref_up = if forward.y.abs() > 0.999 {
            glam::Vec3::Z
        } else {
            glam::Vec3::Y
        };
        let right = forward.cross(ref_up).normalize();
        let up = right.cross(forward);
        // Columns [right, up, -forward]: local +Z maps to -forward so the avatar faces
        // away from the peer's look target (toward the viewer).  det=+1 preserves CCW
        // winding of the greedy mesh, keeping normals pointing outward.
        let rot = glam::Mat4::from_cols(
            right.extend(0.0),
            up.extend(0.0),
            (-forward).extend(0.0),
            glam::Vec4::W,
        );
        // 90° CCW local-space yaw fix: avatar mesh is oriented 90° off.
        // rot * R_y(π/2) remaps columns to [forward, up, right].
        let rot_fixed = rot * glam::Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2);

        // Look up mesh scale/offset from cache (fall back to default glow dot).
        let (scale, center_offset) = viewer
            .avatar_mesh_cache
            .get(&avatar_name)
            .or_else(|| viewer.avatar_mesh_cache.get(""))
            .map(|m| (m.scale, m.center_offset))
            .unwrap_or((1.5, glam::Vec3::ZERO));

        let model = glam::Mat4::from_translation(eye)
            * rot_fixed
            * glam::Mat4::from_scale(glam::Vec3::splat(scale))
            * glam::Mat4::from_translation(center_offset);
        let mvp = vp * model;

        // Extract rotation columns for the normal matrix (std140: each column padded to vec4).
        // After the 90° fix, columns are [forward, up, right].
        let rot_cols = [
            [forward.x, forward.y, forward.z, 0.0],
            [up.x, up.y, up.z, 0.0],
            [right.x, right.y, right.z, 0.0],
        ];

        viewer.update_avatar_peer(
            pid,
            avatar_name.clone(),
            mvp.to_cols_array_2d(),
            tint,
            rot_cols,
        );

        // If the named mesh isn't cached yet, kick off a background load.
        if !avatar_name.is_empty() && !viewer.avatar_mesh_cache.contains_key(&avatar_name) {
            if super::embedded_avatar_bytes(&avatar_name).is_some() {
                super::spawn_load_avatar_mesh(Arc::clone(state), &avatar_name);
            } else if let Some(bytes) = avatar_data.get(&avatar_name) {
                super::spawn_load_avatar_from_bytes(
                    Arc::clone(state),
                    avatar_name.clone(),
                    bytes.clone(),
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Overlay caching helpers
// ---------------------------------------------------------------------------

fn selection_overlay_cache_fingerprint(
    sel: &AHashSet<greedy_mesh::VoxelCoord>,
    mesh_gen: u64,
    pending_offset: (i32, i32, i32),
) -> u64 {
    let mut h = AHasher::default();
    sel.len().hash(&mut h);
    let mut v: Vec<_> = sel.iter().copied().collect();
    v.sort_unstable();
    for c in v {
        c.hash(&mut h);
    }
    mesh_gen.hash(&mut h);
    pending_offset.hash(&mut h);
    h.finish()
}

fn grid_border_overlay_cache_fingerprint(
    world: &AHashMap<greedy_mesh::VoxelCoord, voxelle::Voxel>,
    mesh_gen: u64,
) -> u64 {
    let mut h = AHasher::default();
    world.len().hash(&mut h);
    let mut v: Vec<_> = world.keys().copied().collect();
    v.sort_unstable();
    for c in v {
        c.hash(&mut h);
    }
    mesh_gen.hash(&mut h);
    h.finish()
}

// ---------------------------------------------------------------------------
// Grid border overlay
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) enum GridBorderPrepared {
    Clear,
    Unchanged,
    Draw {
        fp: u64,
        verts: Vec<f32>,
        indices: Vec<u32>,
    },
}

pub(crate) fn prepare_grid_border_overlay(state: &ViewerState) -> GridBorderPrepared {
    let show = state.gpu.show_grid_borders.load(Ordering::Relaxed);
    if !show {
        return GridBorderPrepared::Clear;
    }
    let mesh_gen = state.gpu.mesh_refresh_generation.load(Ordering::Relaxed);
    let file_guard = state.file.current_file.lock();
    let map_guard = state.file.voxel_map.lock();
    let Some(file) = file_guard.as_ref() else {
        return GridBorderPrepared::Clear;
    };
    let Some(vmap) = map_guard.as_ref() else {
        return GridBorderPrepared::Clear;
    };
    let mut world: AHashMap<greedy_mesh::VoxelCoord, voxelle::Voxel> =
        AHashMap::with_capacity(vmap.len());
    for (coord, &idx) in vmap.iter() {
        world.insert(*coord, file.voxels[idx]);
    }
    drop(file_guard);
    drop(map_guard);

    let fp = grid_border_overlay_cache_fingerprint(&world, mesh_gen);
    if *state.gpu.grid_overlay_cache_key.lock() == Some(fp) {
        return GridBorderPrepared::Unchanged;
    }
    let (verts, indices) = greedy_mesh::voxel_surface_grid_line_vertices(&world);
    GridBorderPrepared::Draw { fp, verts, indices }
}

pub(crate) fn apply_grid_border_overlay(
    viewer: &mut WgpuViewer,
    state: &ViewerState,
    prep: GridBorderPrepared,
) {
    match prep {
        GridBorderPrepared::Clear => {
            viewer.clear_grid_border_lines();
            *state.gpu.grid_overlay_cache_key.lock() = None;
        }
        GridBorderPrepared::Unchanged => {}
        GridBorderPrepared::Draw { fp, verts, indices } => {
            if viewer.grid_border_cache_key == Some(fp) {
                return;
            }
            viewer.upload_grid_border_lines(&verts, &indices);
            viewer.grid_border_cache_key = Some(fp);
            *state.gpu.grid_overlay_cache_key.lock() = Some(fp);
        }
    }
}

// ---------------------------------------------------------------------------
// Selection overlay
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) enum SelectionOverlayPrepared {
    Clear,
    Unchanged,
    Draw {
        fp: u64,
        solid: greedy_mesh::MeshBuffers,
        line_verts: Vec<f32>,
    },
}

pub(crate) fn prepare_selection_overlay(state: &ViewerState) -> SelectionOverlayPrepared {
    let sel = state.selection.selection_cells.lock().clone();
    if sel.is_empty() {
        return SelectionOverlayPrepared::Clear;
    }
    let mesh_gen = state.gpu.mesh_refresh_generation.load(Ordering::Relaxed);
    let pending = pending_gizmo_translate(state);
    let fp = selection_overlay_cache_fingerprint(&sel, mesh_gen, pending);
    if *state.gpu.selection_overlay_cache_key.lock() == Some(fp) {
        return SelectionOverlayPrepared::Unchanged;
    }
    // Apply the pending drag offset to cell positions for preview rendering.
    let effective_sel: AHashSet<greedy_mesh::VoxelCoord> = if pending != (0, 0, 0) {
        sel.iter()
            .map(|&(x, y, z)| (x + pending.0, y + pending.1, z + pending.2))
            .collect()
    } else {
        sel
    };
    let file_guard = state.file.current_file.lock();
    let map_guard = state.file.voxel_map.lock();
    let Some(file) = file_guard.as_ref() else {
        return SelectionOverlayPrepared::Clear;
    };
    let Some(vmap) = map_guard.as_ref() else {
        return SelectionOverlayPrepared::Clear;
    };
    let mut world: AHashMap<greedy_mesh::VoxelCoord, voxelle::Voxel> =
        AHashMap::with_capacity(vmap.len());
    for (coord, &idx) in vmap.iter() {
        world.insert(*coord, file.voxels[idx]);
    }
    let solid = greedy_mesh::mesh_buffers_selection_overlay_solid(&effective_sel, &world);
    let line_verts = if let Some((min_x, min_y, min_z, max_x, max_y, max_z)) =
        greedy_mesh::selection_bounds(&effective_sel)
    {
        greedy_mesh::selection_aabb_line_vertices(min_x, min_y, min_z, max_x, max_y, max_z)
    } else {
        Vec::new()
    };
    SelectionOverlayPrepared::Draw {
        fp,
        solid,
        line_verts,
    }
}

pub(crate) fn apply_selection_overlay(
    viewer: &mut WgpuViewer,
    state: &ViewerState,
    prep: SelectionOverlayPrepared,
) {
    match prep {
        SelectionOverlayPrepared::Clear => {
            viewer.clear_selection_overlay();
            *state.gpu.selection_overlay_cache_key.lock() = None;
        }
        SelectionOverlayPrepared::Unchanged => {}
        SelectionOverlayPrepared::Draw {
            fp,
            solid,
            line_verts,
        } => {
            if viewer.selection_overlay_cache_key == Some(fp) {
                return;
            }
            viewer.upload_selection_overlay_solid(&solid);
            viewer.upload_selection_overlay_lines(&line_verts);
            viewer.selection_overlay_cache_key = Some(fp);
            *state.gpu.selection_overlay_cache_key.lock() = Some(fp);
        }
    }
}
