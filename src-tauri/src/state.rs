use crate::camera;
use crate::collab;
use crate::generators;
use crate::greedy_mesh;
use crate::gpu_brick;
use crate::render::WgpuViewer;
use crate::stroke_modes;
use crate::voxel_edit;
use crate::voxelle;

use ahash::{AHashMap, AHashSet};
use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter};

pub(crate) struct FpsCounter {
    pub(crate) period_start: Option<Instant>,
    pub(crate) accum_frames: u32,
    /// Last computed viewport FPS (updated when we emit `viewport-fps`).
    pub(crate) last_fps: u32,
}

pub(crate) fn sample_fps_and_emit(app: &AppHandle, counter: &Mutex<FpsCounter>) {
    let now = Instant::now();
    let mut c = counter.lock();
    if c.period_start.is_none() {
        c.period_start = Some(now);
    }
    c.accum_frames += 1;
    let Some(start) = c.period_start else {
        return;
    };
    let elapsed = now.saturating_duration_since(start);
    if elapsed >= Duration::from_secs(1) {
        let elapsed_ms = elapsed.as_millis().max(1) as f64;
        let fps = ((c.accum_frames as f64 * 1000.0) / elapsed_ms).round() as u32;
        c.last_fps = fps;
        let _ = app.emit("viewport-fps", fps);
        c.accum_frames = 0;
        c.period_start = Some(now);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub(crate) enum PreviewMode {
    #[default]
    Navigate,
    Add,
    Remove,
    Paint,
    /// Solid hit preview for selection stroke tools.
    Select,
    Fly,
    /// Metaball field preview + edit gizmo (Squishy tool).
    Squishy,
    /// Ghost overlay of clipboard entries at the add-tool anchor (empty cell in front of hit).
    Stamp,
    /// Red overlay of clipboard entries at the hit cell (voxels that would be removed).
    Punch,
    /// Extrude selection via axis-locked gizmo arrows.
    SelectExtrude,
}

impl PreviewMode {
    pub(crate) fn parse(s: &str) -> Self {
        match s {
            "add" => Self::Add,
            "remove" => Self::Remove,
            "paint" => Self::Paint,
            "select" => Self::Select,
            "fly" => Self::Fly,
            "squishy" => Self::Squishy,
            "stamp" => Self::Stamp,
            "punch" => Self::Punch,
            "selectExtrude" => Self::SelectExtrude,
            _ => Self::Navigate,
        }
    }
}

/// Viewport meshing mode (matches web Voxelle `renderingMode`; ray-traced mode is not implemented on desktop).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RenderingMode {
    #[default]
    Greedy,
    MarchingCubes,
    DualContour,
    /// Not implemented on desktop — meshed as [`Greedy`](RenderingMode::Greedy).
    Ray,
}

impl RenderingMode {
    pub(crate) fn uses_smooth_surface(self) -> bool {
        matches!(
            self,
            RenderingMode::MarchingCubes | RenderingMode::DualContour
        )
    }
}

/// How additive selection tools merge with the current selection (matches web Voxelle).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SelectionCombineMode {
    #[default]
    Replace,
    Add,
    Subtract,
    Intersect,
}

/// Interleaved solo undo: voxel edit batches and selection snapshots (web parity).
#[derive(Clone)]
pub(crate) enum SoloUndoEntry {
    VoxelDeltas(Vec<voxel_edit::VoxelEditDelta>),
    SelectionBefore(AHashSet<greedy_mesh::VoxelCoord>),
    /// Atomic selection-transform: voxel deltas + selection snapshot bundled together.
    SelectionTransform {
        before: AHashSet<greedy_mesh::VoxelCoord>,
        deltas: Vec<voxel_edit::VoxelEditDelta>,
    },
}

#[derive(Clone)]
pub(crate) enum SoloRedoEntry {
    VoxelDeltas(Vec<voxel_edit::VoxelEditDelta>),
    SelectionAfter(AHashSet<greedy_mesh::VoxelCoord>),
    /// Atomic selection-transform redo: voxel deltas + selection snapshot bundled together.
    SelectionTransform {
        after: AHashSet<greedy_mesh::VoxelCoord>,
        deltas: Vec<voxel_edit::VoxelEditDelta>,
    },
}

/// Millisecond timings for the last successful voxel add/remove (`voxel_edit_at_screen`).
#[derive(Clone, Debug)]
pub struct EditPerfBreakdown {
    pub apply_edit_ms: f64,
    /// Scene AABB in world space (`scene_bounds_for_edit`) + brick patch args; excludes viewer lock and GPU brick upload.
    pub prepare_ms: f64,
    /// Time blocked acquiring `viewer` mutex (often competes with the render loop).
    pub viewer_lock_wait_ms: f64,
    pub brick_ms: f64,
    /// Wall time for the mesh section (sub-fields below; excludes `preview_clear_ms`).
    pub mesh_ms: f64,
    pub total_ms: f64,
    /// Opaque mesh rebuild path after the edit (see `WgpuViewer::last_mesh_route`).
    pub mesh_route: String,

    // --- Mesh sub-phases (zeros when not used; see performance snapshot) ---
    /// `cpu_chunked_incremental`: O(1) [`WgpuViewer::apply_spatial_cache_edit`].
    pub mesh_voxel_map_ms: f64,
    /// `cpu_chunked_incremental`: cold [`greedy_mesh::SpatialMeshCache::from_voxels`] inside remesh (rare).
    pub mesh_buckets_ms: f64,
    /// `cpu_chunked_incremental`: wall time in greedy phase (GPU pack+compute + CPU fallback per chunk).
    pub mesh_greedy_ms: f64,
    /// `cpu_chunked_incremental`: GPU [`WgpuViewer::run_mesh_greedy_compute_with_brick`] / internal dispatch (subset of `mesh_greedy_ms` when env allows).
    pub mesh_greedy_gpu_ms: f64,
    /// `cpu_chunked_incremental`: CPU [`greedy_mesh::mesh_buffers_for_chunk_key`] (subset of `mesh_greedy_ms`).
    pub mesh_greedy_cpu_ms: f64,
    /// `cpu_chunked_incremental`: interleaved mesh → `wgpu` chunk buffers.
    pub mesh_chunk_buffers_ms: f64,
    /// Full [`WgpuViewer::upload_cpu_mesh_chunked_full`] inside remesh (origin drift).
    pub mesh_full_chunked_rebuild_ms: f64,
    /// Non-incremental [`WgpuViewer::rebuild_mesh_gpu_greedy`] wall time (includes internal CPU fallback).
    pub mesh_pipeline_ms: f64,
    pub preview_clear_ms: f64,
}

/// Cached global voxel AABB min and single-object id (see [`resolve_voxel_edit_stats`]).
#[derive(Clone, Copy)]
pub(crate) struct VoxelEditStatsCache {
    pub(crate) aabb_min: (i32, i32, i32),
    /// `Some(id)` iff every voxel has `object_id == id`.
    pub(crate) common_object_id: Option<u32>,
}

pub(crate) fn voxel_aabb_min_and_single_object_one_pass(
    voxels: &[voxelle::Voxel],
) -> VoxelEditStatsCache {
    if voxels.is_empty() {
        return VoxelEditStatsCache {
            aabb_min: (0, 0, 0),
            common_object_id: Some(0),
        };
    }
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut min_z = i32::MAX;
    let first = voxels[0].object_id;
    let mut single = true;
    for v in voxels {
        min_x = min_x.min(v.x);
        min_y = min_y.min(v.y);
        min_z = min_z.min(v.z);
        if v.object_id != first {
            single = false;
        }
    }
    VoxelEditStatsCache {
        aabb_min: (min_x, min_y, min_z),
        common_object_id: if single { Some(first) } else { None },
    }
}

pub(crate) fn resolve_voxel_edit_stats(
    voxels: &[voxelle::Voxel],
    delta: &voxel_edit::VoxelEditDelta,
    cached: Option<VoxelEditStatsCache>,
) -> VoxelEditStatsCache {
    let Some(cached) = cached else {
        return voxel_aabb_min_and_single_object_one_pass(voxels);
    };
    match delta {
        voxel_edit::VoxelEditDelta::Added(v) => {
            let (ox, oy, oz) = cached.aabb_min;
            let new_min = (ox.min(v.x), oy.min(v.y), oz.min(v.z));
            let common_object_id = match cached.common_object_id {
                Some(oid) if v.object_id == oid => Some(oid),
                _ => None,
            };
            VoxelEditStatsCache {
                aabb_min: new_min,
                common_object_id,
            }
        }
        voxel_edit::VoxelEditDelta::Removed { voxel } => {
            let (ox, oy, oz) = cached.aabb_min;
            if voxel.x == ox || voxel.y == oy || voxel.z == oz {
                return voxel_aabb_min_and_single_object_one_pass(voxels);
            }
            cached
        }
        voxel_edit::VoxelEditDelta::Painted { .. } => cached,
    }
}

pub(crate) fn union_dirty_chunk_keys_for_deltas(
    deltas: &[voxel_edit::VoxelEditDelta],
    origin: (i32, i32, i32),
    cs: i32,
) -> Vec<greedy_mesh::ChunkKey> {
    use std::collections::BTreeSet;
    let mut set: BTreeSet<greedy_mesh::ChunkKey> = BTreeSet::new();
    for d in deltas {
        let (x, y, z) = match d {
            voxel_edit::VoxelEditDelta::Added(v) => (v.x, v.y, v.z),
            voxel_edit::VoxelEditDelta::Removed { voxel } => (voxel.x, voxel.y, voxel.z),
            voxel_edit::VoxelEditDelta::Painted { after, .. } => (after.x, after.y, after.z),
        };
        for k in greedy_mesh::dirty_chunk_keys_for_voxel(x, y, z, origin, cs) {
            set.insert(k);
        }
    }
    set.into_iter().collect()
}

pub(crate) fn deltas_to_brick_patches(
    deltas: &[voxel_edit::VoxelEditDelta],
) -> Vec<gpu_brick::BrickCellWrite> {
    deltas
        .iter()
        .map(|d| match d {
            voxel_edit::VoxelEditDelta::Added(v) => gpu_brick::BrickCellWrite {
                x: v.x,
                y: v.y,
                z: v.z,
                packed: gpu_brick::pack_cell(v.color, v.material),
            },
            voxel_edit::VoxelEditDelta::Removed { voxel } => gpu_brick::BrickCellWrite {
                x: voxel.x,
                y: voxel.y,
                z: voxel.z,
                packed: gpu_brick::pack_empty(),
            },
            voxel_edit::VoxelEditDelta::Painted { after, .. } => gpu_brick::BrickCellWrite {
                x: after.x,
                y: after.y,
                z: after.z,
                packed: gpu_brick::pack_cell(after.color, after.material),
            },
        })
        .collect()
}

pub(crate) fn scene_bounds_for_edits(
    state: &ViewerState,
    file: &voxelle::VoxelleFile,
    deltas: &[voxel_edit::VoxelEditDelta],
) -> Result<greedy_mesh::MeshBounds, String> {
    if file.voxels.is_empty() {
        return Ok(greedy_mesh::mesh_bounds_for_cube_side(file.grid_size));
    }
    if deltas.is_empty() {
        return Ok(
            greedy_mesh::mesh_bounds_from_voxels_world(&file.voxels, &file.objects)
                .or_else(|| greedy_mesh::mesh_bounds_from_voxels(&file.voxels))
                .unwrap_or_else(|| greedy_mesh::mesh_bounds_for_cube_side(file.grid_size)),
        );
    }
    if deltas.len() == 1 {
        return crate::scene_bounds_for_edit(state, file, &deltas[0]);
    }
    // Multi-delta fast path: try incremental update from cached bounds.
    let default_objs = voxelle::default_scene_objects();
    let objs: &[voxelle::SceneObject] = if file.objects.is_empty() {
        default_objs.as_slice()
    } else {
        &file.objects
    };
    if voxelle::scene::scene_objects_identity_for_bounds_fast_path(objs) {
        let guard = state.last_scene_bounds.lock();
        if let Some(prev) = guard.as_ref() {
            let mut bounds = *prev;
            let mut needs_full_recompute = false;
            for d in deltas {
                match d {
                    voxel_edit::VoxelEditDelta::Added(v) => {
                        bounds = greedy_mesh::mesh_bounds_expand_with_voxel(&bounds, v);
                    }
                    voxel_edit::VoxelEditDelta::Removed { voxel } => {
                        if !greedy_mesh::mesh_bounds_remove_is_strict_interior(
                            &bounds, voxel.x, voxel.y, voxel.z,
                        ) {
                            needs_full_recompute = true;
                            break;
                        }
                    }
                    voxel_edit::VoxelEditDelta::Painted { .. } => {}
                }
            }
            if !needs_full_recompute {
                return Ok(bounds);
            }
        }
    }
    Ok(
        greedy_mesh::mesh_bounds_from_voxels_world(&file.voxels, objs)
            .or_else(|| greedy_mesh::mesh_bounds_from_voxels(&file.voxels))
            .unwrap_or_else(|| greedy_mesh::mesh_bounds_for_cube_side(file.grid_size)),
    )
}

/// Normalized viewport coords (0..=1) → texels using the same `w,h` as projection / [`voxel_edit::screen_to_world_ray`].
#[inline]
pub(crate) fn viewport_texels_from_norm(nx: f32, ny: f32, w: f32, h: f32) -> (f32, f32) {
    // Pixel-center convention:
    // - `screen_to_world_ray` samples `(sx + 0.5, sy + 0.5) / (w,h)`.
    // - Therefore `sx/sy` must live in `[0, w-1] / [0, h-1]` so `nx=0.5` maps to exact center.
    // Mapping to `[0, w]` introduced a systematic half-pixel bias in rays.
    let sx = nx.clamp(0.0, 1.0) * (w.max(1.0) - 1.0);
    let sy = ny.clamp(0.0, 1.0) * (h.max(1.0) - 1.0);
    (sx, sy)
}

/// Resolve the spray constraint plane for the invisible hit plane trick (web parity).
///
/// On the first spray anchor of a stroke (no plane stored yet): does a normal voxel raycast,
/// computes the plane through the hit using `constrain_to_plane_ref`, and stores it in state.
/// On subsequent calls: returns the stored plane.
/// Returns `None` when constrain_to_plane is not active or not in spray mode.
pub(crate) fn resolve_spray_constraint_plane(
    state: &ViewerState,
    aux: &stroke_modes::StrokeAux,
    stroke_mode: stroke_modes::DrawStrokeMode,
    tool: voxel_edit::EditTool,
    file: &voxelle::VoxelleFile,
    vmap: &AHashMap<greedy_mesh::VoxelCoord, usize>,
    cam: &camera::OrbitCamera,
    w: f32,
    h: f32,
    sx: f32,
    sy: f32,
) -> Option<(glam::Vec3, glam::Vec3)> {
    if stroke_mode != stroke_modes::DrawStrokeMode::Spray || !aux.constrain_to_plane {
        return None;
    }
    let plane_ref = aux.constrain_to_plane_ref.as_deref().unwrap_or("auto");

    // Return stored plane if already established this stroke.
    {
        let stored = state.spray_constraint_plane.lock();
        if stored.is_some() {
            return *stored;
        }
    }

    // First anchor: raycast to find the hit position + face normal, then compute the plane.
    let anchor = voxel_edit::anchor_for_stroke_edit(
        tool,
        aux.stroke_snap_to_surface,
        file,
        vmap,
        cam,
        w,
        h,
        sx,
        sy,
    )?;
    let face_n = voxel_edit::pick_extrude_start(file, vmap, cam, w, h, sx, sy).and_then(|(_, n)| n);
    let plane_normal = voxel_edit::constrain_plane_normal(plane_ref, cam, face_n)?;
    let plane_point = glam::Vec3::new(anchor.0 as f32, anchor.1 as f32, anchor.2 as f32);

    let plane = (plane_point, plane_normal);
    *state.spray_constraint_plane.lock() = Some(plane);
    Some(plane)
}

#[derive(Clone, Copy)]
pub(crate) struct FlyInputState {
    pub forward: f32,
    pub right: f32,
    pub up: f32,
    pub speed_scale: f32,
    pub jump: bool,
}

impl Default for FlyInputState {
    fn default() -> Self {
        Self {
            forward: 0.0,
            right: 0.0,
            up: 0.0,
            speed_scale: 1.0,
            jump: false,
        }
    }
}

// CSS-pixel constants for gizmo interaction (multiplied by dpr when comparing physical px).
pub(crate) const GIZMO_MOVE_HIT_CSS: f32 = 16.0;
pub(crate) const GIZMO_RING_HIT_CSS: f32 = 11.0;
pub(crate) const GIZMO_PX_PER_MOVE_STEP_CSS: f32 = 26.0;
pub(crate) const GIZMO_PX_PER_ROTATE_STEP_CSS: f32 = 65.0;
pub(crate) const GIZMO_RING_SAMPLES: usize = 16;

/// Active drag state for the selection transform gizmo.
#[derive(Clone, Debug, Default)]
pub(crate) enum SelectionGizmoDrag {
    #[default]
    None,
    Move {
        /// Normalized screen-space direction (center→handle tip) at drag start.
        axis_sx: f32,
        axis_sy: f32,
        /// World axis: 0=X, 1=Y, 2=Z
        world_axis: u8,
        /// true for +axis handles (indices 0,2,4), false for -axis (1,3,5)
        positive: bool,
        accum: f32,
        /// GIZMO_PX_PER_MOVE_STEP_CSS * dpr, captured at drag start
        step_threshold: f32,
        /// Accumulated integer steps since drag start — applied as one translate on pointer-up.
        /// The selection overlay is rebuilt at this offset each frame; voxel data is not touched
        /// until the drag is committed.
        pending_dx: i32,
        pending_dy: i32,
        pending_dz: i32,
    },
    Rotate {
        ring: u8,
        tangent_x: f32,
        tangent_y: f32,
        accum: f32,
        /// GIZMO_PX_PER_ROTATE_STEP_CSS * dpr, captured at drag start
        step_threshold: f32,
    },
}

/// Active drag state for the selection extrude gizmo.
#[derive(Clone, Debug, Default)]
pub(crate) enum ExtrudeGizmoDrag {
    #[default]
    None,
    Drag {
        /// Normalized screen-space direction (center→handle tip) at drag start.
        axis_sx: f32,
        axis_sy: f32,
        /// World axis: 0=X, 1=Y, 2=Z
        world_axis: u8,
        /// true for +axis handles (indices 0,2,4), false for -axis (1,3,5)
        positive: bool,
        accum: f32,
        /// GIZMO_PX_PER_MOVE_STEP_CSS * dpr, captured at drag start
        step_threshold: f32,
        /// Signed extrude depth: >0 = along positive axis, <0 = along negative axis.
        depth: i32,
    },
}

pub struct ViewerState {
    pub viewer: Mutex<Option<WgpuViewer>>,
    pub camera: Mutex<camera::OrbitCamera>,
    pub file_label: Mutex<String>,
    /// Latest loaded model for CPU-side edits (add/remove voxels).
    pub current_file: Mutex<Option<voxelle::VoxelleFile>>,
    /// Spatial index: coord → index in `current_file.voxels` (kept in sync; used for raycasts + O(1) remove).
    pub voxel_map: Mutex<Option<AHashMap<greedy_mesh::VoxelCoord, usize>>>,
    /// Latest pointer position in physical pixels (for hover preview; updated from UI, read each frame).
    pub preview_cursor: Mutex<Option<(f32, f32)>>,
    /// True while the user is orbit/pan/dolly-dragging the camera via `viewport_pointer`.
    /// Used by `prepare_preview_mesh` to suppress cursor-attached previews during orbiting.
    pub camera_dragging: AtomicBool,
    pub(crate) preview_mode: Mutex<PreviewMode>,
    /// Brush / stroke params for hover preview (updated from [`sync_preview_input`]).
    pub(crate) preview_hover: Mutex<crate::PreviewHoverContext>,
    pub rendering_mode: Mutex<RenderingMode>,
    pub(crate) fps: Mutex<FpsCounter>,
    /// Last successful edit timings (updated each time `voxel_edit_at_screen` applies a change).
    pub last_edit_perf: Mutex<Option<EditPerfBreakdown>>,
    /// Last scene AABB used for lighting/brick; drives incremental bounds on edit when possible.
    pub last_scene_bounds: Mutex<Option<greedy_mesh::MeshBounds>>,
    /// Bumps when a background opaque-mesh refresh is scheduled; stale applies are skipped.
    pub mesh_refresh_generation: AtomicU64,
    /// Bumps each time a file/project load begins; stale loads bail out instead of overwriting a newer load.
    pub load_generation: AtomicU64,
    /// Chunks built by background meshing thread, waiting for main-thread GPU upload.
    pub(crate) chunk_mesh_inbox: Mutex<VecDeque<(greedy_mesh::ChunkKey, greedy_mesh::MeshBuffers)>>,
    /// Guest edit/undo/redo items waiting for main-thread processing (same pattern as `chunk_mesh_inbox`).
    pub(crate) collab_edit_inbox: Mutex<VecDeque<collab::CollabInboxItem>>,
    /// SpatialMeshCache from background streaming load; moved to viewer after all chunks are uploaded.
    pub(crate) deferred_spatial_cache: Mutex<Option<greedy_mesh::SpatialMeshCache>>,
    /// Incremental [`greedy_mesh::voxel_aabb_min_int`] + single-object detection for chunked mesh path.
    pub(crate) voxel_edit_stats_cache: Mutex<Option<VoxelEditStatsCache>>,
    /// Solo undo stack: voxel batches and selection snapshots (interleaved).
    pub(crate) solo_undo: Mutex<Vec<SoloUndoEntry>>,
    pub(crate) solo_redo: Mutex<Vec<SoloRedoEntry>>,
    /// When true, successful edits append to `stroke_buffer` instead of pushing `solo_undo` immediately.
    pub stroke_active: Mutex<bool>,
    pub stroke_buffer: Mutex<Vec<voxel_edit::VoxelEditDelta>>,
    /// Accumulated stroke preview cells (add/remove/paint drag; committed on pointer up).
    pub stroke_preview_union: Mutex<AHashSet<greedy_mesh::VoxelCoord>>,
    pub(crate) stroke_preview_last_args: Mutex<Option<crate::VoxelEditAtScreen>>,
    /// When set, hover preview must not overwrite the stroke preview mesh each frame.
    pub stroke_preview_suppresses_hover: AtomicBool,
    /// Throttled sculpt samples during drag; replayed on pointer up as one undo step.
    pub(crate) sculpt_stroke_replay: Mutex<Vec<crate::SculptStrokeAtScreenArgs>>,
    /// Stored ray spine for straight-line extrude (used by ray-based extrude preview/recompute).
    pub(crate) extrude_ray_spine: Mutex<Option<Vec<greedy_mesh::VoxelCoord>>>,
    pub collab: Arc<Mutex<collab::CollabRuntime>>,
    /// Raw `.voxelle` bytes for custom avatars the local user has loaded, keyed by name.
    /// Persists across collab sessions so the bytes can be re-sent to new peers.
    pub local_avatar_data: Mutex<HashMap<String, Vec<u8>>>,
    /// Smoothed peer camera presence for frustum rendering (lerped each frame).
    pub smooth_presence: Mutex<HashMap<u32, collab::CameraPresence>>,
    /// Short-lived voxel highlight when a peer sends a world ping (see [`collab::record_ping_flash`]).
    pub ping_flash: Mutex<Option<collab::PingFlash>>,
    /// Host-only autosave to app-local backups (`0` = never when disabled or interval 0).
    pub autosave_interval_secs: Mutex<u64>,
    pub last_autosave: Mutex<Option<Instant>>,
    /// When false, autosave timer does not run.
    pub autosave_enabled: Mutex<bool>,
    /// Rotating slot count per document (`{hash}.0.voxelle` … `{hash}.(n-1).voxelle`).
    pub autosave_keep_count: Mutex<u32>,
    /// Next slot index per stable path hash (see `stable_path_key`).
    pub autosave_slot: Mutex<HashMap<String, u64>>,
    /// True after a scene is fully applied ([`apply_mesh_and_camera`]); false during unload and before the first successful load.
    pub active_project: AtomicBool,
    /// When true, orbit / wheel camera IPC is ignored (WASD fly movement).
    pub fly_mode: Mutex<bool>,
    /// Latest fly WASD input from the webview; movement uses wall-clock dt in the native loop.
    pub(crate) fly_input: Mutex<FlyInputState>,
    /// Previous [`Instant`] for native fly physics (`None` until first step after fly mode enables).
    pub(crate) fly_last_physics: Mutex<Option<Instant>>,
    /// Walk mode: first-person with gravity, collision, and jumping.
    pub walk_mode: Mutex<bool>,
    /// Walk physics state (feet position, velocity, on_ground).
    pub(crate) walk_physics: Mutex<camera::WalkPhysicsState>,
    /// Previous [`Instant`] for walk physics dt.
    pub(crate) walk_last_physics: Mutex<Option<Instant>>,
    /// Selected solid cells (world grid); used for copy / stamp source.
    pub selection_cells: Mutex<AHashSet<greedy_mesh::VoxelCoord>>,
    /// Snapshot at `selection_stroke_begin` for undo + detecting no-op end.
    pub selection_stroke_before: Mutex<Option<AHashSet<greedy_mesh::VoxelCoord>>>,
    /// Accumulates all coords touched during a single stroke so that intersect
    /// mode can union the per-sample coords and apply the intersection once
    /// (rather than shrinking the selection on every pointer-move sample).
    pub selection_stroke_accum: Mutex<Option<AHashSet<greedy_mesh::VoxelCoord>>>,
    pub selection_combine_mode: Mutex<SelectionCombineMode>,
    /// Matches native "Match Material" for color / connected selection (synced with menu + webview).
    pub selection_match_material: Mutex<bool>,
    /// Last copy from [`Self::selection_cells`] (relative offsets).
    pub stamp_clipboard: Mutex<Option<voxel_edit::StampClipboard>>,
    /// Multi-metaball squishy editor (Squishy mode).
    pub squishy_session: Mutex<generators::SquishySession>,
    /// Pointer drag on squishy move/scale handles ([`generators::squishy_gizmo`]).
    pub squishy_gizmo_drag: Mutex<Option<generators::SquishyGizmoDrag>>,
    /// Active pointer drag on the selection transform gizmo (move/rotate handles).
    pub(crate) selection_gizmo_drag: Mutex<SelectionGizmoDrag>,
    /// Active pointer drag on the selection extrude gizmo.
    pub(crate) extrude_gizmo_drag: Mutex<ExtrudeGizmoDrag>,
    /// Accumulated depth from previous drags in the same extrude session (settings phase).
    /// Reset to 0 when the session is committed or cancelled.
    pub(crate) extrude_gizmo_base_depth: Mutex<i32>,
    /// Extrude gizmo axis (0=X,1=Y,2=Z) currently under cursor; 255=none.
    pub hovered_extrude_axis: AtomicU8,
    /// When true, draw the start-screen gradient instead of the scene sky (default true; cleared when a real document loads).
    pub start_screen_logo_transparent: std::sync::atomic::AtomicBool,
    /// Cold-start gradient: light (paper) vs dark — synced from webview appearance preference.
    pub start_screen_light: std::sync::atomic::AtomicBool,
    /// **Debug → Viewport cursor debug overlay**: use bright red ray-hover preview (menu + webview).
    pub viewport_cursor_debug_overlay: AtomicBool,
    /// **View → Show borders**: per-voxel cell wireframe (matches web `showGrid` / `gridLines.ts`).
    pub show_grid_borders: AtomicBool,
    /// Gizmo axis (0=X, 1=Y, 2=Z) currently under the cursor; 255 = none.
    /// Written by `gizmo_hit_test`; read by `sync_gizmo_gpu` to brighten the hovered axis.
    pub hovered_gizmo_axis: AtomicU8,
    /// Mirrors overlay cache keys on [`WgpuViewer`] so prepare steps can run without the viewer mutex (see frame loop).
    pub(crate) grid_overlay_cache_key: Mutex<Option<u64>>,
    pub(crate) selection_overlay_cache_key: Mutex<Option<u64>>,
    pub(crate) preview_overlay_cache_key: Mutex<Option<u64>>,
    /// Camera snapshot taken when a single-click generator enters its confirm phase.
    /// While set, `prepare_preview_mesh` uses this camera for all generator raycasts so that
    /// orbiting or panning the viewport does not change the preview world position.
    pub generator_preview_locked_camera: Mutex<Option<camera::OrbitCamera>>,
    /// Set by [`voxel_fill_cancel`] during a long flood fill so BFS can exit cooperatively.
    pub fill_operation_cancel: Arc<AtomicBool>,
    /// Invisible hit plane for spray constrain-to-plane: `(plane_point, plane_normal)`.
    /// Set on the first spray anchor of a stroke when constrain_to_plane is active; cleared on stroke end.
    pub spray_constraint_plane: Mutex<Option<(glam::Vec3, glam::Vec3)>>,
    /// Locked face normal for wall strokes. `None` = not yet captured this stroke.
    /// `Some(v)` = locked on the first preview frame; reused for all subsequent drag frames
    /// so the wall orientation doesn't flip as the cursor crosses different faces.
    /// Cleared on stroke begin/end. Ignored during hover (stroke_active = false).
    pub wall_stroke_face_snapped: Mutex<Option<Option<(i32, i32, i32)>>>,
    /// Per-column fractional accumulation for terrain sub-voxel raise/lower precision.
    /// Cleared on stroke begin/end; ignored when `terrain_sub_voxel` is false.
    pub terrain_accum: Mutex<AHashMap<(i32, i32), f32>>,
}
