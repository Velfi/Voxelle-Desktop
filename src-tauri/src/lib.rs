mod camera;
mod collab;
#[cfg(desktop)]
mod headless_server;
mod gpu_brick;
/// Greedy CPU meshing (public for `cargo bench`).
pub mod greedy_mesh;
mod marching_tables;
mod smooth_mesh;
#[cfg(target_os = "macos")]
mod macos_undo;
mod render;
mod render_constants;
mod voxel_edit;
mod stroke_modes;
mod generators;
mod export_glb;
/// Voxel format / types (public for `cargo bench` and tests).
pub mod voxelle;

use camera::OrbitCamera;
use gpu_brick::{BrickCellWrite, GpuVoxelBrick};
use render::{compute_greedy_rebuild_cpu, PreparedGreedyRebuild, PreparedOpaqueUpload, WgpuViewer};
use std::any::Any;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::path::BaseDirectory;
use tauri::{AppHandle, Emitter, EventTarget, Manager, RunEvent, Runtime, State};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

use ahash::{AHashMap, AHashSet, AHasher};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use voxelle::{
    decode_payload, encode_payload_v4, focal_length_to_fov_y_radians, start_shape::StartShape,
};

struct FpsCounter {
    period_start: Option<Instant>,
    accum_frames: u32,
    /// Last computed viewport FPS (updated when we emit `viewport-fps`).
    last_fps: u32,
}

fn sample_fps_and_emit(app: &AppHandle, counter: &Mutex<FpsCounter>) {
    let now = Instant::now();
    let mut c = counter.lock().unwrap();
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
}

impl PreviewMode {
    fn parse(s: &str) -> Self {
        match s {
            "add" => Self::Add,
            "remove" => Self::Remove,
            "paint" => Self::Paint,
            "select" => Self::Select,
            "fly" => Self::Fly,
            "squishy" => Self::Squishy,
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
    fn uses_smooth_surface(self) -> bool {
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
}

#[derive(Clone)]
pub(crate) enum SoloRedoEntry {
    VoxelDeltas(Vec<voxel_edit::VoxelEditDelta>),
    SelectionAfter(AHashSet<greedy_mesh::VoxelCoord>),
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
struct VoxelEditStatsCache {
    aabb_min: (i32, i32, i32),
    /// `Some(id)` iff every voxel has `object_id == id`.
    common_object_id: Option<u32>,
}

fn voxel_aabb_min_and_single_object_one_pass(voxels: &[voxelle::Voxel]) -> VoxelEditStatsCache {
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

fn resolve_voxel_edit_stats(
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

fn union_dirty_chunk_keys_for_deltas(
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
        let center = greedy_mesh::chunk_key_from_world(x, y, z, origin, cs);
        for k in greedy_mesh::dirty_chunk_keys_3x3(center) {
            set.insert(k);
        }
    }
    set.into_iter().collect()
}

fn deltas_to_brick_patches(deltas: &[voxel_edit::VoxelEditDelta]) -> Vec<gpu_brick::BrickCellWrite> {
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

fn scene_bounds_for_edits(
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
        return scene_bounds_for_edit(state, file, &deltas[0]);
    }
    let all_paint = deltas
        .iter()
        .all(|d| matches!(d, voxel_edit::VoxelEditDelta::Painted { .. }));
    if all_paint {
        if let Ok(guard) = state.last_scene_bounds.lock() {
            if let Some(prev) = guard.as_ref() {
                return Ok(*prev);
            }
        }
    }
    let default_objs = voxelle::default_scene_objects();
    let objs: &[voxelle::SceneObject] = if file.objects.is_empty() {
        default_objs.as_slice()
    } else {
        &file.objects
    };
    Ok(
        greedy_mesh::mesh_bounds_from_voxels_world(&file.voxels, objs)
            .or_else(|| greedy_mesh::mesh_bounds_from_voxels(&file.voxels))
            .unwrap_or_else(|| greedy_mesh::mesh_bounds_for_cube_side(file.grid_size)),
    )
}

/// Normalized viewport coords (0..=1) → texels using the same `w,h` as projection / [`voxel_edit::screen_to_world_ray`].
#[inline]
fn viewport_texels_from_norm(nx: f32, ny: f32, w: f32, h: f32) -> (f32, f32) {
    (nx.clamp(0.0, 1.0) * w, ny.clamp(0.0, 1.0) * h)
}

#[derive(serde::Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct VoxelEditAtScreen {
    nx: f32,
    ny: f32,
    tool: voxel_edit::EditTool,
    color: u32,
    material: String,
    brush_radius: u32,
    brush_shape: voxel_edit::BrushShape,
    /// 0 = full brush; (0,1] = deterministic spray thinning.
    #[serde(default)]
    spray_density: f32,
    #[serde(default)]
    stroke_line_start_nx: Option<f32>,
    #[serde(default)]
    stroke_line_start_ny: Option<f32>,
    #[serde(default)]
    stroke_segment_prev_nx: Option<f32>,
    #[serde(default)]
    stroke_segment_prev_ny: Option<f32>,
    #[serde(default)]
    stroke_mode: stroke_modes::DrawStrokeMode,
    #[serde(default)]
    plane_axis: stroke_modes::PlaneAxis,
    #[serde(default)]
    stroke_aux: stroke_modes::StrokeAux,
    /// When `stroke_mode` is fill + paint: match material as well as color.
    #[serde(default)]
    match_material: bool,
}

fn default_terrain_strength_sculpt() -> i32 {
    4
}

fn default_smooth_passes_sculpt() -> u32 {
    1
}

fn default_brush_strength_sculpt() -> u32 {
    100
}

fn default_wall_height_vox_sculpt() -> u32 {
    2
}

#[derive(serde::Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct SculptStrokeAtScreenArgs {
    nx: f32,
    ny: f32,
    sculpt_mode: voxel_edit::SculptStrokeMode,
    color: u32,
    material: String,
    brush_radius: u32,
    brush_shape: voxel_edit::BrushShape,
    #[serde(default)]
    spray_density: f32,
    #[serde(default)]
    stroke_line_start_nx: Option<f32>,
    #[serde(default)]
    stroke_line_start_ny: Option<f32>,
    #[serde(default)]
    stroke_segment_prev_nx: Option<f32>,
    #[serde(default)]
    stroke_segment_prev_ny: Option<f32>,
    #[serde(default)]
    terrain_op: Option<voxel_edit::TerrainSculptOp>,
    #[serde(default)]
    terrain_base_y: i32,
    #[serde(default = "default_terrain_strength_sculpt")]
    terrain_strength: i32,
    #[serde(default)]
    terrain_smooth_radius: i32,
    #[serde(default = "default_smooth_passes_sculpt")]
    smooth_neighbor_passes: u32,
    #[serde(default = "default_brush_strength_sculpt")]
    brush_strength: u32,
    #[serde(default)]
    brush_falloff: u32,
    #[serde(default)]
    stroke_seed: u32,
    #[serde(default)]
    wall_area_shape: voxel_edit::WallAreaShape,
    #[serde(default)]
    spray_direction: voxel_edit::SprayDirection,
    #[serde(default)]
    wall_width_index: u32,
    #[serde(default = "default_wall_height_vox_sculpt")]
    wall_height_vox: u32,
    #[serde(default)]
    wall_lock_start_height: bool,
    #[serde(default)]
    wall_axis_align: bool,
}

/// Hover preview uses the same brush/stroke inputs as [`voxel_edit_at_screen`] / [`voxel_stroke_preview_at_screen`].
#[derive(Clone, Debug)]
struct PreviewHoverContext {
    brush_radius: u32,
    brush_shape: voxel_edit::BrushShape,
    spray_density: f32,
    stroke_mode: stroke_modes::DrawStrokeMode,
    plane_axis: stroke_modes::PlaneAxis,
    stroke_aux: stroke_modes::StrokeAux,
    color: u32,
    material: String,
    match_material: bool,
    /// When false (e.g. sculpt), hover uses the legacy single-cell preview.
    use_brush_preview: bool,
}

impl Default for PreviewHoverContext {
    fn default() -> Self {
        Self {
            brush_radius: 0,
            brush_shape: voxel_edit::BrushShape::default(),
            spray_density: 0.0,
            stroke_mode: stroke_modes::DrawStrokeMode::default(),
            plane_axis: stroke_modes::PlaneAxis::default(),
            stroke_aux: stroke_modes::StrokeAux::default(),
            color: 0,
            material: String::new(),
            match_material: false,
            use_brush_preview: true,
        }
    }
}

fn hash_single_cell_preview(
    mode: PreviewMode,
    cx: i32,
    cy: i32,
    cz: i32,
    tag: u8,
    debug_overlay: bool,
) -> u64 {
    let mut h = AHasher::default();
    mode.hash(&mut h);
    cx.hash(&mut h);
    cy.hash(&mut h);
    cz.hash(&mut h);
    tag.hash(&mut h);
    debug_overlay.hash(&mut h);
    h.finish()
}

fn hash_preview_miss(mode: PreviewMode, debug_overlay: bool) -> u64 {
    let mut h = AHasher::default();
    mode.hash(&mut h);
    0x7Fu8.hash(&mut h);
    debug_overlay.hash(&mut h);
    h.finish()
}

fn hash_squishy_preview(
    session: &generators::SquishySession,
    sx: f32,
    sy: f32,
    add_anchor: Option<(i32, i32, i32)>,
    preview_radius_i: u32,
    gizmo_drag: bool,
    delete_hover_id: Option<u32>,
    debug_overlay: bool,
) -> u64 {
    let mut h = AHasher::default();
    PreviewMode::Squishy.hash(&mut h);
    debug_overlay.hash(&mut h);
    sx.to_bits().hash(&mut h);
    sy.to_bits().hash(&mut h);
    preview_radius_i.hash(&mut h);
    gizmo_drag.hash(&mut h);
    delete_hover_id.hash(&mut h);
    (session.mode as u8).hash(&mut h);
    session.hollow.hash(&mut h);
    session.wall_thickness.hash(&mut h);
    session.add_snap_to_surface.hash(&mut h);
    session.selected_id.hash(&mut h);
    for b in &session.balls {
        b.id.hash(&mut h);
        b.x.hash(&mut h);
        b.y.hash(&mut h);
        b.z.hash(&mut h);
        b.radius.to_bits().hash(&mut h);
    }
    if let Some((ax, ay, az)) = add_anchor {
        ax.hash(&mut h);
        ay.hash(&mut h);
        az.hash(&mut h);
    } else {
        0x5Eu8.hash(&mut h);
    }
    h.finish()
}

fn hash_brush_hover_targets(
    mode: PreviewMode,
    ctx: &PreviewHoverContext,
    targets: &[greedy_mesh::VoxelCoord],
    debug_overlay: bool,
) -> u64 {
    let mut sorted: Vec<_> = targets.to_vec();
    sorted.sort_unstable();
    let mut h = AHasher::default();
    mode.hash(&mut h);
    debug_overlay.hash(&mut h);
    ctx.use_brush_preview.hash(&mut h);
    ctx.brush_radius.hash(&mut h);
    (ctx.brush_shape as u8).hash(&mut h);
    ctx.spray_density.to_bits().hash(&mut h);
    (ctx.stroke_mode as u8).hash(&mut h);
    (ctx.plane_axis as u8).hash(&mut h);
    ctx.color.hash(&mut h);
    ctx.material.hash(&mut h);
    ctx.match_material.hash(&mut h);
    sorted.hash(&mut h);
    if let Ok(s) = serde_json::to_string(&ctx.stroke_aux) {
        s.hash(&mut h);
    }
    h.finish()
}

#[derive(Clone, Copy)]
pub(crate) struct FlyInputState {
    pub forward: f32,
    pub right: f32,
    pub up: f32,
    pub speed_scale: f32,
}

impl Default for FlyInputState {
    fn default() -> Self {
        Self {
            forward: 0.0,
            right: 0.0,
            up: 0.0,
            speed_scale: 1.0,
        }
    }
}

pub struct ViewerState {
    pub viewer: Mutex<Option<WgpuViewer>>,
    pub camera: Mutex<OrbitCamera>,
    pub file_label: Mutex<String>,
    /// Latest loaded model for CPU-side edits (add/remove voxels).
    pub current_file: Mutex<Option<voxelle::VoxelleFile>>,
    /// Spatial index: coord → index in `current_file.voxels` (kept in sync; used for raycasts + O(1) remove).
    pub voxel_map: Mutex<Option<AHashMap<greedy_mesh::VoxelCoord, usize>>>,
    /// Latest pointer position in physical pixels (for hover preview; updated from UI, read each frame).
    pub preview_cursor: Mutex<Option<(f32, f32)>>,
    pub(crate) preview_mode: Mutex<PreviewMode>,
    /// Brush / stroke params for hover preview (updated from [`sync_preview_input`]).
    preview_hover: Mutex<PreviewHoverContext>,
    pub rendering_mode: Mutex<RenderingMode>,
    fps: Mutex<FpsCounter>,
    /// Last successful edit timings (updated each time `voxel_edit_at_screen` applies a change).
    pub last_edit_perf: Mutex<Option<EditPerfBreakdown>>,
    /// Last scene AABB used for lighting/brick; drives incremental bounds on edit when possible.
    pub last_scene_bounds: Mutex<Option<greedy_mesh::MeshBounds>>,
    /// Bumps when a background opaque-mesh refresh is scheduled; stale applies are skipped.
    pub mesh_refresh_generation: AtomicU64,
    /// Incremental [`greedy_mesh::voxel_aabb_min_int`] + single-object detection for chunked mesh path.
    voxel_edit_stats_cache: Mutex<Option<VoxelEditStatsCache>>,
    /// Solo undo stack: voxel batches and selection snapshots (interleaved).
    pub(crate) solo_undo: Mutex<Vec<SoloUndoEntry>>,
    pub(crate) solo_redo: Mutex<Vec<SoloRedoEntry>>,
    /// When true, successful edits append to `stroke_buffer` instead of pushing `solo_undo` immediately.
    pub stroke_active: Mutex<bool>,
    pub stroke_buffer: Mutex<Vec<voxel_edit::VoxelEditDelta>>,
    /// Accumulated stroke preview cells (add/remove/paint drag; committed on pointer up).
    pub stroke_preview_union: Mutex<AHashSet<greedy_mesh::VoxelCoord>>,
    pub(crate) stroke_preview_last_args: Mutex<Option<VoxelEditAtScreen>>,
    /// When set, hover preview must not overwrite the stroke preview mesh each frame.
    pub stroke_preview_suppresses_hover: AtomicBool,
    /// Throttled sculpt samples during drag; replayed on pointer up as one undo step.
    pub(crate) sculpt_stroke_replay: Mutex<Vec<SculptStrokeAtScreenArgs>>,
    pub collab: Arc<std::sync::Mutex<collab::CollabRuntime>>,
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
    /// Selected solid cells (world grid); used for copy / stamp source.
    pub selection_cells: Mutex<AHashSet<greedy_mesh::VoxelCoord>>,
    /// Snapshot at `selection_stroke_begin` for undo + detecting no-op end.
    pub selection_stroke_before: Mutex<Option<AHashSet<greedy_mesh::VoxelCoord>>>,
    pub selection_combine_mode: Mutex<SelectionCombineMode>,
    /// Matches native "Match Material" for color / connected selection (synced with menu + webview).
    pub selection_match_material: Mutex<bool>,
    /// Last copy from [`Self::selection_cells`] (relative offsets).
    pub stamp_clipboard: Mutex<Option<voxel_edit::StampClipboard>>,
    /// Multi-metaball squishy editor (Squishy mode).
    pub squishy_session: Mutex<generators::SquishySession>,
    /// Pointer drag on squishy move/scale handles ([`generators::squishy_gizmo`]).
    pub squishy_gizmo_drag: Mutex<Option<generators::SquishyGizmoDrag>>,
    /// When true, draw the start-screen gradient instead of the scene sky (default true; cleared when a real document loads).
    pub start_screen_logo_transparent: std::sync::atomic::AtomicBool,
    /// Cold-start gradient: light (paper) vs dark — synced from webview appearance preference.
    pub start_screen_light: std::sync::atomic::AtomicBool,
    /// **Debug → Viewport cursor debug overlay**: use bright red ray-hover preview (menu + webview).
    pub viewport_cursor_debug_overlay: AtomicBool,
}

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ViewportPixelSize {
    width: u32,
    height: u32,
    surface_width: u32,
    surface_height: u32,
}

/// Authoritative swapchain size in physical pixels (from the viewer; matches `frame.texture` after render).
#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct SurfacePixelSize {
    width: u32,
    height: u32,
}

/// Last known `.viewport` size and swapchain size in physical pixels (matches projection / picking / blit).
#[tauri::command]
fn get_viewport_pixel_size(state: State<'_, Arc<ViewerState>>) -> Result<ViewportPixelSize, String> {
    let v = state.viewer.lock().map_err(|e| e.to_string())?;
    let Some(viewer) = v.as_ref() else {
        return Err("viewer not ready".into());
    };
    let (w, h) = viewer.viewport_size();
    let (sw, sh) = viewer.surface_pixel_size();
    Ok(ViewportPixelSize {
        width: w,
        height: h,
        surface_width: sw,
        surface_height: sh,
    })
}

/// Debug: last `sync_preview_input` cursor (normalized) and matching texels (same as picking `screen_to_world_ray`).
#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ViewportCursorDebug {
    viewport_width: u32,
    viewport_height: u32,
    preview_nx: Option<f32>,
    preview_ny: Option<f32>,
    texel_sx: Option<f32>,
    texel_sy: Option<f32>,
    /// [`voxel_edit::screen_to_world_ray`] at `texel_s*` (same as picking).
    ray_origin_x: Option<f32>,
    ray_origin_y: Option<f32>,
    ray_origin_z: Option<f32>,
    ray_dir_x: Option<f32>,
    ray_dir_y: Option<f32>,
    ray_dir_z: Option<f32>,
}

#[tauri::command]
fn get_viewport_cursor_debug(
    state: State<'_, Arc<ViewerState>>,
) -> Result<ViewportCursorDebug, String> {
    let cam = state.camera.lock().map_err(|e| e.to_string())?;
    let (vw, vh, wf, hf) = {
        let v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (vw, vh) = viewer.viewport_size();
        (vw, vh, vw as f32, vh as f32)
    };
    let pc = state.preview_cursor.lock().map_err(|e| e.to_string())?;
    let (
        preview_nx,
        preview_ny,
        texel_sx,
        texel_sy,
        ray_origin_x,
        ray_origin_y,
        ray_origin_z,
        ray_dir_x,
        ray_dir_y,
        ray_dir_z,
    ) = match *pc {
        Some((nx, ny)) => {
            let (sx, sy) = viewport_texels_from_norm(nx, ny, wf, hf);
            let (o, d) = voxel_edit::screen_to_world_ray(&cam, wf, hf, sx, sy);
            (
                Some(nx),
                Some(ny),
                Some(sx),
                Some(sy),
                Some(o.x),
                Some(o.y),
                Some(o.z),
                Some(d.x),
                Some(d.y),
                Some(d.z),
            )
        }
        None => (
            None, None, None, None, None, None, None, None, None, None,
        ),
    };
    Ok(ViewportCursorDebug {
        viewport_width: vw,
        viewport_height: vh,
        preview_nx,
        preview_ny,
        texel_sx,
        texel_sy,
        ray_origin_x,
        ray_origin_y,
        ray_origin_z,
        ray_dir_x,
        ray_dir_y,
        ray_dir_z,
    })
}

#[tauri::command]
fn get_surface_pixel_size(state: State<'_, Arc<ViewerState>>) -> Result<SurfacePixelSize, String> {
    let v = state.viewer.lock().map_err(|e| e.to_string())?;
    let Some(viewer) = v.as_ref() else {
        return Err("viewer not ready".into());
    };
    let (sw, sh) = viewer.surface_pixel_size();
    Ok(SurfacePixelSize {
        width: sw,
        height: sh,
    })
}

#[tauri::command]
fn set_start_screen_light(state: State<'_, Arc<ViewerState>>, light: bool) -> Result<(), String> {
    state
        .start_screen_light
        .store(light, Ordering::Relaxed);
    Ok(())
}

#[tauri::command]
fn viewer_resize(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    surface_width: u32,
    surface_height: u32,
    viewport_x: u32,
    viewport_y: u32,
    viewport_width: u32,
    viewport_height: u32,
) -> Result<(), String> {
    let sw = surface_width.max(1);
    let sh = surface_height.max(1);
    let mut g = state.viewer.lock().map_err(|e| e.to_string())?;
    if let Some(v) = g.as_mut() {
        v.resize(sw, sh, viewport_x, viewport_y, viewport_width, viewport_height);
        let (vw, vh) = v.viewport_size();
        let (sur_w, sur_h) = v.surface_pixel_size();
        let _ = app.emit_to(
            EventTarget::webview_window("main"),
            "viewport-pixel-size",
            ViewportPixelSize {
                width: vw,
                height: vh,
                surface_width: sur_w,
                surface_height: sur_h,
            },
        );
    }
    Ok(())
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct PointerEvent {
    kind: String,
    nx: f32,
    ny: f32,
    dx: f32,
    dy: f32,
    button: i32,
    buttons: u16,
    /// Left-drag pans when true (Three.js-style); otherwise left-drag orbits.
    #[serde(default, rename = "shiftKey")]
    shift_key: bool,
}

#[tauri::command]
fn viewport_pointer(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    ev: PointerEvent,
) -> Result<(), String> {
    if *state.fly_mode.lock().map_err(|e| e.to_string())? {
        return Ok(());
    }
    // Read size without holding `camera` — the run loop locks `viewer` then `camera`; taking
    // `camera` then `viewer` here deadlocks with the render tick and freezes orbit input.
    let (vw, vh) = {
        let v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_ref() else {
            return Ok(());
        };
        let (w, h) = viewer.viewport_size();
        let w = w as f32;
        let h = h as f32;
        (w, h.max(1.0))
    };

    let (x, y) = viewport_texels_from_norm(ev.nx, ev.ny, vw, vh);
    let mut cam = state.camera.lock().map_err(|e| e.to_string())?;
    let logo_splash = cam.logo_splash_rest.is_some();

    match ev.kind.as_str() {
        "down" | "move" => {
            if logo_splash {
                if ev.buttons & 1 != 0 && !ev.shift_key {
                    cam.rotate_screen_logo_splash(ev.dx, ev.dy, vh);
                } else if ev.kind == "move" && ev.buttons & 1 == 0 {
                    cam.set_logo_splash_hover_from_viewport_px(x, y, vw, vh);
                }
            } else {
                // bitmask: 1=left orbit (or shift+left pan), 2=right pan, 4=middle dolly (Three.js OrbitControls defaults)
                if ev.buttons & 1 != 0 {
                    if ev.shift_key {
                        cam.pan_screen(ev.dx, ev.dy, vw, vh);
                    } else {
                        cam.rotate_screen(ev.dx, ev.dy, vh);
                    }
                } else if ev.buttons & 4 != 0 {
                    cam.dolly_delta(ev.dy);
                } else if ev.buttons & 2 != 0 {
                    cam.pan_screen(ev.dx, ev.dy, vw, vh);
                }
            }
        }
        "up" => {
            if logo_splash {
                cam.reset_logo_splash_orbit();
            }
        }
        "leave" => {
            if logo_splash {
                cam.set_logo_splash_hover_from_viewport_px(vw * 0.5, vh * 0.5, vw, vh);
            }
        }
        _ => {}
    }
    wake_viewport_loop(&app);
    Ok(())
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct WheelEvent {
    delta_x: f32,
    delta_y: f32,
}

#[tauri::command]
fn viewport_wheel(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    ev: WheelEvent,
) -> Result<(), String> {
    if *state.fly_mode.lock().map_err(|e| e.to_string())? {
        return Ok(());
    }
    let mut cam = state.camera.lock().map_err(|e| e.to_string())?;
    if cam.logo_splash_rest.is_some() {
        return Ok(());
    }
    // Same `deltaY` semantics as the browser / Three.js `onMouseWheel`.
    cam.dolly_delta(ev.delta_y);
    wake_viewport_loop(&app);
    Ok(())
}

fn scene_bounds_min_max_grid(state: &ViewerState) -> (glam::Vec3, glam::Vec3, i32) {
    if let Ok(guard) = state.last_scene_bounds.lock() {
        if let Some(b) = guard.as_ref() {
            let grid = state
                .current_file
                .lock()
                .ok()
                .and_then(|f| f.as_ref().map(|file| file.grid_size))
                .unwrap_or(64);
            return (b.min, b.max, grid);
        }
    }
    if let Ok(fg) = state.current_file.lock() {
        if let Some(ref file) = *fg {
            let b = greedy_mesh::mesh_bounds_from_voxels_world(&file.voxels, &file.objects)
                .or_else(|| greedy_mesh::mesh_bounds_from_voxels(&file.voxels))
                .unwrap_or_else(|| greedy_mesh::mesh_bounds_for_cube_side(file.grid_size));
            return (b.min, b.max, file.grid_size);
        }
    }
    let grid = 64_i32;
    let b = greedy_mesh::mesh_bounds_for_cube_side(grid);
    (b.min, b.max, grid)
}

fn perspective_zoom_base_dist(min: glam::Vec3, max: glam::Vec3, grid: i32) -> f32 {
    if (max - min).length() > 1e-3 {
        let dx = max.x - min.x;
        let dy = max.y - min.y;
        let dz = max.z - min.z;
        dx.max(dy).max(dz) * 1.5 + 10.0
    } else {
        grid as f32 * 2.5
    }
}

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct OrbitGizmoProjectionItem {
    sx: f32,
    sy: f32,
    depth: f32,
}

#[tauri::command]
fn get_orbit_gizmo_projection(state: State<'_, Arc<ViewerState>>) -> Result<Vec<OrbitGizmoProjectionItem>, String> {
    let cam = state.camera.lock().map_err(|e| e.to_string())?;
    let axes = cam.gizmo_axis_projections();
    const R: f32 = 40.0;
    Ok(axes
        .into_iter()
        .map(|a| OrbitGizmoProjectionItem {
            sx: a[0] * R,
            sy: -a[1] * R,
            depth: a[2],
        })
        .collect())
}

#[tauri::command]
fn get_camera_zoom_percent(state: State<'_, Arc<ViewerState>>) -> Result<i32, String> {
    let (min, max, grid) = scene_bounds_min_max_grid(state.inner());
    let cam = state.camera.lock().map_err(|e| e.to_string())?;
    let base = perspective_zoom_base_dist(min, max, grid);
    let r = (max - min).length() * 0.5;
    let ortho_ref = if r > 1e-3 {
        r * 1.1
    } else {
        (grid as f32) * 1.1
    };
    Ok(cam.zoom_percent_for_display(base, ortho_ref))
}

#[tauri::command]
fn camera_fit_to_scene(app: AppHandle, state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
    if *state.fly_mode.lock().map_err(|e| e.to_string())? {
        return Ok(());
    }
    let (min, max, _) = scene_bounds_min_max_grid(state.inner());
    let (w, h) = {
        let v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_ref() else {
            return Ok(());
        };
        viewer.viewport_size()
    };
    let mut cam = state.camera.lock().map_err(|e| e.to_string())?;
    cam.fit_to_aabb_preserving_view(min, max, w as f32, h as f32);
    drop(cam);
    wake_viewport_loop(&app);
    Ok(())
}

#[tauri::command]
fn camera_reset_view(app: AppHandle, state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
    if *state.fly_mode.lock().map_err(|e| e.to_string())? {
        return Ok(());
    }
    let (min, max, grid) = scene_bounds_min_max_grid(state.inner());
    let mut cam = state.camera.lock().map_err(|e| e.to_string())?;
    cam.reset_view_to_bounds(min, max, grid as f32);
    drop(cam);
    wake_viewport_loop(&app);
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct OrbitGizmoDragArgs {
    dx: f32,
    dy: f32,
    theta_only: bool,
}

#[tauri::command]
fn camera_orbit_gizmo_drag(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: OrbitGizmoDragArgs,
) -> Result<(), String> {
    if *state.fly_mode.lock().map_err(|e| e.to_string())? {
        return Ok(());
    }
    let mut cam = state.camera.lock().map_err(|e| e.to_string())?;
    cam.orbit_gizmo_drag(args.dx, args.dy, args.theta_only);
    drop(cam);
    wake_viewport_loop(&app);
    Ok(())
}

#[tauri::command]
fn camera_snap_orbit_axis(app: AppHandle, state: State<'_, Arc<ViewerState>>, axis: u8) -> Result<(), String> {
    if *state.fly_mode.lock().map_err(|e| e.to_string())? {
        return Ok(());
    }
    let mut cam = state.camera.lock().map_err(|e| e.to_string())?;
    cam.snap_to_axis(axis);
    drop(cam);
    wake_viewport_loop(&app);
    Ok(())
}

#[tauri::command]
fn camera_zoom_step(app: AppHandle, state: State<'_, Arc<ViewerState>>, inward: bool) -> Result<(), String> {
    if *state.fly_mode.lock().map_err(|e| e.to_string())? {
        return Ok(());
    }
    let mut cam = state.camera.lock().map_err(|e| e.to_string())?;
    cam.zoom_step(inward);
    drop(cam);
    wake_viewport_loop(&app);
    Ok(())
}

/// Brick axis cap must match [`WgpuViewer::upload_scene_data`] (`MAX_AXIS` 512).
const LOAD_SCENE_BRICK_MAX_AXIS: u32 = 512;

/// Load pipeline progress for the webview (`voxelle-load-progress`): overall fraction and short phase label.
#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LoadProgressPayload {
    pub fraction: f32,
    pub phase: String,
}

pub(crate) fn emit_load_progress<R: Runtime>(app: &AppHandle<R>, fraction: f32, phase: impl Into<String>) {
    let _ = app.emit(
        "voxelle-load-progress",
        LoadProgressPayload {
            fraction: fraction.clamp(0.0, 1.0),
            phase: phase.into(),
        },
    );
}

/// Status bar progress for save, heavy mesh refresh, undo/redo (webview `voxelle-work-progress`).
pub(crate) fn emit_work_progress<R: Runtime>(app: &AppHandle<R>, fraction: f32, phase: impl Into<String>) {
    let _ = app.emit(
        "voxelle-work-progress",
        LoadProgressPayload {
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

#[derive(Clone, Copy)]
pub(crate) enum VoxelGpuRefreshReason {
    SoloEdit,
    Undo,
    Redo,
    CollabApply,
}

impl VoxelGpuRefreshReason {
    fn label(self) -> &'static str {
        match self {
            VoxelGpuRefreshReason::SoloEdit => "Applying edit…",
            VoxelGpuRefreshReason::Undo => "Undo…",
            VoxelGpuRefreshReason::Redo => "Redo…",
            VoxelGpuRefreshReason::CollabApply => "Applying remote edit…",
        }
    }
}

/// Show status progress for voxel GPU refresh when the scene is large or a full mesh rebuild is required.
fn work_progress_for_voxel_refresh(
    viewer: &WgpuViewer,
    file: &voxelle::VoxelleFile,
    rm: RenderingMode,
) -> bool {
    let nv = file.voxels.len();
    if nv < 4_000 {
        return false;
    }
    if rm.uses_smooth_surface() {
        return nv >= 6_000;
    }
    let Some(origin_new) = greedy_mesh::voxel_aabb_min_int(&file.voxels) else {
        return false;
    };
    let origin_iv = glam::IVec3::new(origin_new.0, origin_new.1, origin_new.2);
    let single_object = file
        .voxels
        .iter()
        .map(|v| v.object_id)
        .collect::<std::collections::HashSet<_>>()
        .len()
        <= 1;
    let use_incremental = viewer.opaque_chunked
        && single_object
        && nv >= greedy_mesh::CHUNKED_CPU_MESH_MIN_VOXELS
        && viewer.chunk_grid_origin == origin_iv;
    !use_incremental || nv >= 32_768
}

/// Progress band inside [`prepare_load_scene_cpu`] (mesh build uses [`LOAD_P_MESH_START`]..[`LOAD_P_MESH_END`]).
const LOAD_P_BOUNDS: f32 = 0.22;
const LOAD_P_BRICK: f32 = 0.26;
const LOAD_P_MESH_START: f32 = 0.28;
const LOAD_P_MESH_END: f32 = 0.74;

pub(crate) struct PreparedLoadScene {
    pub bounds: greedy_mesh::MeshBounds,
    pub brick: GpuVoxelBrick,
    pub opaque: PreparedOpaqueUpload,
}

/// CPU-only work for open/new/collab snapshot: greedy mesh + brick layout off the AppKit main thread.
pub(crate) fn prepare_load_scene_cpu<R: Runtime>(
    grid_size: i32,
    voxels: &[voxelle::Voxel],
    objects: &[voxelle::SceneObject],
    mode: RenderingMode,
    app: Option<&AppHandle<R>>,
) -> Result<PreparedLoadScene, String> {
    let emit = |frac: f32, phase: &str| {
        if let Some(a) = app {
            emit_load_progress(a, frac, phase);
        }
    };

    let t_prep = Instant::now();
    let nv = voxels.len();
    log::info!(
        target: "voxelle_load",
        "prepare_load_scene_cpu: start voxels={nv} grid_size={grid_size} mode={mode:?}"
    );

    let t = Instant::now();
    let bounds = if voxels.is_empty() {
        greedy_mesh::mesh_bounds_for_cube_side(grid_size)
    } else {
        greedy_mesh::mesh_bounds_from_voxels_world(voxels, objects)
            .or_else(|| greedy_mesh::mesh_bounds_from_voxels(voxels))
            .unwrap_or_else(|| greedy_mesh::mesh_bounds_for_cube_side(grid_size))
    };
    emit(LOAD_P_BOUNDS, "Computing bounds…");
    log::info!(
        target: "voxelle_load",
        "prepare_load_scene_cpu: bounds {:?}",
        t.elapsed()
    );

    let t = Instant::now();
    let brick = GpuVoxelBrick::from_voxels(voxels, LOAD_SCENE_BRICK_MAX_AXIS).unwrap_or(
        GpuVoxelBrick {
            origin: glam::IVec3::ZERO,
            dims: (0, 0, 0),
            cells: vec![0u32],
        },
    );
    emit(LOAD_P_BRICK, "Packing voxel brick…");
    log::info!(
        target: "voxelle_load",
        "prepare_load_scene_cpu: gpu brick {:?}",
        t.elapsed()
    );

    let mesh_span = LOAD_P_MESH_END - LOAD_P_MESH_START;
    let visible_voxels = voxelle::scene::visible_voxels_for_meshing(voxels, objects);
    let opaque = if voxels.is_empty() {
        log::info!(target: "voxelle_load", "prepare_load_scene_cpu: mesh route = (no voxels)");
        PreparedOpaqueUpload::Empty
    } else if mode.uses_smooth_surface() {
        log::info!(
            target: "voxelle_load",
            "prepare_load_scene_cpu: mesh route = smooth ({mode:?})"
        );
        emit(LOAD_P_MESH_START, "Building surface mesh…");
        let t = Instant::now();
        let mesh = match mode {
            RenderingMode::MarchingCubes => smooth_mesh::build_marching_cubes_merged(voxels),
            RenderingMode::DualContour => smooth_mesh::build_dual_contour_merged(voxels),
            _ => unreachable!(),
        };
        emit(LOAD_P_MESH_END, "Surface mesh ready…");
        log::info!(
            target: "voxelle_load",
            "prepare_load_scene_cpu: smooth surface {:?}",
            t.elapsed()
        );
        if mesh.indices.is_empty() {
            PreparedOpaqueUpload::Empty
        } else {
            PreparedOpaqueUpload::Single(mesh)
        }
    } else if visible_voxels.len() >= greedy_mesh::CHUNKED_CPU_MESH_MIN_VOXELS
        && visible_voxels
            .iter()
            .map(|v| v.object_id)
            .collect::<std::collections::HashSet<_>>()
            .len()
            <= 1
    {
        log::info!(
            target: "voxelle_load",
            "prepare_load_scene_cpu: mesh route = chunked greedy (>={} voxels, chunk {})",
            greedy_mesh::CHUNKED_CPU_MESH_MIN_VOXELS,
            greedy_mesh::SPATIAL_CHUNK_SIZE
        );
        let t = Instant::now();
        let chunked = greedy_mesh::build_chunk_meshes_and_spatial_cache(
            &visible_voxels,
            greedy_mesh::SPATIAL_CHUNK_SIZE,
            |frac, done, total| {
                let g = LOAD_P_MESH_START + frac * mesh_span;
                let pct = (frac * 100.0).min(100.0) as u32;
                emit(
                    g,
                    &format!("Building mesh chunks {done}/{total} ({pct}%)"),
                );
            },
        );
        log::info!(
            target: "voxelle_load",
            "prepare_load_scene_cpu: chunked build {:?}",
            t.elapsed()
        );
        match chunked {
            Some((origin, meshes, spatial_cache)) if !meshes.is_empty() => {
                let chunk_origin = glam::IVec3::new(origin.0, origin.1, origin.2);
                PreparedOpaqueUpload::Chunked {
                    chunk_origin,
                    meshes,
                    spatial_cache,
                }
            }
            _ => PreparedOpaqueUpload::Empty,
        }
    } else {
        log::info!(target: "voxelle_load", "prepare_load_scene_cpu: mesh route = greedy (single pass)");
        emit(LOAD_P_MESH_START, "Building mesh…");
        let t = Instant::now();
        let (mesh, _) = greedy_mesh::build_greedy_mesh(voxels, objects);
        emit(LOAD_P_MESH_END, "Mesh ready…");
        log::info!(
            target: "voxelle_load",
            "prepare_load_scene_cpu: greedy mesh {:?}",
            t.elapsed()
        );
        PreparedOpaqueUpload::Single(mesh)
    };

    log::info!(
        target: "voxelle_load",
        "prepare_load_scene_cpu: total {:?}",
        t_prep.elapsed()
    );

    Ok(PreparedLoadScene {
        bounds,
        brick,
        opaque,
    })
}

/// Clears the loaded model, GPU meshes, and editing state. Must run on the main thread (GPU + AppKit undo).
fn unload_current_project<R: Runtime>(state: &Arc<ViewerState>, app: &AppHandle<R>) -> Result<(), String> {
    let mode = *state.rendering_mode.lock().map_err(|e| e.to_string())?;
    let objects = voxelle::default_scene_objects();
    let prepared = prepare_load_scene_cpu::<R>(
        MAX_GRID_SIZE as i32,
        &[],
        &objects,
        mode,
        None,
    )?;
    {
        let mut cf = state.current_file.lock().map_err(|e| e.to_string())?;
        let mut vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
        *cf = None;
        *vm = None;
    }
    state.active_project.store(false, Ordering::Release);
    let mut v = state.viewer.lock().map_err(|e| e.to_string())?;
    let Some(viewer) = v.as_mut() else {
        return Err("viewer not ready".into());
    };
    viewer.upload_scene_data_from_brick(prepared.bounds, prepared.brick);
    viewer.upload_prepared_opaque(prepared.opaque);
    viewer.clear_preview_mesh();
    viewer.clear_selection_overlay();
    viewer.clear_collab_peer_lines();
    viewer.clear_ping_mesh();
    viewer.set_mood_params(0.0, 0.0, 0.0, 0.0, 0.0);
    drop(v);

    *state.last_scene_bounds.lock().map_err(|e| e.to_string())? = Some(prepared.bounds);
    *state.voxel_edit_stats_cache.lock().map_err(|e| e.to_string())? = None;
    *state.last_edit_perf.lock().map_err(|e| e.to_string())? = None;
    state.mesh_refresh_generation.fetch_add(1, Ordering::Release);

    if let Ok(mut u) = state.solo_undo.lock() {
        u.clear();
    }
    if let Ok(mut r) = state.solo_redo.lock() {
        r.clear();
    }
    #[cfg(target_os = "macos")]
    macos_undo::clear_all(app);

    *state.selection_cells.lock().map_err(|e| e.to_string())? = AHashSet::default();
    *state.selection_stroke_before.lock().map_err(|e| e.to_string())? = None;
    *state.selection_combine_mode.lock().map_err(|e| e.to_string())? = SelectionCombineMode::default();
    *state.stamp_clipboard.lock().map_err(|e| e.to_string())? = None;
    *state.stroke_buffer.lock().map_err(|e| e.to_string())? = Vec::new();
    *state.stroke_preview_union.lock().map_err(|e| e.to_string())? = AHashSet::default();
    *state.stroke_preview_last_args.lock().map_err(|e| e.to_string())? = None;
    state
        .stroke_preview_suppresses_hover
        .store(false, Ordering::Release);
    *state.sculpt_stroke_replay.lock().map_err(|e| e.to_string())? = Vec::new();
    *state.stroke_active.lock().map_err(|e| e.to_string())? = false;
    *state.ping_flash.lock().map_err(|e| e.to_string())? = None;
    *state.preview_cursor.lock().map_err(|e| e.to_string())? = None;

    if let Ok(mut g) = state.squishy_session.lock() {
        g.clear();
    }

    log::info!(target: "voxelle_load", "unload_current_project: done");
    Ok(())
}

fn run_unload_on_main_thread<R: Runtime>(state: &Arc<ViewerState>, app: &AppHandle<R>) -> Result<(), String> {
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let state_c = Arc::clone(state);
    let app_c = app.clone();
    app
        .run_on_main_thread(move || {
            let r = unload_current_project(&state_c, &app_c);
            let _ = done_tx.send(r);
        })
        .map_err(|e| format!("could not schedule unload: {e}"))?;
    done_rx
        .recv()
        .map_err(|_| "unload disconnected".to_string())?
}

fn load_thread_panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        return (*s).to_string();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    "load thread panicked (details on stderr)".to_string()
}

fn start_shape_label(shape: StartShape) -> &'static str {
    match shape {
        StartShape::Cube => "cube",
        StartShape::Orb => "orb",
        StartShape::Cylinder => "cylinder",
        StartShape::HollowCube => "hollow cube",
        StartShape::Plane => "plane",
        StartShape::Circle => "circle",
        StartShape::Empty => "empty",
    }
}

fn spawn_new_project(state: Arc<ViewerState>, app: AppHandle, grid_size: u32, shape: StartShape) {
    let shape_l = start_shape_label(shape);
    let label = format!("New project ({grid_size}³, {shape_l})");
    let app_spawn_err = app.clone();
    match std::thread::Builder::new()
        .name("voxelle-new-project".into())
        .spawn(move || {
            if let Err(e) = run_unload_on_main_thread(&state, &app) {
                let _ = app.emit("voxelle-load-error", e);
                return;
            }
            state
                .start_screen_logo_transparent
                .store(false, Ordering::Release);
            emit_load_progress(&app, 0.05, "Starting…");

            let mesh_result: Result<(), String> = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || {
                    (|| -> Result<(), String> {
                        let size = grid_size as i32;
                        let voxels = voxelle::start_shape::voxels_for_start_shape(size, shape)?;
                        let file = voxelle::VoxelleFile {
                            version: 3,
                            grid_size: size,
                            scene: Default::default(),
                            scene_extra: None,
                            mood: None,
                            lighting: None,
                            voxels,
                            objects: voxelle::default_scene_objects(),
                            active_object_id: 0,
                        };

                        let mode = *state
                            .rendering_mode
                            .lock()
                            .map_err(|e| e.to_string())?;
                        let prepared = prepare_load_scene_cpu(
                            file.grid_size,
                            &file.voxels,
                            &file.objects,
                            mode,
                            Some(&app),
                        )?;

                        if file.voxels.is_empty() {
                            let (done_tx, done_rx) = std::sync::mpsc::channel();
                            let app_c = app.clone();
                            let app_mesh = app_c.clone();
                            let state_c = Arc::clone(&state);
                            let file_c = file;
                            let _ = app_c.run_on_main_thread(move || {
                                let res = apply_mesh_and_camera(&state_c, &app_mesh, file_c, prepared, false);
                                let _ = done_tx.send(res);
                            });
                            return match done_rx.recv() {
                                Ok(Ok(())) => Ok(()),
                                Ok(Err(e)) => Err(e),
                                Err(_) => Err("main thread disconnected".into()),
                            };
                        }

                        run_v3_mesh_on_main(&state, &app, file, prepared, false)?;
                        Ok(())
                    })()
                },
            )) {
                Ok(inner) => inner,
                Err(payload) => Err(load_thread_panic_message(payload)),
            };

            let (done_tx, done_rx) = std::sync::mpsc::channel();
            if let Err(e) = app.run_on_main_thread(move || {
                let out = match mesh_result {
                    Ok(()) => Ok(()),
                    Err(e) => Err(e),
                };
                let _ = done_tx.send(out);
            }) {
                log::warn!(
                    target: "voxelle_load",
                    "new project: run_on_main_thread failed: {e}"
                );
                let _ = app.emit(
                    "voxelle-load-error",
                    format!("could not finish new project: {e}"),
                );
                return;
            }
            match done_rx.recv() {
                Ok(Ok(())) => {
                    emit_voxelle_loaded(&app, label, &state, false);
                }
                Ok(Err(e)) => {
                    let _ = app.emit("voxelle-load-error", e);
                }
                Err(_) => {
                    let _ = app.emit(
                        "voxelle-load-error",
                        "new project pipeline disconnected".to_string(),
                    );
                }
            }
        }) {
        Ok(_) => {}
        Err(e) => {
            let _ = app_spawn_err.emit(
                "voxelle-load-error",
                format!("could not start new-project thread: {e}"),
            );
        }
    }
}

enum DecodeMeshOutcome {
    /// Main thread only uploads GPU buffers; mesh/brick built on load thread.
    ApplyOnce {
        file: voxelle::VoxelleFile,
        prepared: PreparedLoadScene,
    },
    /// v3 with voxels: mesh already applied inside `run_v3_mesh_on_main`.
    Done,
}

fn run_v3_mesh_on_main(
    state: &Arc<ViewerState>,
    app: &AppHandle,
    file: voxelle::VoxelleFile,
    prepared: PreparedLoadScene,
    logo_splash: bool,
) -> Result<(), String> {
    log::info!(
        target: "voxelle_load",
        "run_v3_mesh_on_main: dispatch upload to main thread (voxels={})",
        file.voxels.len()
    );
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let app_c = app.clone();
    let app_mesh = app_c.clone();
    let state_c = Arc::clone(state);
    let t_main = Instant::now();
    let _ = app_c.run_on_main_thread(move || {
        let res = apply_mesh_and_camera(&state_c, &app_mesh, file, prepared, logo_splash);
        let _ = done_tx.send(res);
    });
    match done_rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err("main thread disconnected".into()),
    }
    log::info!(
        target: "voxelle_load",
        "run_v3_mesh_on_main: main-thread apply_mesh_and_camera {:?}",
        t_main.elapsed()
    );
    Ok(())
}

fn spawn_decode_and_mesh(state: Arc<ViewerState>, app: AppHandle, path: PathBuf) {
    let label = path.to_string_lossy().to_string();
    spawn_decode_and_mesh_with_label(state, app, path, label, false);
}

fn spawn_decode_and_mesh_with_label(
    state: Arc<ViewerState>,
    app: AppHandle,
    read_from: PathBuf,
    file_label: String,
    start_screen_logo: bool,
) {
    let app_spawn_err = app.clone();
    match std::thread::Builder::new()
        .name("voxelle-load".into())
        .spawn(move || {
            if let Err(e) = run_unload_on_main_thread(&state, &app) {
                let _ = app.emit("voxelle-load-error", e);
                return;
            }
            let label = file_label;
            emit_load_progress(&app, 0.05, "Starting…");

            let mesh_result: Result<DecodeMeshOutcome, String> = match std::panic::catch_unwind(
                std::panic::AssertUnwindSafe(|| {
                    (|| -> Result<DecodeMeshOutcome, String> {
                        let t = Instant::now();
                        let bytes = std::fs::read(&read_from).map_err(|e| e.to_string())?;
                        log::info!(
                            target: "voxelle_load",
                            "load file: read {} bytes from disk {:?}",
                            bytes.len(),
                            t.elapsed()
                        );
                        emit_load_progress(&app, 0.12, "Reading file…");
                        let t = Instant::now();
                        let file = decode_payload(&bytes).map_err(|e| e.to_string())?;
                        log::info!(
                            target: "voxelle_load",
                            "load file: decoded v{} grid_size={} voxels={} {:?}",
                            file.version,
                            file.grid_size,
                            file.voxels.len(),
                            t.elapsed()
                        );
                        emit_load_progress(&app, 0.18, "Preparing scene…");
                        let mode = *state
                            .rendering_mode
                            .lock()
                            .map_err(|e| e.to_string())?;
                        let prepared = prepare_load_scene_cpu(
                            file.grid_size,
                            &file.voxels,
                            &file.objects,
                            mode,
                            Some(&app),
                        )?;

                        if file.version == 3 && !file.voxels.is_empty() {
                            run_v3_mesh_on_main(&state, &app, file, prepared, start_screen_logo)?;
                            return Ok(DecodeMeshOutcome::Done);
                        }

                        Ok(DecodeMeshOutcome::ApplyOnce { file, prepared })
                    })()
                }),
            ) {
                Ok(inner) => inner,
                Err(payload) => Err(load_thread_panic_message(payload)),
            };

            let (done_tx, done_rx) = std::sync::mpsc::channel();
            let state_c = Arc::clone(&state);
            let app_emit = app.clone();
            if let Err(e) = app.run_on_main_thread(move || {
                let res: Result<(), String> = match mesh_result {
                    Ok(DecodeMeshOutcome::ApplyOnce { file, prepared }) => {
                        let t = Instant::now();
                        let r = apply_mesh_and_camera(&state_c, &app_emit, file, prepared, start_screen_logo);
                        log::info!(
                            target: "voxelle_load",
                            "load file: ApplyOnce apply_mesh_and_camera {:?}",
                            t.elapsed()
                        );
                        r
                    }
                    Ok(DecodeMeshOutcome::Done) => Ok(()),
                    Err(e) => Err(e),
                };
                let _ = done_tx.send(res);
            }) {
                log::warn!(
                    target: "voxelle_load",
                    "open file: run_on_main_thread failed after decode: {e}"
                );
                let _ = app.emit(
                    "voxelle-load-error",
                    format!("could not finish loading: {e}"),
                );
                return;
            }
            match done_rx.recv() {
                Ok(Ok(())) => {
                    if start_screen_logo {
                        emit_voxelle_loaded(&app, String::new(), &state, true);
                    } else {
                        if label.ends_with(".voxelle") {
                            persist_last_document_path(&app, &label);
                        }
                        emit_voxelle_loaded(&app, label, &state, false);
                    }
                }
                Ok(Err(e)) => {
                    let _ = app.emit("voxelle-load-error", e);
                }
                Err(_) => {
                    let _ = app.emit(
                        "voxelle-load-error",
                        "load pipeline disconnected".to_string(),
                    );
                }
            }
        }) {
        Ok(_) => {}
        Err(e) => {
            let _ = app_spawn_err.emit(
                "voxelle-load-error",
                format!("could not start load thread: {e}"),
            );
        }
    }
}

pub(crate) fn apply_mesh_and_camera<R: Runtime>(
    state: &Arc<ViewerState>,
    app: &AppHandle<R>,
    file: voxelle::VoxelleFile,
    prepared: PreparedLoadScene,
    logo_splash: bool,
) -> Result<(), String> {
    emit_load_progress(app, 0.76, "Uploading scene to GPU…");
    let PreparedLoadScene {
        bounds,
        brick,
        opaque,
    } = prepared;
    let fl = file.scene.focal_length_mm.unwrap_or(29.0);
    let orthographic = file.scene.orthographic;
    let voxel_edit_stats_cache = if file.voxels.is_empty() {
        None
    } else {
        Some(voxel_aabb_min_and_single_object_one_pass(&file.voxels))
    };
    let voxel_map = greedy_mesh::voxel_map_indices(&file.voxels);
    let mood = file.mood;
    let lighting = file.lighting.clone().unwrap_or_default();
    {
        let mut cf = state.current_file.lock().map_err(|e| e.to_string())?;
        let mut vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
        *cf = Some(file);
        *vm = Some(voxel_map);
    }
    let mut v = state.viewer.lock().map_err(|e| e.to_string())?;
    let Some(viewer) = v.as_mut() else {
        return Err("viewer not ready".into());
    };
    viewer.upload_scene_data_from_brick(bounds, brick);
    viewer.upload_prepared_opaque(opaque);
    viewer.clear_preview_mesh();
    if let Some(m) = mood {
        viewer.set_mood_params(
            m.grain,
            m.vignette,
            m.distance_tint,
            m.atmosphere,
            m.sun_shafts,
        );
    }
    viewer.apply_lighting_settings(&lighting);

    let mut cam = state.camera.lock().map_err(|e| e.to_string())?;
    cam.fov_y = focal_length_to_fov_y_radians(fl);
    cam.perspective = !orthographic;
    if orthographic {
        let r = bounds.radius().max(1.0);
        cam.ortho_half_height = r * 1.1;
    }

    let center = bounds.center();
    let r = bounds.radius().max(1.0);
    let (w, h) = viewer.viewport_size();
    cam.fit_sphere(center, r, w as f32, h as f32);
    if logo_splash {
        cam.configure_logo_splash_after_fit();
    } else {
        cam.logo_splash_rest = None;
    }
    *state.last_scene_bounds.lock().map_err(|e| e.to_string())? = Some(bounds);
    *state.voxel_edit_stats_cache.lock().map_err(|e| e.to_string())? = voxel_edit_stats_cache;
    if let Ok(mut u) = state.solo_undo.lock() {
        u.clear();
    }
    if let Ok(mut r) = state.solo_redo.lock() {
        r.clear();
    }
    #[cfg(target_os = "macos")]
    macos_undo::clear_all(app);
    collab::broadcast_snapshot_to_guests(state);
    state.active_project.store(true, Ordering::Release);
    emit_load_progress(app, 0.97, "Finishing…");
    emit_load_progress(app, 1.0, "");
    Ok(())
}

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct VoxelleLoadedEvent {
    path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    mood: Option<voxelle::MoodSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lighting: Option<voxelle::LightingSettings>,
    #[serde(default)]
    start_screen_logo: bool,
}

pub(crate) fn emit_voxelle_loaded<R: Runtime>(
    app: &AppHandle<R>,
    path: String,
    state: &ViewerState,
    start_screen_logo: bool,
) {
    state.start_screen_logo_transparent.store(
        start_screen_logo,
        Ordering::Release,
    );
    let (mood, lighting) = match state.current_file.lock().ok().as_ref().and_then(|g| g.as_ref()) {
        Some(f) => (f.mood, f.lighting.clone()),
        None => (None, None),
    };
    let _ = app.emit(
        "voxelle-loaded",
        VoxelleLoadedEvent {
            path,
            mood,
            lighting,
            start_screen_logo,
        },
    );
}

fn scene_bounds_for_edit(
    state: &ViewerState,
    file: &voxelle::VoxelleFile,
    delta: &voxel_edit::VoxelEditDelta,
) -> Result<greedy_mesh::MeshBounds, String> {
    if file.voxels.is_empty() {
        return Ok(greedy_mesh::mesh_bounds_for_cube_side(file.grid_size));
    }
    let default_objs = voxelle::default_scene_objects();
    let objs: &[voxelle::SceneObject] = if file.objects.is_empty() {
        default_objs.as_slice()
    } else {
        &file.objects
    };
    if voxelle::scene::scene_objects_identity_for_bounds_fast_path(objs) {
        if let Ok(guard) = state.last_scene_bounds.lock() {
            if let Some(prev) = guard.as_ref() {
                match delta {
                    voxel_edit::VoxelEditDelta::Added(v) => {
                        return Ok(greedy_mesh::mesh_bounds_expand_with_voxel(prev, v));
                    }
                    voxel_edit::VoxelEditDelta::Removed { voxel } => {
                        if greedy_mesh::mesh_bounds_remove_is_strict_interior(
                            prev,
                            voxel.x,
                            voxel.y,
                            voxel.z,
                        ) {
                            return Ok(*prev);
                        }
                    }
                    voxel_edit::VoxelEditDelta::Painted { .. } => {
                        return Ok(*prev);
                    }
                }
            }
        }
    }
    Ok(
        greedy_mesh::mesh_bounds_from_voxels_world(&file.voxels, &file.objects)
            .or_else(|| greedy_mesh::mesh_bounds_from_voxels(&file.voxels))
            .unwrap_or_else(|| greedy_mesh::mesh_bounds_for_cube_side(file.grid_size)),
    )
}

fn resolve_voxel_edit_stats_batch(
    voxels: &[voxelle::Voxel],
    deltas: &[voxel_edit::VoxelEditDelta],
    cached: Option<VoxelEditStatsCache>,
) -> VoxelEditStatsCache {
    if deltas.len() == 1 {
        return resolve_voxel_edit_stats(voxels, &deltas[0], cached);
    }
    voxel_aabb_min_and_single_object_one_pass(voxels)
}

/// GPU upload + mesh rebuild after one or more voxel changes (shared by edit, undo, redo, collab).
pub(crate) fn finish_voxel_edit_gpu_deltas<R: Runtime>(
    state: &Arc<ViewerState>,
    deltas: &[voxel_edit::VoxelEditDelta],
    apply_edit_ms: f64,
    t_total: Instant,
    app: &AppHandle<R>,
    reason: VoxelGpuRefreshReason,
) -> Result<(), String> {
    if deltas.is_empty() {
        return Ok(());
    }
    let t_prep_start = Instant::now();
    let fg = state.current_file.lock().map_err(|e| e.to_string())?;
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };

    let bounds = scene_bounds_for_edits(state.as_ref(), file, deltas)?;

    let prepare_ms = t_prep_start.elapsed().as_secs_f64() * 1000.0;

    let t_lock_start = Instant::now();
    let mut v = state.viewer.lock().map_err(|e| e.to_string())?;
    let viewer_lock_wait_ms = t_lock_start.elapsed().as_secs_f64() * 1000.0;
    let Some(viewer) = v.as_mut() else {
        return Err("viewer not ready".into());
    };

    let rm = *state.rendering_mode.lock().map_err(|e| e.to_string())?;
    let show_work = work_progress_for_voxel_refresh(viewer, file, rm);
    let mut wp = WorkProgressGuard::new(app);
    if show_work {
        wp.arm();
        emit_work_progress(app, 0.12, reason.label());
    }

    let t_brick = Instant::now();
    if file.voxels.is_empty() {
        viewer.upload_scene_data(bounds, &file.voxels, None);
    } else if deltas.len() == 1 {
        let brick_patch = Some(match &deltas[0] {
            voxel_edit::VoxelEditDelta::Added(v) => BrickCellWrite {
                x: v.x,
                y: v.y,
                z: v.z,
                packed: gpu_brick::pack_cell(v.color, v.material),
            },
            voxel_edit::VoxelEditDelta::Removed { voxel } => BrickCellWrite {
                x: voxel.x,
                y: voxel.y,
                z: voxel.z,
                packed: gpu_brick::pack_empty(),
            },
            voxel_edit::VoxelEditDelta::Painted { after, .. } => BrickCellWrite {
                x: after.x,
                y: after.y,
                z: after.z,
                packed: gpu_brick::pack_cell(after.color, after.material),
            },
        });
        viewer.upload_scene_data(bounds, &file.voxels, brick_patch);
    } else {
        let patches = deltas_to_brick_patches(deltas);
        viewer.upload_scene_data_patches(bounds, &file.voxels, &patches);
    }
    let brick_ms = t_brick.elapsed().as_secs_f64() * 1000.0;

    if show_work {
        emit_work_progress(app, 0.38, "Rebuilding mesh…");
    }

    let t_mesh = Instant::now();
    let mut mesh_voxel_map_ms = 0.0;
    let mut mesh_buckets_ms = 0.0;
    let mut mesh_greedy_ms = 0.0;
    let mut mesh_greedy_gpu_ms = 0.0;
    let mut mesh_greedy_cpu_ms = 0.0;
    let mut mesh_chunk_buffers_ms = 0.0;
    let mut mesh_full_chunked_rebuild_ms = 0.0;
    let mut mesh_pipeline_ms = 0.0;

    if file.voxels.is_empty() {
        viewer.upload_mesh(&greedy_mesh::MeshBuffers::default());
        viewer.last_mesh_route = "clear".to_string();
        if let Ok(mut g) = state.voxel_edit_stats_cache.lock() {
            *g = None;
        }
    } else if rm.uses_smooth_surface() {
        let mesh = match rm {
            RenderingMode::MarchingCubes => smooth_mesh::build_marching_cubes_merged(&file.voxels),
            RenderingMode::DualContour => smooth_mesh::build_dual_contour_merged(&file.voxels),
            _ => unreachable!(),
        };
        viewer.upload_mesh(&mesh);
        viewer.last_mesh_route = match rm {
            RenderingMode::MarchingCubes => "marching_cubes".to_string(),
            RenderingMode::DualContour => "dual_contour".to_string(),
            _ => unreachable!(),
        };
        if let Ok(mut g) = state.voxel_edit_stats_cache.lock() {
            *g = Some(voxel_aabb_min_and_single_object_one_pass(&file.voxels));
        }
    } else {
        let cached_stats = state
            .voxel_edit_stats_cache
            .lock()
            .ok()
            .and_then(|g| *g);
        let voxel_stats =
            resolve_voxel_edit_stats_batch(&file.voxels, deltas, cached_stats);
        let origin_new = voxel_stats.aabb_min;
        let single_object = voxel_stats.common_object_id.is_some();
        let origin_iv = glam::IVec3::new(origin_new.0, origin_new.1, origin_new.2);
        let use_incremental = viewer.opaque_chunked
            && single_object
            && file.voxels.len() >= greedy_mesh::CHUNKED_CPU_MESH_MIN_VOXELS
            && viewer.chunk_grid_origin == origin_iv;

        if use_incremental {
            let t_cache = Instant::now();
            for d in deltas {
                viewer.apply_spatial_cache_edit(d);
            }
            mesh_voxel_map_ms = t_cache.elapsed().as_secs_f64() * 1000.0;
            let dirty = union_dirty_chunk_keys_for_deltas(
                deltas,
                origin_new,
                greedy_mesh::SPATIAL_CHUNK_SIZE,
            );
            let (ok, rperf) = viewer.remesh_opaque_chunks(
                &dirty,
                &file.voxels,
                if show_work { Some(app) } else { None },
            );
            mesh_buckets_ms = rperf.buckets_ms;
            mesh_greedy_ms = rperf.greedy_ms;
            mesh_greedy_gpu_ms = rperf.greedy_gpu_ms;
            mesh_greedy_cpu_ms = rperf.greedy_cpu_ms;
            mesh_chunk_buffers_ms = rperf.chunk_buffers_ms;
            mesh_full_chunked_rebuild_ms = rperf.full_chunked_rebuild_ms;
            if ok {
                viewer.last_mesh_route = "cpu_chunked_incremental".to_string();
            }
        } else {
            let t_pipe = Instant::now();
            let _ = viewer.rebuild_mesh_gpu_greedy(&file.voxels, &file.objects, file.grid_size);
            mesh_pipeline_ms = t_pipe.elapsed().as_secs_f64() * 1000.0;
        }
        if let Ok(mut g) = state.voxel_edit_stats_cache.lock() {
            *g = Some(voxel_stats);
        }
    }
    let mesh_ms = t_mesh.elapsed().as_secs_f64() * 1000.0;

    let t_preview_clear = Instant::now();
    viewer.clear_preview_mesh();
    let preview_clear_ms = t_preview_clear.elapsed().as_secs_f64() * 1000.0;

    let mesh_route = viewer.last_mesh_route.clone();
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    *state.last_edit_perf.lock().map_err(|e| e.to_string())? = Some(EditPerfBreakdown {
        apply_edit_ms,
        prepare_ms,
        viewer_lock_wait_ms,
        brick_ms,
        mesh_ms,
        total_ms,
        mesh_route,
        mesh_voxel_map_ms,
        mesh_buckets_ms,
        mesh_greedy_ms,
        mesh_greedy_gpu_ms,
        mesh_greedy_cpu_ms,
        mesh_chunk_buffers_ms,
        mesh_full_chunked_rebuild_ms,
        mesh_pipeline_ms,
        preview_clear_ms,
    });

    *state.last_scene_bounds.lock().map_err(|e| e.to_string())? = Some(bounds);

    Ok(())
}

/// Rebuild opaque mesh from current voxels + [`RenderingMode`] (after switching view mode in the UI).
pub(crate) fn refresh_opaque_mesh<R: Runtime>(
    state: &Arc<ViewerState>,
    app: Option<&AppHandle<R>>,
) -> Result<(), String> {
    let rm = *state.rendering_mode.lock().map_err(|e| e.to_string())?;
    let fg = state.current_file.lock().map_err(|e| e.to_string())?;
    let Some(file) = fg.as_ref() else {
        return Ok(());
    };
    let mut v = state.viewer.lock().map_err(|e| e.to_string())?;
    let Some(viewer) = v.as_mut() else {
        return Err("viewer not ready".into());
    };
    let mut wp: Option<WorkProgressGuard<R>> = None;
    if let Some(a) = app {
        if work_progress_for_voxel_refresh(viewer, file, rm) {
            let mut g = WorkProgressGuard::new(a);
            g.arm();
            emit_work_progress(a, 0.15, "Rebuilding mesh…");
            wp = Some(g);
        }
    }
    if file.voxels.is_empty() {
        viewer.upload_mesh(&greedy_mesh::MeshBuffers::default());
        viewer.last_mesh_route = "clear".to_string();
        if let Ok(mut g) = state.voxel_edit_stats_cache.lock() {
            *g = None;
        }
        drop(wp);
        return Ok(());
    }
    if rm.uses_smooth_surface() {
        let mesh = match rm {
            RenderingMode::MarchingCubes => smooth_mesh::build_marching_cubes_merged(&file.voxels),
            RenderingMode::DualContour => smooth_mesh::build_dual_contour_merged(&file.voxels),
            _ => unreachable!(),
        };
        viewer.upload_mesh(&mesh);
        viewer.last_mesh_route = match rm {
            RenderingMode::MarchingCubes => "marching_cubes".to_string(),
            RenderingMode::DualContour => "dual_contour".to_string(),
            _ => unreachable!(),
        };
    } else {
        match viewer.rebuild_mesh_gpu_greedy(&file.voxels, &file.objects, file.grid_size) {
            Ok(b) => {
                viewer.set_scene_bounds(b);
                if let Ok(mut g) = state.last_scene_bounds.lock() {
                    *g = Some(b);
                }
            }
            Err(_) => {
                let work = voxelle::scene::visible_voxels_for_meshing(&file.voxels, &file.objects);
                let b = if work.is_empty() {
                    greedy_mesh::mesh_bounds_for_cube_side(file.grid_size)
                } else {
                    greedy_mesh::mesh_bounds_from_voxels_world(&file.voxels, &file.objects)
                        .or_else(|| greedy_mesh::mesh_bounds_from_voxels(&work))
                        .unwrap_or_else(|| greedy_mesh::mesh_bounds_for_cube_side(file.grid_size))
                };
                viewer.set_scene_bounds(b);
                if let Ok(mut g) = state.last_scene_bounds.lock() {
                    *g = Some(b);
                }
            }
        }
    }
    if let Ok(mut g) = state.voxel_edit_stats_cache.lock() {
        *g = Some(voxel_aabb_min_and_single_object_one_pass(&file.voxels));
    }
    drop(wp);
    Ok(())
}

enum OpaqueRefreshWork {
    Smooth {
        mesh: greedy_mesh::MeshBuffers,
        bounds: greedy_mesh::MeshBounds,
        route: String,
    },
    Greedy(PreparedGreedyRebuild),
}

/// Heavy CPU mesh work runs on a side thread; GPU upload runs on the main thread via [`AppHandle::run_on_main_thread`].
fn schedule_opaque_mesh_refresh(state: &Arc<ViewerState>, app: &AppHandle) {
    let token = state.mesh_refresh_generation.fetch_add(1, Ordering::SeqCst) + 1;
    let state_c = Arc::clone(state);
    let app = app.clone();
    let file = state
        .current_file
        .lock()
        .ok()
        .and_then(|g| g.clone());
    let Some(file) = file else {
        return;
    };
    let rm = state
        .rendering_mode
        .lock()
        .ok()
        .map(|g| *g)
        .unwrap_or(RenderingMode::Greedy);
    if std::thread::Builder::new()
        .name("voxelle-opaque-refresh".into())
        .spawn(move || {
            let work: Result<OpaqueRefreshWork, String> = if file.voxels.is_empty() {
                Ok(OpaqueRefreshWork::Greedy(PreparedGreedyRebuild::NoVoxels))
            } else if rm.uses_smooth_surface() {
                let mesh = match rm {
                    RenderingMode::MarchingCubes => {
                        smooth_mesh::build_marching_cubes_merged(&file.voxels)
                    }
                    RenderingMode::DualContour => smooth_mesh::build_dual_contour_merged(&file.voxels),
                    _ => {
                        log::warn!(target: "voxelle", "opaque refresh: unexpected smooth mode");
                        return;
                    }
                };
                let bounds = greedy_mesh::mesh_bounds_from_voxels(&file.voxels).unwrap_or_else(|| {
                    greedy_mesh::mesh_bounds_for_cube_side(file.grid_size)
                });
                let route = match rm {
                    RenderingMode::MarchingCubes => "marching_cubes",
                    RenderingMode::DualContour => "dual_contour",
                    _ => "marching_cubes",
                }
                .to_string();
                Ok(OpaqueRefreshWork::Smooth {
                    mesh,
                    bounds,
                    route,
                })
            } else {
                use std::sync::atomic::{AtomicU32, Ordering};
                let last_permille = AtomicU32::new(0);
                let app_pb = app.clone();
                emit_work_progress(&app_pb, 0.08, "Rebuilding mesh…");
                let chunk_progress = move |frac: f32, done: u32, total: u32| {
                    let permille = (frac * 1000.0).min(1000.0) as u32;
                    let prev = last_permille.load(Ordering::Relaxed);
                    if permille.saturating_sub(prev) >= 40 || done == total {
                        last_permille.store(permille, Ordering::Relaxed);
                        emit_work_progress(
                            &app_pb,
                            0.1 + 0.85 * frac,
                            format!("Building mesh chunks {done}/{total}…"),
                        );
                    }
                };
                compute_greedy_rebuild_cpu(
                    &file.voxels,
                    &file.objects,
                    file.grid_size,
                    Some(&chunk_progress),
                )
                .map(OpaqueRefreshWork::Greedy)
            };
            let work = match work {
                Ok(w) => w,
                Err(e) => {
                    log::warn!(target: "voxelle", "opaque refresh prepare: {e}");
                    return;
                }
            };
            let file_snapshot = file.clone();
            let app_emit = app.clone();
            if let Err(e) = app.run_on_main_thread(move || {
                if state_c.mesh_refresh_generation.load(Ordering::SeqCst) != token {
                    return;
                }
                let mut vl = state_c.viewer.lock().ok();
                let Some(viewer) = vl.as_mut().and_then(|v| v.as_mut()) else {
                    return;
                };
                match work {
                    OpaqueRefreshWork::Smooth {
                        mesh,
                        bounds,
                        route,
                    } => {
                        viewer.upload_mesh(&mesh);
                        viewer.set_scene_bounds(bounds);
                        viewer.last_mesh_route = route;
                        if let Ok(mut g) = state_c.last_scene_bounds.lock() {
                            *g = Some(bounds);
                        }
                    }
                    OpaqueRefreshWork::Greedy(prepared) => {
                        match viewer.apply_prepared_greedy_rebuild(prepared) {
                            Ok(b) => {
                                viewer.set_scene_bounds(b);
                                if let Ok(mut g) = state_c.last_scene_bounds.lock() {
                                    *g = Some(b);
                                }
                            }
                            Err(_) => {
                                let w = voxelle::scene::visible_voxels_for_meshing(
                                    &file_snapshot.voxels,
                                    &file_snapshot.objects,
                                );
                                let b = if w.is_empty() {
                                    greedy_mesh::mesh_bounds_for_cube_side(file_snapshot.grid_size)
                                } else {
                                    greedy_mesh::mesh_bounds_from_voxels_world(
                                        &file_snapshot.voxels,
                                        &file_snapshot.objects,
                                    )
                                    .or_else(|| greedy_mesh::mesh_bounds_from_voxels(&w))
                                    .unwrap_or_else(|| {
                                        greedy_mesh::mesh_bounds_for_cube_side(file_snapshot.grid_size)
                                    })
                                };
                                viewer.set_scene_bounds(b);
                                if let Ok(mut g) = state_c.last_scene_bounds.lock() {
                                    *g = Some(b);
                                }
                            }
                        }
                    }
                }
                if file_snapshot.voxels.is_empty() {
                    if let Ok(mut g) = state_c.voxel_edit_stats_cache.lock() {
                        *g = None;
                    }
                } else if let Ok(mut g) = state_c.voxel_edit_stats_cache.lock() {
                    *g = Some(voxel_aabb_min_and_single_object_one_pass(&file_snapshot.voxels));
                }
                emit_work_progress(&app_emit, 1.0, "");
            }) {
                log::warn!(target: "voxelle", "opaque refresh run_on_main_thread: {e}");
            }
        })
        .is_err()
    {
        log::warn!(target: "voxelle", "failed to spawn opaque refresh thread");
    }
}

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct SceneObjectsPayload {
    objects: Vec<voxelle::SceneObject>,
    active_object_id: u32,
}

#[tauri::command]
fn get_scene_objects(state: State<'_, Arc<ViewerState>>) -> Result<SceneObjectsPayload, String> {
    let fg = state.current_file.lock().map_err(|e| e.to_string())?;
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };
    Ok(SceneObjectsPayload {
        objects: file.objects.clone(),
        active_object_id: file.active_object_id,
    })
}

#[tauri::command]
fn set_active_object(state: State<'_, Arc<ViewerState>>, id: u32) -> Result<(), String> {
    let mut fg = state.current_file.lock().map_err(|e| e.to_string())?;
    let Some(file) = fg.as_mut() else {
        return Err("no model loaded".into());
    };
    if !file.objects.iter().any(|o| o.id == id) {
        return Err("unknown object".into());
    }
    file.active_object_id = id;
    Ok(())
}

#[tauri::command]
fn set_object_visible(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    id: u32,
    visible: bool,
) -> Result<(), String> {
    {
        let mut fg = state.current_file.lock().map_err(|e| e.to_string())?;
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(obj) = file.objects.iter_mut().find(|o| o.id == id) else {
            return Err("unknown object".into());
        };
        obj.visible = visible;
    }
    schedule_opaque_mesh_refresh(state.inner(), &app);
    Ok(())
}

#[tauri::command]
fn create_scene_object(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    name: String,
) -> Result<u32, String> {
    let next_id = {
        let mut fg = state.current_file.lock().map_err(|e| e.to_string())?;
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let next_id = file
            .objects
            .iter()
            .map(|o| o.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        let sort_order = file
            .objects
            .iter()
            .map(|o| o.sort_order)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        file.objects.push(voxelle::SceneObject {
            id: next_id,
            parent_id: None,
            name: if name.is_empty() {
                format!("Object {next_id}")
            } else {
                name
            },
            visible: true,
            sort_order,
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
        });
        file.active_object_id = next_id;
        next_id
    };
    refresh_opaque_mesh(state.inner(), Some(&app))?;
    Ok(next_id)
}

fn apply_rendering_mode(
    state: &Arc<ViewerState>,
    app: &AppHandle,
    mode: RenderingMode,
) -> Result<(), String> {
    *state.rendering_mode.lock().map_err(|e| e.to_string())? = mode;
    refresh_opaque_mesh(state, Some(app))
}

fn apply_orthographic(state: &Arc<ViewerState>, orthographic: bool) -> Result<(), String> {
    {
        let mut cam = state.camera.lock().map_err(|e| e.to_string())?;
        cam.perspective = !orthographic;
        if orthographic {
            if let Ok(g) = state.last_scene_bounds.lock() {
                if let Some(b) = g.as_ref() {
                    let r = b.radius().max(1.0);
                    cam.ortho_half_height = r * 1.1;
                }
            }
        }
    }
    if let Ok(mut fg) = state.current_file.lock() {
        if let Some(ref mut file) = *fg {
            file.scene.orthographic = orthographic;
        }
    }
    Ok(())
}

/// Wake the winit/tauri main loop so the next `MainEventsCleared` runs (projection / mesh / preview refresh).
fn wake_viewport_loop(app: &AppHandle) {
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let _ = app.run_on_main_thread(|| {});
    });
}

#[tauri::command]
fn get_rendering_mode(state: State<'_, Arc<ViewerState>>) -> Result<RenderingMode, String> {
    Ok(*state.rendering_mode.lock().map_err(|e| e.to_string())?)
}

#[tauri::command]
fn set_rendering_mode(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    mode: RenderingMode,
) -> Result<(), String> {
    apply_rendering_mode(state.inner(), &app, mode)?;
    wake_viewport_loop(&app);
    Ok(())
}

#[tauri::command]
fn get_orthographic(state: State<'_, Arc<ViewerState>>) -> Result<bool, String> {
    Ok(!state.camera.lock().map_err(|e| e.to_string())?.perspective)
}

#[tauri::command]
fn set_orthographic(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    orthographic: bool,
) -> Result<(), String> {
    apply_orthographic(state.inner(), orthographic)?;
    wake_viewport_loop(&app);
    Ok(())
}

/// Keeps the native **Match Material** menu checkbox in sync with app state.
#[tauri::command]
fn selection_menu_sync_match_material(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    checked: bool,
) -> Result<(), String> {
    *state.selection_match_material.lock().map_err(|e| e.to_string())? = checked;
    #[cfg(desktop)]
    {
        if let Some(menu) = app.try_state::<SelectionMenuState>() {
            menu.match_material
                .set_checked(checked)
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

/// Keeps **Debug → Viewport cursor debug overlay** in sync with webview / `localStorage`.
#[tauri::command]
fn debug_menu_sync_viewport_cursor_overlay(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    enabled: bool,
) -> Result<(), String> {
    state
        .viewport_cursor_debug_overlay
        .store(enabled, Ordering::Relaxed);
    #[cfg(desktop)]
    {
        if let Some(menu) = app.try_state::<SelectionMenuState>() {
            menu.viewport_cursor_debug
                .set_checked(enabled)
                .map_err(|e| e.to_string())?;
        }
    }
    #[cfg(not(desktop))]
    {
        let _ = app;
    }
    Ok(())
}

#[tauri::command]
fn set_tone_mapping(state: State<'_, Arc<ViewerState>>, mode: u32) -> Result<(), String> {
    let mut v = state.viewer.lock().map_err(|e| e.to_string())?;
    let Some(viewer) = v.as_mut() else {
        return Err("viewer not ready".into());
    };
    viewer.set_tone_mapping_mode(mode);
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct MoodParamsArgs {
    grain: f32,
    vignette: f32,
    /// Screen-space distance tint (0–1), lerps toward horizon color toward edges.
    distance_tint: f32,
    #[serde(default)]
    atmosphere: f32,
    #[serde(default)]
    sun_shafts: f32,
}

#[tauri::command]
fn set_mood_params(
    state: State<'_, Arc<ViewerState>>,
    args: MoodParamsArgs,
) -> Result<(), String> {
    let mut v = state.viewer.lock().map_err(|e| e.to_string())?;
    let Some(viewer) = v.as_mut() else {
        return Err("viewer not ready".into());
    };
    viewer.set_mood_params(
        args.grain,
        args.vignette,
        args.distance_tint,
        args.atmosphere,
        args.sun_shafts,
    );
    drop(v);
    let m = voxelle::MoodSettings {
        grain: args.grain,
        vignette: args.vignette,
        distance_tint: args.distance_tint,
        atmosphere: args.atmosphere,
        sun_shafts: args.sun_shafts,
    };
    if let Ok(mut cf) = state.current_file.lock() {
        if let Some(f) = cf.as_mut() {
            f.mood = Some(m);
        }
    }
    Ok(())
}

#[tauri::command]
fn set_scene_lighting(
    state: State<'_, Arc<ViewerState>>,
    args: voxelle::LightingSettings,
) -> Result<(), String> {
    let mut v = state.viewer.lock().map_err(|e| e.to_string())?;
    let Some(viewer) = v.as_mut() else {
        return Err("viewer not ready".into());
    };
    viewer.apply_lighting_settings(&args);
    drop(v);
    if let Ok(mut cf) = state.current_file.lock() {
        if let Some(f) = cf.as_mut() {
            f.lighting = Some(args);
        }
    }
    Ok(())
}

#[tauri::command]
fn get_scene_lighting(state: State<'_, Arc<ViewerState>>) -> Result<voxelle::LightingSettings, String> {
    let g = state.current_file.lock().map_err(|e| e.to_string())?;
    let Some(f) = g.as_ref() else {
        return Ok(voxelle::LightingSettings::default());
    };
    Ok(f.lighting.clone().unwrap_or_default())
}

#[tauri::command]
fn set_focal_length_mm(state: State<'_, Arc<ViewerState>>, mm: f32) -> Result<(), String> {
    let mm = mm.clamp(15.0, 200.0);
    let mut cam = state.camera.lock().map_err(|e| e.to_string())?;
    if !cam.perspective {
        return Ok(());
    }
    cam.fov_y = focal_length_to_fov_y_radians(mm);
    if let Ok(mut cf) = state.current_file.lock() {
        if let Some(f) = cf.as_mut() {
            f.scene.focal_length_mm = Some(mm);
        }
    }
    Ok(())
}

#[tauri::command]
fn get_focal_length_mm(state: State<'_, Arc<ViewerState>>) -> Result<f32, String> {
    let g = state.current_file.lock().map_err(|e| e.to_string())?;
    let Some(f) = g.as_ref() else {
        return Ok(29.0);
    };
    Ok(f.scene.focal_length_mm.unwrap_or(29.0))
}

#[tauri::command]
fn set_fly_mode(state: State<'_, Arc<ViewerState>>, enabled: bool) -> Result<(), String> {
    *state.fly_mode.lock().map_err(|e| e.to_string())? = enabled;
    if enabled {
        *state.fly_last_physics.lock().map_err(|e| e.to_string())? = None;
    }
    Ok(())
}

#[tauri::command]
fn get_fly_mode(state: State<'_, Arc<ViewerState>>) -> Result<bool, String> {
    Ok(*state.fly_mode.lock().map_err(|e| e.to_string())?)
}

fn fly_speed_scale_default() -> f32 {
    1.0
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SyncFlyInputArgs {
    forward: f32,
    right: f32,
    up: f32,
    #[serde(default = "fly_speed_scale_default")]
    speed_scale: f32,
}

/// WASD / shift state only. Translation integrates on the native event loop with real elapsed time.
#[tauri::command]
fn sync_fly_input(state: State<'_, Arc<ViewerState>>, args: SyncFlyInputArgs) -> Result<(), String> {
    if !*state.fly_mode.lock().map_err(|e| e.to_string())? {
        return Ok(());
    }
    let scale = args.speed_scale;
    let speed_scale = if scale.is_finite() {
        scale.clamp(0.0, 1e6)
    } else {
        1.0
    };
    *state.fly_input.lock().map_err(|e| e.to_string())? = FlyInputState {
        forward: args.forward,
        right: args.right,
        up: args.up,
        speed_scale,
    };
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct FlyLookArgs {
    dx: f32,
    dy: f32,
}

#[tauri::command]
fn camera_fly_look(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: FlyLookArgs,
) -> Result<(), String> {
    if !*state.fly_mode.lock().map_err(|e| e.to_string())? {
        return Ok(());
    }
    let vh = {
        let v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_ref() else {
            return Ok(());
        };
        let (_, h) = viewer.viewport_size();
        h as f32
    };
    let mut cam = state.camera.lock().map_err(|e| e.to_string())?;
    cam.fly_look_rotate_screen(args.dx, args.dy, vh.max(1.0));
    wake_viewport_loop(&app);
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct PickAtScreen {
    nx: f32,
    ny: f32,
}

/// Whether the camera ray from this screen point hits solid geometry (voxel) — used to choose camera vs edit.
#[tauri::command]
fn voxel_pick_probe(
    state: State<'_, Arc<ViewerState>>,
    args: PickAtScreen,
) -> Result<bool, String> {
    let (w, h) = {
        let v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_ref() else {
            return Ok(false);
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let fg = state.current_file.lock().map_err(|e| e.to_string())?;
    let vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
    let Some(file) = fg.as_ref() else {
        return Ok(false);
    };
    let Some(vmap) = vm.as_ref() else {
        return Ok(false);
    };
    let cam = state.camera.lock().map_err(|e| e.to_string())?;
    Ok(voxel_edit::probe_solid_hit(
        file, vmap, &cam, w, h, sx, sy,
    ))
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct StrokeAnchorAtScreen {
    nx: f32,
    ny: f32,
    tool: voxel_edit::EditTool,
}

/// Anchor voxel for multi-click stroke geometry (add → placement cell; remove/paint → solid under ray).
#[tauri::command]
fn voxel_stroke_anchor_coord_at_screen(
    state: State<'_, Arc<ViewerState>>,
    args: StrokeAnchorAtScreen,
) -> Result<Option<[i32; 3]>, String> {
    let (w, h) = {
        let v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_ref() else {
            return Ok(None);
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let fg = state.current_file.lock().map_err(|e| e.to_string())?;
    let Some(file) = fg.as_ref() else {
        return Ok(None);
    };
    let vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
    let Some(vmap) = vm.as_ref() else {
        return Ok(None);
    };
    let cam = state.camera.lock().map_err(|e| e.to_string())?;
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let c = match args.tool {
        voxel_edit::EditTool::Add => {
            voxel_edit::preview_add_cell(file, vmap, &cam, w, h, sx, sy)
        }
        voxel_edit::EditTool::Remove | voxel_edit::EditTool::Paint => {
            voxel_edit::preview_remove_cell(file, vmap, &cam, w, h, sx, sy)
        }
    };
    Ok(c.map(|(x, y, z)| [x, y, z]))
}

fn pick_cell_for_ping(
    mode: PreviewMode,
    file: &voxelle::VoxelleFile,
    vmap: &AHashMap<greedy_mesh::VoxelCoord, usize>,
    cam: &OrbitCamera,
    w: f32,
    h: f32,
    sx: f32,
    sy: f32,
) -> Option<(i32, i32, i32)> {
    match mode {
        PreviewMode::Add => voxel_edit::preview_add_cell(file, vmap, cam, w, h, sx, sy),
        PreviewMode::Remove | PreviewMode::Paint | PreviewMode::Select => {
            voxel_edit::preview_remove_cell(file, vmap, cam, w, h, sx, sy)
        }
        PreviewMode::Navigate | PreviewMode::Fly | PreviewMode::Squishy => {
            voxel_edit::preview_remove_cell(file, vmap, cam, w, h, sx, sy)
                .or_else(|| voxel_edit::preview_add_cell(file, vmap, cam, w, h, sx, sy))
        }
    }
}

fn local_accent_ping_color(state: &ViewerState) -> u32 {
    state
        .collab
        .lock()
        .ok()
        .and_then(|c| {
            c.roster
                .iter()
                .find(|r| r.peer_id == c.local_peer_id)
                .map(|r| r.color_rgb)
        })
        .unwrap_or(0x66ccff)
}

fn local_accent_ping_display_name(state: &ViewerState) -> String {
    state
        .collab
        .lock()
        .ok()
        .and_then(|c| {
            c.roster.iter().find(|r| r.peer_id == c.local_peer_id).map(|r| {
                if r.display_name.trim().is_empty() {
                    "You".to_string()
                } else {
                    r.display_name.clone()
                }
            })
        })
        .unwrap_or_else(|| "You".to_string())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct PingCursorPickArgs {
    nx: f32,
    ny: f32,
    #[serde(default)]
    display_name: String,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct PingCursorPickResult {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    x: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    y: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    z: Option<i32>,
}

/// Brief highlight at the voxel cell under the cursor ray (add / remove / navigate semantics).
#[tauri::command]
fn ping_cursor_pick(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: PingCursorPickArgs,
) -> Result<PingCursorPickResult, String> {
    let (w, h) = {
        let v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_ref() else {
            return Ok(PingCursorPickResult {
                ok: false,
                x: None,
                y: None,
                z: None,
            });
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let mode = *state.preview_mode.lock().map_err(|e| e.to_string())?;
    let coords = {
        let fg = state.current_file.lock().map_err(|e| e.to_string())?;
        let vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
        let Some(file) = fg.as_ref() else {
            return Ok(PingCursorPickResult {
                ok: false,
                x: None,
                y: None,
                z: None,
            });
        };
        let Some(vmap) = vm.as_ref() else {
            return Ok(PingCursorPickResult {
                ok: false,
                x: None,
                y: None,
                z: None,
            });
        };
        let cam = state.camera.lock().map_err(|e| e.to_string())?;
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        pick_cell_for_ping(mode, file, vmap, &cam, w, h, sx, sy)
    };
    let Some((x, y, z)) = coords else {
        return Ok(PingCursorPickResult {
            ok: false,
            x: None,
            y: None,
            z: None,
        });
    };
    let color = local_accent_ping_color(&state);
    let label = if !args.display_name.trim().is_empty() {
        args.display_name.trim().to_string()
    } else {
        local_accent_ping_display_name(&state)
    };
    collab::record_ping_flash_colored(Arc::as_ref(&*state), x, y, z, color, label);
    wake_viewport_loop(&app);
    Ok(PingCursorPickResult {
        ok: true,
        x: Some(x),
        y: Some(y),
        z: Some(z),
    })
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct WorldPointArgs {
    x: f32,
    y: f32,
    z: f32,
}

#[tauri::command]
fn world_to_viewport_pixels(
    state: State<'_, Arc<ViewerState>>,
    args: WorldPointArgs,
) -> Result<Option<(f32, f32)>, String> {
    let (w, h) = {
        let v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_ref() else {
            return Ok(None);
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let cam = state.camera.lock().map_err(|e| e.to_string())?;
    Ok(voxel_edit::world_to_viewport_pixels(
        &cam,
        w,
        h,
        args.x,
        args.y,
        args.z,
    ))
}

fn push_solo_undo_step(
    state: &Arc<ViewerState>,
    app: &AppHandle,
    deltas: Vec<voxel_edit::VoxelEditDelta>,
) -> Result<(), String> {
    if deltas.is_empty() {
        return Ok(());
    }
    state
        .solo_undo
        .lock()
        .map_err(|e| e.to_string())?
        .push(SoloUndoEntry::VoxelDeltas(deltas));
    state.solo_redo.lock().map_err(|e| e.to_string())?.clear();
    #[cfg(target_os = "macos")]
    macos_undo::register_solo_edit_completed(app, state);
    Ok(())
}

fn push_solo_selection_undo_step(
    state: &Arc<ViewerState>,
    app: &AppHandle,
    before: AHashSet<greedy_mesh::VoxelCoord>,
) -> Result<(), String> {
    state
        .solo_undo
        .lock()
        .map_err(|e| e.to_string())?
        .push(SoloUndoEntry::SelectionBefore(before));
    state.solo_redo.lock().map_err(|e| e.to_string())?.clear();
    #[cfg(target_os = "macos")]
    macos_undo::register_solo_edit_completed(app, state);
    Ok(())
}

fn commit_voxel_edits(
    state: &Arc<ViewerState>,
    app: &AppHandle,
    deltas: Vec<voxel_edit::VoxelEditDelta>,
) -> Result<bool, String> {
    if deltas.is_empty() {
        return Ok(false);
    }
    let t_total = Instant::now();
    finish_voxel_edit_gpu_deltas(
        state,
        &deltas,
        0.0,
        t_total,
        app,
        VoxelGpuRefreshReason::SoloEdit,
    )?;
    let stroke_on = *state.stroke_active.lock().map_err(|e| e.to_string())?;
    if stroke_on {
        state
            .stroke_buffer
            .lock()
            .map_err(|e| e.to_string())?
            .extend(deltas.iter().copied());
        return Ok(true);
    }
    let cm = Arc::clone(&state.collab);
    let mut cb = cm.lock().map_err(|e| e.to_string())?;
    if cb.is_client() {
        if let Some(tx) = &cb.client_tx {
            let msg = serde_json::to_string(&collab::ClientToHost::Edit {
                deltas: deltas.clone(),
            })
            .unwrap();
            let _ = tx.send(msg);
        }
    } else if cb.is_host() {
        cb.next_seq += 1;
        let seq = cb.next_seq;
        cb.host_undo
            .entry(collab::HOST_PEER_ID)
            .or_default()
            .push(deltas.clone());
        cb.host_redo.remove(&collab::HOST_PEER_ID);
        drop(cb);
        collab::host_emit_edit_batch(&cm, app, seq, collab::HOST_PEER_ID, &deltas);
    } else {
        drop(cb);
        push_solo_undo_step(state, app, deltas)?;
    }
    Ok(true)
}

#[tauri::command]
fn voxel_stroke_begin(state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
    *state.stroke_active.lock().map_err(|e| e.to_string())? = true;
    state.stroke_buffer.lock().map_err(|e| e.to_string())?.clear();
    state
        .stroke_preview_union
        .lock()
        .map_err(|e| e.to_string())?
        .clear();
    *state
        .stroke_preview_last_args
        .lock()
        .map_err(|e| e.to_string())? = None;
    state
        .stroke_preview_suppresses_hover
        .store(false, Ordering::Relaxed);
    state
        .sculpt_stroke_replay
        .lock()
        .map_err(|e| e.to_string())?
        .clear();
    Ok(())
}

const STROKE_PREVIEW_MAX_CELLS: usize = 25_000;

/// Solid + wire RGB, cube half-extent, wire thickness scale — hover and stroke preview cubes.
fn preview_tool_colors(
    tool: voxel_edit::EditTool,
    debug_pick_highlight: bool,
) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    if debug_pick_highlight {
        // Bright red fill, dark red wire — stands out for viewport cursor debug.
        return (1.0, 0.12, 0.1, 0.55, 0.0, 0.0, 0.56, 3.5);
    }
    match tool {
        voxel_edit::EditTool::Add => (0.25f32, 0.92, 0.4, 0.02, 0.09, 0.05, 0.5f32, 2.0f32),
        voxel_edit::EditTool::Remove => (0.95, 0.28, 0.22, 0.14, 0.03, 0.03, 0.53, 2.0),
        voxel_edit::EditTool::Paint => (0.35, 0.55, 0.98, 0.05, 0.08, 0.2, 0.53, 2.0),
    }
}

fn stroke_preview_meshes_for_union(
    tool: voxel_edit::EditTool,
    union: &AHashSet<greedy_mesh::VoxelCoord>,
    debug_pick_highlight: bool,
) -> (greedy_mesh::MeshBuffers, greedy_mesh::MeshBuffers) {
    let mut solid = greedy_mesh::MeshBuffers::default();
    let mut wire = greedy_mesh::MeshBuffers::default();
    let (sr, sg, sb, wr, wg, wb, size, wem) =
        preview_tool_colors(tool, debug_pick_highlight);
    let mut sorted: Vec<_> = union.iter().copied().collect();
    sorted.sort_unstable_by_key(|&(x, y, z)| (x, y, z));
    for (cx, cy, cz) in sorted.into_iter().take(STROKE_PREVIEW_MAX_CELLS) {
        let s = greedy_mesh::preview_cube_mesh(
            cx as f32,
            cy as f32,
            cz as f32,
            size,
            [sr, sg, sb],
            1.0,
        );
        let w = greedy_mesh::preview_cube_wireframe_mesh(
            cx as f32,
            cy as f32,
            cz as f32,
            size,
            [wr, wg, wb],
            wem,
        );
        greedy_mesh::append_mesh_buffers(&mut solid, s);
        greedy_mesh::append_mesh_buffers(&mut wire, w);
    }
    (solid, wire)
}

/// Preview-only stroke update during drag (commit on [`voxel_stroke_end`]).
#[tauri::command]
fn voxel_stroke_preview_at_screen(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: VoxelEditAtScreen,
) -> Result<(), String> {
    {
        let cm = state.collab.lock().map_err(|e| e.to_string())?;
        if cm.is_client() {
            return Ok(());
        }
    }

    let material = voxelle::MaterialId::from_str_id(&args.material);
    let stroke_line_start_meta = match (args.stroke_line_start_nx, args.stroke_line_start_ny) {
        (Some(_), Some(_)) => Some((0.0_f32, 0.0_f32)),
        _ => None,
    };
    let targets = {
        let fg = state.current_file.lock().map_err(|e| e.to_string())?;
        let vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
        let Some(file) = fg.as_ref() else {
            return Ok(());
        };
        let Some(vmap) = vm.as_ref() else {
            return Ok(());
        };
        let cam = state.camera.lock().map_err(|e| e.to_string())?;
        let (w, h) = {
            let v = state.viewer.lock().map_err(|e| e.to_string())?;
            let Some(viewer) = v.as_ref() else {
                return Ok(());
            };
            viewer.viewport_size()
        };
        let w = w as f32;
        let h = h as f32;
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        let stroke_line_start = match (args.stroke_line_start_nx, args.stroke_line_start_ny) {
            (Some(lnx), Some(lny)) => Some(viewport_texels_from_norm(lnx, lny, w, h)),
            _ => None,
        };
        let stroke_segment_prev = match (args.stroke_segment_prev_nx, args.stroke_segment_prev_ny) {
            (Some(pnx), Some(pny)) => Some(viewport_texels_from_norm(pnx, pny, w, h)),
            _ => None,
        };
        voxel_edit::collect_stroke_edit_targets(
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
            args.tool,
            args.color,
            material,
            args.brush_radius,
            args.brush_shape,
            args.spray_density,
            stroke_line_start,
            stroke_segment_prev,
            args.stroke_mode,
            args.plane_axis,
            &args.stroke_aux,
        )
    };

    {
        let mut union = state.stroke_preview_union.lock().map_err(|e| e.to_string())?;
        let accumulate =
            voxel_edit::stroke_preview_accumulates_samples(args.stroke_mode, stroke_line_start_meta);
        if accumulate {
            for c in targets {
                union.insert(c);
            }
        } else {
            union.clear();
            for c in targets {
                union.insert(c);
            }
        }
    }

    *state
        .stroke_preview_last_args
        .lock()
        .map_err(|e| e.to_string())? = Some(args.clone());

    let (solid, wire) = {
        let union = state.stroke_preview_union.lock().map_err(|e| e.to_string())?;
        stroke_preview_meshes_for_union(args.tool, &union, false)
    };

    {
        let mut v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_mut() else {
            return Ok(());
        };
        if solid.positions.is_empty() {
            viewer.clear_preview_mesh();
            state
                .stroke_preview_suppresses_hover
                .store(false, Ordering::Relaxed);
        } else {
            viewer.upload_preview_mesh(&solid, &wire);
            viewer.preview_cache_key = None;
            state
                .stroke_preview_suppresses_hover
                .store(true, Ordering::Relaxed);
        }
    }

    wake_viewport_loop(&app);
    Ok(())
}

#[tauri::command]
fn voxel_stroke_end(state: State<'_, Arc<ViewerState>>, app: AppHandle) -> Result<(), String> {
    *state.stroke_active.lock().map_err(|e| e.to_string())? = false;
    let had_stroke_preview = state
        .stroke_preview_suppresses_hover
        .swap(false, Ordering::Relaxed);
    let union = std::mem::take(&mut *state.stroke_preview_union.lock().map_err(|e| e.to_string())?);
    let last_args = state
        .stroke_preview_last_args
        .lock()
        .map_err(|e| e.to_string())?
        .take();
    let buf = std::mem::take(&mut *state.stroke_buffer.lock().map_err(|e| e.to_string())?);
    let sculpt_replay = std::mem::take(
        &mut *state
            .sculpt_stroke_replay
            .lock()
            .map_err(|e| e.to_string())?,
    );

    if had_stroke_preview {
        if let Ok(mut v) = state.viewer.lock() {
            if let Some(viewer) = v.as_mut() {
                viewer.clear_preview_mesh();
            }
        }
    }

    if !sculpt_replay.is_empty() {
        commit_sculpt_stroke_replay(&state, &app, sculpt_replay)?;
        return Ok(());
    }

    if !union.is_empty() {
        if let Some(args) = last_args {
            let material = voxelle::MaterialId::from_str_id(&args.material);
            let deltas = {
                let mut fg = state.current_file.lock().map_err(|e| e.to_string())?;
                let mut vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
                let Some(file) = fg.as_mut() else {
                    return Ok(());
                };
                let Some(vmap) = vm.as_mut() else {
                    return Ok(());
                };
                voxel_edit::apply_edits_to_coords(
                    file,
                    vmap,
                    args.tool,
                    args.color,
                    material,
                    &union,
                )
            };
            if !deltas.is_empty() {
                commit_voxel_edits(&state, &app, deltas)?;
            }
        }
        return Ok(());
    }

    if buf.is_empty() {
        return Ok(());
    }
    let cm = Arc::clone(&state.collab);
    let mut cb = cm.lock().map_err(|e| e.to_string())?;
    if cb.is_client() {
        if let Some(tx) = &cb.client_tx {
            let msg = serde_json::to_string(&collab::ClientToHost::Edit {
                deltas: buf.clone(),
            })
            .unwrap();
            let _ = tx.send(msg);
        }
    } else if cb.is_host() {
        cb.next_seq += 1;
        let seq = cb.next_seq;
        cb.host_undo
            .entry(collab::HOST_PEER_ID)
            .or_default()
            .push(buf.clone());
        cb.host_redo.remove(&collab::HOST_PEER_ID);
        drop(cb);
        collab::host_emit_edit_batch(&cm, &app, seq, collab::HOST_PEER_ID, &buf);
    } else {
        drop(cb);
        push_solo_undo_step(&state, &app, buf)?;
    }
    Ok(())
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct VoxelPickColorResult {
    color: u32,
    material: String,
}

#[tauri::command]
fn voxel_pick_color_at_screen(
    state: State<'_, Arc<ViewerState>>,
    args: VoxelEditAtScreen,
) -> Result<Option<VoxelPickColorResult>, String> {
    let (w, h) = {
        let v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let fg = state.current_file.lock().map_err(|e| e.to_string())?;
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };
    let vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let cam = state.camera.lock().map_err(|e| e.to_string())?;
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let Some(v) = voxel_edit::pick_voxel_at_screen(file, vmap, &cam, w, h, sx, sy) else {
        return Ok(None);
    };
    Ok(Some(VoxelPickColorResult {
        color: v.color,
        material: v.material.as_str_id().to_string(),
    }))
}

fn merge_coords_into_selection(
    sel: &mut AHashSet<greedy_mesh::VoxelCoord>,
    coords: Vec<greedy_mesh::VoxelCoord>,
    mode: SelectionCombineMode,
) {
    let set: AHashSet<_> = coords.iter().copied().collect();
    match mode {
        SelectionCombineMode::Replace => {
            sel.clear();
            sel.extend(coords);
        }
        SelectionCombineMode::Add => {
            sel.extend(coords);
        }
        SelectionCombineMode::Subtract => {
            for c in coords {
                sel.remove(&c);
            }
        }
        SelectionCombineMode::Intersect => {
            sel.retain(|c| set.contains(c));
        }
    }
}

fn emit_selection_updated(app: &AppHandle, state: &Arc<ViewerState>) {
    let n = state
        .selection_cells
        .lock()
        .map(|s| s.len() as u32)
        .unwrap_or(0);
    let _ = app.emit_to(
        EventTarget::webview_window("main"),
        "voxelle-selection-updated",
        n,
    );
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SelectionStrokeAtScreen {
    nx: f32,
    ny: f32,
    brush_radius: u32,
    brush_shape: voxel_edit::BrushShape,
    #[serde(default)]
    spray_density: f32,
    #[serde(default)]
    stroke_line_start_nx: Option<f32>,
    #[serde(default)]
    stroke_line_start_ny: Option<f32>,
    #[serde(default)]
    stroke_segment_prev_nx: Option<f32>,
    #[serde(default)]
    stroke_segment_prev_ny: Option<f32>,
    #[serde(default)]
    stroke_mode: stroke_modes::DrawStrokeMode,
    #[serde(default)]
    plane_axis: stroke_modes::PlaneAxis,
    #[serde(default)]
    stroke_aux: stroke_modes::StrokeAux,
    #[serde(default)]
    fill_select_diagonals: bool,
    #[serde(default = "default_fill_respects_color")]
    fill_respects_color: bool,
    #[serde(default)]
    match_material: bool,
    /// `select` | `selectByColor` | `selectCoplanar` | `selectCoplanarEmpty`
    #[serde(default)]
    interaction: String,
}

fn default_fill_respects_color() -> bool {
    true
}

#[tauri::command]
fn selection_stroke_begin(state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
    let snap = state
        .selection_cells
        .lock()
        .map_err(|e| e.to_string())?
        .clone();
    *state
        .selection_stroke_before
        .lock()
        .map_err(|e| e.to_string())? = Some(snap);
    Ok(())
}

#[tauri::command]
fn selection_stroke_end(app: AppHandle, state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
    let before = state
        .selection_stroke_before
        .lock()
        .map_err(|e| e.to_string())?
        .take();
    let Some(before) = before else {
        return Ok(());
    };
    let after = state
        .selection_cells
        .lock()
        .map_err(|e| e.to_string())?
        .clone();
    if after == before {
        return Ok(());
    }
    push_solo_selection_undo_step(state.inner(), &app, before)
}

#[tauri::command]
fn selection_stroke_at_screen(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: SelectionStrokeAtScreen,
) -> Result<u32, String> {
    {
        let cm = state.collab.lock().map_err(|e| e.to_string())?;
        if cm.is_client() {
            return Ok(0);
        }
    }

    let (w, h) = {
        let v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let fg = state.current_file.lock().map_err(|e| e.to_string())?;
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };
    let vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let cam = state.camera.lock().map_err(|e| e.to_string())?;

    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let stroke_line_start = match (args.stroke_line_start_nx, args.stroke_line_start_ny) {
        (Some(lnx), Some(lny)) => Some(viewport_texels_from_norm(lnx, lny, w, h)),
        _ => None,
    };
    let stroke_segment_prev = match (args.stroke_segment_prev_nx, args.stroke_segment_prev_ny) {
        (Some(pnx), Some(pny)) => Some(viewport_texels_from_norm(pnx, pny, w, h)),
        _ => None,
    };

    let interaction = args.interaction.as_str();

    let coords: Vec<greedy_mesh::VoxelCoord> =
        if matches!(args.stroke_mode, stroke_modes::DrawStrokeMode::Fill) {
            match interaction {
                "selectCoplanar" => voxel_edit::coplanar_connected_from_screen(
                    file, vmap, &cam, w, h, sx, sy,
                )
                .unwrap_or_default(),
                "selectCoplanarEmpty" => voxel_edit::coplanar_empty_connected_from_screen(
                    file, vmap, &cam, w, h, sx, sy,
                )
                .unwrap_or_default(),
                _ => {
                    let mut c = voxel_edit::flood_fill_selection_coords(
                        file,
                        vmap,
                        &cam,
                        w,
                        h,
                        sx,
                        sy,
                        args.fill_select_diagonals,
                        args.fill_respects_color,
                        args.match_material,
                    );
                    if interaction == "selectByColor" {
                        if let Some(seed) =
                            voxel_edit::pick_voxel_at_screen(file, vmap, &cam, w, h, sx, sy)
                        {
                            c = voxel_edit::filter_coords_by_seed_color(
                                file,
                                vmap,
                                &c,
                                seed,
                                args.match_material,
                            );
                        } else {
                            c.clear();
                        }
                    }
                    c
                }
            }
        } else if interaction == "selectCoplanarEmpty" {
            let c = voxel_edit::selection_stroke_sample_empty_coords(
                file,
                vmap,
                &cam,
                w,
                h,
                sx,
                sy,
                args.brush_radius,
                args.brush_shape,
                args.spray_density,
                stroke_line_start,
                stroke_segment_prev,
                args.stroke_mode,
                args.plane_axis,
                &args.stroke_aux,
            );
            voxel_edit::filter_coords_coplanar_empty_from_screen(
                file, vmap, &cam, w, h, sx, sy, &c,
            )
        } else {
            let mut c = voxel_edit::selection_stroke_sample_coords(
                file,
                vmap,
                &cam,
                w,
                h,
                sx,
                sy,
                args.brush_radius,
                args.brush_shape,
                args.spray_density,
                stroke_line_start,
                stroke_segment_prev,
                args.stroke_mode,
                args.plane_axis,
                &args.stroke_aux,
            );
            match interaction {
                "selectByColor" => {
                    if let Some(seed) =
                        voxel_edit::pick_voxel_at_screen(file, vmap, &cam, w, h, sx, sy)
                    {
                        c = voxel_edit::filter_coords_by_seed_color(
                            file,
                            vmap,
                            &c,
                            seed,
                            args.match_material,
                        );
                    } else {
                        c.clear();
                    }
                }
                "selectCoplanar" => {
                    c = voxel_edit::filter_coords_coplanar_solid_from_screen(
                        file, vmap, &cam, w, h, sx, sy, &c,
                    );
                }
                _ => {}
            }
            c
        };

    if coords.is_empty() {
        return Ok(0);
    }

    let mode = *state
        .selection_combine_mode
        .lock()
        .map_err(|e| e.to_string())?;
    let mut sel = state.selection_cells.lock().map_err(|e| e.to_string())?;
    merge_coords_into_selection(&mut sel, coords, mode);
    let n = sel.len() as u32;
    drop(sel);
    emit_selection_updated(&app, state.inner());
    Ok(n)
}

#[tauri::command]
fn selection_toggle_at_screen(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: PickAtScreen,
) -> Result<bool, String> {
    let (w, h) = {
        let v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let fg = state.current_file.lock().map_err(|e| e.to_string())?;
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };
    let vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let cam = state.camera.lock().map_err(|e| e.to_string())?;
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let Some(c) = voxel_edit::pick_solid_coord_at_screen(file, vmap, &cam, w, h, sx, sy)
    else {
        return Ok(false);
    };
    let mode = *state.selection_combine_mode.lock().map_err(|e| e.to_string())?;
    let mut sel = state.selection_cells.lock().map_err(|e| e.to_string())?;
    match mode {
        SelectionCombineMode::Replace => {
            sel.clear();
            sel.insert(c);
        }
        SelectionCombineMode::Add => {
            if sel.contains(&c) {
                sel.remove(&c);
            } else {
                sel.insert(c);
            }
        }
        SelectionCombineMode::Subtract => {
            sel.remove(&c);
        }
        SelectionCombineMode::Intersect => {
            if sel.contains(&c) {
                sel.clear();
                sel.insert(c);
            } else {
                sel.clear();
            }
        }
    }
    emit_selection_updated(&app, state.inner());
    Ok(true)
}

#[tauri::command]
fn selection_clear(app: AppHandle, state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
    state.selection_cells.lock().map_err(|e| e.to_string())?.clear();
    emit_selection_updated(&app, state.inner());
    Ok(())
}

#[tauri::command]
fn selection_delete_selected_voxels(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
) -> Result<u32, String> {
    let t_total = Instant::now();
    let coords: Vec<greedy_mesh::VoxelCoord> = {
        let sel = state.selection_cells.lock().map_err(|e| e.to_string())?;
        if sel.is_empty() {
            return Ok(0);
        }
        sel.iter().copied().collect()
    };
    let t_apply_start = Instant::now();
    let deltas = {
        let mut fg = state.current_file.lock().map_err(|e| e.to_string())?;
        let mut vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        voxel_edit::remove_voxels_at_coords(file, vmap, coords)
    };
    let apply_edit_ms = t_apply_start.elapsed().as_secs_f64() * 1000.0;

    if deltas.is_empty() {
        return Ok(0);
    }

    finish_voxel_edit_gpu_deltas(
        &state,
        &deltas,
        apply_edit_ms,
        t_total,
        &app,
        VoxelGpuRefreshReason::SoloEdit,
    )?;

    let stroke_on = *state.stroke_active.lock().map_err(|e| e.to_string())?;
    let n = deltas.len() as u32;
    if stroke_on {
        state
            .stroke_buffer
            .lock()
            .map_err(|e| e.to_string())?
            .extend(deltas.iter().copied());
        return Ok(n);
    }

    let cm = Arc::clone(&state.collab);
    let mut cb = cm.lock().map_err(|e| e.to_string())?;
    if cb.is_client() {
        if let Some(tx) = &cb.client_tx {
            let msg = serde_json::to_string(&collab::ClientToHost::Edit {
                deltas: deltas.clone(),
            })
            .unwrap();
            let _ = tx.send(msg);
        }
    } else if cb.is_host() {
        cb.next_seq += 1;
        let seq = cb.next_seq;
        cb.host_undo
            .entry(collab::HOST_PEER_ID)
            .or_default()
            .push(deltas.clone());
        cb.host_redo.remove(&collab::HOST_PEER_ID);
        drop(cb);
        collab::host_emit_edit_batch(&cm, &app, seq, collab::HOST_PEER_ID, &deltas);
    } else {
        drop(cb);
        push_solo_undo_step(&state, &app, deltas)?;
    }

    Ok(n)
}

#[tauri::command]
fn selection_get_count(state: State<'_, Arc<ViewerState>>) -> Result<u32, String> {
    Ok(state
        .selection_cells
        .lock()
        .map_err(|e| e.to_string())?
        .len() as u32)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SelectByColorArgs {
    nx: f32,
    ny: f32,
    match_material: bool,
}

#[tauri::command]
fn selection_add_by_color_at_screen(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: SelectByColorArgs,
) -> Result<u32, String> {
    let (w, h) = {
        let v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let fg = state.current_file.lock().map_err(|e| e.to_string())?;
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };
    let vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let cam = state.camera.lock().map_err(|e| e.to_string())?;
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let Some(v) = voxel_edit::pick_voxel_at_screen(file, vmap, &cam, w, h, sx, sy) else {
        return Ok(0);
    };
    let coords = voxel_edit::coords_matching_color(
        file,
        v.color,
        args.match_material,
        v.material,
    );
    let mode = *state.selection_combine_mode.lock().map_err(|e| e.to_string())?;
    let mut sel = state.selection_cells.lock().map_err(|e| e.to_string())?;
    merge_coords_into_selection(&mut sel, coords, mode);
    let n = sel.len() as u32;
    drop(sel);
    emit_selection_updated(&app, state.inner());
    Ok(n)
}

#[tauri::command]
fn selection_add_coplanar_at_screen(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: PickAtScreen,
) -> Result<u32, String> {
    let (w, h) = {
        let v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let fg = state.current_file.lock().map_err(|e| e.to_string())?;
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };
    let vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let cam = state.camera.lock().map_err(|e| e.to_string())?;
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let Some(coords) = voxel_edit::coplanar_connected_from_screen(
        file, vmap, &cam, w, h, sx, sy,
    ) else {
        return Ok(0);
    };
    let mode = *state.selection_combine_mode.lock().map_err(|e| e.to_string())?;
    let mut sel = state.selection_cells.lock().map_err(|e| e.to_string())?;
    merge_coords_into_selection(&mut sel, coords, mode);
    let n = sel.len() as u32;
    drop(sel);
    emit_selection_updated(&app, state.inner());
    Ok(n)
}

#[tauri::command]
fn selection_add_coplanar_empty_at_screen(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: PickAtScreen,
) -> Result<u32, String> {
    let (w, h) = {
        let v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let fg = state.current_file.lock().map_err(|e| e.to_string())?;
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };
    let vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let cam = state.camera.lock().map_err(|e| e.to_string())?;
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let Some(coords) = voxel_edit::coplanar_empty_connected_from_screen(
        file, vmap, &cam, w, h, sx, sy,
    ) else {
        return Ok(0);
    };
    let mode = *state.selection_combine_mode.lock().map_err(|e| e.to_string())?;
    let mut sel = state.selection_cells.lock().map_err(|e| e.to_string())?;
    merge_coords_into_selection(&mut sel, coords, mode);
    let n = sel.len() as u32;
    drop(sel);
    emit_selection_updated(&app, state.inner());
    Ok(n)
}

#[tauri::command]
fn selection_select_all(app: AppHandle, state: State<'_, Arc<ViewerState>>) -> Result<u32, String> {
    let vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let mut sel = state.selection_cells.lock().map_err(|e| e.to_string())?;
    sel.clear();
    sel.extend(vmap.keys().copied());
    let n = sel.len() as u32;
    drop(sel);
    emit_selection_updated(&app, state.inner());
    Ok(n)
}

#[tauri::command]
fn selection_invert(app: AppHandle, state: State<'_, Arc<ViewerState>>) -> Result<u32, String> {
    let vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let all: AHashSet<_> = vmap.keys().copied().collect();
    let mut sel = state.selection_cells.lock().map_err(|e| e.to_string())?;
    let new_sel: AHashSet<_> = all.difference(&sel).copied().collect();
    *sel = new_sel;
    let n = sel.len() as u32;
    drop(sel);
    emit_selection_updated(&app, state.inner());
    Ok(n)
}

#[tauri::command]
fn selection_grow(app: AppHandle, state: State<'_, Arc<ViewerState>>) -> Result<u32, String> {
    let fg = state.current_file.lock().map_err(|e| e.to_string())?;
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };
    let grid_size = file.grid_size.max(1);
    let vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let mut sel = state.selection_cells.lock().map_err(|e| e.to_string())?;
    if sel.is_empty() {
        return Ok(0);
    }
    let mut to_add = AHashSet::new();
    for &c in sel.iter() {
        for n in voxel_edit::neighbors_6(c) {
            if voxel_edit::in_grid(n.0, n.1, n.2, grid_size) && vmap.contains_key(&n) {
                to_add.insert(n);
            }
        }
    }
    sel.extend(to_add);
    let n = sel.len() as u32;
    drop(sel);
    emit_selection_updated(&app, state.inner());
    Ok(n)
}

#[tauri::command]
fn selection_shrink(app: AppHandle, state: State<'_, Arc<ViewerState>>) -> Result<u32, String> {
    let fg = state.current_file.lock().map_err(|e| e.to_string())?;
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };
    let grid_size = file.grid_size.max(1);
    let mut sel = state.selection_cells.lock().map_err(|e| e.to_string())?;
    if sel.is_empty() {
        return Ok(0);
    }
    let mut next = AHashSet::new();
    for &c in sel.iter() {
        let mut boundary = false;
        for n in voxel_edit::neighbors_6(c) {
            if !voxel_edit::in_grid(n.0, n.1, n.2, grid_size) {
                boundary = true;
                break;
            }
            if !sel.contains(&n) {
                boundary = true;
                break;
            }
        }
        if !boundary {
            next.insert(c);
        }
    }
    *sel = next;
    let n = sel.len() as u32;
    drop(sel);
    emit_selection_updated(&app, state.inner());
    Ok(n)
}

#[tauri::command]
fn selection_deselect_inner_voxels(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
) -> Result<u32, String> {
    let mut sel = state.selection_cells.lock().map_err(|e| e.to_string())?;
    if sel.is_empty() {
        return Ok(0);
    }
    let mut next = AHashSet::new();
    for &c in sel.iter() {
        let mut inner = true;
        for n in voxel_edit::neighbors_6(c) {
            if !sel.contains(&n) {
                inner = false;
                break;
            }
        }
        if !inner {
            next.insert(c);
        }
    }
    *sel = next;
    let n = sel.len() as u32;
    drop(sel);
    emit_selection_updated(&app, state.inner());
    Ok(n)
}

/// Keep only selected cells that are empty (no solid voxel) — matches web "Deselect voxels".
#[tauri::command]
fn selection_retain_empty_only(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
) -> Result<u32, String> {
    let vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let mut sel = state.selection_cells.lock().map_err(|e| e.to_string())?;
    sel.retain(|c| !vmap.contains_key(c));
    let n = sel.len() as u32;
    drop(sel);
    emit_selection_updated(&app, state.inner());
    Ok(n)
}

/// Keep only selected cells that have a solid voxel — matches web "Deselect empty spaces".
#[tauri::command]
fn selection_retain_solid_only(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
) -> Result<u32, String> {
    let vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let mut sel = state.selection_cells.lock().map_err(|e| e.to_string())?;
    sel.retain(|c| vmap.contains_key(c));
    let n = sel.len() as u32;
    drop(sel);
    emit_selection_updated(&app, state.inner());
    Ok(n)
}

fn run_selection_add_connected(
    state: &Arc<ViewerState>,
    args: PickAtScreen,
) -> Result<u32, String> {
    let (w, h) = {
        let v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let fg = state.current_file.lock().map_err(|e| e.to_string())?;
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };
    let vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let cam = state.camera.lock().map_err(|e| e.to_string())?;
    let mm = *state.selection_match_material.lock().map_err(|e| e.to_string())?;
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let Some(coords) = voxel_edit::connected_solid_same_color_from_screen(
        file, vmap, &cam, w, h, sx, sy, mm,
    ) else {
        return Ok(0);
    };
    let mode = *state.selection_combine_mode.lock().map_err(|e| e.to_string())?;
    let mut sel = state.selection_cells.lock().map_err(|e| e.to_string())?;
    merge_coords_into_selection(&mut sel, coords, mode);
    Ok(sel.len() as u32)
}

#[tauri::command]
fn selection_add_connected_at_screen(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: PickAtScreen,
) -> Result<u32, String> {
    let n = run_selection_add_connected(state.inner(), args)?;
    emit_selection_updated(&app, state.inner());
    Ok(n)
}

#[tauri::command]
fn selection_add_connected_at_cursor(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
) -> Result<u32, String> {
    let (nx, ny) = state
        .preview_cursor
        .lock()
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "Move the pointer over the viewport first.".to_string())?;
    let n = run_selection_add_connected(state.inner(), PickAtScreen { nx, ny })?;
    emit_selection_updated(&app, state.inner());
    Ok(n)
}

#[tauri::command]
fn selection_set_combine_mode(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    mode: SelectionCombineMode,
) -> Result<(), String> {
    *state.selection_combine_mode.lock().map_err(|e| e.to_string())? = mode;
    let payload = match mode {
        SelectionCombineMode::Replace => "replace",
        SelectionCombineMode::Add => "add",
        SelectionCombineMode::Subtract => "subtract",
        SelectionCombineMode::Intersect => "intersect",
    };
    let _ = app.emit_to(
        EventTarget::webview_window("main"),
        "voxelle-selection-combine-mode",
        payload,
    );
    Ok(())
}

#[tauri::command]
fn get_selection_combine_mode(
    state: State<'_, Arc<ViewerState>>,
) -> Result<SelectionCombineMode, String> {
    state
        .selection_combine_mode
        .lock()
        .map_err(|e| e.to_string())
        .map(|m| *m)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct VoxelFillAtScreen {
    nx: f32,
    ny: f32,
    color: u32,
    material: String,
    match_material: bool,
}

#[tauri::command]
fn voxel_fill_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: VoxelFillAtScreen,
) -> Result<bool, String> {
    let t_total = Instant::now();
    let (w, h) = {
        let v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let mut fg = state.current_file.lock().map_err(|e| e.to_string())?;
        let mut vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock().map_err(|e| e.to_string())?;
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        voxel_edit::flood_fill_paint_at_screen(
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
            args.color,
            material,
            args.match_material,
        )?
    };
    if deltas.is_empty() {
        return Ok(false);
    }
    finish_voxel_edit_gpu_deltas(
        &state,
        &deltas,
        0.0,
        t_total,
        &app,
        VoxelGpuRefreshReason::SoloEdit,
    )?;
    let cm = Arc::clone(&state.collab);
    let mut cb = cm.lock().map_err(|e| e.to_string())?;
    if cb.is_client() {
        if let Some(tx) = &cb.client_tx {
            let msg = serde_json::to_string(&collab::ClientToHost::Edit {
                deltas: deltas.clone(),
            })
            .unwrap();
            let _ = tx.send(msg);
        }
    } else if cb.is_host() {
        cb.next_seq += 1;
        let seq = cb.next_seq;
        cb.host_undo
            .entry(collab::HOST_PEER_ID)
            .or_default()
            .push(deltas.clone());
        cb.host_redo.remove(&collab::HOST_PEER_ID);
        drop(cb);
        collab::host_emit_edit_batch(&cm, &app, seq, collab::HOST_PEER_ID, &deltas);
    } else {
        drop(cb);
        push_solo_undo_step(&state, &app, deltas)?;
    }
    Ok(true)
}

#[tauri::command]
fn clipboard_copy_selection(state: State<'_, Arc<ViewerState>>) -> Result<bool, String> {
    let fg = state.current_file.lock().map_err(|e| e.to_string())?;
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };
    let vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let sel = state.selection_cells.lock().map_err(|e| e.to_string())?;
    let Some(clip) = voxel_edit::selection_to_clipboard(file, vmap, &sel) else {
        return Ok(false);
    };
    *state.stamp_clipboard.lock().map_err(|e| e.to_string())? = Some(clip);
    Ok(true)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct StampAtScreenArgs {
    nx: f32,
    ny: f32,
    color: u32,
    material: String,
}

#[tauri::command]
fn clipboard_stamp_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: StampAtScreenArgs,
) -> Result<bool, String> {
    let clip = state
        .stamp_clipboard
        .lock()
        .map_err(|e| e.to_string())?
        .clone();
    let Some(clip) = clip else {
        return Ok(false);
    };
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let (w, h) = {
            let v = state.viewer.lock().map_err(|e| e.to_string())?;
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.current_file.lock().map_err(|e| e.to_string())?;
        let mut vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock().map_err(|e| e.to_string())?;
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        voxel_edit::stamp_clipboard_at_screen(
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
            &clip,
            args.color,
            material,
        )?
    };
    commit_voxel_edits(&state, &app, deltas)
}

#[tauri::command]
fn clipboard_punch_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: PickAtScreen,
) -> Result<bool, String> {
    let clip = state
        .stamp_clipboard
        .lock()
        .map_err(|e| e.to_string())?
        .clone();
    let Some(clip) = clip else {
        return Ok(false);
    };
    let deltas = {
        let (w, h) = {
            let v = state.viewer.lock().map_err(|e| e.to_string())?;
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.current_file.lock().map_err(|e| e.to_string())?;
        let mut vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock().map_err(|e| e.to_string())?;
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        voxel_edit::punch_clipboard_at_screen(file, vmap, &cam, w, h, sx, sy, &clip)?
    };
    commit_voxel_edits(&state, &app, deltas)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SculptRaiseArgs {
    nx: f32,
    ny: f32,
    color: u32,
    material: String,
}

#[tauri::command]
fn voxel_sculpt_raise_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: SculptRaiseArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let (w, h) = {
            let v = state.viewer.lock().map_err(|e| e.to_string())?;
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.current_file.lock().map_err(|e| e.to_string())?;
        let mut vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock().map_err(|e| e.to_string())?;
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        voxel_edit::sculpt_raise_at_screen(
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
            args.color,
            material,
        )?
    };
    commit_voxel_edits(&state, &app, deltas)
}

#[tauri::command]
fn voxel_sculpt_stroke_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: SculptStrokeAtScreenArgs,
) -> Result<bool, String> {
    let t_total = Instant::now();
    let t_apply_start = Instant::now();
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let (w, h) = {
            let v = state.viewer.lock().map_err(|e| e.to_string())?;
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let line = match (args.stroke_line_start_nx, args.stroke_line_start_ny) {
            (Some(lnx), Some(lny)) => Some(viewport_texels_from_norm(lnx, lny, w, h)),
            _ => None,
        };
        let seg = match (args.stroke_segment_prev_nx, args.stroke_segment_prev_ny) {
            (Some(pnx), Some(pny)) => Some(viewport_texels_from_norm(pnx, pny, w, h)),
            _ => None,
        };
        let mut fg = state.current_file.lock().map_err(|e| e.to_string())?;
        let mut vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock().map_err(|e| e.to_string())?;
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        voxel_edit::apply_sculpt_stroke(
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
            args.sculpt_mode,
            args.color,
            material,
            args.brush_radius,
            args.brush_shape,
            args.spray_density,
            line,
            seg,
            args.terrain_op,
            args.terrain_base_y,
            args.terrain_strength,
            args.terrain_smooth_radius,
            args.smooth_neighbor_passes,
            args.brush_strength,
            args.brush_falloff,
            args.stroke_seed,
            args.wall_area_shape,
            args.spray_direction,
            args.wall_width_index,
            args.wall_height_vox,
            args.wall_lock_start_height,
            args.wall_axis_align,
        )?
    };
    let apply_edit_ms = t_apply_start.elapsed().as_secs_f64() * 1000.0;
    if deltas.is_empty() {
        return Ok(false);
    }
    finish_voxel_edit_gpu_deltas(
        &state,
        &deltas,
        apply_edit_ms,
        t_total,
        &app,
        VoxelGpuRefreshReason::SoloEdit,
    )?;
    let stroke_on = *state.stroke_active.lock().map_err(|e| e.to_string())?;
    if stroke_on {
        state
            .stroke_buffer
            .lock()
            .map_err(|e| e.to_string())?
            .extend(deltas.iter().copied());
        return Ok(true);
    }
    let cm = Arc::clone(&state.collab);
    let mut cb = cm.lock().map_err(|e| e.to_string())?;
    if cb.is_client() {
        if let Some(tx) = &cb.client_tx {
            let msg = serde_json::to_string(&collab::ClientToHost::Edit {
                deltas: deltas.clone(),
            })
            .unwrap();
            let _ = tx.send(msg);
        }
    } else if cb.is_host() {
        cb.next_seq += 1;
        let seq = cb.next_seq;
        cb.host_undo
            .entry(collab::HOST_PEER_ID)
            .or_default()
            .push(deltas.clone());
        cb.host_redo.remove(&collab::HOST_PEER_ID);
        drop(cb);
        collab::host_emit_edit_batch(&cm, &app, seq, collab::HOST_PEER_ID, &deltas);
    } else {
        drop(cb);
        push_solo_undo_step(&state, &app, deltas)?;
    }
    Ok(true)
}

fn commit_sculpt_stroke_replay(
    state: &Arc<ViewerState>,
    app: &AppHandle,
    replay: Vec<SculptStrokeAtScreenArgs>,
) -> Result<(), String> {
    if replay.is_empty() {
        return Ok(());
    }
    let mut all_deltas: Vec<voxel_edit::VoxelEditDelta> = Vec::new();
    for args in replay {
        let material = voxelle::MaterialId::from_str_id(&args.material);
        let deltas = {
            let (w, h) = {
                let v = state.viewer.lock().map_err(|e| e.to_string())?;
                let Some(viewer) = v.as_ref() else {
                    return Err("viewer not ready".into());
                };
                viewer.viewport_size()
            };
            let w = w as f32;
            let h = h as f32;
            let line = match (args.stroke_line_start_nx, args.stroke_line_start_ny) {
                (Some(lnx), Some(lny)) => Some(viewport_texels_from_norm(lnx, lny, w, h)),
                _ => None,
            };
            let seg = match (args.stroke_segment_prev_nx, args.stroke_segment_prev_ny) {
                (Some(pnx), Some(pny)) => Some(viewport_texels_from_norm(pnx, pny, w, h)),
                _ => None,
            };
            let mut fg = state.current_file.lock().map_err(|e| e.to_string())?;
            let mut vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
            let Some(file) = fg.as_mut() else {
                return Ok(());
            };
            let Some(vmap) = vm.as_mut() else {
                return Ok(());
            };
            let cam = state.camera.lock().map_err(|e| e.to_string())?;
            let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
            voxel_edit::apply_sculpt_stroke(
                file,
                vmap,
                &cam,
                w,
                h,
                sx,
                sy,
                args.sculpt_mode,
                args.color,
                material,
                args.brush_radius,
                args.brush_shape,
                args.spray_density,
                line,
                seg,
                args.terrain_op,
                args.terrain_base_y,
                args.terrain_strength,
                args.terrain_smooth_radius,
                args.smooth_neighbor_passes,
                args.brush_strength,
                args.brush_falloff,
                args.stroke_seed,
                args.wall_area_shape,
                args.spray_direction,
                args.wall_width_index,
                args.wall_height_vox,
                args.wall_lock_start_height,
                args.wall_axis_align,
            )?
        };
        all_deltas.extend(deltas);
    }
    if !all_deltas.is_empty() {
        commit_voxel_edits(state, app, all_deltas)?;
    }
    Ok(())
}

#[tauri::command]
fn voxel_sculpt_stroke_preview_at_screen(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: SculptStrokeAtScreenArgs,
) -> Result<(), String> {
    {
        let cm = state.collab.lock().map_err(|e| e.to_string())?;
        if cm.is_client() {
            return Ok(());
        }
    }

    let stroke_line_start_meta = match (args.stroke_line_start_nx, args.stroke_line_start_ny) {
        (Some(_), Some(_)) => Some((0.0_f32, 0.0_f32)),
        _ => None,
    };

    state
        .sculpt_stroke_replay
        .lock()
        .map_err(|e| e.to_string())?
        .push(args.clone());

    let footprint = {
        let fg = state.current_file.lock().map_err(|e| e.to_string())?;
        let vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
        let Some(file) = fg.as_ref() else {
            return Ok(());
        };
        let Some(vmap) = vm.as_ref() else {
            return Ok(());
        };
        let cam = state.camera.lock().map_err(|e| e.to_string())?;
        let (w, h) = {
            let v = state.viewer.lock().map_err(|e| e.to_string())?;
            let Some(viewer) = v.as_ref() else {
                return Ok(());
            };
            viewer.viewport_size()
        };
        let w = w as f32;
        let h = h as f32;
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        let line = match (args.stroke_line_start_nx, args.stroke_line_start_ny) {
            (Some(lnx), Some(lny)) => Some(viewport_texels_from_norm(lnx, lny, w, h)),
            _ => None,
        };
        let seg = match (args.stroke_segment_prev_nx, args.stroke_segment_prev_ny) {
            (Some(pnx), Some(pny)) => Some(viewport_texels_from_norm(pnx, pny, w, h)),
            _ => None,
        };
        if args.sculpt_mode == voxel_edit::SculptStrokeMode::Wall {
            voxel_edit::compute_wall_sculpt_footprint(
                file,
                vmap,
                &cam,
                w,
                h,
                sx,
                sy,
                line,
                seg,
                args.wall_area_shape,
                args.spray_direction,
                args.wall_width_index,
                args.wall_height_vox,
                args.wall_lock_start_height,
                args.wall_axis_align,
                args.brush_radius,
                args.brush_falloff,
                args.brush_strength,
                args.stroke_seed,
            )
        } else {
            voxel_edit::sculpt_stroke_effective_footprint(
                file,
                vmap,
                &cam,
                w,
                h,
                sx,
                sy,
                args.sculpt_mode,
                args.brush_radius,
                args.brush_shape,
                args.spray_density,
                line,
                seg,
                args.brush_strength,
                args.brush_falloff,
                args.stroke_seed,
            )
        }
    };

    {
        let mut union = state.stroke_preview_union.lock().map_err(|e| e.to_string())?;
        let accumulate = voxel_edit::stroke_preview_accumulates_samples(
            stroke_modes::DrawStrokeMode::Line,
            stroke_line_start_meta,
        );
        if accumulate {
            for c in footprint {
                union.insert(c);
            }
        } else {
            union.clear();
            for c in footprint {
                union.insert(c);
            }
        }
    }

    let preview_tool = match args.sculpt_mode {
        voxel_edit::SculptStrokeMode::Draw
        | voxel_edit::SculptStrokeMode::Wall
        | voxel_edit::SculptStrokeMode::Extrude
        | voxel_edit::SculptStrokeMode::Terrain => voxel_edit::EditTool::Add,
        voxel_edit::SculptStrokeMode::Gouge | voxel_edit::SculptStrokeMode::Smooth => {
            voxel_edit::EditTool::Remove
        }
    };

    let (solid, wire) = {
        let union = state.stroke_preview_union.lock().map_err(|e| e.to_string())?;
        stroke_preview_meshes_for_union(preview_tool, &union, false)
    };

    {
        let mut v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_mut() else {
            return Ok(());
        };
        if solid.positions.is_empty() {
            viewer.clear_preview_mesh();
            state
                .stroke_preview_suppresses_hover
                .store(false, Ordering::Relaxed);
        } else {
            viewer.upload_preview_mesh(&solid, &wire);
            viewer.preview_cache_key = None;
            state
                .stroke_preview_suppresses_hover
                .store(true, Ordering::Relaxed);
        }
    }

    wake_viewport_loop(&app);
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeneratorRocksArgs {
    nx: f32,
    ny: f32,
    #[serde(default)]
    seed: i32,
    #[serde(default = "default_rock_size")]
    size: i32,
    #[serde(default = "default_roughness")]
    roughness: f32,
    color: u32,
    material: String,
}

fn default_rock_size() -> i32 {
    4
}

fn default_roughness() -> f32 {
    0.45
}

#[tauri::command]
fn generator_rocks_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: GeneratorRocksArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let (w, h) = {
            let v = state.viewer.lock().map_err(|e| e.to_string())?;
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.current_file.lock().map_err(|e| e.to_string())?;
        let mut vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock().map_err(|e| e.to_string())?;
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        crate::generators::generator_rocks_at_screen(
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
            args.seed,
            args.size,
            args.roughness,
            args.color,
            material,
        )?
    };
    commit_voxel_edits(&state, &app, deltas)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeneratorGrassArgs {
    nx: f32,
    ny: f32,
    #[serde(default)]
    seed: i32,
    #[serde(default = "default_grass_density")]
    density: i32,
    #[serde(default = "default_grass_height")]
    max_height: i32,
    color: u32,
    material: String,
}

fn default_grass_density() -> i32 {
    4
}

fn default_grass_height() -> i32 {
    3
}

#[tauri::command]
fn generator_grass_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: GeneratorGrassArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let (w, h) = {
            let v = state.viewer.lock().map_err(|e| e.to_string())?;
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.current_file.lock().map_err(|e| e.to_string())?;
        let mut vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock().map_err(|e| e.to_string())?;
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        crate::generators::generator_grass_at_screen(
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
            args.seed,
            args.density,
            args.max_height,
            args.color,
            material,
        )?
    };
    commit_voxel_edits(&state, &app, deltas)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeneratorRopeArgs {
    nx1: f32,
    ny1: f32,
    nx2: f32,
    ny2: f32,
    #[serde(default = "default_rope_sag")]
    sag: f32,
    color: u32,
    material: String,
}

fn default_rope_sag() -> f32 {
    2.5
}

#[tauri::command]
fn generator_rope_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: GeneratorRopeArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let (w, h) = {
            let v = state.viewer.lock().map_err(|e| e.to_string())?;
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.current_file.lock().map_err(|e| e.to_string())?;
        let mut vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock().map_err(|e| e.to_string())?;
        let (sx1, sy1) = viewport_texels_from_norm(args.nx1, args.ny1, w, h);
        let (sx2, sy2) = viewport_texels_from_norm(args.nx2, args.ny2, w, h);
        crate::generators::generator_rope_between_screens(
            file,
            vmap,
            &cam,
            w,
            h,
            sx1,
            sy1,
            sx2,
            sy2,
            args.sag,
            args.color,
            material,
        )?
    };
    commit_voxel_edits(&state, &app, deltas)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeneratorSquishyArgs {
    nx: f32,
    ny: f32,
    #[serde(default = "default_squishy_radius")]
    radius: i32,
    color: u32,
    material: String,
}

fn default_squishy_radius() -> i32 {
    5
}

#[tauri::command]
fn generator_squishy_metaball_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: GeneratorSquishyArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let (w, h) = {
            let v = state.viewer.lock().map_err(|e| e.to_string())?;
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.current_file.lock().map_err(|e| e.to_string())?;
        let mut vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock().map_err(|e| e.to_string())?;
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        crate::generators::squishy_metaball_at_screen(
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
            args.radius,
            args.color,
            material,
        )?
    };
    commit_voxel_edits(&state, &app, deltas)
}

#[tauri::command]
fn squishy_session_get(
    state: State<'_, Arc<ViewerState>>,
) -> Result<generators::SquishySession, String> {
    state
        .squishy_session
        .lock()
        .map_err(|e| e.to_string())
        .map(|g| g.clone())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SquishySetModeArgs {
    mode: String,
}

#[tauri::command]
fn squishy_session_set_mode(
    state: State<'_, Arc<ViewerState>>,
    args: SquishySetModeArgs,
) -> Result<(), String> {
    let mut g = state.squishy_session.lock().map_err(|e| e.to_string())?;
    g.mode = match args.mode.as_str() {
        "edit" => generators::SquishyMode::Edit,
        "delete" => generators::SquishyMode::Delete,
        _ => generators::SquishyMode::Add,
    };
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SquishySessionFlagsArgs {
    #[serde(default)]
    hollow: Option<bool>,
    #[serde(default)]
    wall_thickness: Option<i32>,
    #[serde(default)]
    add_snap_to_surface: Option<bool>,
}

#[tauri::command]
fn squishy_session_set_flags(
    state: State<'_, Arc<ViewerState>>,
    args: SquishySessionFlagsArgs,
) -> Result<(), String> {
    let mut g = state.squishy_session.lock().map_err(|e| e.to_string())?;
    if let Some(h) = args.hollow {
        g.hollow = h;
    }
    if let Some(w) = args.wall_thickness {
        g.wall_thickness = w.max(1);
    }
    if let Some(a) = args.add_snap_to_surface {
        g.add_snap_to_surface = a;
    }
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SquishyMetaballAddArgs {
    nx: f32,
    ny: f32,
    #[serde(default = "default_squishy_radius")]
    radius: i32,
}

#[tauri::command]
fn squishy_metaball_add_at_screen(
    state: State<'_, Arc<ViewerState>>,
    args: SquishyMetaballAddArgs,
) -> Result<Option<u32>, String> {
    let (w, h) = {
        let v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let mut sg = state.squishy_session.lock().map_err(|e| e.to_string())?;
    let fg = state.current_file.lock().map_err(|e| e.to_string())?;
    let vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let cam = state.camera.lock().map_err(|e| e.to_string())?;
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let id = generators::squishy_add_ball_at_screen(
        &mut *sg,
        file,
        vmap,
        &cam,
        w,
        h,
        sx,
        sy,
        args.radius,
    );
    Ok(id)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SquishyMetaballIdArgs {
    id: u32,
}

#[tauri::command]
fn squishy_metaball_remove(
    state: State<'_, Arc<ViewerState>>,
    args: SquishyMetaballIdArgs,
) -> Result<bool, String> {
    let mut g = state.squishy_session.lock().map_err(|e| e.to_string())?;
    Ok(g.remove_ball(args.id))
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SquishySelectArgs {
    id: Option<u32>,
}

#[tauri::command]
fn squishy_metaball_select(
    state: State<'_, Arc<ViewerState>>,
    args: SquishySelectArgs,
) -> Result<(), String> {
    let mut g = state.squishy_session.lock().map_err(|e| e.to_string())?;
    g.selected_id = args.id;
    Ok(())
}

#[tauri::command]
fn squishy_session_clear(state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
    let mut g = state.squishy_session.lock().map_err(|e| e.to_string())?;
    g.clear();
    if let Ok(mut d) = state.squishy_gizmo_drag.lock() {
        *d = None;
    }
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SquishyCommitArgs {
    color: u32,
    material: String,
}

#[tauri::command]
fn squishy_session_commit(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: SquishyCommitArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let sg = state.squishy_session.lock().map_err(|e| e.to_string())?;
        let mut fg = state.current_file.lock().map_err(|e| e.to_string())?;
        let mut vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        generators::squishy_commit_session(&sg, file, vmap, args.color, material)?
    };
    if deltas.is_empty() {
        return Ok(false);
    }
    commit_voxel_edits(&state, &app, deltas)?;
    let mut g = state.squishy_session.lock().map_err(|e| e.to_string())?;
    g.clear();
    if let Ok(mut d) = state.squishy_gizmo_drag.lock() {
        *d = None;
    }
    Ok(true)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SquishyPickArgs {
    nx: f32,
    ny: f32,
}

#[tauri::command]
fn squishy_pick_at_screen(
    state: State<'_, Arc<ViewerState>>,
    args: SquishyPickArgs,
) -> Result<Option<u32>, String> {
    let (w, h) = {
        let v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let sg = state.squishy_session.lock().map_err(|e| e.to_string())?;
    let cam = state.camera.lock().map_err(|e| e.to_string())?;
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    Ok(generators::pick_metaball_at_screen(
        &sg, &cam, w, h, sx, sy,
    ))
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SquishyGizmoPointerArgs {
    nx: f32,
    ny: f32,
}

#[tauri::command]
fn squishy_gizmo_pointer_down(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: SquishyGizmoPointerArgs,
) -> Result<bool, String> {
    let (w, h) = {
        let v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let cam = state.camera.lock().map_err(|e| e.to_string())?;
    let sg = state.squishy_session.lock().map_err(|e| e.to_string())?;
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let Some(handle) =
        generators::pick_squishy_gizmo_handle(&sg, &cam, w, h, sx, sy)
    else {
        return Ok(false);
    };
    let Some(drag) =
        generators::squishy_gizmo_begin_drag(&sg, &cam, w, h, sx, sy, handle)
    else {
        return Ok(false);
    };
    drop(sg);
    drop(cam);
    *state
        .squishy_gizmo_drag
        .lock()
        .map_err(|e| e.to_string())? = Some(drag);
    wake_viewport_loop(&app);
    Ok(true)
}

#[tauri::command]
fn squishy_gizmo_pointer_move(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: SquishyGizmoPointerArgs,
) -> Result<(), String> {
    let drag = state
        .squishy_gizmo_drag
        .lock()
        .map_err(|e| e.to_string())?
        .clone();
    let Some(drag) = drag else {
        return Ok(());
    };
    let (w, h) = {
        let v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_ref() else {
            return Ok(());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let cam = state.camera.lock().map_err(|e| e.to_string())?;
    let mut sg = state.squishy_session.lock().map_err(|e| e.to_string())?;
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    generators::squishy_gizmo_apply_drag(&mut sg, &cam, w, h, sx, sy, &drag);
    wake_viewport_loop(&app);
    Ok(())
}

#[tauri::command]
fn squishy_gizmo_pointer_up(state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
    *state
        .squishy_gizmo_drag
        .lock()
        .map_err(|e| e.to_string())? = None;
    Ok(())
}

#[tauri::command]
fn voxel_edit_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: VoxelEditAtScreen,
) -> Result<bool, String> {
    let t_total = Instant::now();
    let (w, h) = {
        let v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };

    let t_apply_start = Instant::now();
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let mut fg = state.current_file.lock().map_err(|e| e.to_string())?;
        let mut vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock().map_err(|e| e.to_string())?;
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        let stroke_line_start = match (args.stroke_line_start_nx, args.stroke_line_start_ny) {
            (Some(lnx), Some(lny)) => Some(viewport_texels_from_norm(lnx, lny, w, h)),
            _ => None,
        };
        let stroke_segment_prev = match (args.stroke_segment_prev_nx, args.stroke_segment_prev_ny) {
            (Some(pnx), Some(pny)) => Some(viewport_texels_from_norm(pnx, pny, w, h)),
            _ => None,
        };
        if matches!(args.stroke_mode, stroke_modes::DrawStrokeMode::Fill)
            && matches!(args.tool, voxel_edit::EditTool::Paint)
        {
            voxel_edit::flood_fill_paint_at_screen(
                file,
                vmap,
                &cam,
                w,
                h,
                sx,
                sy,
                args.color,
                material,
                args.match_material,
            )?
        } else {
            voxel_edit::apply_edit(
                file,
                vmap,
                &cam,
                w,
                h,
                sx,
                sy,
                args.tool,
                args.color,
                material,
                args.brush_radius,
                args.brush_shape,
                args.spray_density,
                stroke_line_start,
                stroke_segment_prev,
                args.stroke_mode,
                args.plane_axis,
                &args.stroke_aux,
            )?
        }
    };
    let apply_edit_ms = t_apply_start.elapsed().as_secs_f64() * 1000.0;

    if deltas.is_empty() {
        return Ok(false);
    }

    finish_voxel_edit_gpu_deltas(
        &state,
        &deltas,
        apply_edit_ms,
        t_total,
        &app,
        VoxelGpuRefreshReason::SoloEdit,
    )?;

    let stroke_on = *state.stroke_active.lock().map_err(|e| e.to_string())?;
    if stroke_on {
        state
            .stroke_buffer
            .lock()
            .map_err(|e| e.to_string())?
            .extend(deltas.iter().copied());
        return Ok(true);
    }

    let cm = Arc::clone(&state.collab);
    let mut cb = cm.lock().map_err(|e| e.to_string())?;
    if cb.is_client() {
        if let Some(tx) = &cb.client_tx {
            let msg = serde_json::to_string(&collab::ClientToHost::Edit {
                deltas: deltas.clone(),
            })
            .unwrap();
            let _ = tx.send(msg);
        }
    } else if cb.is_host() {
        cb.next_seq += 1;
        let seq = cb.next_seq;
        cb.host_undo
            .entry(collab::HOST_PEER_ID)
            .or_default()
            .push(deltas.clone());
        cb.host_redo.remove(&collab::HOST_PEER_ID);
        drop(cb);
        collab::host_emit_edit_batch(&cm, &app, seq, collab::HOST_PEER_ID, &deltas);
    } else {
        drop(cb);
        push_solo_undo_step(&state, &app, deltas)?;
    }

    Ok(true)
}

/// Solo (non-collab) undo: pop `solo_undo` — voxel inverse or selection restore.
pub(crate) fn perform_solo_voxel_undo(
    state: &Arc<ViewerState>,
    app: &AppHandle,
) -> Result<bool, String> {
    let t_total = Instant::now();
    let step = {
        let mut u = state.solo_undo.lock().map_err(|e| e.to_string())?;
        u.pop()
    };
    let Some(step) = step else {
        return Ok(false);
    };
    match step {
        SoloUndoEntry::VoxelDeltas(original) => {
            let mesh_refresh: Vec<voxel_edit::VoxelEditDelta> = {
                let mut fg = state.current_file.lock().map_err(|e| e.to_string())?;
                let mut vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
                let Some(file) = fg.as_mut() else {
                    return Err("no model loaded".into());
                };
                let Some(vmap) = vm.as_mut() else {
                    return Err("voxel index not ready".into());
                };
                let mut mesh = Vec::with_capacity(original.len());
                for d in original.iter().rev() {
                    voxel_edit::apply_inverse_delta(file, vmap, d)?;
                    mesh.push(voxel_edit::mesh_delta_after_inverse_of(d));
                }
                mesh
            };
            finish_voxel_edit_gpu_deltas(
                state,
                &mesh_refresh,
                0.0,
                t_total,
                app,
                VoxelGpuRefreshReason::Undo,
            )?;
            state
                .solo_redo
                .lock()
                .map_err(|e| e.to_string())?
                .push(SoloRedoEntry::VoxelDeltas(original));
            Ok(true)
        }
        SoloUndoEntry::SelectionBefore(before) => {
            let cur = {
                let mut sel = state.selection_cells.lock().map_err(|e| e.to_string())?;
                let cur = sel.clone();
                *sel = before;
                cur
            };
            emit_selection_updated(app, state);
            state
                .solo_redo
                .lock()
                .map_err(|e| e.to_string())?
                .push(SoloRedoEntry::SelectionAfter(cur));
            Ok(true)
        }
    }
}

/// Solo redo: pop `solo_redo`.
pub(crate) fn perform_solo_voxel_redo(
    state: &Arc<ViewerState>,
    app: &AppHandle,
) -> Result<bool, String> {
    let t_total = Instant::now();
    let step = {
        let mut r = state.solo_redo.lock().map_err(|e| e.to_string())?;
        r.pop()
    };
    let Some(step) = step else {
        return Ok(false);
    };
    match step {
        SoloRedoEntry::VoxelDeltas(forward_batch) => {
            {
                let mut fg = state.current_file.lock().map_err(|e| e.to_string())?;
                let mut vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
                let Some(file) = fg.as_mut() else {
                    return Err("no model loaded".into());
                };
                let Some(vmap) = vm.as_mut() else {
                    return Err("voxel index not ready".into());
                };
                for d in &forward_batch {
                    voxel_edit::apply_forward_delta(file, vmap, d)?;
                }
            }
            finish_voxel_edit_gpu_deltas(
                state,
                &forward_batch,
                0.0,
                t_total,
                app,
                VoxelGpuRefreshReason::Redo,
            )?;
            state
                .solo_undo
                .lock()
                .map_err(|e| e.to_string())?
                .push(SoloUndoEntry::VoxelDeltas(forward_batch));
            Ok(true)
        }
        SoloRedoEntry::SelectionAfter(after) => {
            let cur = {
                let mut sel = state.selection_cells.lock().map_err(|e| e.to_string())?;
                let cur = sel.clone();
                *sel = after;
                cur
            };
            emit_selection_updated(app, state);
            state
                .solo_undo
                .lock()
                .map_err(|e| e.to_string())?
                .push(SoloUndoEntry::SelectionBefore(cur));
            Ok(true)
        }
    }
}

#[tauri::command]
fn voxel_undo(state: State<'_, Arc<ViewerState>>, app: AppHandle) -> Result<bool, String> {
    let cm = Arc::clone(&state.collab);
    {
        let mut c = cm.lock().map_err(|e| e.to_string())?;
        if c.is_client() {
            if let Some(tx) = &c.client_tx {
                let _ = tx.send(serde_json::to_string(&collab::ClientToHost::Undo).unwrap());
            }
            return Ok(true);
        }
        if c.is_host() {
            let mesh = collab::host_undo_peer(&app, &state, &mut c, collab::HOST_PEER_ID)?;
            let Some(d) = mesh else {
                return Ok(false);
            };
            c.next_seq += 1;
            let seq = c.next_seq;
            drop(c);
            collab::host_emit_edit_batch(&cm, &app, seq, collab::HOST_PEER_ID, &d);
            return Ok(true);
        }
    }

    #[cfg(target_os = "macos")]
    {
        if macos_undo::solo_undo_via_system(&app) {
            return Ok(true);
        }
    }
    perform_solo_voxel_undo(&state, &app)
}

#[tauri::command]
fn voxel_redo(state: State<'_, Arc<ViewerState>>, app: AppHandle) -> Result<bool, String> {
    let cm = Arc::clone(&state.collab);
    {
        let mut c = cm.lock().map_err(|e| e.to_string())?;
        if c.is_client() {
            if let Some(tx) = &c.client_tx {
                let _ = tx.send(serde_json::to_string(&collab::ClientToHost::Redo).unwrap());
            }
            return Ok(true);
        }
        if c.is_host() {
            let mesh = collab::host_redo_peer(&app, &state, &mut c, collab::HOST_PEER_ID)?;
            let Some(d) = mesh else {
                return Ok(false);
            };
            c.next_seq += 1;
            let seq = c.next_seq;
            drop(c);
            collab::host_emit_edit_batch(&cm, &app, seq, collab::HOST_PEER_ID, &d);
            return Ok(true);
        }
    }

    #[cfg(target_os = "macos")]
    {
        if macos_undo::solo_redo_via_system(&app) {
            return Ok(true);
        }
    }
    perform_solo_voxel_redo(&state, &app)
}

fn write_voxelle_file_to_path(
    progress: Option<&AppHandle>,
    state: &ViewerState,
    path: &std::path::Path,
) -> Result<(), String> {
    let wp = match progress {
        Some(app) => {
            let mut g = WorkProgressGuard::new(app);
            g.arm();
            emit_work_progress(app, 0.1, "Saving…");
            Some(g)
        }
        None => None,
    };
    let file = {
        let g = state.current_file.lock().map_err(|e| e.to_string())?;
        g.as_ref()
            .ok_or_else(|| "no model loaded".to_string())?
            .clone()
    };
    if let Some(app) = progress {
        emit_work_progress(app, 0.35, "Saving — encoding…");
    }
    let bytes = encode_payload_v4(&file).map_err(|e| e.to_string())?;
    if let Some(app) = progress {
        emit_work_progress(app, 0.7, "Saving — writing file…");
    }
    std::fs::write(path, bytes).map_err(|e| e.to_string())?;
    drop(wp);
    Ok(())
}

#[derive(serde::Serialize, serde::Deserialize)]
struct LastSessionFile {
    last_document_path: String,
}

fn session_state_path(app: &AppHandle) -> Result<PathBuf, String> {
    let mut p = app.path().app_data_dir().map_err(|e| e.to_string())?;
    p.push("last_session.json");
    Ok(p)
}

fn persist_last_document_path(app: &AppHandle, document_path: &str) {
    if !document_path.ends_with(".voxelle") {
        return;
    }
    let Ok(path) = session_state_path(app) else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let data = LastSessionFile {
        last_document_path: document_path.to_string(),
    };
    if let Ok(s) = serde_json::to_string_pretty(&data) {
        let _ = std::fs::write(path, s);
    }
}

fn read_last_document_path(app: &AppHandle) -> Option<String> {
    let path = session_state_path(app).ok()?;
    let bytes = std::fs::read(path).ok()?;
    let f: LastSessionFile = serde_json::from_slice(&bytes).ok()?;
    Some(f.last_document_path)
}

fn autosave_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let mut d = app.path().app_data_dir().map_err(|e| e.to_string())?;
    d.push("autosaves");
    std::fs::create_dir_all(&d).map_err(|e| e.to_string())?;
    Ok(d)
}

fn stable_path_key(path: &Path) -> String {
    let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let key = canon.to_string_lossy();
    format!("{:016x}", crc32fast::hash(key.as_bytes()))
}

/// Legacy single backup before per-slot rotation (`{hash}.voxelle`).
fn legacy_autosave_path(app: &AppHandle, document_path: &Path) -> Result<PathBuf, String> {
    let h = stable_path_key(document_path);
    let mut p = autosave_dir(app)?;
    p.push(format!("{h}.voxelle"));
    Ok(p)
}

fn collect_autosave_paths_for_document(
    app: &AppHandle,
    state: &ViewerState,
    document_path: &Path,
) -> Result<Vec<PathBuf>, String> {
    let keep = *state.autosave_keep_count.lock().map_err(|e| e.to_string())?;
    let keep = keep.max(1);
    let mut out = Vec::new();
    let leg = legacy_autosave_path(app, document_path)?;
    if leg.exists() {
        out.push(leg);
    }
    let h = stable_path_key(document_path);
    let dir = autosave_dir(app)?;
    for i in 0..keep {
        let p = dir.join(format!("{h}.{i}.voxelle"));
        if p.exists() {
            out.push(p);
        }
    }
    Ok(out)
}

fn newest_autosave_path(
    app: &AppHandle,
    state: &ViewerState,
    document_path: &Path,
) -> Option<PathBuf> {
    let paths = collect_autosave_paths_for_document(app, state, document_path).ok()?;
    let epoch = std::time::UNIX_EPOCH;
    paths
        .into_iter()
        .max_by_key(|p| file_mtime(p).unwrap_or(epoch))
}

fn next_rotating_autosave_path(
    app: &AppHandle,
    state: &ViewerState,
    document_path: &Path,
) -> Result<PathBuf, String> {
    let h = stable_path_key(document_path);
    let keep = *state
        .autosave_keep_count
        .lock()
        .map_err(|e| e.to_string())?;
    let k = (keep.max(1)) as u64;
    let idx = {
        let mut map = state.autosave_slot.lock().map_err(|e| e.to_string())?;
        let n = map.entry(h.clone()).or_insert(0);
        let slot = (*n % k) as u32;
        *n = n.wrapping_add(1);
        slot
    };
    let mut dir = autosave_dir(app)?;
    dir.push(format!("{h}.{idx}.voxelle"));
    Ok(dir)
}

fn file_mtime(path: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(path).ok()?.modified().ok()
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct LastSessionInfo {
    last_document_path: Option<String>,
    document_basename: Option<String>,
    autosave_path: Option<String>,
    document_exists: bool,
    autosave_exists: bool,
    autosave_newer_than_document: bool,
}

#[tauri::command]
fn get_last_session_info(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
) -> Result<LastSessionInfo, String> {
    let Some(doc_str) = read_last_document_path(&app) else {
        return Ok(LastSessionInfo {
            last_document_path: None,
            document_basename: None,
            autosave_path: None,
            document_exists: false,
            autosave_exists: false,
            autosave_newer_than_document: false,
        });
    };
    let doc_path = PathBuf::from(&doc_str);
    let basename = doc_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned());
    let document_exists = doc_path.exists();
    let (autosave_str, autosave_exists, newer) =
        match newest_autosave_path(&app, state.inner().as_ref(), &doc_path) {
            Some(ap) => {
                let aex = ap.exists();
                let s = ap.to_string_lossy().into_owned();
                let newer = match (document_exists, aex) {
                    (true, true) => match (file_mtime(&doc_path), file_mtime(&ap)) {
                        (Some(dm), Some(am)) => am > dm,
                        (None, Some(_)) => true,
                        _ => false,
                    },
                    (false, true) => true,
                    _ => false,
                };
                (Some(s), aex, newer)
            }
            None => (None, false, false),
        };
    Ok(LastSessionInfo {
        last_document_path: Some(doc_str),
        document_basename: basename,
        autosave_path: autosave_str,
        document_exists,
        autosave_exists,
        autosave_newer_than_document: newer,
    })
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct LoadVoxelleRecoveryArgs {
    document_path: String,
    autosave_path: String,
}

#[tauri::command]
fn load_voxelle_recovery(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: LoadVoxelleRecoveryArgs,
) -> Result<(), String> {
    state
        .start_screen_logo_transparent
        .store(false, Ordering::Release);
    let read_from = PathBuf::from(&args.autosave_path);
    if !read_from.is_file() {
        return Err("Autosave file not found.".into());
    }
    *state.file_label.lock().map_err(|e| e.to_string())? = args.document_path.clone();
    let _ = app.emit("voxelle-load-start", args.document_path.clone());
    spawn_decode_and_mesh_with_label(
        Arc::clone(&*state),
        app,
        read_from,
        args.document_path,
        false,
    );
    Ok(())
}

#[tauri::command]
fn save_voxelle(app: AppHandle, state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
    let label = state.file_label.lock().map_err(|e| e.to_string())?;
    if label.starts_with("New project") || !label.ends_with(".voxelle") {
        return Err("Use “Save As…” for new or unsaved projects.".into());
    }
    let s = label.clone();
    drop(label);
    write_voxelle_file_to_path(Some(&app), &state, Path::new(s.as_str()))?;
    persist_last_document_path(&app, s.as_str());
    Ok(())
}

#[tauri::command]
fn save_voxelle_as(state: State<'_, Arc<ViewerState>>, app: AppHandle) -> Result<(), String> {
    let state_c = Arc::clone(&*state);
    let app_c = app.clone();
    let mut builder = app
        .dialog()
        .file()
        .add_filter("Voxelle", &["voxelle"])
        .set_file_name("untitled.voxelle");
    if let Some(window) = app.get_webview_window("main") {
        builder = builder.set_parent(&window);
    }
    builder.save_file(move |file_path| {
        let Some(file_path) = file_path else {
            return;
        };
        let Ok(path) = file_path.into_path() else {
            let _ = app_c.emit("voxelle-load-error", "could not resolve save path");
            return;
        };
        if let Err(e) = write_voxelle_file_to_path(Some(&app_c), &state_c, &path) {
            let _ = app_c.emit("voxelle-load-error", e);
            return;
        }
        let s = path.to_string_lossy().to_string();
        if let Ok(mut g) = state_c.file_label.lock() {
            *g = s.clone();
        }
        persist_last_document_path(&app_c, &s);
        emit_voxelle_loaded(&app_c, s, &state_c, false);
    });
    Ok(())
}

fn mesh_for_export(state: &Arc<ViewerState>) -> Result<greedy_mesh::MeshBuffers, String> {
    let fg = state.current_file.lock().map_err(|e| e.to_string())?;
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };
    let rm = *state.rendering_mode.lock().map_err(|e| e.to_string())?;
    let mesh = match rm {
        RenderingMode::Greedy | RenderingMode::Ray => {
            greedy_mesh::build_greedy_mesh(&file.voxels, &file.objects).0
        }
        RenderingMode::MarchingCubes => {
            crate::smooth_mesh::build_marching_cubes_merged(&file.voxels)
        }
        RenderingMode::DualContour => crate::smooth_mesh::build_dual_contour_merged(&file.voxels),
    };
    Ok(mesh)
}

#[tauri::command]
fn export_mesh_glb(state: State<'_, Arc<ViewerState>>, app: AppHandle) -> Result<(), String> {
    let state_c = Arc::clone(&*state);
    let app_c = app.clone();
    let mut builder = app
        .dialog()
        .file()
        .add_filter("glTF Binary", &["glb"])
        .set_file_name("export.glb");
    if let Some(window) = app.get_webview_window("main") {
        builder = builder.set_parent(&window);
    }
    builder.save_file(move |file_path| {
        let Some(file_path) = file_path else {
            return;
        };
        let Ok(path) = file_path.into_path() else {
            let _ = app_c.emit("voxelle-load-error", "could not resolve export path");
            return;
        };
        let mesh = match mesh_for_export(&state_c) {
            Ok(m) => m,
            Err(e) => {
                let _ = app_c.emit("voxelle-load-error", e);
                return;
            }
        };
        let glb = match export_glb::mesh_buffers_to_glb(&mesh) {
            Ok(b) => b,
            Err(e) => {
                let _ = app_c.emit("voxelle-load-error", e);
                return;
            }
        };
        if let Err(e) = std::fs::write(&path, glb) {
            let _ = app_c.emit("voxelle-load-error", e.to_string());
        }
    });
    Ok(())
}

fn default_true() -> bool {
    true
}

/// Cursor + mode + brush/stroke state for hover preview (mesh work runs on the viewport thread).
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SyncPreviewInput {
    nx: f32,
    ny: f32,
    mode: String,
    #[serde(default)]
    brush_radius: u32,
    #[serde(default)]
    brush_shape: voxel_edit::BrushShape,
    #[serde(default)]
    spray_density: f32,
    #[serde(default)]
    stroke_mode: stroke_modes::DrawStrokeMode,
    #[serde(default)]
    plane_axis: stroke_modes::PlaneAxis,
    #[serde(default)]
    stroke_aux: stroke_modes::StrokeAux,
    #[serde(default)]
    color: u32,
    #[serde(default)]
    material: String,
    #[serde(default)]
    match_material: bool,
    #[serde(default = "default_true")]
    use_brush_preview: bool,
}

#[tauri::command]
fn sync_preview_input(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: SyncPreviewInput,
) -> Result<(), String> {
    let new_mode = PreviewMode::parse(&args.mode);
    {
        let mut pm = state.preview_mode.lock().map_err(|e| e.to_string())?;
        let changed = *pm != new_mode;
        *pm = new_mode;
        if changed {
            wake_viewport_loop(&app);
        }
    }
    {
        let mut ph = state.preview_hover.lock().map_err(|e| e.to_string())?;
        ph.brush_radius = args.brush_radius;
        ph.brush_shape = args.brush_shape;
        ph.spray_density = args.spray_density;
        ph.stroke_mode = args.stroke_mode;
        ph.plane_axis = args.plane_axis;
        ph.stroke_aux = args.stroke_aux;
        ph.color = args.color;
        ph.material = args.material;
        ph.match_material = args.match_material;
        ph.use_brush_preview = args.use_brush_preview;
    }
    if args.nx < 0.0 {
        *state.preview_cursor.lock().map_err(|e| e.to_string())? = None;
    } else {
        *state.preview_cursor.lock().map_err(|e| e.to_string())? = Some((args.nx, args.ny));
    }
    Ok(())
}

fn sync_ping_flash(viewer: &mut WgpuViewer, state: &ViewerState) {
    let snap = state.ping_flash.lock().ok().and_then(|g| g.clone());
    let Some(f) = snap else {
        viewer.clear_ping_mesh();
        return;
    };
    if std::time::Instant::now() > f.until {
        *state.ping_flash.lock().unwrap() = None;
        viewer.clear_ping_mesh();
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
}

fn sync_collab_peer_lines(viewer: &mut WgpuViewer, state: &ViewerState) {
    let (local_id, roster, presence) = {
        let c = match state.collab.lock() {
            Ok(g) => g,
            Err(_) => {
                viewer.clear_collab_peer_lines();
                return;
            }
        };
        if !c.is_active() {
            viewer.clear_collab_peer_lines();
            return;
        }
        (c.local_peer_id, c.roster.clone(), c.presence.clone())
    };
    let mut verts: Vec<f32> = Vec::with_capacity(presence.len().saturating_mul(12));
    for (pid, pr) in presence {
        if pid == local_id {
            continue;
        }
        let color = roster
            .iter()
            .find(|r| r.peer_id == pid)
            .map(|r| r.color_rgb)
            .unwrap_or(0x888888);
        let rf = ((color >> 16) & 0xff) as f32 / 255.0;
        let gf = ((color >> 8) & 0xff) as f32 / 255.0;
        let bf = (color & 0xff) as f32 / 255.0;
        let eye = collab::presence_eye(&pr);
        let target = glam::Vec3::new(pr.target[0], pr.target[1], pr.target[2]);
        if (target - eye).length_squared() < 1e-8 {
            continue;
        }
        verts.extend_from_slice(&[eye.x, eye.y, eye.z, rf, gf, bf]);
        verts.extend_from_slice(&[target.x, target.y, target.z, rf, gf, bf]);
    }
    viewer.upload_collab_peer_lines(&verts);
}

fn selection_overlay_cache_fingerprint(
    sel: &AHashSet<greedy_mesh::VoxelCoord>,
    mesh_gen: u64,
) -> u64 {
    let mut h = AHasher::default();
    sel.len().hash(&mut h);
    let mut v: Vec<_> = sel.iter().copied().collect();
    v.sort_unstable();
    for c in v {
        c.hash(&mut h);
    }
    mesh_gen.hash(&mut h);
    h.finish()
}

fn refresh_selection_overlay(viewer: &mut WgpuViewer, state: &ViewerState) {
    let sel = match state.selection_cells.lock() {
        Ok(g) => g.clone(),
        Err(_) => return,
    };
    if sel.is_empty() {
        viewer.clear_selection_overlay();
        return;
    }
    let mesh_gen = state
        .mesh_refresh_generation
        .load(Ordering::Relaxed);
    let fp = selection_overlay_cache_fingerprint(&sel, mesh_gen);
    if viewer.selection_overlay_cache_key == Some(fp) {
        return;
    }
    let file_guard = state.current_file.lock().unwrap();
    let map_guard = state.voxel_map.lock().unwrap();
    let Some(file) = file_guard.as_ref() else {
        viewer.clear_selection_overlay();
        return;
    };
    let Some(vmap) = map_guard.as_ref() else {
        viewer.clear_selection_overlay();
        return;
    };
    let mut world: AHashMap<greedy_mesh::VoxelCoord, voxelle::Voxel> =
        AHashMap::with_capacity(vmap.len());
    for (coord, &idx) in vmap.iter() {
        world.insert(*coord, file.voxels[idx]);
    }
    let solid = greedy_mesh::mesh_buffers_selection_overlay_solid(&sel, &world);
    let line_verts = if let Some((min_x, min_y, min_z, max_x, max_y, max_z)) =
        greedy_mesh::selection_bounds(&sel)
    {
        greedy_mesh::selection_aabb_line_vertices(min_x, min_y, min_z, max_x, max_y, max_z)
    } else {
        Vec::new()
    };
    viewer.upload_selection_overlay_solid(&solid);
    viewer.upload_selection_overlay_lines(&line_verts);
    viewer.selection_overlay_cache_key = Some(fp);
}

fn refresh_preview_mesh(viewer: &mut WgpuViewer, state: &ViewerState, cam: &OrbitCamera) {
    if state
        .stroke_preview_suppresses_hover
        .load(Ordering::Relaxed)
    {
        return;
    }
    let dbg = state
        .viewport_cursor_debug_overlay
        .load(Ordering::Relaxed);
    let (cursor, mode) = {
        let c = state.preview_cursor.lock().unwrap();
        let m = state.preview_mode.lock().unwrap();
        (*c, *m)
    };

    if matches!(mode, PreviewMode::Navigate | PreviewMode::Fly) {
        viewer.clear_preview_mesh();
        return;
    }

    let Some((nx, ny)) = cursor else {
        viewer.clear_preview_mesh();
        return;
    };

    let file_guard = state.current_file.lock().unwrap();
    let map_guard = state.voxel_map.lock().unwrap();
    let Some(file) = file_guard.as_ref() else {
        viewer.clear_preview_mesh();
        return;
    };
    let Some(vmap) = map_guard.as_ref() else {
        viewer.clear_preview_mesh();
        return;
    };

    let (w, h) = viewer.viewport_size();
    let w = w as f32;
    let h = h as f32;
    let (sx, sy) = viewport_texels_from_norm(nx, ny, w, h);

    if matches!(mode, PreviewMode::Squishy) {
        let hover = state.preview_hover.lock().unwrap();
        let preview_radius_i = hover.brush_radius.max(2).min(64);
        let gizmo_drag = state
            .squishy_gizmo_drag
            .lock()
            .ok()
            .map(|g| g.is_some())
            .unwrap_or(false);
        let max_v = if gizmo_drag { 12_000 } else { 24_000 };

        let session_snap = state.squishy_session.lock().unwrap().clone();

        let add_anchor = if session_snap.mode == generators::SquishyMode::Add {
            if session_snap.add_snap_to_surface {
                voxel_edit::preview_add_cell(file, vmap, cam, w, h, sx, sy)
            } else {
                voxel_edit::pick_solid_coord_at_screen(file, vmap, cam, w, h, sx, sy)
            }
        } else {
            None
        };

        let delete_hover_id = if session_snap.mode == generators::SquishyMode::Delete {
            generators::pick_metaball_at_screen(&session_snap, cam, w, h, sx, sy)
        } else {
            None
        };

        let key = hash_squishy_preview(
            &session_snap,
            sx,
            sy,
            add_anchor,
            preview_radius_i,
            gizmo_drag,
            delete_hover_id,
            dbg,
        );
        if viewer.preview_cache_key == Some(key) {
            return;
        }
        viewer.preview_cache_key = Some(key);

        let mut temp_session = session_snap.clone();
        temp_session.hollow = false;
        if let Some((ax, ay, az)) = add_anchor {
            if session_snap.mode == generators::SquishyMode::Add {
                temp_session.balls.push(generators::Metaball {
                    id: 0,
                    x: ax,
                    y: ay,
                    z: az,
                    radius: preview_radius_i as f32,
                });
            }
        }

        let coords = generators::voxel_coords_for_session_with_limit(
            &temp_session,
            file.grid_size.max(1),
            max_v,
        );

        let show_gizmo = session_snap.mode == generators::SquishyMode::Edit
            && session_snap.selected_id.is_some();

        let has_pick_chrome = !session_snap.balls.is_empty()
            || (session_snap.mode == generators::SquishyMode::Add && add_anchor.is_some());

        if coords.is_empty() && !show_gizmo && !has_pick_chrome {
            viewer.clear_preview_mesh();
            return;
        }

        let set: AHashSet<_> = coords.iter().copied().collect();
        let (solid, mut wire) =
            stroke_preview_meshes_for_union(voxel_edit::EditTool::Add, &set, dbg);

        if show_gizmo {
            generators::append_squishy_gizmo_wire(&session_snap, cam, &mut wire);
        }

        if has_pick_chrome {
            generators::append_squishy_metaball_pick_rings(
                &mut wire,
                &session_snap,
                add_anchor,
                preview_radius_i as i32,
                delete_hover_id,
            );
        }

        viewer.upload_preview_mesh(&solid, &wire);
        return;
    }

    let hover = state.preview_hover.lock().unwrap();
    let ctx = &*hover;

    if matches!(mode, PreviewMode::Select) {
        let key_cell = voxel_edit::preview_remove_cell(file, vmap, cam, w, h, sx, sy);
        let key = match key_cell {
            Some((cx, cy, cz)) => hash_single_cell_preview(mode, cx, cy, cz, 3, dbg),
            None => hash_preview_miss(mode, dbg),
        };
        if viewer.preview_cache_key == Some(key) {
            return;
        }
        viewer.preview_cache_key = Some(key);
        if let Some((cx, cy, cz)) = key_cell {
            let (sr, sg, sb, wr, wg, wb, size, wem) = if dbg {
                (1.0f32, 0.12, 0.1, 0.55, 0.0, 0.0, 0.56f32, 3.5f32)
            } else {
                (0.95, 0.75, 0.2, 0.2, 0.15, 0.02, 0.53, 2.0)
            };
            let solid = greedy_mesh::preview_cube_mesh(
                cx as f32,
                cy as f32,
                cz as f32,
                size,
                [sr, sg, sb],
                1.0,
            );
            let wire = greedy_mesh::preview_cube_wireframe_mesh(
                cx as f32,
                cy as f32,
                cz as f32,
                size,
                [wr, wg, wb],
                wem,
            );
            viewer.upload_preview_mesh(&solid, &wire);
        } else {
            viewer.clear_preview_mesh();
        }
        return;
    }

    let tool = match mode {
        PreviewMode::Add => voxel_edit::EditTool::Add,
        PreviewMode::Remove => voxel_edit::EditTool::Remove,
        PreviewMode::Paint => voxel_edit::EditTool::Paint,
        PreviewMode::Navigate | PreviewMode::Fly | PreviewMode::Select | PreviewMode::Squishy => {
            unreachable!()
        }
    };
    let mode_tag: u8 = match mode {
        PreviewMode::Add => 0,
        PreviewMode::Remove => 1,
        PreviewMode::Paint => 2,
        _ => 0,
    };

    if !ctx.use_brush_preview {
        let key_cell = match mode {
            PreviewMode::Add => voxel_edit::preview_add_cell(file, vmap, cam, w, h, sx, sy),
            PreviewMode::Remove | PreviewMode::Paint => {
                voxel_edit::preview_remove_cell(file, vmap, cam, w, h, sx, sy)
            }
            _ => None,
        };
        let key = match key_cell {
            Some((cx, cy, cz)) => hash_single_cell_preview(mode, cx, cy, cz, mode_tag, dbg),
            None => hash_preview_miss(mode, dbg),
        };
        if viewer.preview_cache_key == Some(key) {
            return;
        }
        viewer.preview_cache_key = Some(key);
        match key_cell {
            Some((cx, cy, cz)) => {
                let (sr, sg, sb, wr, wg, wb, size, wem) =
                    preview_tool_colors(tool, dbg);
                let solid = greedy_mesh::preview_cube_mesh(
                    cx as f32,
                    cy as f32,
                    cz as f32,
                    size,
                    [sr, sg, sb],
                    1.0,
                );
                let wire = greedy_mesh::preview_cube_wireframe_mesh(
                    cx as f32,
                    cy as f32,
                    cz as f32,
                    size,
                    [wr, wg, wb],
                    wem,
                );
                viewer.upload_preview_mesh(&solid, &wire);
            }
            None => viewer.clear_preview_mesh(),
        }
        return;
    }

    let material = voxelle::MaterialId::from_str_id(&ctx.material);
    let targets = voxel_edit::collect_stroke_edit_targets(
        file,
        vmap,
        cam,
        w,
        h,
        sx,
        sy,
        tool,
        ctx.color,
        material,
        ctx.brush_radius,
        ctx.brush_shape,
        ctx.spray_density,
        None,
        None,
        ctx.stroke_mode,
        ctx.plane_axis,
        &ctx.stroke_aux,
    );
    let key = hash_brush_hover_targets(mode, ctx, &targets, dbg);
    if viewer.preview_cache_key == Some(key) {
        return;
    }
    viewer.preview_cache_key = Some(key);
    if targets.is_empty() {
        viewer.clear_preview_mesh();
        return;
    }
    let set: AHashSet<_> = targets.iter().copied().collect();
    let (solid, wire) = stroke_preview_meshes_for_union(tool, &set, dbg);
    if solid.positions.is_empty() {
        viewer.clear_preview_mesh();
    } else {
        viewer.upload_preview_mesh(&solid, &wire);
    }
}

/// Non-blocking `pick_file` — `blocking_pick_file` stalls the wry event loop and freezes the
/// window (spinner) on macOS while the sheet is open.
fn open_voxelle_file_dialog(app: AppHandle, state: Arc<ViewerState>) {
    let state = Arc::clone(&state);
    let app_cb = app.clone();
    let mut builder = app.dialog().file().add_filter("Voxelle", &["voxelle"]);
    if let Some(window) = app.get_webview_window("main") {
        builder = builder.set_parent(&window);
    }
    builder.pick_file(move |file_path| {
        let Some(file_path) = file_path else {
            return;
        };
        let Ok(path) = file_path.into_path() else {
            let _ = app_cb.emit("voxelle-load-error", "could not resolve file path");
            return;
        };
        let label = path.to_string_lossy().to_string();
        if let Ok(mut g) = state.file_label.lock() {
            *g = label.clone();
        }
        let _ = app_cb.emit("voxelle-load-start", label);
        spawn_decode_and_mesh(state, app_cb, path);
    });
}

#[cfg(desktop)]
fn vd_about_metadata(app: &AppHandle) -> tauri::Result<tauri::menu::AboutMetadata<'_>> {
    use tauri::menu::AboutMetadata;
    // Public repo (matches updater endpoint in `tauri.conf.json`).
    const GITHUB_VD: &str = "https://github.com/Velfi/Voxelle-Desktop";
    let pkg = app.package_info();
    let mut m = AboutMetadata {
        name: Some(pkg.name.clone()),
        version: Some(pkg.version.to_string()),
        website: Some(GITHUB_VD.into()),
        website_label: Some("GitHub".into()),
        comments: Some("Voxel art, together on the desktop.".into()),
        copyright: app.config().bundle.copyright.clone(),
        ..Default::default()
    };
    #[cfg(target_os = "macos")]
    {
        // NSAboutPanel only shows a subset of fields; `credits` is the scrollable body with the link.
        m.website = None;
        m.website_label = None;
        m.comments = None;
        m.credits = Some(format!(
            "Voxel art, together on the desktop.\n\n{GITHUB_VD}"
        ));
    }
    Ok(m)
}

/// Native menu handles for [`CheckMenuItem`] sync (match material, debug overlay).
#[cfg(desktop)]
pub struct SelectionMenuState {
    pub match_material: tauri::menu::CheckMenuItem<tauri::Wry>,
    pub viewport_cursor_debug: tauri::menu::CheckMenuItem<tauri::Wry>,
}

/// Order top-level menus as: … File, Edit, **Selection**, View, Window, **Voxels**, Help, **Debug**
/// (after the app menu on macOS). [`Menu::default`] would leave Help before appended items; we
/// insert Selection / Voxels / Debug at the correct indices instead of appending at the end.
#[cfg(desktop)]
fn place_voxelle_custom_top_level_menus<R: tauri::Runtime>(
    menu: &tauri::menu::Menu<R>,
    selection_submenu: &tauri::menu::Submenu<R>,
    voxels_submenu: &tauri::menu::Submenu<R>,
    debug_menu: &tauri::menu::Submenu<R>,
) -> tauri::Result<()> {
    use tauri::menu::MenuItemKind;

    fn submenu_title<R2: tauri::Runtime>(kind: &MenuItemKind<R2>) -> Option<String> {
        match kind {
            MenuItemKind::Submenu(s) => s.text().ok(),
            _ => None,
        }
    }

    #[cfg(target_os = "macos")]
    {
        let items = menu.items()?;
        let edit_idx = items
            .iter()
            .position(|i| submenu_title(i).as_deref() == Some("Edit"))
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "menubar: Edit submenu not found",
                )
            })?;
        menu.insert(selection_submenu, edit_idx + 1)?;

        let items = menu.items()?;
        let window_idx = items
            .iter()
            .position(|i| submenu_title(i).as_deref() == Some("Window"))
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "menubar: Window submenu not found",
                )
            })?;
        menu.insert(voxels_submenu, window_idx + 1)?;

        let items = menu.items()?;
        let help_idx = items
            .iter()
            .position(|i| submenu_title(i).as_deref() == Some("Help"))
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "menubar: Help submenu not found",
                )
            })?;
        menu.insert(debug_menu, help_idx + 1)?;
        return Ok(());
    }

    #[cfg(not(target_os = "macos"))]
    {
        let items = menu.items()?;
        if let Some(edit_idx) = items
            .iter()
            .position(|i| submenu_title(i).as_deref() == Some("Edit"))
        {
            menu.insert(selection_submenu, edit_idx + 1)?;
        } else {
            menu.append(selection_submenu)?;
        }
        menu.append(voxels_submenu)?;
        menu.append(debug_menu)?;
        Ok(())
    }
}

#[cfg(desktop)]
fn install_app_menu(app: &AppHandle) -> tauri::Result<SelectionMenuState> {
    use tauri::menu::{
        CheckMenuItem, Menu, MenuItem, MenuItemKind, PredefinedMenuItem, Submenu,
    };

    let menu = Menu::default(app)?;
    let about_item = PredefinedMenuItem::about(app, None, Some(vd_about_metadata(app)?))?;
    let app_menu_title = app.package_info().name.clone();
    let new_item = MenuItem::with_id(app, "new_project", "New Project…", true, None::<&str>)?;
    let open_item = MenuItem::with_id(app, "open_voxelle", "Open…", true, Some("CommandOrCtrl+O"))?;
    let save_item = MenuItem::with_id(app, "menu_save", "Save", true, Some("CommandOrCtrl+S"))?;
    let save_as_item = MenuItem::with_id(
        app,
        "menu_save_as",
        "Save As…",
        true,
        Some("CommandOrCtrl+Shift+S"),
    )?;
    let export_glb_item = MenuItem::with_id(app, "menu_export_glb", "Export GLB…", true, None::<&str>)?;
    let undo_item = MenuItem::with_id(app, "menu_undo", "Undo", true, Some("CommandOrCtrl+Z"))?;
    let redo_item = MenuItem::with_id(
        app,
        "menu_redo",
        "Redo",
        true,
        Some("CommandOrCtrl+Shift+Z"),
    )?;
    let collab_start_item = MenuItem::with_id(
        app,
        "menu_collab_start",
        "Start Session",
        true,
        Some("CommandOrCtrl+Shift+L"),
    )?;
    let collab_join_item = MenuItem::with_id(
        app,
        "menu_collab_join",
        "Join Session…",
        true,
        None::<&str>,
    )?;
    let collab_leave_item = MenuItem::with_id(
        app,
        "menu_collab_leave",
        "Leave Session",
        true,
        None::<&str>,
    )?;
    let collab_submenu = Submenu::with_items(
        app,
        "Collaboration",
        true,
        &[&collab_start_item, &collab_join_item, &collab_leave_item],
    )?;
    let chat_panel_item = MenuItem::with_id(app, "menu_chat_panel", "Chat", true, None::<&str>)?;
    let check_updates_item = MenuItem::with_id(
        app,
        "menu_check_updates",
        "Check for Updates…",
        true,
        None::<&str>,
    )?;
    let preferences_item = MenuItem::with_id(
        app,
        "menu_preferences",
        "Preferences…",
        true,
        Some("CommandOrCtrl+,"),
    )?;
    let debug_viewport_cursor = CheckMenuItem::with_id(
        app,
        "debug_viewport_cursor_overlay",
        "Viewport cursor debug overlay",
        true,
        false,
        None::<&str>,
    )?;
    let debug_copy_perf = MenuItem::with_id(
        app,
        "debug_copy_performance",
        "Copy performance info",
        true,
        None::<&str>,
    )?;
    let debug_menu = Submenu::with_items(
        app,
        "Debug",
        true,
        &[&debug_viewport_cursor, &debug_copy_perf],
    )?;
    let sep = PredefinedMenuItem::separator(app)?;
    let view_render_greedy = MenuItem::with_id(app, "view_render_greedy", "Blocky", true, None::<&str>)?;
    let view_render_marching = MenuItem::with_id(
        app,
        "view_render_marching",
        "Smooth",
        true,
        None::<&str>,
    )?;
    let view_render_dual = MenuItem::with_id(
        app,
        "view_render_dual",
        "Crisp",
        true,
        None::<&str>,
    )?;
    let rendering_submenu = Submenu::with_items(
        app,
        "Rendering",
        true,
        &[&view_render_greedy, &view_render_marching, &view_render_dual],
    )?;
    let view_perspective_item =
        MenuItem::with_id(app, "menu_view_perspective", "Perspective", true, None::<&str>)?;
    let ortho_view_item = MenuItem::with_id(app, "menu_view_ortho", "Orthographic", true, None::<&str>)?;
    let view_render_ray =
        MenuItem::with_id(app, "menu_view_render_ray", "Ray (WebGPU)", true, None::<&str>)?;
    let sep_view_extras = PredefinedMenuItem::separator(app)?;
    let view_borders_show =
        MenuItem::with_id(app, "menu_view_borders_show", "Show borders", true, None::<&str>)?;
    let view_borders_hide =
        MenuItem::with_id(app, "menu_view_borders_hide", "Hide borders", true, None::<&str>)?;
    let sep_view_stamp = PredefinedMenuItem::separator(app)?;
    let view_stamp_book =
        MenuItem::with_id(app, "menu_view_stamp_book", "Stamp book…", true, None::<&str>)?;
    let view_project_stats =
        MenuItem::with_id(app, "menu_view_project_stats", "Project stats…", true, None::<&str>)?;
    let sep_before_chat = PredefinedMenuItem::separator(app)?;

    let mut file_inserted = false;
    let mut edit_inserted = false;
    let mut view_inserted = false;
    for item in menu.items()? {
        if let MenuItemKind::Submenu(sub) = item {
            let text = sub.text()?;
            #[cfg(target_os = "macos")]
            if text == app_menu_title {
                sub.remove_at(0)?;
                sub.insert(&about_item, 0)?;
                sub.insert(&preferences_item, 1)?;
                sub.insert(&check_updates_item, 2)?;
            }
            #[cfg(not(target_os = "macos"))]
            if text == "Help" {
                sub.remove_at(0)?;
                sub.insert(&about_item, 0)?;
            }
            if text == "File" {
                sub.prepend_items(&[
                    &new_item,
                    &open_item,
                    &save_item,
                    &save_as_item,
                    &export_glb_item,
                    &sep,
                ])?;
                // Also under File so it works when OS menus are localized (no reliance on "View").
                sub.append(&collab_submenu)?;
                #[cfg(not(target_os = "macos"))]
                sub.append(&check_updates_item)?;
                file_inserted = true;
            } else if text == "Edit" {
                #[cfg(not(target_os = "macos"))]
                {
                    let sep_edit = PredefinedMenuItem::separator(app)?;
                    sub.append(&sep_edit)?;
                    sub.append(&preferences_item)?;
                }
                // macOS (and many platforms) already ship Undo/Redo in Edit. Do not append ours — that
                // duplicates entries. Voxel undo/redo uses the same shortcuts via the webview (`App.tsx`).
                edit_inserted = true;
            } else if text == "View" {
                sub.append(&rendering_submenu)?;
                sub.append(&view_perspective_item)?;
                sub.append(&ortho_view_item)?;
                sub.append(&view_render_ray)?;
                sub.append(&sep_view_extras)?;
                sub.append(&view_borders_show)?;
                sub.append(&view_borders_hide)?;
                sub.append(&sep_view_stamp)?;
                sub.append(&view_stamp_book)?;
                sub.append(&view_project_stats)?;
                sub.append(&sep_before_chat)?;
                sub.append(&chat_panel_item)?;
                view_inserted = true;
            }
        }
    }

    if !view_inserted {
        let view_menu = Submenu::with_items(
            app,
            "View",
            true,
            &[
                &rendering_submenu,
                &view_perspective_item,
                &ortho_view_item,
                &view_render_ray,
                &sep_view_extras,
                &view_borders_show,
                &view_borders_hide,
                &sep_view_stamp,
                &view_stamp_book,
                &view_project_stats,
                &sep_before_chat,
                &chat_panel_item,
            ],
        )?;
        menu.append(&view_menu)?;
    }

    if !file_inserted {
        let close = PredefinedMenuItem::close_window(app, None)?;
        #[cfg(target_os = "macos")]
        {
            let file_menu = Submenu::with_items(
                app,
                "File",
                true,
                &[
                    &new_item,
                    &open_item,
                    &save_item,
                    &save_as_item,
                    &export_glb_item,
                    &sep,
                    &collab_submenu,
                    &close,
                ],
            )?;
            menu.prepend(&file_menu)?;
        }
        #[cfg(not(target_os = "macos"))]
        {
            let file_menu = Submenu::with_items(
                app,
                "File",
                true,
                &[
                    &new_item,
                    &open_item,
                    &save_item,
                    &save_as_item,
                    &export_glb_item,
                    &sep,
                    &collab_submenu,
                    &check_updates_item,
                    &close,
                ],
            )?;
            menu.prepend(&file_menu)?;
        }
    }

    if !edit_inserted {
        #[cfg(target_os = "macos")]
        {
            let edit_menu = Submenu::with_items(app, "Edit", true, &[&undo_item, &redo_item])?;
            menu.append(&edit_menu)?;
        }
        #[cfg(not(target_os = "macos"))]
        {
            let sep_edit = PredefinedMenuItem::separator(app)?;
            let edit_menu = Submenu::with_items(
                app,
                "Edit",
                true,
                &[&undo_item, &redo_item, &sep_edit, &preferences_item],
            )?;
            menu.append(&edit_menu)?;
        }
    }

    let voxel_hide_selected = MenuItem::with_id(
        app,
        "menu_voxel_hide_selected",
        "Hide selected",
        true,
        None::<&str>,
    )?;
    let voxel_unhide_all = MenuItem::with_id(
        app,
        "menu_voxel_unhide_all",
        "Unhide all",
        true,
        None::<&str>,
    )?;
    let sep_voxel_1 = PredefinedMenuItem::separator(app)?;
    let voxel_hollow = MenuItem::with_id(app, "menu_voxel_hollow", "Hollow out", true, None::<&str>)?;
    let voxel_scale = MenuItem::with_id(
        app,
        "menu_voxel_scale",
        "Scale by factor…",
        true,
        None::<&str>,
    )?;
    let voxel_rotate = MenuItem::with_id(
        app,
        "menu_voxel_rotate",
        "Rotate by degrees…",
        true,
        None::<&str>,
    )?;
    let sep_voxel_2 = PredefinedMenuItem::separator(app)?;
    let voxel_mirror_hdr = MenuItem::with_id(app, "menu_voxel_mirror_hdr", "Mirror", false, None::<&str>)?;
    let voxel_mirror_x = MenuItem::with_id(
        app,
        "menu_voxel_mirror_x",
        "Across X (YZ plane)",
        true,
        None::<&str>,
    )?;
    let voxel_mirror_y = MenuItem::with_id(
        app,
        "menu_voxel_mirror_y",
        "Across Y (XZ plane)",
        true,
        None::<&str>,
    )?;
    let voxel_mirror_z = MenuItem::with_id(
        app,
        "menu_voxel_mirror_z",
        "Across Z (XY plane)",
        true,
        None::<&str>,
    )?;
    let voxels_submenu = Submenu::with_items(
        app,
        "Voxels",
        true,
        &[
            &voxel_hide_selected,
            &voxel_unhide_all,
            &sep_voxel_1,
            &voxel_hollow,
            &voxel_scale,
            &voxel_rotate,
            &sep_voxel_2,
            &voxel_mirror_hdr,
            &voxel_mirror_x,
            &voxel_mirror_y,
            &voxel_mirror_z,
        ],
    )?;

    let menu_sel_all = MenuItem::with_id(app, "menu_sel_all", "Select All", true, None::<&str>)?;
    let menu_sel_by_color = MenuItem::with_id(
        app,
        "menu_sel_by_color",
        "Select by Color",
        true,
        None::<&str>,
    )?;
    let menu_sel_connected = MenuItem::with_id(
        app,
        "menu_sel_connected",
        "Select Connected",
        true,
        None::<&str>,
    )?;
    let menu_sel_coplanar = MenuItem::with_id(
        app,
        "menu_sel_coplanar",
        "Select Coplanar Faces",
        true,
        None::<&str>,
    )?;
    let menu_sel_coplanar_empty = MenuItem::with_id(
        app,
        "menu_sel_coplanar_empty",
        "Select Coplanar Void",
        true,
        None::<&str>,
    )?;
    let menu_sel_sep1 = PredefinedMenuItem::separator(app)?;
    let menu_sel_grow = MenuItem::with_id(app, "menu_sel_grow", "Grow", true, None::<&str>)?;
    let menu_sel_shrink = MenuItem::with_id(app, "menu_sel_shrink", "Shrink", true, None::<&str>)?;
    let menu_sel_invert = MenuItem::with_id(app, "menu_sel_invert", "Invert", true, None::<&str>)?;
    let menu_sel_sep2 = PredefinedMenuItem::separator(app)?;
    let menu_sel_deselect_all =
        MenuItem::with_id(app, "menu_sel_deselect_all", "Deselect All", true, None::<&str>)?;
    let menu_sel_deselect_inner = MenuItem::with_id(
        app,
        "menu_sel_deselect_inner",
        "Deselect Inner Voxels",
        true,
        None::<&str>,
    )?;
    let menu_sel_deselect_voxels = MenuItem::with_id(
        app,
        "menu_sel_deselect_voxels",
        "Deselect Voxels",
        true,
        None::<&str>,
    )?;
    let menu_sel_deselect_empty = MenuItem::with_id(
        app,
        "menu_sel_deselect_empty",
        "Deselect Empty Spaces",
        true,
        None::<&str>,
    )?;
    let menu_sel_sep3 = PredefinedMenuItem::separator(app)?;
    let menu_sel_mode_replace =
        MenuItem::with_id(app, "menu_sel_mode_replace", "Replace", true, None::<&str>)?;
    let menu_sel_mode_add =
        MenuItem::with_id(app, "menu_sel_mode_add", "Add to Selection", true, None::<&str>)?;
    let menu_sel_mode_subtract = MenuItem::with_id(
        app,
        "menu_sel_mode_subtract",
        "Subtract from Selection",
        true,
        None::<&str>,
    )?;
    let menu_sel_mode_intersect = MenuItem::with_id(
        app,
        "menu_sel_mode_intersect",
        "Intersect with Selection",
        true,
        None::<&str>,
    )?;
    let menu_sel_sep4 = PredefinedMenuItem::separator(app)?;
    let menu_sel_match_material = CheckMenuItem::with_id(
        app,
        "menu_sel_match_material",
        "Match Material",
        true,
        false,
        None::<&str>,
    )?;
    let selection_submenu = Submenu::with_items(
        app,
        "Selection",
        true,
        &[
            &menu_sel_all,
            &menu_sel_by_color,
            &menu_sel_connected,
            &menu_sel_coplanar,
            &menu_sel_coplanar_empty,
            &menu_sel_sep1,
            &menu_sel_grow,
            &menu_sel_shrink,
            &menu_sel_invert,
            &menu_sel_sep2,
            &menu_sel_deselect_all,
            &menu_sel_deselect_inner,
            &menu_sel_deselect_voxels,
            &menu_sel_deselect_empty,
            &menu_sel_sep3,
            &menu_sel_mode_replace,
            &menu_sel_mode_add,
            &menu_sel_mode_subtract,
            &menu_sel_mode_intersect,
            &menu_sel_sep4,
            &menu_sel_match_material,
        ],
    )?;

    place_voxelle_custom_top_level_menus(
        &menu,
        &selection_submenu,
        &voxels_submenu,
        &debug_menu,
    )?;
    menu.set_as_app_menu()?;
    Ok(SelectionMenuState {
        match_material: menu_sel_match_material.clone(),
        viewport_cursor_debug: debug_viewport_cursor.clone(),
    })
}

#[cfg(desktop)]
fn performance_report_text(state: &ViewerState) -> String {
    let fps = state.fps.lock().map(|c| c.last_fps).unwrap_or(0);
    let file_label = state
        .file_label
        .lock()
        .map(|g| g.clone())
        .unwrap_or_default();
    let (vw, vh, idx_count, vtx_buf_verts) = state
        .viewer
        .lock()
        .ok()
        .and_then(|v| {
            v.as_ref().map(|viewer| {
                let (vw, vh) = viewer.viewport_size();
                (
                    vw,
                    vh,
                    viewer.opaque_index_count(),
                    viewer.opaque_vertex_buffer_vertices(),
                )
            })
        })
        .unwrap_or((0, 0, 0, 0));
    let (voxel_n, grid_size) = state
        .current_file
        .lock()
        .ok()
        .and_then(|g| g.as_ref().map(|f| (f.voxels.len(), f.grid_size)))
        .unwrap_or((0, 0));
    let unix_s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let edit_block = state
        .last_edit_perf
        .lock()
        .ok()
        .and_then(|g| g.clone())
        .map(|e| {
            format!(
                "\nLast voxel edit (ms):\n\
                 \tapply_edit (ray + data): {:.2}\n\
                 \tprepare (scene bounds world + brick patch args): {:.2}\n\
                 \tviewer lock wait: {:.2}\n\
                 \tbrick (upload_scene_data): {:.2}\n\
                 \tmesh (total): {:.2}\n\
                 \t  spatial cache delta: {:.2}\n\
                 \t  spatial cache cold init: {:.2}\n\
                 \t  greedy (dirty chunks): {:.2}\n\
                 \t  greedy GPU (dirty chunks): {:.2}\n\
                 \t  greedy CPU (dirty chunks): {:.2}\n\
                 \t  chunk GPU buffers: {:.2}\n\
                 \t  full chunked rebuild: {:.2}\n\
                 \t  pipeline (rebuild_mesh_gpu_greedy): {:.2}\n\
                 \tpreview clear: {:.2}\n\
                 \tmesh route: {}\n\
                 \ttotal: {:.2}\n",
                e.apply_edit_ms,
                e.prepare_ms,
                e.viewer_lock_wait_ms,
                e.brick_ms,
                e.mesh_ms,
                e.mesh_voxel_map_ms,
                e.mesh_buckets_ms,
                e.mesh_greedy_ms,
                e.mesh_greedy_gpu_ms,
                e.mesh_greedy_cpu_ms,
                e.mesh_chunk_buffers_ms,
                e.mesh_full_chunked_rebuild_ms,
                e.mesh_pipeline_ms,
                e.preview_clear_ms,
                e.mesh_route,
                e.total_ms,
            )
        })
        .unwrap_or_else(|| "\nLast voxel edit (ms): (none yet this session)\n".to_string());
    format!(
        "Voxelle Desktop — performance snapshot\n\
         \n\
         Timestamp (UTC, Unix s): {unix_s}\n\
         Viewport FPS (last 1s avg): {fps}\n\
         Viewport size (physical px): {vw}×{vh}\n\
         Opaque mesh: index count = {idx_count}, vertex buffer slots ≈ {vtx_buf_verts}\n\
         Scene: voxel count = {voxel_n}, grid_size = {grid_size}\n\
         File label: {file_label}\n\
         Platform: {} / {}{edit_block}",
        std::env::consts::OS,
        std::env::consts::ARCH,
    )
}

#[cfg(desktop)]
fn copy_performance_data_to_clipboard(state: &Arc<ViewerState>) -> Result<(), String> {
    let text = performance_report_text(state);
    arboard::Clipboard::new()
        .map_err(|e| e.to_string())?
        .set_text(text)
        .map_err(|e| e.to_string())
}

/// Ok/Cancel prompt **without** parenting to the webview window. The JS `confirm` API always
/// attaches to the main window, which on macOS uses a sheet; keyboard/focus churn after the
/// app menu can activate the default OK before the user intends to.
#[tauri::command]
fn confirm_app_update_dialog(app: AppHandle, message: String, title: String) -> bool {
    app.dialog()
        .message(message)
        .title(title)
        .kind(MessageDialogKind::Info)
        .buttons(MessageDialogButtons::OkCancel)
        .blocking_show()
}

#[tauri::command]
fn open_voxelle_dialog(state: State<'_, Arc<ViewerState>>, app: AppHandle) -> Result<(), String> {
    open_voxelle_file_dialog(app, Arc::clone(&*state));
    Ok(())
}

fn resolve_start_screen_logo_path(app: &AppHandle) -> Option<PathBuf> {
    let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../public/Logo.voxelle");
    if dev.is_file() {
        return Some(dev);
    }
    let res = app.path().resolve("Logo.voxelle", BaseDirectory::Resource).ok()?;
    if res.is_file() {
        return Some(res);
    }
    None
}

/// Loads bundled `Logo.voxelle` for the cold-start screen (no `voxelle-load-start`, empty `file_label`).
#[tauri::command]
fn load_start_screen_logo(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
) -> Result<(), String> {
    let Some(p) = resolve_start_screen_logo_path(&app) else {
        // Let the webview clear any cold-start loading UI (same as a successful splash load).
        emit_voxelle_loaded(&app, String::new(), &state, true);
        return Ok(());
    };
    *state.file_label.lock().map_err(|e| e.to_string())? = String::new();
    spawn_decode_and_mesh_with_label(Arc::clone(&*state), app, p, String::new(), true);
    Ok(())
}

#[tauri::command]
fn load_voxelle_path(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    path: String,
) -> Result<(), String> {
    state
        .start_screen_logo_transparent
        .store(false, Ordering::Release);
    let p = std::path::PathBuf::from(&path);
    *state.file_label.lock().map_err(|e| e.to_string())? = path.clone();
    let _ = app.emit("voxelle-load-start", path.clone());
    spawn_decode_and_mesh(Arc::clone(&*state), app, p);
    Ok(())
}

const MAX_GRID_SIZE: u32 = 256;

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct NewProjectArgs {
    grid_size: u32,
    shape: StartShape,
}

#[tauri::command]
fn collab_host_start(
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
fn collab_join(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    url: String,
    display_name: String,
    color_rgb: u32,
) -> Result<(), String> {
    let vs = Arc::clone(&*state);
    let cm = Arc::clone(&vs.collab);
    tauri::async_runtime::spawn(async move {
        if let Err(e) =
            collab::client_connect_blocking(&url, app.clone(), vs, cm, display_name, color_rgb)
                .await
        {
            let _ = app.emit("collab-error", e);
        }
    });
    Ok(())
}

#[tauri::command]
fn collab_leave(state: State<'_, Arc<ViewerState>>, app: AppHandle) -> Result<(), String> {
    let (was_host, was_client, upnp_port) = {
        let mut c = state.collab.lock().map_err(|e| e.to_string())?;
        let wh = c.is_host();
        let wc = c.is_client();
        let upnp = c.upnp_external_tcp_port;
        if wc {
            if let Some(tx) = &c.client_tx {
                let msg = serde_json::to_string(&collab::ClientToHost::Leave).unwrap();
                let _ = tx.send(msg);
            }
        }
        c.leave();
        (wh, wc, upnp)
    };
    if let Some(p) = upnp_port {
        collab::schedule_remove_upnp_mapping(p);
    }
    *state.ping_flash.lock().map_err(|e| e.to_string())? = None;
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

#[tauri::command]
fn collab_local_peer_id(state: State<'_, Arc<ViewerState>>) -> u32 {
    state.collab.lock().map(|c| c.local_peer_id).unwrap_or(0)
}

#[tauri::command]
fn collab_kick_peer(state: State<'_, Arc<ViewerState>>, target_peer: u32) -> Result<(), String> {
    collab::host_kick_peer(&state.collab, target_peer)
}

#[tauri::command]
fn collab_update_profile(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    display_name: String,
    color_rgb: u32,
) -> Result<(), String> {
    let mut c = state.collab.lock().map_err(|e| e.to_string())?;
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
            let _ = tx.send(msg);
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
fn collab_set_can_edit(
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
    let mut c = state.collab.lock().map_err(|e| e.to_string())?;
    if c.is_client() {
        if let Some(tx) = &c.client_tx {
            let _ = tx.send(msg);
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

#[tauri::command]
fn collab_push_camera(state: State<'_, Arc<ViewerState>>, app: AppHandle) -> Result<(), String> {
    let vs = Arc::clone(&*state);
    let mut c = vs.collab.lock().map_err(|e| e.to_string())?;
    if !c.is_active() {
        return Ok(());
    }
    let cam = state.camera.lock().map_err(|e| e.to_string())?;
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
            let _ = tx.send(msg);
        }
    } else if c.is_host() {
        // Guests only receive camera updates via WebSocket; without this broadcast the host's
        // peer id never appears in guests' `presence`, so "snap to host" always failed.
        let cam_ev = collab::HostToClient::Camera { peer_id: pid, presence };
        let json = serde_json::to_string(&cam_ev).unwrap();
        let _ = app.emit("collab-camera", &json);
        if let Some(tx) = &c.host_broadcast {
            let _ = tx.send(json);
        }
    }
    Ok(())
}

#[tauri::command]
fn collab_snap_camera(state: State<'_, Arc<ViewerState>>, peer_id: u32) -> Result<(), String> {
    let pr = {
        let c = state.collab.lock().map_err(|e| e.to_string())?;
        c.presence.get(&peer_id).copied()
    };
    let Some(pr) = pr else {
        return Err("no camera data for peer".into());
    };
    let mut cam = state.camera.lock().map_err(|e| e.to_string())?;
    cam.target = glam::Vec3::new(pr.target[0], pr.target[1], pr.target[2]);
    cam.spherical.radius = pr.radius;
    cam.spherical.theta = pr.theta;
    cam.spherical.phi = pr.phi;
    cam.smooth_target = cam.target;
    cam.smooth_spherical = cam.spherical;
    cam.perspective = pr.perspective;
    cam.fov_y = pr.fov_y;
    cam.ortho_half_height = pr.ortho_half_height;
    Ok(())
}

#[tauri::command]
fn collab_send_chat(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    text: String,
) -> Result<(), String> {
    let c = state.collab.lock().map_err(|e| e.to_string())?;
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
            let _ = tx.send(json);
        }
        return Ok(());
    }
    if let Some(tx) = &c.client_tx {
        let msg = serde_json::to_string(&collab::ClientToHost::Chat { text }).unwrap();
        let _ = tx.send(msg);
    }
    Ok(())
}

#[tauri::command]
fn collab_send_ping(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    x: i32,
    y: i32,
    z: i32,
) -> Result<(), String> {
    let mut c = state.collab.lock().map_err(|e| e.to_string())?;
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
        );
        c = state.collab.lock().map_err(|e| e.to_string())?;
        let ping = collab::HostToClient::Ping {
            peer_id: collab::HOST_PEER_ID,
            x,
            y,
            z,
            display_name: host_name,
        };
        let json = serde_json::to_string(&ping).unwrap();
        let _ = app.emit("collab-ping", &json);
        if let Some(tx) = &c.host_broadcast {
            let _ = tx.send(json);
        }
        return Ok(());
    }
    if let Some(tx) = &c.client_tx {
        let msg = serde_json::to_string(&collab::ClientToHost::Ping { x, y, z }).unwrap();
        let _ = tx.send(msg);
    }
    Ok(())
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct AutosaveSettings {
    enabled: bool,
    interval_secs: u64,
    keep_count: u32,
}

#[tauri::command]
fn get_autosave_settings(state: State<'_, Arc<ViewerState>>) -> Result<AutosaveSettings, String> {
    Ok(AutosaveSettings {
        enabled: *state.autosave_enabled.lock().map_err(|e| e.to_string())?,
        interval_secs: *state
            .autosave_interval_secs
            .lock()
            .map_err(|e| e.to_string())?,
        keep_count: *state
            .autosave_keep_count
            .lock()
            .map_err(|e| e.to_string())?,
    })
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct AutosaveSettingsArgs {
    enabled: bool,
    interval_secs: u64,
    keep_count: u32,
}

#[tauri::command]
fn set_autosave_settings(
    state: State<'_, Arc<ViewerState>>,
    args: AutosaveSettingsArgs,
) -> Result<(), String> {
    *state.autosave_enabled.lock().map_err(|e| e.to_string())? = args.enabled;
    *state
        .autosave_interval_secs
        .lock()
        .map_err(|e| e.to_string())? = args.interval_secs;
    let k = args.keep_count.max(1).min(64);
    *state.autosave_keep_count.lock().map_err(|e| e.to_string())? = k;
    Ok(())
}

#[tauri::command]
fn create_new_project(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: NewProjectArgs,
) -> Result<(), String> {
    let grid_size = args.grid_size.clamp(1, MAX_GRID_SIZE);
    let shape_l = start_shape_label(args.shape);
    let label = format!("New project ({grid_size}³, {shape_l})");
    *state.file_label.lock().map_err(|e| e.to_string())? = label.clone();
    let _ = app.emit("voxelle-load-start", label);
    spawn_new_project(Arc::clone(&*state), app, grid_size, args.shape);
    Ok(())
}

/// stderr logger: debug builds default to `warn` + `voxelle_load=info`. Override with `RUST_LOG`, e.g. `RUST_LOG=voxelle_load=debug`.
fn init_load_logging() {
    let default_filter = if cfg!(debug_assertions) {
        "warn,voxelle_load=info"
    } else {
        "warn"
    };
    let _ = env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or(default_filter),
    )
    .format_timestamp_millis()
    .try_init();
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    init_load_logging();
    #[cfg(desktop)]
    let headless_server_port: Option<u16> = headless_server::parse_config();
    #[cfg(not(desktop))]
    let headless_server_port: Option<u16> = None;

    let viewer_state = Arc::new(ViewerState {
        viewer: Mutex::new(None),
        camera: Mutex::new(OrbitCamera::new()),
        file_label: Mutex::new(String::new()),
        current_file: Mutex::new(None),
        voxel_map: Mutex::new(None),
        preview_cursor: Mutex::new(None),
        preview_mode: Mutex::new(PreviewMode::Navigate),
        preview_hover: Mutex::new(PreviewHoverContext::default()),
        rendering_mode: Mutex::new(RenderingMode::Greedy),
        fps: Mutex::new(FpsCounter {
            period_start: None,
            accum_frames: 0,
            last_fps: 0,
        }),
        last_edit_perf: Mutex::new(None),
        last_scene_bounds: Mutex::new(None),
        mesh_refresh_generation: AtomicU64::new(0),
        voxel_edit_stats_cache: Mutex::new(None),
        solo_undo: Mutex::new(Vec::new()),
        solo_redo: Mutex::new(Vec::new()),
        stroke_active: Mutex::new(false),
        stroke_buffer: Mutex::new(Vec::new()),
        stroke_preview_union: Mutex::new(AHashSet::new()),
        stroke_preview_last_args: Mutex::new(None),
        stroke_preview_suppresses_hover: AtomicBool::new(false),
        sculpt_stroke_replay: Mutex::new(Vec::new()),
        collab: Arc::new(std::sync::Mutex::new(collab::CollabRuntime::default())),
        ping_flash: Mutex::new(None),
        autosave_interval_secs: Mutex::new(120),
        last_autosave: Mutex::new(None),
        autosave_enabled: Mutex::new(true),
        autosave_keep_count: Mutex::new(5),
        autosave_slot: Mutex::new(HashMap::new()),
        active_project: AtomicBool::new(false),
        fly_mode: Mutex::new(false),
        fly_input: Mutex::new(FlyInputState::default()),
        fly_last_physics: Mutex::new(None),
        selection_cells: Mutex::new(AHashSet::new()),
        selection_stroke_before: Mutex::new(None),
        selection_combine_mode: Mutex::new(SelectionCombineMode::Replace),
        selection_match_material: Mutex::new(false),
        stamp_clipboard: Mutex::new(None),
        squishy_session: Mutex::new(generators::SquishySession::new()),
        squishy_gizmo_drag: Mutex::new(None),
        start_screen_logo_transparent: std::sync::atomic::AtomicBool::new(true),
        start_screen_light: std::sync::atomic::AtomicBool::new(false),
        viewport_cursor_debug_overlay: AtomicBool::new(false),
    });
    let vs = viewer_state.clone();

    let mut builder = tauri::Builder::default();
    #[cfg(desktop)]
    {
        builder = builder
            .plugin(tauri_plugin_process::init())
            .plugin(tauri_plugin_updater::Builder::new().build());
    }
    builder
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(viewer_state.clone())
        .on_menu_event(|app, event| {
            if event.id() == "open_voxelle" {
                let state = app.state::<Arc<ViewerState>>();
                open_voxelle_file_dialog(app.clone(), state.inner().clone());
            } else if event.id() == "new_project" {
                let _ = app.emit("voxelle-open-new-project", ());
            } else if event.id() == "menu_undo" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = voxel_undo(state, app.clone());
            } else if event.id() == "menu_redo" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = voxel_redo(state, app.clone());
            } else if event.id() == "menu_save" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                if let Err(e) = save_voxelle(app.clone(), state) {
                    let _ = app.emit("voxelle-load-error", e);
                }
            } else if event.id() == "menu_save_as" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = save_voxelle_as(state, app.clone());
            } else if event.id() == "menu_export_glb" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = export_mesh_glb(state, app.clone());
            } else if event.id() == "menu_collab_start" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-collab-start-session",
                    (),
                );
            } else if event.id() == "menu_collab_join" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-collab-join-session",
                    (),
                );
            } else if event.id() == "menu_collab_leave" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-collab-leave-session",
                    (),
                );
            } else if event.id() == "menu_chat_panel" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-show-chat-panel",
                    true,
                );
            } else if event.id() == "menu_check_updates" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-check-updates",
                    (),
                );
            } else if event.id() == "menu_preferences" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-open-preferences",
                    (),
                );
            } else if event.id() == "debug_copy_performance" {
                let state = app.state::<Arc<ViewerState>>();
                if let Err(e) = copy_performance_data_to_clipboard(state.inner()) {
                    eprintln!("copy performance data: {e}");
                }
            } else if event.id() == "debug_viewport_cursor_overlay" {
                #[cfg(desktop)]
                if let Some(sel) = app.try_state::<SelectionMenuState>() {
                    if let Ok(enabled) = sel.viewport_cursor_debug.is_checked() {
                        let state = app.state::<Arc<ViewerState>>();
                        state
                            .viewport_cursor_debug_overlay
                            .store(enabled, Ordering::Relaxed);
                        let _ = app.emit_to(
                            EventTarget::webview_window("main"),
                            "voxelle-debug-viewport-cursor-overlay",
                            enabled,
                        );
                    }
                }
            } else if event.id() == "view_render_greedy" {
                let state = app.state::<Arc<ViewerState>>();
                let _ = apply_rendering_mode(&state, &app, RenderingMode::Greedy);
                wake_viewport_loop(&app);
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-rendering-mode-changed",
                    "greedy",
                );
            } else if event.id() == "view_render_marching" {
                let state = app.state::<Arc<ViewerState>>();
                let _ = apply_rendering_mode(&state, &app, RenderingMode::MarchingCubes);
                wake_viewport_loop(&app);
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-rendering-mode-changed",
                    "marchingCubes",
                );
            } else if event.id() == "view_render_dual" {
                let state = app.state::<Arc<ViewerState>>();
                let _ = apply_rendering_mode(&state, &app, RenderingMode::DualContour);
                wake_viewport_loop(&app);
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-rendering-mode-changed",
                    "dualContour",
                );
            } else if event.id() == "menu_view_perspective" {
                let state = app.state::<Arc<ViewerState>>();
                let _ = apply_orthographic(state.inner(), false);
                wake_viewport_loop(&app);
            } else if event.id() == "menu_view_ortho" {
                let state = app.state::<Arc<ViewerState>>();
                let new_o = state
                    .camera
                    .lock()
                    .map(|c| c.perspective)
                    .unwrap_or(true);
                let _ = apply_orthographic(&state, new_o);
                wake_viewport_loop(&app);
            } else if event.id() == "menu_view_render_ray" {
                let state = app.state::<Arc<ViewerState>>();
                let _ = apply_rendering_mode(&state, &app, RenderingMode::Ray);
                wake_viewport_loop(&app);
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-rendering-mode-changed",
                    "ray",
                );
            } else if event.id() == "menu_view_borders_show"
                || event.id() == "menu_view_borders_hide"
            {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-menu-not-implemented",
                    "Viewport border overlay is not available in the desktop build yet.",
                );
            } else if event.id() == "menu_view_stamp_book" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-menu-not-implemented",
                    "Stamp book is not available in the desktop build yet.",
                );
            } else if event.id() == "menu_view_project_stats" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-open-project-stats",
                    (),
                );
            } else if event.id() == "menu_voxel_hide_selected"
                || event.id() == "menu_voxel_unhide_all"
                || event.id() == "menu_voxel_hollow"
                || event.id() == "menu_voxel_scale"
                || event.id() == "menu_voxel_rotate"
                || event.id() == "menu_voxel_mirror_x"
                || event.id() == "menu_voxel_mirror_y"
                || event.id() == "menu_voxel_mirror_z"
            {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-menu-not-implemented",
                    "This voxel transform is not wired up in the desktop build yet.",
                );
            } else if event.id() == "menu_sel_all" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_select_all(app.clone(), state);
            } else if event.id() == "menu_sel_connected" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_add_connected_at_cursor(app.clone(), state);
            } else if event.id() == "menu_sel_grow" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_grow(app.clone(), state);
            } else if event.id() == "menu_sel_shrink" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_shrink(app.clone(), state);
            } else if event.id() == "menu_sel_invert" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_invert(app.clone(), state);
            } else if event.id() == "menu_sel_deselect_all" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_clear(app.clone(), state);
            } else if event.id() == "menu_sel_deselect_inner" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_deselect_inner_voxels(app.clone(), state);
            } else if event.id() == "menu_sel_deselect_voxels" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_retain_empty_only(app.clone(), state);
            } else if event.id() == "menu_sel_deselect_empty" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_retain_solid_only(app.clone(), state);
            } else if event.id() == "menu_sel_mode_replace" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_set_combine_mode(app.clone(), state, SelectionCombineMode::Replace);
            } else if event.id() == "menu_sel_mode_add" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_set_combine_mode(app.clone(), state, SelectionCombineMode::Add);
            } else if event.id() == "menu_sel_mode_subtract" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_set_combine_mode(app.clone(), state, SelectionCombineMode::Subtract);
            } else if event.id() == "menu_sel_mode_intersect" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_set_combine_mode(
                    app.clone(),
                    state,
                    SelectionCombineMode::Intersect,
                );
            } else if event.id() == "menu_sel_by_color" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-menu-selection-mode",
                    "selectByColor",
                );
            } else if event.id() == "menu_sel_coplanar" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-menu-selection-mode",
                    "selectCoplanar",
                );
            } else if event.id() == "menu_sel_coplanar_empty" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-menu-selection-mode",
                    "selectCoplanarEmpty",
                );
            } else if event.id() == "menu_sel_match_material" {
                #[cfg(desktop)]
                if let Some(sel) = app.try_state::<SelectionMenuState>() {
                    if let Ok(checked) = sel.match_material.is_checked() {
                        let state = app.state::<Arc<ViewerState>>();
                        if let Ok(mut g) = state.selection_match_material.lock() {
                            *g = checked;
                        }
                        let _ = app.emit_to(
                            EventTarget::webview_window("main"),
                            "voxelle-menu-match-material",
                            checked,
                        );
                    }
                }
            }
        })
        .setup(move |app| {
            #[cfg(desktop)]
            {
                let selection_menu_state = install_app_menu(app.handle())?;
                app.manage(selection_menu_state);
            }

            let window = app.get_webview_window("main").expect("main window");
            if headless_server_port.is_some() {
                let _ = window.hide();
            }
            let w = window.clone();
            let viewer =
                tauri::async_runtime::block_on(async move { WgpuViewer::new(w).await })
                    .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
            // Do not resize to `inner_size()` here: the 3D view matches the `.viewport` div (below
            // toolbar / beside sidebar), not the full window. Wrong dimensions break screen→world
            // raycasts until the frontend sends `viewer_resize`.
            *vs.viewer.lock().unwrap() = Some(viewer);

            #[cfg(desktop)]
            if let Some(port) = headless_server_port {
                let listener = tauri::async_runtime::block_on(tokio::net::TcpListener::bind(
                    ("127.0.0.1", port),
                ))
                .map_err(|e| format!("headless server bind 127.0.0.1:{port}: {e}"))?;
                headless_server::start(listener)?;
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            viewer_resize,
            get_viewport_pixel_size,
            get_viewport_cursor_debug,
            get_surface_pixel_size,
            set_start_screen_light,
            viewport_pointer,
            viewport_wheel,
            get_orbit_gizmo_projection,
            get_camera_zoom_percent,
            camera_fit_to_scene,
            camera_reset_view,
            camera_orbit_gizmo_drag,
            camera_snap_orbit_axis,
            camera_zoom_step,
            open_voxelle_dialog,
            confirm_app_update_dialog,
            load_voxelle_path,
            load_start_screen_logo,
            load_voxelle_recovery,
            get_last_session_info,
            create_new_project,
            voxel_pick_probe,
            voxel_stroke_anchor_coord_at_screen,
            ping_cursor_pick,
            world_to_viewport_pixels,
            sync_preview_input,
            voxel_stroke_begin,
            voxel_stroke_preview_at_screen,
            voxel_stroke_end,
            voxel_pick_color_at_screen,
            voxel_edit_at_screen,
            voxel_undo,
            voxel_redo,
            save_voxelle,
            save_voxelle_as,
            collab_host_start,
            collab_join,
            collab_leave,
            collab_local_peer_id,
            collab_kick_peer,
            collab_update_profile,
            collab_set_can_edit,
            collab_push_camera,
            collab_snap_camera,
            collab_send_chat,
            collab_send_ping,
            get_autosave_settings,
            set_autosave_settings,
            get_rendering_mode,
            set_rendering_mode,
            get_orthographic,
            set_orthographic,
            selection_menu_sync_match_material,
            debug_menu_sync_viewport_cursor_overlay,
            set_tone_mapping,
            set_mood_params,
            set_scene_lighting,
            get_scene_lighting,
            set_focal_length_mm,
            get_focal_length_mm,
            set_fly_mode,
            get_fly_mode,
            sync_fly_input,
            camera_fly_look,
            selection_toggle_at_screen,
            selection_clear,
            selection_delete_selected_voxels,
            selection_get_count,
            selection_add_by_color_at_screen,
            selection_add_coplanar_at_screen,
            selection_add_coplanar_empty_at_screen,
            selection_select_all,
            selection_invert,
            selection_grow,
            selection_shrink,
            selection_deselect_inner_voxels,
            selection_retain_empty_only,
            selection_retain_solid_only,
            selection_add_connected_at_screen,
            selection_add_connected_at_cursor,
            selection_set_combine_mode,
            get_selection_combine_mode,
            selection_stroke_begin,
            selection_stroke_end,
            selection_stroke_at_screen,
            voxel_fill_at_screen,
            clipboard_copy_selection,
            clipboard_stamp_at_screen,
            clipboard_punch_at_screen,
            voxel_sculpt_raise_at_screen,
            voxel_sculpt_stroke_at_screen,
            voxel_sculpt_stroke_preview_at_screen,
            generator_rocks_at_screen,
            generator_grass_at_screen,
            generator_rope_at_screen,
            generator_squishy_metaball_at_screen,
            squishy_session_get,
            squishy_session_set_mode,
            squishy_session_set_flags,
            squishy_metaball_add_at_screen,
            squishy_metaball_remove,
            squishy_metaball_select,
            squishy_session_clear,
            squishy_session_commit,
            squishy_pick_at_screen,
            squishy_gizmo_pointer_down,
            squishy_gizmo_pointer_move,
            squishy_gizmo_pointer_up,
            export_mesh_glb,
            get_scene_objects,
            set_active_object,
            set_object_visible,
            create_scene_object,
        ])
        .build(tauri::generate_context!())
        .expect("error building app")
        .run(move |app, event| {
            if let RunEvent::MainEventsCleared = event {
                let app_wake = app.clone();
                let state = app.state::<Arc<ViewerState>>();
                {
                    let mut cam = state.camera.lock().unwrap();
                    cam.update_damping();
                }
                // Fly WASD: integrate here with wall-clock dt between native iterations (not webview RAF).
                if *state.fly_mode.lock().unwrap() {
                    let now = Instant::now();
                    let dt = {
                        let mut last = state.fly_last_physics.lock().unwrap();
                        match *last {
                            None => {
                                *last = Some(now);
                                0.0
                            }
                            Some(t) => {
                                let d = (now - t).as_secs_f32();
                                *last = Some(now);
                                d.max(0.0)
                            }
                        }
                    };
                    let input = *state.fly_input.lock().unwrap();
                    let scale = if input.speed_scale.is_finite() {
                        input.speed_scale.clamp(0.0, 1e6)
                    } else {
                        1.0
                    };
                    if dt > 0.0
                        && (input.forward != 0.0
                            || input.right != 0.0
                            || input.up != 0.0)
                    {
                        const SPEED: f32 = 26.0;
                        let mut cam = state.camera.lock().unwrap();
                        cam.fly_move(
                            input.forward,
                            input.right,
                            input.up,
                            dt,
                            SPEED * scale,
                        );
                    }
                }
                let mut v = state.viewer.lock().unwrap();
                if let Some(viewer) = v.as_mut() {
                    let cam = state.camera.lock().unwrap();
                    viewer.update_uniforms(&cam);
                    refresh_preview_mesh(viewer, Arc::as_ref(&state), &cam);
                    refresh_selection_overlay(viewer, Arc::as_ref(&state));
                    sync_collab_peer_lines(viewer, Arc::as_ref(&state));
                    sync_ping_flash(viewer, Arc::as_ref(&state));
                    let transparent = state
                        .start_screen_logo_transparent
                        .load(Ordering::Relaxed);
                    viewer.set_start_screen_transparent(transparent);
                    let start_light = state.start_screen_light.load(Ordering::Relaxed);
                    viewer.set_start_screen_appearance(if start_light {
                        1.0
                    } else {
                        0.0
                    });
                    let sz_before = viewer.surface_size;
                    let _ = viewer.render();
                    let (vw, vh) = viewer.viewport_size();
                    if viewer.surface_size != sz_before {
                        let (sur_w, sur_h) = viewer.surface_pixel_size();
                        let _ = app.emit_to(
                            EventTarget::webview_window("main"),
                            "viewport-pixel-size",
                            ViewportPixelSize {
                                width: vw,
                                height: vh,
                                surface_width: sur_w,
                                surface_height: sur_h,
                            },
                        );
                    }
                    sample_fps_and_emit(app, &state.fps);

                    let enabled = *state.autosave_enabled.lock().unwrap();
                    let interval = *state.autosave_interval_secs.lock().unwrap();
                    let (collab_on, is_host) = {
                        let c = state.collab.lock().unwrap();
                        (c.is_active(), c.is_host())
                    };
                    if enabled
                        && interval > 0
                        && (!collab_on || is_host)
                        && state.active_project.load(Ordering::Relaxed)
                    {
                        let label = state.file_label.lock().unwrap().clone();
                        if label.ends_with(".voxelle") {
                            let now = Instant::now();
                            let last = state.last_autosave.lock().unwrap();
                            let do_save = last
                                .map(|t| now.duration_since(t).as_secs() >= interval)
                                .unwrap_or(true);
                            if do_save {
                                drop(last);
                                let doc = std::path::Path::new(&label);
                                if let Ok(dest) =
                                    next_rotating_autosave_path(&app, Arc::as_ref(&state), doc)
                                {
                                    if write_voxelle_file_to_path(None, &state, &dest).is_ok() {
                                        *state.last_autosave.lock().unwrap() = Some(now);
                                    }
                                }
                            }
                        }
                    }
                }
                // While orbit damping runs, no pointer IPC wakes the Wry loop (`ControlFlow::Wait`).
                // Queue a no-op on the main thread from a background context so the proxy wakes
                // another iteration at display rate (see `send_user_message` vs main thread).
                // Fly mode: camera look snaps smooth state so `needs_redraw` is often false; WebKit
                // may throttle RAF; fly movement uses native loop dt, not the webview clock.
                // Keep spinning while fly is on so the viewport and FPS/status UI stay live.
                let fly_on = *state.fly_mode.lock().unwrap();
                let needs_next = state.camera.lock().unwrap().needs_redraw() || fly_on;
                if needs_next {
                    tauri::async_runtime::spawn(async move {
                        let _ = app_wake.run_on_main_thread(|| {});
                    });
                }
            }
        });
}

#[cfg(test)]
pub(crate) fn minimal_viewer_state_for_collab_tests() -> Arc<ViewerState> {
    Arc::new(ViewerState {
        viewer: Mutex::new(None),
        camera: Mutex::new(OrbitCamera::new()),
        file_label: Mutex::new(String::new()),
        current_file: Mutex::new(None),
        voxel_map: Mutex::new(None),
        preview_cursor: Mutex::new(None),
        preview_mode: Mutex::new(PreviewMode::Navigate),
        preview_hover: Mutex::new(PreviewHoverContext::default()),
        rendering_mode: Mutex::new(RenderingMode::Greedy),
        fps: Mutex::new(FpsCounter {
            period_start: None,
            accum_frames: 0,
            last_fps: 0,
        }),
        last_edit_perf: Mutex::new(None),
        last_scene_bounds: Mutex::new(None),
        mesh_refresh_generation: AtomicU64::new(0),
        voxel_edit_stats_cache: Mutex::new(None),
        solo_undo: Mutex::new(Vec::new()),
        solo_redo: Mutex::new(Vec::new()),
        stroke_active: Mutex::new(false),
        stroke_buffer: Mutex::new(Vec::new()),
        stroke_preview_union: Mutex::new(AHashSet::new()),
        stroke_preview_last_args: Mutex::new(None),
        stroke_preview_suppresses_hover: AtomicBool::new(false),
        sculpt_stroke_replay: Mutex::new(Vec::new()),
        collab: Arc::new(std::sync::Mutex::new(collab::CollabRuntime::default())),
        ping_flash: Mutex::new(None),
        autosave_interval_secs: Mutex::new(120),
        last_autosave: Mutex::new(None),
        autosave_enabled: Mutex::new(true),
        autosave_keep_count: Mutex::new(5),
        autosave_slot: Mutex::new(HashMap::new()),
        active_project: AtomicBool::new(false),
        fly_mode: Mutex::new(false),
        fly_input: Mutex::new(FlyInputState::default()),
        fly_last_physics: Mutex::new(None),
        selection_cells: Mutex::new(AHashSet::new()),
        selection_stroke_before: Mutex::new(None),
        selection_combine_mode: Mutex::new(SelectionCombineMode::Replace),
        selection_match_material: Mutex::new(false),
        stamp_clipboard: Mutex::new(None),
        squishy_session: Mutex::new(generators::SquishySession::new()),
        squishy_gizmo_drag: Mutex::new(None),
        start_screen_logo_transparent: std::sync::atomic::AtomicBool::new(true),
        start_screen_light: std::sync::atomic::AtomicBool::new(false),
        viewport_cursor_debug_overlay: AtomicBool::new(false),
    })
}

#[cfg(test)]
mod edit_perf_tests {
    use super::*;
    use voxelle::{MaterialId, Voxel};

    fn voxel_at(x: i32, y: i32, z: i32, object_id: u32) -> Voxel {
        Voxel {
            x,
            y,
            z,
            color: 1,
            material: MaterialId::Plastic,
            object_id,
        }
    }

    #[test]
    fn resolve_stats_incremental_add_shrinks_aabb_min() {
        let cache = Some(VoxelEditStatsCache {
            aabb_min: (5, 5, 5),
            common_object_id: Some(0),
        });
        let added = voxel_at(2, 5, 5, 0);
        let delta = voxel_edit::VoxelEditDelta::Added(added);
        let voxels = vec![voxel_at(5, 5, 5, 0), added];
        let s = resolve_voxel_edit_stats(&voxels, &delta, cache);
        assert_eq!(s.aabb_min, (2, 5, 5));
        assert_eq!(s.common_object_id, Some(0));
    }

    #[test]
    fn resolve_stats_remove_interior_preserves_cache() {
        let cache = Some(VoxelEditStatsCache {
            aabb_min: (0, 0, 0),
            common_object_id: Some(0),
        });
        let voxels = vec![voxel_at(0, 0, 0, 0), voxel_at(5, 5, 5, 0)];
        let delta = voxel_edit::VoxelEditDelta::Removed {
            voxel: voxel_at(5, 5, 5, 0),
        };
        let s = resolve_voxel_edit_stats(&voxels, &delta, cache);
        assert_eq!(s.aabb_min, (0, 0, 0));
        assert_eq!(s.common_object_id, Some(0));
    }

    #[test]
    fn resolve_stats_remove_on_min_face_rescans() {
        let cache = Some(VoxelEditStatsCache {
            aabb_min: (0, 0, 0),
            common_object_id: Some(0),
        });
        let voxels = vec![voxel_at(5, 5, 5, 0)];
        let delta = voxel_edit::VoxelEditDelta::Removed {
            voxel: voxel_at(0, 0, 0, 0),
        };
        let s = resolve_voxel_edit_stats(&voxels, &delta, cache);
        assert_eq!(s.aabb_min, (5, 5, 5));
        assert_eq!(s.common_object_id, Some(0));
    }

    #[test]
    fn resolve_stats_add_second_object_id_clears_common() {
        let cache = Some(VoxelEditStatsCache {
            aabb_min: (0, 0, 0),
            common_object_id: Some(0),
        });
        let added = voxel_at(1, 1, 1, 1);
        let delta = voxel_edit::VoxelEditDelta::Added(added);
        let voxels = vec![voxel_at(0, 0, 0, 0), added];
        let s = resolve_voxel_edit_stats(&voxels, &delta, cache);
        assert_eq!(s.common_object_id, None);
    }
}
