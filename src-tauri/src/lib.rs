mod camera;
mod collab;
pub mod crash_guard;
mod export_glb;
mod generators;
mod gpu_brick;
/// Greedy CPU meshing (public for `cargo bench`).
pub mod greedy_mesh;
#[cfg(desktop)]
mod headless_server;
#[cfg(target_os = "macos")]
mod macos_titlebar;
#[cfg(target_os = "macos")]
mod macos_undo;
mod marching_tables;
mod paint_color_distrib;
mod render;
mod render_constants;
mod sculpt_mesh_smooth;
mod smooth_mesh;
mod stroke_modes;
mod voxel_edit;
/// Voxel format / types (public for `cargo bench` and tests).
pub mod voxelle;

use camera::OrbitCamera;
use gpu_brick::{BrickCellWrite, GpuVoxelBrick};
use render::{
    compute_greedy_rebuild_cpu, GpuPeerLabel, MoodParams, PreparedGreedyRebuild,
    PreparedOpaqueUpload, WgpuViewer,
};
use std::any::Any;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, EventTarget, Manager, RunEvent, Runtime, State};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

use ahash::{AHashMap, AHashSet, AHasher};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};

use voxelle::scene::object_world_matrix;
use voxelle::{
    decode_payload, encode_payload_v4, focal_length_to_fov_y_radians, start_shape::StartShape,
};

/// Convert file-format `MoodSettings` → GPU-ready `MoodParams`.
fn mood_settings_to_params(m: &voxelle::MoodSettings) -> MoodParams {
    MoodParams {
        vignette: m.vignette,
        grain_enabled: m.grain_enabled,
        grain_strength: m.grain_strength,
        grain_animated: m.grain_animated,
        grain_speed: m.grain_speed,
        grain_colorful: m.grain_colorful,
        atm_enabled: m.atm_enabled,
        atm_color: m.atm_color.clone(),
        atm_thickness: m.atm_thickness,
        atm_density: m.atm_density,
        atm_aerial: m.atm_aerial,
        atm_positive_side: m.atm_positive_side,
        atm_plane_nx: m.atm_plane_nx,
        atm_plane_ny: m.atm_plane_ny,
        atm_plane_nz: m.atm_plane_nz,
        atm_plane_c: m.atm_plane_c,
        atm_height_bias: m.atm_height_bias,
        atm_height_falloff: m.atm_height_falloff,
        atm_drift_enabled: m.atm_drift_enabled,
        atm_drift_amount: m.atm_drift_amount,
        atm_drift_scale: m.atm_drift_scale,
        atm_drift_speed: m.atm_drift_speed,
        dt_enabled: m.dt_enabled,
        dt_near_color: m.dt_near_color.clone(),
        dt_mid_color: m.dt_mid_color.clone(),
        dt_far_color: m.dt_far_color.clone(),
        dt_near_dist: m.dt_near_dist,
        dt_far_dist: m.dt_far_dist,
        dt_strength: m.dt_strength,
        ss_enabled: m.ss_enabled,
        ss_strength: m.ss_strength,
        ss_decay: m.ss_decay,
        ss_density: m.ss_density,
        ss_weight: m.ss_weight,
        ss_samples: m.ss_samples,
        ssr_enabled: m.ssr_enabled,
        ssr_strength: m.ssr_strength,
        bloom_strength: m.bloom_strength,
    }
}

/// Convert `MoodParams` (from frontend) → file-format `MoodSettings`.
fn mood_params_to_settings(p: &MoodParams) -> voxelle::MoodSettings {
    voxelle::MoodSettings {
        vignette: p.vignette,
        grain_enabled: p.grain_enabled,
        grain_strength: p.grain_strength,
        grain_animated: p.grain_animated,
        grain_speed: p.grain_speed,
        grain_colorful: p.grain_colorful,
        atm_enabled: p.atm_enabled,
        atm_color: p.atm_color.clone(),
        atm_thickness: p.atm_thickness,
        atm_density: p.atm_density,
        atm_aerial: p.atm_aerial,
        atm_positive_side: p.atm_positive_side,
        atm_plane_nx: p.atm_plane_nx,
        atm_plane_ny: p.atm_plane_ny,
        atm_plane_nz: p.atm_plane_nz,
        atm_plane_c: p.atm_plane_c,
        atm_height_bias: p.atm_height_bias,
        atm_height_falloff: p.atm_height_falloff,
        atm_drift_enabled: p.atm_drift_enabled,
        atm_drift_amount: p.atm_drift_amount,
        atm_drift_scale: p.atm_drift_scale,
        atm_drift_speed: p.atm_drift_speed,
        dt_enabled: p.dt_enabled,
        dt_near_color: p.dt_near_color.clone(),
        dt_mid_color: p.dt_mid_color.clone(),
        dt_far_color: p.dt_far_color.clone(),
        dt_near_dist: p.dt_near_dist,
        dt_far_dist: p.dt_far_dist,
        dt_strength: p.dt_strength,
        ss_enabled: p.ss_enabled,
        ss_strength: p.ss_strength,
        ss_decay: p.ss_decay,
        ss_density: p.ss_density,
        ss_weight: p.ss_weight,
        ss_samples: p.ss_samples,
        ssr_enabled: p.ssr_enabled,
        ssr_strength: p.ssr_strength,
        bloom_strength: p.bloom_strength,
    }
}

struct FpsCounter {
    period_start: Option<Instant>,
    accum_frames: u32,
    /// Last computed viewport FPS (updated when we emit `viewport-fps`).
    last_fps: u32,
}

fn sample_fps_and_emit(app: &AppHandle, counter: &Mutex<FpsCounter>) {
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
            "stamp" => Self::Stamp,
            "punch" => Self::Punch,
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
        for k in greedy_mesh::dirty_chunk_keys_for_voxel(x, y, z, origin, cs) {
            set.insert(k);
        }
    }
    set.into_iter().collect()
}

fn deltas_to_brick_patches(
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
fn viewport_texels_from_norm(nx: f32, ny: f32, w: f32, h: f32) -> (f32, f32) {
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
fn resolve_spray_constraint_plane(
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

#[derive(serde::Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct VoxelEditAtScreen {
    nx: f32,
    ny: f32,
    tool: voxel_edit::EditTool,
    color: u32,
    /// Multi-color palette (when non-empty and len > 1, overrides `color`).
    #[serde(default)]
    palette: Vec<u32>,
    /// Color distribution mode + params; used only when `palette.len() > 1`.
    #[serde(default)]
    paint_color_distrib: Option<paint_color_distrib::PaintColorDistrib>,
    /// Deterministic seed for the current stroke (for randomSingle / preview consistency).
    #[serde(default)]
    stroke_seed: u32,
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
    /// Solid/empty flood adjacency (fill stroke); mirrors selection fill.
    #[serde(default)]
    fill_select_diagonals: bool,
    #[serde(default = "default_fill_respects_color")]
    fill_respects_color: bool,
}

/// Build a per-voxel color resolver from the palette + distribution args.
/// Falls back to `color_single` when palette has 0 or 1 entry.
fn build_color_resolver(
    color_single: u32,
    palette: Vec<u32>,
    distrib: Option<paint_color_distrib::PaintColorDistrib>,
    stroke_seed: u32,
) -> impl Fn(i32, i32, i32) -> u32 {
    move |x, y, z| {
        if palette.len() > 1 {
            if let Some(ref d) = distrib {
                d.resolve(&palette, stroke_seed, x, y, z)
            } else {
                let idx = paint_color_distrib::paint_color_index(x, y, z, palette.len());
                palette[idx] & 0x00ff_ffff
            }
        } else {
            color_single
        }
    }
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

fn default_smooth_aggressiveness_sculpt() -> u32 {
    100
}

fn default_laplacian_iterations_sculpt() -> u32 {
    4
}

fn default_laplacian_relax_sculpt() -> u32 {
    50
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
    brush_clip_bottom_half: bool,
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
    #[serde(default)]
    terrain_flatten_use_base_y: bool,
    #[serde(default)]
    terrain_sub_voxel: bool,
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
    #[serde(default)]
    sculpt_smooth_variant: crate::sculpt_mesh_smooth::SculptSmoothVariant,
    #[serde(default)]
    smooth_neighbor_radius: u32,
    #[serde(default = "default_smooth_aggressiveness_sculpt")]
    smooth_aggressiveness: u32,
    #[serde(default = "default_laplacian_iterations_sculpt")]
    smooth_laplacian_iterations: u32,
    #[serde(default = "default_laplacian_relax_sculpt")]
    smooth_laplacian_relax_pct: u32,
    #[serde(default)]
    wall_polygon_vertices: Option<Vec<[i32; 3]>>,
    #[serde(default)]
    extrude_profile: voxel_edit::ExtrudeProfile,
    #[serde(default)]
    extrude_end_cap: voxel_edit::ExtrudeEndCap,
    #[serde(default)]
    extrude_taper: bool,
    #[serde(default)]
    extrude_taper_start: f32,
    #[serde(default)]
    extrude_taper_end: f32,
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
    palette: Vec<u32>,
    paint_color_distrib: Option<paint_color_distrib::PaintColorDistrib>,
    material: String,
    match_material: bool,
    /// When false (e.g. sculpt), hover uses the legacy single-cell preview.
    use_brush_preview: bool,
    /// `Some("rope" | "cloth" | "rocks" | "grass")` when the generator tool is active (webview sync).
    generator_kind: Option<String>,
    generator_rope_first_nx: Option<f32>,
    generator_rope_first_ny: Option<f32>,
    generator_rope_sag: f32,
    generator_rope_tension: f32,
    generator_rope_gravity_direction: String,
    generator_cloth_pins: Vec<[i32; 3]>,
    generator_cloth_tension: f32,
    generator_cloth_gravity_direction: String,
    generator_cloth_gravity_scale: f64,
    generator_cloth_stiffness_scale: f64,
    generator_cloth_iterations: u32,
    generator_cloth_constraint_passes: u32,
    generator_rock_size: i32,
    generator_rock_roughness: f32,
    generator_rock_seed: i32,
    generator_rock_count: i32,
    generator_rock_cluster_radius: i32,
    generator_rock_sink_direction: i32,
    generator_rock_sink_amount: i32,
    generator_grass_radius: i32,
    generator_grass_density: f32,
    generator_grass_max_height: i32,
    generator_grass_seed: i32,
    generator_roof_pins: Vec<[i32; 3]>,
    generator_roof_style: String,
    generator_roof_height: i32,
    generator_roof_thickness: i32,
    generator_roof_break_ratio: f32,
    generator_roof_wall_height: i32,
    generator_roof_parapet_height: i32,
    generator_roof_salt_skew: f32,
    generator_roof_hollow: bool,
    generator_ashlar_size: i32,
    generator_ashlar_roughness: f32,
    generator_ashlar_seed: i32,
    generator_ashlar_thickness: i32,
    // Flora
    generator_flora_seed: i32,
    generator_flora_height: i32,
    generator_flora_girth: i32,
    generator_flora_wobble: f32,
    generator_flora_taper: f32,
    generator_flora_stem_count: i32,
    generator_flora_cluster_radius: i32,
    generator_flora_branch_count: i32,
    generator_flora_branch_depth: i32,
    generator_flora_branch_start: f32,
    generator_flora_branch_spread: f32,
    generator_flora_braid_strands: i32,
    generator_flora_braid_twist: f32,
    generator_flora_canopy: f32,
    // Insecta
    generator_insecta_species: String,
    generator_insecta_total_length: i32,
    generator_insecta_head_ratio: f32,
    generator_insecta_thorax_ratio: f32,
    generator_insecta_abdomen_ratio: f32,
    generator_insecta_body_half_width: i32,
    generator_insecta_body_half_height: i32,
    generator_insecta_abdomen_taper: f32,
    generator_insecta_head_shape: i32,
    generator_insecta_anchor_offset_u: i32,
    generator_insecta_anchor_offset_v: i32,
    generator_insecta_body_yaw: f32,
    generator_insecta_body_arch: f32,
    generator_insecta_antenna_length: i32,
    generator_insecta_antenna_spread: f32,
    generator_insecta_antenna_pitch: f32,
    generator_insecta_antenna_root: i32,
    generator_insecta_mandible_length: i32,
    generator_insecta_mandible_spread: f32,
    generator_insecta_mandible_forward: i32,
    generator_insecta_wing_shape: i32,
    generator_insecta_show_wing_fore: bool,
    generator_insecta_wing_fore_length: i32,
    generator_insecta_wing_fore_width: i32,
    generator_insecta_wing_fore_spread: f32,
    generator_insecta_wing_fore_pitch: f32,
    generator_insecta_wing_fore_offset: i32,
    generator_insecta_wing_fore_forward_cant: f32,
    generator_insecta_show_wing_hind: bool,
    generator_insecta_wing_hind_length: i32,
    generator_insecta_wing_hind_width: i32,
    generator_insecta_wing_hind_spread: f32,
    generator_insecta_wing_hind_pitch: f32,
    generator_insecta_wing_hind_offset: i32,
    // Fauna
    generator_fauna_stance: String,
    generator_fauna_archetype: String,
    generator_fauna_anchor_offset_u: i32,
    generator_fauna_anchor_offset_v: i32,
    generator_fauna_body_yaw: f32,
    generator_fauna_body_arch: f32,
    generator_fauna_spine_segments: i32,
    generator_fauna_body_length: i32,
    generator_fauna_body_half_width: i32,
    generator_fauna_body_half_height: i32,
    generator_fauna_neck_length: i32,
    generator_fauna_neck_half_width: i32,
    generator_fauna_neck_half_height: i32,
    generator_fauna_head_length: i32,
    generator_fauna_head_half_width: i32,
    generator_fauna_head_half_height: i32,
    generator_fauna_tail_length: i32,
    generator_fauna_shoulder_offset_forward: i32,
    generator_fauna_hip_offset_forward: i32,
    generator_fauna_front_upper_length: i32,
    generator_fauna_front_lower_length: i32,
    generator_fauna_hind_upper_length: i32,
    generator_fauna_hind_lower_length: i32,
    generator_fauna_auto_foot_placement: bool,
    // Piscina
    generator_piscina_seed: i32,
    generator_piscina_species: String,
    generator_piscina_length: i32,
    generator_piscina_width: i32,
    generator_piscina_thickness: i32,
    generator_piscina_spine_bend: f32,
    generator_piscina_spine_s_curve: f32,
    generator_piscina_fin_dorsal: i32,
    generator_piscina_fin_anal: i32,
    generator_piscina_fin_caudal: i32,
    generator_piscina_fin_pectoral: i32,
    generator_piscina_fin_pelvic: i32,
    generator_piscina_fin_adipose: i32,
    generator_piscina_show_fin_dorsal: bool,
    generator_piscina_show_fin_anal: bool,
    generator_piscina_show_fin_caudal: bool,
    generator_piscina_show_fin_pectoral: bool,
    generator_piscina_show_fin_pelvic: bool,
    generator_piscina_show_fin_adipose: bool,
    generator_piscina_anchor_offset_u: i32,
    generator_piscina_anchor_offset_v: i32,
    /// Stamp placement origin X: 0 = min edge, 1 = center, 2 = max edge.
    stamp_origin_x: i32,
    /// Stamp placement origin Z: 0 = min edge, 1 = center, 2 = max edge.
    stamp_origin_z: i32,
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
            palette: Vec::new(),
            paint_color_distrib: None,
            material: String::new(),
            match_material: false,
            use_brush_preview: true,
            generator_kind: None,
            generator_rope_first_nx: None,
            generator_rope_first_ny: None,
            generator_rope_sag: 2.5,
            generator_rope_tension: 0.5,
            generator_rope_gravity_direction: "down".into(),
            generator_cloth_pins: Vec::new(),
            generator_cloth_tension: 0.5,
            generator_cloth_gravity_direction: "down".into(),
            generator_cloth_gravity_scale: 1.0,
            generator_cloth_stiffness_scale: 1.0,
            generator_cloth_iterations: 0,
            generator_cloth_constraint_passes: 2,
            generator_rock_size: 4,
            generator_rock_roughness: 0.4,
            generator_rock_seed: 42,
            generator_rock_count: 1,
            generator_rock_cluster_radius: 1,
            generator_rock_sink_direction: 0,
            generator_rock_sink_amount: 0,
            generator_grass_radius: 4,
            generator_grass_density: 0.6,
            generator_grass_max_height: 3,
            generator_grass_seed: 42,
            generator_roof_pins: Vec::new(),
            generator_roof_style: "gable".into(),
            generator_roof_height: 6,
            generator_roof_thickness: 1,
            generator_roof_break_ratio: 0.5,
            generator_roof_wall_height: 3,
            generator_roof_parapet_height: 2,
            generator_roof_salt_skew: 0.0,
            generator_roof_hollow: false,
            generator_ashlar_size: 4,
            generator_ashlar_roughness: 0.3,
            generator_ashlar_seed: 42,
            generator_ashlar_thickness: 3,
            // Flora
            generator_flora_seed: 42,
            generator_flora_height: 10,
            generator_flora_girth: 2,
            generator_flora_wobble: 0.3,
            generator_flora_taper: 0.5,
            generator_flora_stem_count: 1,
            generator_flora_cluster_radius: 0,
            generator_flora_branch_count: 4,
            generator_flora_branch_depth: 2,
            generator_flora_branch_start: 0.3,
            generator_flora_branch_spread: 0.5,
            generator_flora_braid_strands: 0,
            generator_flora_braid_twist: 0.5,
            generator_flora_canopy: 2.0,
            // Insecta
            generator_insecta_species: "beetle".into(),
            generator_insecta_total_length: 12,
            generator_insecta_head_ratio: 1.0,
            generator_insecta_thorax_ratio: 1.0,
            generator_insecta_abdomen_ratio: 2.0,
            generator_insecta_body_half_width: 2,
            generator_insecta_body_half_height: 2,
            generator_insecta_abdomen_taper: 0.5,
            generator_insecta_head_shape: 0,
            generator_insecta_anchor_offset_u: 0,
            generator_insecta_anchor_offset_v: 0,
            generator_insecta_body_yaw: 0.0,
            generator_insecta_body_arch: 0.0,
            generator_insecta_antenna_length: 4,
            generator_insecta_antenna_spread: 0.4,
            generator_insecta_antenna_pitch: 0.3,
            generator_insecta_antenna_root: 1,
            generator_insecta_mandible_length: 2,
            generator_insecta_mandible_spread: 0.3,
            generator_insecta_mandible_forward: 1,
            generator_insecta_wing_shape: 0,
            generator_insecta_show_wing_fore: true,
            generator_insecta_wing_fore_length: 8,
            generator_insecta_wing_fore_width: 4,
            generator_insecta_wing_fore_spread: 0.5,
            generator_insecta_wing_fore_pitch: 0.1,
            generator_insecta_wing_fore_offset: 0,
            generator_insecta_wing_fore_forward_cant: 0.0,
            generator_insecta_show_wing_hind: true,
            generator_insecta_wing_hind_length: 6,
            generator_insecta_wing_hind_width: 4,
            generator_insecta_wing_hind_spread: 0.6,
            generator_insecta_wing_hind_pitch: 0.2,
            generator_insecta_wing_hind_offset: 0,
            // Fauna
            generator_fauna_stance: "quadruped".into(),
            generator_fauna_archetype: "mammal".into(),
            generator_fauna_anchor_offset_u: 0,
            generator_fauna_anchor_offset_v: 0,
            generator_fauna_body_yaw: 0.0,
            generator_fauna_body_arch: 0.0,
            generator_fauna_spine_segments: 5,
            generator_fauna_body_length: 10,
            generator_fauna_body_half_width: 2,
            generator_fauna_body_half_height: 2,
            generator_fauna_neck_length: 3,
            generator_fauna_neck_half_width: 1,
            generator_fauna_neck_half_height: 1,
            generator_fauna_head_length: 3,
            generator_fauna_head_half_width: 2,
            generator_fauna_head_half_height: 2,
            generator_fauna_tail_length: 4,
            generator_fauna_shoulder_offset_forward: 3,
            generator_fauna_hip_offset_forward: -3,
            generator_fauna_front_upper_length: 4,
            generator_fauna_front_lower_length: 4,
            generator_fauna_hind_upper_length: 4,
            generator_fauna_hind_lower_length: 4,
            generator_fauna_auto_foot_placement: true,
            // Piscina
            generator_piscina_seed: 42,
            generator_piscina_species: "bass".into(),
            generator_piscina_length: 14,
            generator_piscina_width: 4,
            generator_piscina_thickness: 3,
            generator_piscina_spine_bend: 0.1,
            generator_piscina_spine_s_curve: 0.0,
            generator_piscina_fin_dorsal: 4,
            generator_piscina_fin_anal: 4,
            generator_piscina_fin_caudal: 4,
            generator_piscina_fin_pectoral: 4,
            generator_piscina_fin_pelvic: 4,
            generator_piscina_fin_adipose: 4,
            generator_piscina_show_fin_dorsal: true,
            generator_piscina_show_fin_anal: true,
            generator_piscina_show_fin_caudal: true,
            generator_piscina_show_fin_pectoral: true,
            generator_piscina_show_fin_pelvic: true,
            generator_piscina_show_fin_adipose: false,
            generator_piscina_anchor_offset_u: 0,
            generator_piscina_anchor_offset_v: 0,
            stamp_origin_x: 0,
            stamp_origin_z: 0,
        }
    }
}

/// `texel_s*` included because single-cell hover is anchored at the ray face hit (moves within a cell).
fn hash_single_cell_preview(
    mode: PreviewMode,
    cx: i32,
    cy: i32,
    cz: i32,
    tag: u8,
    debug_overlay: bool,
    palette_color: u32,
    object_id: u32,
    texel_sx: f32,
    texel_sy: f32,
) -> u64 {
    let mut h = AHasher::default();
    mode.hash(&mut h);
    cx.hash(&mut h);
    cy.hash(&mut h);
    cz.hash(&mut h);
    tag.hash(&mut h);
    debug_overlay.hash(&mut h);
    palette_color.hash(&mut h);
    object_id.hash(&mut h);
    texel_sx.to_bits().hash(&mut h);
    texel_sy.to_bits().hash(&mut h);
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
    palette_color: u32,
) -> u64 {
    let mut h = AHasher::default();
    PreviewMode::Squishy.hash(&mut h);
    debug_overlay.hash(&mut h);
    palette_color.hash(&mut h);
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
    voxel_map: &AHashMap<greedy_mesh::VoxelCoord, usize>,
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
    for c in &sorted {
        voxel_map.contains_key(c).hash(&mut h);
    }
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
    /// Bumps each time a file/project load begins; stale loads bail out instead of overwriting a newer load.
    pub load_generation: AtomicU64,
    /// Chunks built by background meshing thread, waiting for main-thread GPU upload.
    pub(crate) chunk_mesh_inbox: Mutex<VecDeque<(greedy_mesh::ChunkKey, greedy_mesh::MeshBuffers)>>,
    /// Guest edit/undo/redo items waiting for main-thread processing (same pattern as `chunk_mesh_inbox`).
    pub(crate) collab_edit_inbox: Mutex<VecDeque<collab::CollabInboxItem>>,
    /// SpatialMeshCache from background streaming load; moved to viewer after all chunks are uploaded.
    pub(crate) deferred_spatial_cache: Mutex<Option<greedy_mesh::SpatialMeshCache>>,
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
    /// Stored ray spine for straight-line extrude (used by ray-based extrude preview/recompute).
    pub(crate) extrude_ray_spine: Mutex<Option<Vec<greedy_mesh::VoxelCoord>>>,
    pub collab: Arc<Mutex<collab::CollabRuntime>>,
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
    grid_overlay_cache_key: Mutex<Option<u64>>,
    selection_overlay_cache_key: Mutex<Option<u64>>,
    preview_overlay_cache_key: Mutex<Option<u64>>,
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
fn get_viewport_pixel_size(
    state: State<'_, Arc<ViewerState>>,
) -> Result<ViewportPixelSize, String> {
    let v = state.viewer.lock();
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
    /// Swapchain drawable size (physical px); see [`WgpuViewer::surface_pixel_size`].
    surface_width: u32,
    surface_height: u32,
    /// Top-left of the viewport texture in surface pixel space (`copy_texture_to_texture` dest origin).
    viewport_origin_x: u32,
    viewport_origin_y: u32,
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
    /// Preview cube world center projected back to normalized viewport coords.
    proj_cube_nx: Option<f32>,
    proj_cube_ny: Option<f32>,
    /// Same projection path as `proj_cube_*`, but voxel **center** in world space (matches hover mesh).
    proj_center_nx: Option<f32>,
    proj_center_ny: Option<f32>,
}

/// #region agent log
fn debug_agent_ndjson_log(payload: serde_json::Value) {
    const PATH: &str = "/Users/zelda/Documents/digital-garden/.cursor/debug-0e537f.log";
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(PATH)
    {
        if let Ok(s) = serde_json::to_string(&payload) {
            let _ = writeln!(f, "{}", s);
        }
    }
}
/// #endregion

#[tauri::command]
fn get_viewport_cursor_debug(
    state: State<'_, Arc<ViewerState>>,
) -> Result<ViewportCursorDebug, String> {
    let cam = state.camera.lock();
    let (vw, vh, wf, hf, viewport_x, viewport_y, surface_w, surface_h) = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (vw, vh) = viewer.viewport_size();
        let (sw, sh) = viewer.surface_pixel_size();
        (
            vw,
            vh,
            vw as f32,
            vh as f32,
            viewer.viewport_x,
            viewer.viewport_y,
            sw,
            sh,
        )
    };
    let pc = state.preview_cursor.lock();
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
        None => (None, None, None, None, None, None, None, None, None, None),
    };
    // #region agent log
    debug_agent_ndjson_log(serde_json::json!({
        "sessionId": "0e537f",
        "hypothesisId": "H_rust_surface",
        "location": "lib.rs:get_viewport_cursor_debug",
        "message": "gpu viewport + texels",
        "data": {
            "viewportWidth": vw,
            "viewportHeight": vh,
            "viewportX": viewport_x,
            "viewportY": viewport_y,
            "surfaceW": surface_w,
            "surfaceH": surface_h,
            "previewNx": preview_nx,
            "previewNy": preview_ny,
            "texelSx": texel_sx,
            "texelSy": texel_sy,
            "aspectWh": (vw as f64 / vh.max(1) as f64),
        },
        "timestamp": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0),
    }));
    // #endregion
    let (proj_cube_nx, proj_cube_ny, proj_center_nx, proj_center_ny) = match (texel_sx, texel_sy) {
        (Some(sx), Some(sy)) => {
            let file_guard = state.current_file.lock();
            let vmap_guard = state.voxel_map.lock();
            match (file_guard.as_ref(), vmap_guard.as_ref()) {
                (Some(file), Some(vmap)) if !file.voxels.is_empty() => {
                    let grid_size = voxel_edit::effective_ray_grid_size(file);
                    let (o, d) = voxel_edit::screen_to_world_ray(&cam, wf, hf, sx, sy);
                    match voxel_edit::ray_first_solid_scene(o, d, file, vmap, grid_size) {
                        Some(((cx, cy, cz), _prev, oid)) => {
                            let m = object_world_matrix(&file.objects, oid);
                            let wp_hit =
                                voxel_edit::world_ray_entry_on_voxel_cell(o, d, cx, cy, cz, m)
                                    .unwrap_or_else(|| {
                                        m.transform_point3(glam::Vec3::new(
                                            cx as f32, cy as f32, cz as f32,
                                        ))
                                    });
                            let wc = m
                                .transform_point3(glam::Vec3::new(cx as f32, cy as f32, cz as f32));
                            let denom_x = (wf - 1.0).max(1.0);
                            let denom_y = (hf - 1.0).max(1.0);
                            let hit_norm = voxel_edit::world_to_viewport_pixels(
                                &cam, wf, hf, wp_hit.x, wp_hit.y, wp_hit.z,
                            )
                            .map(|(px, py)| (px / denom_x, py / denom_y));
                            let center_norm = voxel_edit::world_to_viewport_pixels(
                                &cam, wf, hf, wc.x, wc.y, wc.z,
                            )
                            .map(|(px, py)| (px / denom_x, py / denom_y));
                            // #region agent log
                            if let (Some((hnx, hny)), Some((cnx, cny))) = (hit_norm, center_norm) {
                                debug_agent_ndjson_log(serde_json::json!({
                                        "sessionId": "0e537f",
                                        "runId": "post-fix",
                                        "hypothesisId": "H1_center_vs_hit",
                                        "location": "lib.rs:get_viewport_cursor_debug",
                                        "message": "proj hit vs voxel center (hover mesh uses center)",
                                    "data": {
                                        "cx": cx, "cy": cy, "cz": cz,
                                        "projHitNx": hnx, "projHitNy": hny,
                                        "projCenterNx": cnx, "projCenterNy": cny,
                                        "deltaCenterMinusHitNx": cnx - hnx,
                                        "deltaCenterMinusHitNy": cny - hny,
                                        "previewNx": preview_nx,
                                        "previewNy": preview_ny,
                                    },
                                    "timestamp": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0),
                                }));
                            }
                            // #endregion
                            match (hit_norm, center_norm) {
                                (Some((hnx, hny)), Some((cnx, cny))) => {
                                    (Some(hnx), Some(hny), Some(cnx), Some(cny))
                                }
                                (Some((hnx, hny)), _) => (Some(hnx), Some(hny), None, None),
                                _ => (None, None, None, None),
                            }
                        }
                        None => (None, None, None, None),
                    }
                }
                _ => (None, None, None, None),
            }
        }
        _ => (None, None, None, None),
    };
    Ok(ViewportCursorDebug {
        viewport_width: vw,
        viewport_height: vh,
        surface_width: surface_w,
        surface_height: surface_h,
        viewport_origin_x: viewport_x,
        viewport_origin_y: viewport_y,
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
        proj_cube_nx,
        proj_cube_ny,
        proj_center_nx,
        proj_center_ny,
    })
}

#[tauri::command]
fn get_surface_pixel_size(state: State<'_, Arc<ViewerState>>) -> Result<SurfacePixelSize, String> {
    let v = state.viewer.lock();
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
    state.start_screen_light.store(light, Ordering::Relaxed);
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

    let mut g = state.viewer.lock();
    if let Some(v) = g.as_mut() {
        v.resize(
            sw,
            sh,
            viewport_x,
            viewport_y,
            viewport_width,
            viewport_height,
        );
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
    if *state.fly_mode.lock() || *state.walk_mode.lock() {
        return Ok(());
    }
    // Read size without holding `camera` — the run loop locks `viewer` then `camera`; taking
    // `camera` then `viewer` here deadlocks with the render tick and freezes orbit input.
    let (vw, vh) = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Ok(());
        };
        let (w, h) = viewer.viewport_size();
        let w = w as f32;
        let h = h as f32;
        (w, h.max(1.0))
    };

    let (x, y) = viewport_texels_from_norm(ev.nx, ev.ny, vw, vh);
    let mut cam = state.camera.lock();
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
    if *state.fly_mode.lock() || *state.walk_mode.lock() {
        return Ok(());
    }
    let mut cam = state.camera.lock();
    if cam.logo_splash_rest.is_some() {
        return Ok(());
    }
    // Same `deltaY` semantics as the browser / Three.js `onMouseWheel`.
    cam.dolly_delta(ev.delta_y);
    wake_viewport_loop(&app);
    Ok(())
}

fn scene_bounds_min_max_grid(state: &ViewerState) -> (glam::Vec3, glam::Vec3, i32) {
    let guard = state.last_scene_bounds.lock();
    if let Some(b) = guard.as_ref() {
        let grid = state
            .current_file
            .lock()
            .as_ref()
            .map(|file| file.grid_size)
            .unwrap_or(64);
        return (b.min, b.max, grid);
    }
    let fg = state.current_file.lock();
    if let Some(ref file) = *fg {
        let b = greedy_mesh::mesh_bounds_from_voxels_world(&file.voxels, &file.objects)
            .or_else(|| greedy_mesh::mesh_bounds_from_voxels(&file.voxels))
            .unwrap_or_else(|| greedy_mesh::mesh_bounds_for_cube_side(file.grid_size));
        return (b.min, b.max, file.grid_size);
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
fn get_orbit_gizmo_projection(
    state: State<'_, Arc<ViewerState>>,
) -> Result<Vec<OrbitGizmoProjectionItem>, String> {
    let cam = state.camera.lock();
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
    let cam = state.camera.lock();
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
    if *state.fly_mode.lock() || *state.walk_mode.lock() {
        return Ok(());
    }
    let (min, max, _) = scene_bounds_min_max_grid(state.inner());
    let (w, h) = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Ok(());
        };
        viewer.viewport_size()
    };
    let mut cam = state.camera.lock();
    cam.fit_to_aabb_preserving_view(min, max, w as f32, h as f32);
    drop(cam);
    wake_viewport_loop(&app);
    Ok(())
}

#[tauri::command]
fn camera_reset_view(app: AppHandle, state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
    if *state.fly_mode.lock() || *state.walk_mode.lock() {
        return Ok(());
    }
    let (min, max, grid) = scene_bounds_min_max_grid(state.inner());
    let mut cam = state.camera.lock();
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
    if *state.fly_mode.lock() || *state.walk_mode.lock() {
        return Ok(());
    }
    let mut cam = state.camera.lock();
    cam.orbit_gizmo_drag(args.dx, args.dy, args.theta_only);
    drop(cam);
    wake_viewport_loop(&app);
    Ok(())
}

#[tauri::command]
fn camera_snap_orbit_axis(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    axis: u8,
) -> Result<(), String> {
    if *state.fly_mode.lock() || *state.walk_mode.lock() {
        return Ok(());
    }
    let mut cam = state.camera.lock();
    cam.snap_to_axis(axis);
    drop(cam);
    wake_viewport_loop(&app);
    Ok(())
}

#[tauri::command]
fn camera_zoom_step(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    inward: bool,
) -> Result<(), String> {
    if *state.fly_mode.lock() || *state.walk_mode.lock() {
        return Ok(());
    }
    let mut cam = state.camera.lock();
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

pub(crate) fn emit_load_progress<R: Runtime>(
    app: &AppHandle<R>,
    fraction: f32,
    phase: impl Into<String>,
) {
    let _ = app.emit(
        "voxelle-load-progress",
        LoadProgressPayload {
            fraction: fraction.clamp(0.0, 1.0),
            phase: phase.into(),
        },
    );
}

/// Status bar progress for save, heavy mesh refresh, undo/redo (webview `voxelle-work-progress`).
pub(crate) fn emit_work_progress<R: Runtime>(
    app: &AppHandle<R>,
    fraction: f32,
    phase: impl Into<String>,
) {
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

/// Full greedy rebuild CPU prep runs off-thread above these thresholds (viewer lock released during prep).
const OFF_THREAD_GREEDY_MESH_MIN_VOXELS: usize = 2_000;
const OFF_THREAD_GREEDY_MESH_MIN_DELTAS: usize = 500;
const OFF_THREAD_SMOOTH_MESH_MIN_VOXELS: usize = 4_000;
/// Incremental remesh releases the viewer mutex between batches so the render loop can present.
const INCREMENTAL_REMESH_KEY_BATCH: usize = 8;

fn merge_remesh_opaque_perf(acc: &mut render::RemeshOpaquePerf, p: &render::RemeshOpaquePerf) {
    acc.buckets_ms += p.buckets_ms;
    acc.greedy_ms += p.greedy_ms;
    acc.greedy_gpu_ms += p.greedy_gpu_ms;
    acc.greedy_cpu_ms += p.greedy_cpu_ms;
    acc.chunk_buffers_ms += p.chunk_buffers_ms;
    acc.full_chunked_rebuild_ms += p.full_chunked_rebuild_ms;
}

fn off_thread_prepare_greedy_rebuild<R: Runtime>(
    app: &AppHandle<R>,
    grid_size: i32,
    voxels: Vec<voxelle::Voxel>,
    objects: Vec<voxelle::SceneObject>,
) -> Result<PreparedGreedyRebuild, String> {
    use std::sync::atomic::AtomicU32;
    let app_pb = app.clone();
    std::thread::Builder::new()
        .name("voxelle-edit-greedy-prep".into())
        .spawn(move || {
            let last_permille = AtomicU32::new(0);
            let chunk_progress = |frac: f32, done: u32, total: u32| {
                let permille = (frac * 1000.0).min(1000.0) as u32;
                let prev = last_permille.load(Ordering::Relaxed);
                if permille.saturating_sub(prev) >= 40 || done == total {
                    last_permille.store(permille, Ordering::Relaxed);
                    emit_work_progress(
                        &app_pb,
                        0.38 + 0.52 * frac,
                        format!("Building mesh chunks {done}/{total}…"),
                    );
                }
            };
            compute_greedy_rebuild_cpu(&voxels, &objects, grid_size, Some(&chunk_progress))
        })
        .map_err(|e| e.to_string())?
        .join()
        .map_err(|_| "greedy mesh prep thread panicked".to_string())?
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
    let brick =
        GpuVoxelBrick::from_voxels(voxels, LOAD_SCENE_BRICK_MAX_AXIS).unwrap_or(GpuVoxelBrick {
            origin: glam::IVec3::ZERO,
            dims: (0, 0, 0),
            cells: vec![0u32],
        });
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
        let mesh = {
            let on_bucket = |frac: f32, done: usize, total: usize| {
                let g = LOAD_P_MESH_START + frac * mesh_span;
                let pct = (frac * 100.0).min(100.0) as u32;
                emit(
                    g,
                    &format!("Building surface mesh… ({done}/{total} buckets, {pct}%)"),
                );
            };
            match mode {
                RenderingMode::MarchingCubes => {
                    smooth_mesh::build_marching_cubes_merged_with_progress(voxels, on_bucket)
                }
                RenderingMode::DualContour => {
                    smooth_mesh::build_dual_contour_merged_with_progress(voxels, on_bucket)
                }
                _ => unreachable!(),
            }
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
                emit(g, &format!("Building mesh chunks {done}/{total} ({pct}%)"));
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

/// Like [`prepare_load_scene_cpu`] but returns immediately with bounds + brick + chunk origin.
/// Chunk meshes are built later by [`stream_chunk_meshes_to_inbox`].
/// Returns `(PreparedLoadScene, Option<SpatialMeshCache>)` — the cache is `Some` when
/// streaming will follow; the caller keeps it for background meshing.
pub(crate) fn prepare_load_scene_cpu_streaming<R: Runtime>(
    grid_size: i32,
    voxels: &[voxelle::Voxel],
    objects: &[voxelle::SceneObject],
    mode: RenderingMode,
    app: Option<&AppHandle<R>>,
) -> Result<(PreparedLoadScene, Option<greedy_mesh::SpatialMeshCache>), String> {
    let emit = |frac: f32, phase: &str| {
        if let Some(a) = app {
            emit_load_progress(a, frac, phase);
        }
    };

    let t_prep = Instant::now();
    let nv = voxels.len();
    log::info!(
        target: "voxelle_load",
        "prepare_load_scene_cpu_streaming: start voxels={nv} grid_size={grid_size} mode={mode:?}"
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
    log::info!(target: "voxelle_load", "prepare_load_scene_cpu_streaming: bounds {:?}", t.elapsed());

    let t = Instant::now();
    let brick =
        GpuVoxelBrick::from_voxels(voxels, LOAD_SCENE_BRICK_MAX_AXIS).unwrap_or(GpuVoxelBrick {
            origin: glam::IVec3::ZERO,
            dims: (0, 0, 0),
            cells: vec![0u32],
        });
    emit(LOAD_P_BRICK, "Packing voxel brick…");
    log::info!(target: "voxelle_load", "prepare_load_scene_cpu_streaming: brick {:?}", t.elapsed());

    // Decide whether to stream or fall back to the synchronous path.
    let visible_voxels = voxelle::scene::visible_voxels_for_meshing(voxels, objects);
    let can_stream = !mode.uses_smooth_surface()
        && visible_voxels.len() >= greedy_mesh::CHUNKED_CPU_MESH_MIN_VOXELS
        && visible_voxels
            .iter()
            .map(|v| v.object_id)
            .collect::<std::collections::HashSet<_>>()
            .len()
            <= 1;

    if can_stream {
        // Build SpatialMeshCache (occupancy + buckets) — this is the fast part.
        emit(LOAD_P_MESH_START, "Indexing voxels…");
        let t = Instant::now();
        let cache = greedy_mesh::SpatialMeshCache::from_voxels(
            &visible_voxels,
            greedy_mesh::SPATIAL_CHUNK_SIZE,
        );
        log::info!(
            target: "voxelle_load",
            "prepare_load_scene_cpu_streaming: spatial cache {:?}",
            t.elapsed()
        );
        match cache {
            Some(c) => {
                let origin = c.origin;
                let chunk_origin = glam::IVec3::new(origin.0, origin.1, origin.2);
                log::info!(
                    target: "voxelle_load",
                    "prepare_load_scene_cpu_streaming: total {:?} — {} chunks deferred",
                    t_prep.elapsed(),
                    c.buckets.len()
                );
                let prepared = PreparedLoadScene {
                    bounds,
                    brick,
                    opaque: PreparedOpaqueUpload::ChunkedDeferred { chunk_origin },
                };
                Ok((prepared, Some(c)))
            }
            None => {
                let prepared = PreparedLoadScene {
                    bounds,
                    brick,
                    opaque: PreparedOpaqueUpload::Empty,
                };
                Ok((prepared, None))
            }
        }
    } else {
        // Non-chunked paths (smooth, small models) — fall back to synchronous.
        let mesh_span = LOAD_P_MESH_END - LOAD_P_MESH_START;
        let opaque = if voxels.is_empty() {
            PreparedOpaqueUpload::Empty
        } else if mode.uses_smooth_surface() {
            emit(LOAD_P_MESH_START, "Building surface mesh…");
            let t = Instant::now();
            let mesh = {
                let on_bucket = |frac: f32, done: usize, total: usize| {
                    let g = LOAD_P_MESH_START + frac * mesh_span;
                    let pct = (frac * 100.0).min(100.0) as u32;
                    emit(
                        g,
                        &format!("Building surface mesh… ({done}/{total} buckets, {pct}%)"),
                    );
                };
                match mode {
                    RenderingMode::MarchingCubes => {
                        smooth_mesh::build_marching_cubes_merged_with_progress(voxels, on_bucket)
                    }
                    RenderingMode::DualContour => {
                        smooth_mesh::build_dual_contour_merged_with_progress(voxels, on_bucket)
                    }
                    _ => unreachable!(),
                }
            };
            log::info!(target: "voxelle_load", "prepare_load_scene_cpu_streaming: smooth {:?}", t.elapsed());
            if mesh.indices.is_empty() {
                PreparedOpaqueUpload::Empty
            } else {
                PreparedOpaqueUpload::Single(mesh)
            }
        } else {
            emit(LOAD_P_MESH_START, "Building mesh…");
            let t = Instant::now();
            let (mesh, _) = greedy_mesh::build_greedy_mesh(voxels, objects);
            log::info!(target: "voxelle_load", "prepare_load_scene_cpu_streaming: greedy {:?}", t.elapsed());
            PreparedOpaqueUpload::Single(mesh)
        };

        log::info!(target: "voxelle_load", "prepare_load_scene_cpu_streaming: total {:?}", t_prep.elapsed());
        Ok((
            PreparedLoadScene {
                bounds,
                brick,
                opaque,
            },
            None,
        ))
    }
}

/// Build chunk meshes from a [`SpatialMeshCache`] and push each to `state.chunk_mesh_inbox`.
/// When done, deposits the cache into `state.deferred_spatial_cache`.
/// Respects `load_gen` — bails if a newer load has started.
fn stream_chunk_meshes_to_inbox(
    cache: greedy_mesh::SpatialMeshCache,
    state: &Arc<ViewerState>,
    load_gen: u64,
) {
    use rayon::prelude::*;

    let keys: Vec<greedy_mesh::ChunkKey> = cache.buckets.keys().copied().collect();
    let total = keys.len();
    log::info!(
        target: "voxelle_load",
        "stream_chunk_meshes_to_inbox: start {total} chunks"
    );
    let t = Instant::now();

    // Build meshes in parallel and push to inbox as they complete.
    keys.par_iter().for_each(|&key| {
        if is_load_stale(state, load_gen) {
            return;
        }
        let mesh = greedy_mesh::mesh_buffers_for_chunk_key(&cache.buckets, &cache.occupancy, key);
        if mesh.indices.is_empty() {
            return;
        }
        state.chunk_mesh_inbox.lock().push_back((key, mesh));
    });

    if is_load_stale(state, load_gen) {
        log::info!(target: "voxelle_load", "stream_chunk_meshes_to_inbox: cancelled (stale)");
        return;
    }

    // Hand the cache to the viewer (main thread will pick it up).
    *state.deferred_spatial_cache.lock() = Some(cache);
    log::info!(
        target: "voxelle_load",
        "stream_chunk_meshes_to_inbox: done {total} chunks {:?}",
        t.elapsed()
    );
}

/// Clears the loaded model, GPU meshes, and editing state. Must run on the main thread (GPU + AppKit undo).
fn unload_current_project<R: Runtime>(
    state: &Arc<ViewerState>,
    app: &AppHandle<R>,
) -> Result<(), String> {
    let mode = *state.rendering_mode.lock();
    let objects = voxelle::default_scene_objects();
    let prepared = prepare_load_scene_cpu::<R>(MAX_GRID_SIZE as i32, &[], &objects, mode, None)?;
    {
        let mut cf = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
        *cf = None;
        *vm = None;
    }
    state.active_project.store(false, Ordering::Release);
    let mut v = state.viewer.lock();
    let Some(viewer) = v.as_mut() else {
        return Err("viewer not ready".into());
    };
    viewer.upload_scene_data_from_brick(prepared.bounds, prepared.brick);
    viewer.upload_prepared_opaque(prepared.opaque);
    clear_preview_mesh_sync_cache(viewer, state.as_ref());
    viewer.clear_selection_overlay();
    *state.selection_overlay_cache_key.lock() = None;
    viewer.clear_grid_border_lines();
    *state.grid_overlay_cache_key.lock() = None;
    viewer.clear_collab_peer_lines();
    viewer.clear_ping_mesh();
    viewer.set_mood_params(&MoodParams::default());
    drop(v);

    *state.last_scene_bounds.lock() = Some(prepared.bounds);
    *state.voxel_edit_stats_cache.lock() = None;
    *state.last_edit_perf.lock() = None;
    state
        .mesh_refresh_generation
        .fetch_add(1, Ordering::Release);

    state.solo_undo.lock().clear();
    state.solo_redo.lock().clear();
    #[cfg(target_os = "macos")]
    macos_undo::clear_all(app);

    *state.selection_cells.lock() = AHashSet::default();
    *state.selection_stroke_before.lock() = None;
    *state.selection_stroke_accum.lock() = None;
    *state.selection_combine_mode.lock() = SelectionCombineMode::default();
    *state.stamp_clipboard.lock() = None;
    *state.stroke_buffer.lock() = Vec::new();
    *state.stroke_preview_union.lock() = AHashSet::default();
    *state.stroke_preview_last_args.lock() = None;
    state
        .stroke_preview_suppresses_hover
        .store(false, Ordering::Release);
    *state.sculpt_stroke_replay.lock() = Vec::new();
    *state.stroke_active.lock() = false;
    *state.ping_flash.lock() = None;
    *state.preview_cursor.lock() = None;

    state.squishy_session.lock().clear();

    log::info!(target: "voxelle_load", "unload_current_project: done");
    #[cfg(desktop)]
    selection_menu_sync_enabled_for_scene(app, false, false);
    Ok(())
}

fn run_unload_on_main_thread<R: Runtime>(
    state: &Arc<ViewerState>,
    app: &AppHandle<R>,
) -> Result<(), String> {
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let state_c = Arc::clone(state);
    let app_c = app.clone();
    app.run_on_main_thread(move || {
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
    let load_gen = next_load_generation(&state);
    let app_spawn_err = app.clone();
    match std::thread::Builder::new()
        .name("voxelle-new-project".into())
        .spawn(move || {
            if let Err(e) = run_unload_on_main_thread(&state, &app) {
                let _ = app.emit("voxelle-load-error", e);
                return;
            }
            if is_load_stale(&state, load_gen) {
                log::info!(target: "voxelle_load", "new project cancelled (stale after unload)");
                return;
            }
            state
                .start_screen_logo_transparent
                .store(false, Ordering::Release);
            emit_load_progress(&app, 0.05, "Starting…");

            let mesh_result: Result<(), String> =
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
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

                        let mode = *state.rendering_mode.lock();
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
                                let res = apply_mesh_and_camera(
                                    &state_c, &app_mesh, file_c, prepared, false,
                                );
                                let _ = done_tx.send(res);
                            });
                            return match done_rx.recv() {
                                Ok(Ok(())) => Ok(()),
                                Ok(Err(e)) => Err(e),
                                Err(_) => Err("main thread disconnected".into()),
                            };
                        }

                        run_v3_mesh_on_main(&state, &app, file, prepared, false, load_gen)?;
                        Ok(())
                    })()
                })) {
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
                    emit_voxelle_loaded(&app, label.clone(), &state, false);
                    try_initial_autosave_after_new_project(&app, &state, &label);
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
    /// Main thread uploads brick + camera; background thread continues building chunk meshes.
    Streaming {
        file: voxelle::VoxelleFile,
        prepared: PreparedLoadScene,
        cache: greedy_mesh::SpatialMeshCache,
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
    load_gen: u64,
) -> Result<(), String> {
    if is_load_stale(state, load_gen) {
        return Err("load cancelled".into());
    }
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
        if is_load_stale(&state_c, load_gen) {
            let _ = done_tx.send(Err("load cancelled".into()));
            return;
        }
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

/// Bump the load generation counter and return the new value.
/// Every load entry point should call this so that older in-flight loads can detect they are stale.
fn next_load_generation(state: &ViewerState) -> u64 {
    state.load_generation.fetch_add(1, Ordering::SeqCst) + 1
}

/// Returns true when a newer load has started since `gen` was issued.
fn is_load_stale(state: &ViewerState, gen: u64) -> bool {
    state.load_generation.load(Ordering::SeqCst) != gen
}

fn spawn_decode_and_mesh(state: Arc<ViewerState>, app: AppHandle, path: PathBuf) {
    let label = path.to_string_lossy().to_string();
    spawn_decode_and_mesh_with_label(state, app, path, label, false);
}

fn spawn_decode_and_mesh_from_bytes(
    state: Arc<ViewerState>,
    app: AppHandle,
    bytes: &'static [u8],
    file_label: String,
    start_screen_logo: bool,
) {
    let owned = bytes.to_vec();
    spawn_decode_and_mesh_inner(state, app, owned, file_label, start_screen_logo);
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
        .name("voxelle-load-read".into())
        .spawn(move || {
            let t = Instant::now();
            match std::fs::read(&read_from) {
                Ok(bytes) => {
                    log::info!(
                        target: "voxelle_load",
                        "load file: read {} bytes from disk {:?}",
                        bytes.len(),
                        t.elapsed()
                    );
                    spawn_decode_and_mesh_inner(state, app, bytes, file_label, start_screen_logo);
                }
                Err(e) => {
                    let _ = app.emit("voxelle-load-error", e.to_string());
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

fn spawn_decode_and_mesh_inner(
    state: Arc<ViewerState>,
    app: AppHandle,
    preloaded_bytes: Vec<u8>,
    file_label: String,
    start_screen_logo: bool,
) {
    let load_gen = next_load_generation(&state);
    let app_spawn_err = app.clone();
    match std::thread::Builder::new()
        .name("voxelle-load".into())
        .spawn(move || {
            if let Err(e) = run_unload_on_main_thread(&state, &app) {
                let _ = app.emit("voxelle-load-error", e);
                return;
            }
            if is_load_stale(&state, load_gen) {
                log::info!(target: "voxelle_load", "load cancelled (stale after unload)");
                return;
            }
            let label = file_label;
            emit_load_progress(&app, 0.05, "Starting…");

            let mesh_result: Result<DecodeMeshOutcome, String> =
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    (|| -> Result<DecodeMeshOutcome, String> {
                        let bytes = preloaded_bytes;
                        if is_load_stale(&state, load_gen) {
                            return Err("load cancelled".into());
                        }
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
                        if is_load_stale(&state, load_gen) {
                            return Err("load cancelled".into());
                        }
                        emit_load_progress(&app, 0.18, "Preparing scene…");
                        let mode = *state.rendering_mode.lock();
                        let (prepared, streaming_cache) = prepare_load_scene_cpu_streaming(
                            file.grid_size,
                            &file.voxels,
                            &file.objects,
                            mode,
                            Some(&app),
                        )?;

                        if is_load_stale(&state, load_gen) {
                            return Err("load cancelled".into());
                        }

                        if file.version == 3 && !file.voxels.is_empty() {
                            run_v3_mesh_on_main(
                                &state,
                                &app,
                                file,
                                prepared,
                                start_screen_logo,
                                load_gen,
                            )?;
                            return Ok(DecodeMeshOutcome::Done);
                        }

                        match streaming_cache {
                            Some(cache) => Ok(DecodeMeshOutcome::Streaming {
                                file,
                                prepared,
                                cache,
                            }),
                            None => Ok(DecodeMeshOutcome::ApplyOnce { file, prepared }),
                        }
                    })()
                })) {
                    Ok(inner) => inner,
                    Err(payload) => Err(load_thread_panic_message(payload)),
                };

            // Final stale check before applying to the scene.
            if is_load_stale(&state, load_gen) {
                log::info!(target: "voxelle_load", "load cancelled (stale before apply)");
                return;
            }

            // Extract the streaming cache (if any) so it stays on the background thread
            // while the main thread applies brick + camera.
            let (mesh_result, streaming_cache) = match mesh_result {
                Ok(DecodeMeshOutcome::Streaming {
                    file,
                    prepared,
                    cache,
                }) => (
                    Ok(DecodeMeshOutcome::ApplyOnce { file, prepared }),
                    Some(cache),
                ),
                other => (other, None),
            };

            let (done_tx, done_rx) = std::sync::mpsc::channel();
            let state_c = Arc::clone(&state);
            let app_emit = app.clone();
            if let Err(e) = app.run_on_main_thread(move || {
                // Check once more on the main thread before touching the scene.
                if is_load_stale(&state_c, load_gen) {
                    log::info!(target: "voxelle_load", "load cancelled (stale on main thread)");
                    let _ = done_tx.send(Err("load cancelled".into()));
                    return;
                }
                let res: Result<(), String> = match mesh_result {
                    Ok(DecodeMeshOutcome::ApplyOnce { file, prepared }) => {
                        let t = Instant::now();
                        let r = apply_mesh_and_camera(
                            &state_c,
                            &app_emit,
                            file,
                            prepared,
                            start_screen_logo,
                        );
                        log::info!(
                            target: "voxelle_load",
                            "load file: ApplyOnce apply_mesh_and_camera {:?}",
                            t.elapsed()
                        );
                        r
                    }
                    Ok(DecodeMeshOutcome::Streaming { .. }) => unreachable!("extracted above"),
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
                    // Kick off background chunk meshing (overlaps with first rendered frames).
                    if let Some(cache) = streaming_cache {
                        stream_chunk_meshes_to_inbox(cache, &state, load_gen);
                    }
                    if start_screen_logo {
                        emit_voxelle_loaded(&app, String::new(), &state, true);
                    } else {
                        if label.ends_with(".voxelle") {
                            persist_last_document_path(&app, &label);
                            persist_recent_file(&app, &label);
                            #[cfg(desktop)]
                            if let Some(rm) = app.try_state::<RecentMenuState>() {
                                rebuild_recent_submenu(&app, &rm.submenu);
                            }
                        }
                        emit_voxelle_loaded(&app, label, &state, false);
                    }
                }
                Ok(Err(e)) => {
                    // Don't emit user-facing errors for intentional cancellation.
                    if e != "load cancelled" {
                        let _ = app.emit("voxelle-load-error", e);
                    }
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
    let mood = file.mood.clone();
    let lighting = file.lighting.clone().unwrap_or_default();
    {
        let mut cf = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
        *cf = Some(file);
        *vm = Some(voxel_map);
    }
    let mut v = state.viewer.lock();
    let Some(viewer) = v.as_mut() else {
        return Err("viewer not ready".into());
    };
    viewer.upload_scene_data_from_brick(bounds, brick);
    viewer.upload_prepared_opaque(opaque);
    clear_preview_mesh_sync_cache(viewer, state.as_ref());
    if let Some(m) = mood {
        viewer.set_mood_params(&mood_settings_to_params(&m));
    }
    viewer.apply_lighting_settings(&lighting);

    let mut cam = state.camera.lock();
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
    drop(v);
    *state.last_scene_bounds.lock() = Some(bounds);
    *state.voxel_edit_stats_cache.lock() = voxel_edit_stats_cache;
    state.solo_undo.lock().clear();
    state.solo_redo.lock().clear();
    #[cfg(target_os = "macos")]
    macos_undo::clear_all(app);
    collab::broadcast_snapshot_to_guests(state);
    state.active_project.store(true, Ordering::Release);
    emit_load_progress(app, 0.97, "Finishing…");
    emit_load_progress(app, 1.0, "");
    #[cfg(desktop)]
    {
        let (has_voxels, has_selection) = scene_menu_flags(state.as_ref());
        selection_menu_sync_enabled_for_scene(app, has_voxels, has_selection);
    }
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
    state
        .start_screen_logo_transparent
        .store(start_screen_logo, Ordering::Release);
    let (mood, lighting) = match state.current_file.lock().as_ref() {
        Some(f) => (f.mood.clone(), f.lighting.clone()),
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
        let guard = state.last_scene_bounds.lock();
        if let Some(prev) = guard.as_ref() {
            match delta {
                voxel_edit::VoxelEditDelta::Added(v) => {
                    return Ok(greedy_mesh::mesh_bounds_expand_with_voxel(prev, v));
                }
                voxel_edit::VoxelEditDelta::Removed { voxel } => {
                    if greedy_mesh::mesh_bounds_remove_is_strict_interior(
                        prev, voxel.x, voxel.y, voxel.z,
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
    // Compute bounds with a short `current_file` lock only. Overlay prep locks `current_file`
    // only while the viewer mutex is free (see main-loop `prepare_*_overlay`).
    let bounds = {
        let fg = state.current_file.lock();
        let Some(file) = fg.as_ref() else {
            return Err("no model loaded".into());
        };
        scene_bounds_for_edits(state.as_ref(), file, deltas)?
    };

    let prepare_ms = t_prep_start.elapsed().as_secs_f64() * 1000.0;

    let t_lock_start = Instant::now();
    let mut v = state.viewer.lock();
    let viewer_lock_wait_ms = t_lock_start.elapsed().as_secs_f64() * 1000.0;

    let mut fg = state.current_file.lock();
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };

    let rm = *state.rendering_mode.lock();
    let show_work = {
        let Some(viewer) = v.as_mut() else {
            return Err("viewer not ready".into());
        };
        work_progress_for_voxel_refresh(viewer, file, rm) || deltas.len() >= 1_000
    };
    let mut wp = WorkProgressGuard::new(app);
    if show_work {
        wp.arm();
        emit_work_progress(app, 0.12, reason.label());
    }

    let t_brick = Instant::now();
    {
        let Some(viewer) = v.as_mut() else {
            return Err("viewer not ready".into());
        };
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
    }
    let brick_ms = t_brick.elapsed().as_secs_f64() * 1000.0;

    // Release locks so the render loop can present the brick update before the
    // potentially slower mesh rebuild. This gives instant visual feedback.
    drop(fg);
    drop(v);
    std::thread::yield_now();
    v = state.viewer.lock();
    fg = state.current_file.lock();
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };

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
        {
            let Some(viewer) = v.as_mut() else {
                return Err("viewer not ready".into());
            };
            viewer.upload_mesh(&mut greedy_mesh::MeshBuffers::default());
            viewer.last_mesh_route = "clear".to_string();
        }
        *state.voxel_edit_stats_cache.lock() = None;
    } else if rm.uses_smooth_surface() {
        let nv = file.voxels.len();
        if nv >= OFF_THREAD_SMOOTH_MESH_MIN_VOXELS {
            let voxels = file.voxels.clone();
            let rm_copy = rm;
            let token = state.mesh_refresh_generation.fetch_add(1, Ordering::SeqCst) + 1;
            let app_thread = app.clone();
            let show_work_thread = show_work;
            drop(fg);
            drop(v);
            let mesh_from_thread: greedy_mesh::MeshBuffers = std::thread::Builder::new()
                .name("voxelle-smooth-mesh".into())
                .spawn(move || {
                    use std::sync::atomic::{AtomicU32, Ordering};
                    let last_permille = AtomicU32::new(0);
                    let on_bucket = move |frac: f32, done: usize, total: usize| {
                        if !show_work_thread {
                            return;
                        }
                        let permille = (frac * 1000.0).min(1000.0) as u32;
                        let prev = last_permille.load(Ordering::Relaxed);
                        if permille.saturating_sub(prev) >= 40 || done == total {
                            last_permille.store(permille, Ordering::Relaxed);
                            let pct = (frac * 100.0).min(100.0) as u32;
                            emit_work_progress(
                                &app_thread,
                                0.38 + 0.52 * frac,
                                format!("Building surface mesh… ({done}/{total} buckets, {pct}%)"),
                            );
                        }
                    };
                    match rm_copy {
                        RenderingMode::MarchingCubes => {
                            smooth_mesh::build_marching_cubes_merged_with_progress(
                                &voxels, on_bucket,
                            )
                        }
                        RenderingMode::DualContour => {
                            smooth_mesh::build_dual_contour_merged_with_progress(&voxels, on_bucket)
                        }
                        _ => greedy_mesh::MeshBuffers::default(),
                    }
                })
                .map_err(|e| e.to_string())?
                .join()
                .map_err(|_| "smooth mesh thread panicked".to_string())?;
            v = state.viewer.lock();
            let Some(viewer) = v.as_mut() else {
                return Err("viewer not ready".into());
            };
            fg = state.current_file.lock();
            let Some(file) = fg.as_ref() else {
                return Err("no model loaded".into());
            };
            let mut mesh = if state.mesh_refresh_generation.load(Ordering::SeqCst) == token {
                mesh_from_thread
            } else {
                match rm {
                    RenderingMode::MarchingCubes => {
                        smooth_mesh::build_marching_cubes_merged(&file.voxels)
                    }
                    RenderingMode::DualContour => {
                        smooth_mesh::build_dual_contour_merged(&file.voxels)
                    }
                    _ => unreachable!(),
                }
            };
            viewer.upload_mesh(&mut mesh);
            viewer.last_mesh_route = match rm {
                RenderingMode::MarchingCubes => "marching_cubes".to_string(),
                RenderingMode::DualContour => "dual_contour".to_string(),
                _ => unreachable!(),
            };
            *state.voxel_edit_stats_cache.lock() =
                Some(voxel_aabb_min_and_single_object_one_pass(&file.voxels));
        } else {
            let Some(viewer) = v.as_mut() else {
                return Err("viewer not ready".into());
            };
            let on_bucket = |frac: f32, done: usize, total: usize| {
                if !show_work {
                    return;
                }
                let pct = (frac * 100.0).min(100.0) as u32;
                emit_work_progress(
                    app,
                    0.38 + 0.52 * frac,
                    format!("Building surface mesh… ({done}/{total} buckets, {pct}%)"),
                );
            };
            let mut mesh = match rm {
                RenderingMode::MarchingCubes => {
                    smooth_mesh::build_marching_cubes_merged_with_progress(&file.voxels, on_bucket)
                }
                RenderingMode::DualContour => {
                    smooth_mesh::build_dual_contour_merged_with_progress(&file.voxels, on_bucket)
                }
                _ => unreachable!(),
            };
            viewer.upload_mesh(&mut mesh);
            viewer.last_mesh_route = match rm {
                RenderingMode::MarchingCubes => "marching_cubes".to_string(),
                RenderingMode::DualContour => "dual_contour".to_string(),
                _ => unreachable!(),
            };
            *state.voxel_edit_stats_cache.lock() =
                Some(voxel_aabb_min_and_single_object_one_pass(&file.voxels));
        }
    } else {
        let cached_stats = state.voxel_edit_stats_cache.lock().clone();
        let voxel_stats = resolve_voxel_edit_stats_batch(&file.voxels, deltas, cached_stats);
        let origin_new = voxel_stats.aabb_min;
        let single_object = voxel_stats.common_object_id.is_some();
        let origin_iv = glam::IVec3::new(origin_new.0, origin_new.1, origin_new.2);
        let use_incremental = {
            let Some(viewer) = v.as_mut() else {
                return Err("viewer not ready".into());
            };
            viewer.opaque_chunked
                && single_object
                && file.voxels.len() >= greedy_mesh::CHUNKED_CPU_MESH_MIN_VOXELS
                && viewer.chunk_grid_origin == origin_iv
        };

        if use_incremental {
            let dirty = {
                let Some(viewer) = v.as_mut() else {
                    return Err("viewer not ready".into());
                };
                let t_cache = Instant::now();
                for d in deltas {
                    viewer.apply_spatial_cache_edit(d);
                }
                mesh_voxel_map_ms = t_cache.elapsed().as_secs_f64() * 1000.0;
                union_dirty_chunk_keys_for_deltas(
                    deltas,
                    origin_new,
                    greedy_mesh::SPATIAL_CHUNK_SIZE,
                )
            };

            if dirty.len() <= INCREMENTAL_REMESH_KEY_BATCH {
                let Some(viewer) = v.as_mut() else {
                    return Err("viewer not ready".into());
                };
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
                drop(v);
                let mut rperf_acc = render::RemeshOpaquePerf::default();
                let mut ok_acc = true;
                for (bi, batch) in dirty.chunks(INCREMENTAL_REMESH_KEY_BATCH).enumerate() {
                    if bi > 0 {
                        std::thread::yield_now();
                    }
                    let mut v2 = state.viewer.lock();
                    let Some(viewer2) = v2.as_mut() else {
                        return Err("viewer not ready".into());
                    };
                    let (ok, rperf) = viewer2.remesh_opaque_chunks(
                        batch,
                        &file.voxels,
                        if show_work { Some(app) } else { None },
                    );
                    merge_remesh_opaque_perf(&mut rperf_acc, &rperf);
                    ok_acc = ok_acc && ok;
                    if ok {
                        viewer2.last_mesh_route = "cpu_chunked_incremental".to_string();
                    }
                    drop(v2);
                }
                mesh_buckets_ms = rperf_acc.buckets_ms;
                mesh_greedy_ms = rperf_acc.greedy_ms;
                mesh_greedy_gpu_ms = rperf_acc.greedy_gpu_ms;
                mesh_greedy_cpu_ms = rperf_acc.greedy_cpu_ms;
                mesh_chunk_buffers_ms = rperf_acc.chunk_buffers_ms;
                mesh_full_chunked_rebuild_ms = rperf_acc.full_chunked_rebuild_ms;
                v = state.viewer.lock();
                let Some(viewer) = v.as_mut() else {
                    return Err("viewer not ready".into());
                };
                if ok_acc {
                    viewer.last_mesh_route = "cpu_chunked_incremental".to_string();
                }
            }
        } else {
            let nv = file.voxels.len();
            let off_thread = nv >= OFF_THREAD_GREEDY_MESH_MIN_VOXELS
                || deltas.len() >= OFF_THREAD_GREEDY_MESH_MIN_DELTAS;
            if off_thread {
                let grid_size = file.grid_size;
                let voxels = file.voxels.clone();
                let objects = file.objects.clone();
                let token = state.mesh_refresh_generation.fetch_add(1, Ordering::SeqCst) + 1;
                drop(fg);
                drop(v);
                let prepared_result =
                    off_thread_prepare_greedy_rebuild(app, grid_size, voxels, objects);
                v = state.viewer.lock();
                let Some(viewer) = v.as_mut() else {
                    return Err("viewer not ready".into());
                };
                fg = state.current_file.lock();
                let Some(file) = fg.as_ref() else {
                    return Err("no model loaded".into());
                };
                let t_pipe = Instant::now();
                match prepared_result {
                    Ok(prepared) => {
                        if state.mesh_refresh_generation.load(Ordering::SeqCst) != token {
                            let _ = viewer.rebuild_mesh_gpu_greedy(
                                &file.voxels,
                                &file.objects,
                                file.grid_size,
                            );
                        } else {
                            let _ = viewer.apply_prepared_greedy_rebuild(prepared);
                        }
                    }
                    Err(e) => {
                        log::warn!(target: "voxelle", "off-thread greedy prep failed ({e}); rebuilding inline");
                        let _ = viewer.rebuild_mesh_gpu_greedy(
                            &file.voxels,
                            &file.objects,
                            file.grid_size,
                        );
                    }
                }
                mesh_pipeline_ms = t_pipe.elapsed().as_secs_f64() * 1000.0;
            } else {
                let Some(viewer) = v.as_mut() else {
                    return Err("viewer not ready".into());
                };
                let t_pipe = Instant::now();
                let _ = viewer.rebuild_mesh_gpu_greedy(&file.voxels, &file.objects, file.grid_size);
                mesh_pipeline_ms = t_pipe.elapsed().as_secs_f64() * 1000.0;
            }
        }
        *state.voxel_edit_stats_cache.lock() = Some(voxel_stats);
    }
    // Release `current_file` before other helpers that lock it. Release `viewer` before any
    // follow-up that might contend with the render/preview paths (menu sync is lock-free; keeping
    // `drop(v)` avoids blocking other work on the viewer mutex).
    drop(fg);
    let mesh_ms = t_mesh.elapsed().as_secs_f64() * 1000.0;

    let (preview_clear_ms, mesh_route) = {
        let Some(viewer) = v.as_mut() else {
            return Err("viewer not ready".into());
        };
        let t_preview_clear = Instant::now();
        clear_preview_mesh_sync_cache(viewer, state);
        let preview_clear_ms = t_preview_clear.elapsed().as_secs_f64() * 1000.0;
        (preview_clear_ms, viewer.last_mesh_route.clone())
    };
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    *state.last_edit_perf.lock() = Some(EditPerfBreakdown {
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

    *state.last_scene_bounds.lock() = Some(bounds);

    drop(v);

    #[cfg(desktop)]
    {
        let (has_voxels, has_selection) = scene_menu_flags(state.as_ref());
        selection_menu_sync_enabled_for_scene(app, has_voxels, has_selection);
    }
    Ok(())
}

/// Rebuild opaque mesh from current voxels + [`RenderingMode`] (after switching view mode in the UI).
pub(crate) fn refresh_opaque_mesh<R: Runtime>(
    state: &Arc<ViewerState>,
    app: Option<&AppHandle<R>>,
) -> Result<(), String> {
    let rm = *state.rendering_mode.lock();
    let mut v = state.viewer.lock();
    let Some(viewer) = v.as_mut() else {
        return Err("viewer not ready".into());
    };
    let fg = state.current_file.lock();
    let Some(file) = fg.as_ref() else {
        drop(fg);
        drop(v);
        #[cfg(desktop)]
        if let Some(a) = app {
            let (has_voxels, has_selection) = scene_menu_flags(state.as_ref());
            selection_menu_sync_enabled_for_scene(a, has_voxels, has_selection);
        }
        return Ok(());
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
        viewer.upload_mesh(&mut greedy_mesh::MeshBuffers::default());
        viewer.last_mesh_route = "clear".to_string();
        *state.voxel_edit_stats_cache.lock() = None;
        drop(wp);
        drop(fg);
        drop(v);
        #[cfg(desktop)]
        if let Some(a) = app {
            let (has_voxels, has_selection) = scene_menu_flags(state.as_ref());
            selection_menu_sync_enabled_for_scene(a, has_voxels, has_selection);
        }
        return Ok(());
    }
    if rm.uses_smooth_surface() {
        let mut mesh = match rm {
            RenderingMode::MarchingCubes => smooth_mesh::build_marching_cubes_merged(&file.voxels),
            RenderingMode::DualContour => smooth_mesh::build_dual_contour_merged(&file.voxels),
            _ => unreachable!(),
        };
        viewer.upload_mesh(&mut mesh);
        viewer.last_mesh_route = match rm {
            RenderingMode::MarchingCubes => "marching_cubes".to_string(),
            RenderingMode::DualContour => "dual_contour".to_string(),
            _ => unreachable!(),
        };
    } else {
        match viewer.rebuild_mesh_gpu_greedy(&file.voxels, &file.objects, file.grid_size) {
            Ok(b) => {
                viewer.set_scene_bounds(b);
                *state.last_scene_bounds.lock() = Some(b);
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
                *state.last_scene_bounds.lock() = Some(b);
            }
        }
    }
    *state.voxel_edit_stats_cache.lock() =
        Some(voxel_aabb_min_and_single_object_one_pass(&file.voxels));
    drop(wp);
    drop(fg);
    drop(v);
    #[cfg(desktop)]
    if let Some(a) = app {
        let (has_voxels, has_selection) = scene_menu_flags(state.as_ref());
        selection_menu_sync_enabled_for_scene(a, has_voxels, has_selection);
    }
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
    let file = (*state.current_file.lock()).clone();
    let Some(file) = file else {
        return;
    };
    let rm = *state.rendering_mode.lock();
    if std::thread::Builder::new()
        .name("voxelle-opaque-refresh".into())
        .spawn(move || {
            let work: Result<OpaqueRefreshWork, String> = if file.voxels.is_empty() {
                Ok(OpaqueRefreshWork::Greedy(PreparedGreedyRebuild::NoVoxels))
            } else if rm.uses_smooth_surface() {
                use std::sync::atomic::{AtomicU32, Ordering};
                let last_permille = AtomicU32::new(0);
                let app_pb = app.clone();
                emit_work_progress(&app_pb, 0.08, "Rebuilding mesh…");
                let on_bucket = move |frac: f32, done: usize, total: usize| {
                    let permille = (frac * 1000.0).min(1000.0) as u32;
                    let prev = last_permille.load(Ordering::Relaxed);
                    if permille.saturating_sub(prev) >= 40 || done == total {
                        last_permille.store(permille, Ordering::Relaxed);
                        let pct = (frac * 100.0).min(100.0) as u32;
                        emit_work_progress(
                            &app_pb,
                            0.1 + 0.85 * frac,
                            format!("Building surface mesh… ({done}/{total} buckets, {pct}%)"),
                        );
                    }
                };
                let is_stale = {
                    let state_check = Arc::clone(&state_c);
                    move || state_check.mesh_refresh_generation.load(Ordering::Relaxed) != token
                };
                let mesh = match rm {
                    RenderingMode::MarchingCubes => {
                        smooth_mesh::build_marching_cubes_merged_cancellable(
                            &file.voxels,
                            on_bucket,
                            is_stale,
                        )
                    }
                    RenderingMode::DualContour => {
                        smooth_mesh::build_dual_contour_merged_cancellable(
                            &file.voxels,
                            on_bucket,
                            is_stale,
                        )
                    }
                    _ => {
                        log::warn!(target: "voxelle", "opaque refresh: unexpected smooth mode");
                        return;
                    }
                };
                let bounds = greedy_mesh::mesh_bounds_from_voxels(&file.voxels)
                    .unwrap_or_else(|| greedy_mesh::mesh_bounds_for_cube_side(file.grid_size));
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
                let mut vl = Some(state_c.viewer.lock());
                let Some(viewer) = vl.as_mut().and_then(|v| v.as_mut()) else {
                    return;
                };
                match work {
                    OpaqueRefreshWork::Smooth {
                        mut mesh,
                        bounds,
                        route,
                    } => {
                        viewer.upload_mesh(&mut mesh);
                        viewer.set_scene_bounds(bounds);
                        viewer.last_mesh_route = route;
                        *state_c.last_scene_bounds.lock() = Some(bounds);
                    }
                    OpaqueRefreshWork::Greedy(prepared) => {
                        match viewer.apply_prepared_greedy_rebuild(prepared) {
                            Ok(b) => {
                                viewer.set_scene_bounds(b);
                                *state_c.last_scene_bounds.lock() = Some(b);
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
                                        greedy_mesh::mesh_bounds_for_cube_side(
                                            file_snapshot.grid_size,
                                        )
                                    })
                                };
                                viewer.set_scene_bounds(b);
                                *state_c.last_scene_bounds.lock() = Some(b);
                            }
                        }
                    }
                }
                if file_snapshot.voxels.is_empty() {
                    *state_c.voxel_edit_stats_cache.lock() = None;
                } else {
                    *state_c.voxel_edit_stats_cache.lock() = Some(
                        voxel_aabb_min_and_single_object_one_pass(&file_snapshot.voxels),
                    );
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
    let fg = state.current_file.lock();
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
    let mut fg = state.current_file.lock();
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
        let mut fg = state.current_file.lock();
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
        let mut fg = state.current_file.lock();
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
    *state.rendering_mode.lock() = mode;
    // Ray mode drives the GPU path tracer directly.
    if let Some(viewer) = state.viewer.lock().as_mut() {
        viewer.set_raytrace_mode(matches!(mode, RenderingMode::Ray));
    }
    if mode.uses_smooth_surface() {
        // DC/MC mesh build can take many seconds; run it on a side thread so the
        // main thread stays responsive.  `schedule_opaque_mesh_refresh` handles
        // background compute + main-thread GPU upload with the stale-token guard.
        schedule_opaque_mesh_refresh(state, app);
        return Ok(());
    }
    refresh_opaque_mesh(state, Some(app))
}

fn apply_orthographic(state: &Arc<ViewerState>, orthographic: bool) -> Result<(), String> {
    {
        let mut cam = state.camera.lock();
        cam.perspective = !orthographic;
        if orthographic {
            let g = state.last_scene_bounds.lock();
            if let Some(b) = g.as_ref() {
                let r = b.radius().max(1.0);
                cam.ortho_half_height = r * 1.1;
            }
        }
    }
    {
        let mut fg = state.current_file.lock();
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
    Ok(*state.rendering_mode.lock())
}

#[tauri::command]
fn set_rendering_mode(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    mode: RenderingMode,
) -> Result<(), String> {
    apply_rendering_mode(state.inner(), &app, mode)?;
    wake_viewport_loop(&app);
    #[cfg(desktop)]
    if let Some(sel) = app.try_state::<SelectionMenuState>() {
        let _ = sel
            .render_greedy
            .set_checked(matches!(mode, RenderingMode::Greedy));
        let _ = sel
            .render_marching
            .set_checked(matches!(mode, RenderingMode::MarchingCubes));
        let _ = sel
            .render_dual
            .set_checked(matches!(mode, RenderingMode::DualContour));
        let _ = sel
            .render_ray
            .set_checked(matches!(mode, RenderingMode::Ray));
    }
    Ok(())
}

#[tauri::command]
fn set_raytrace_mode(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    enabled: bool,
) -> Result<(), String> {
    let mut v = state.viewer.lock();
    let viewer = v.as_mut().ok_or("viewer not ready")?;
    viewer.set_raytrace_mode(enabled);
    drop(v);
    wake_viewport_loop(&app);
    Ok(())
}

#[tauri::command]
fn benchmark_raytrace(
    state: State<'_, Arc<ViewerState>>,
    frame_count: Option<u32>,
) -> Result<crate::render::RaytraceBenchmarkResult, String> {
    let mut v = state.viewer.lock();
    let viewer = v.as_mut().ok_or("viewer not ready")?;
    Ok(viewer.run_raytrace_benchmark(frame_count.unwrap_or(50)))
}

#[tauri::command]
fn get_orthographic(state: State<'_, Arc<ViewerState>>) -> Result<bool, String> {
    Ok(!state.camera.lock().perspective)
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

#[tauri::command]
fn get_show_grid_borders(state: State<'_, Arc<ViewerState>>) -> Result<bool, String> {
    Ok(state.show_grid_borders.load(Ordering::Relaxed))
}

/// Keeps **View → Show borders** in sync with webview (e.g. after restoring preferences).
#[tauri::command]
fn view_menu_sync_show_borders(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    show: bool,
) -> Result<(), String> {
    state.show_grid_borders.store(show, Ordering::Relaxed);
    #[cfg(desktop)]
    {
        if let Some(menu) = app.try_state::<SelectionMenuState>() {
            menu.view_show_borders
                .set_checked(show)
                .map_err(|e| e.to_string())?;
        }
        wake_viewport_loop(&app);
    }
    #[cfg(not(desktop))]
    {
        let _ = app;
    }
    Ok(())
}

/// Keeps **View → Hide UI** in sync with webview state.
#[tauri::command]
fn view_menu_sync_hide_ui(app: AppHandle, hidden: bool) -> Result<(), String> {
    #[cfg(desktop)]
    {
        if let Some(menu) = app.try_state::<SelectionMenuState>() {
            menu.view_hide_ui
                .set_checked(hidden)
                .map_err(|e| e.to_string())?;
        }
    }
    #[cfg(not(desktop))]
    {
        let _ = (app, hidden);
    }
    Ok(())
}

/// Keeps the native **Match Material** menu checkbox in sync with app state.
#[tauri::command]
fn selection_menu_sync_match_material(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    checked: bool,
) -> Result<(), String> {
    *state.selection_match_material.lock() = checked;
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
fn set_soft_shadows(state: State<'_, Arc<ViewerState>>, enabled: bool) -> Result<(), String> {
    if let Some(viewer) = state.viewer.lock().as_mut() {
        viewer.soft_shadows = enabled;
    }
    Ok(())
}

#[tauri::command]
fn set_gizmo_on_top(state: State<'_, Arc<ViewerState>>, enabled: bool) -> Result<(), String> {
    if let Some(viewer) = state.viewer.lock().as_mut() {
        viewer.set_gizmo_on_top(enabled);
    }
    Ok(())
}

#[tauri::command]
fn set_soft_sunshafts(state: State<'_, Arc<ViewerState>>, enabled: bool) -> Result<(), String> {
    if let Some(viewer) = state.viewer.lock().as_mut() {
        viewer.set_soft_sunshafts(enabled);
    }
    Ok(())
}

#[tauri::command]
fn set_emission_lighting(state: State<'_, Arc<ViewerState>>, enabled: bool) -> Result<(), String> {
    greedy_mesh::EMISSION_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
    // Invalidate the mesh cache so the next frame triggers a full remesh with the new setting.
    if let Some(viewer) = state.viewer.lock().as_mut() {
        viewer.invalidate_spatial_mesh_cache();
    }
    state
        .mesh_refresh_generation
        .fetch_add(1, std::sync::atomic::Ordering::Release);
    Ok(())
}

#[tauri::command]
fn set_tone_mapping(state: State<'_, Arc<ViewerState>>, mode: u32) -> Result<(), String> {
    let mut v = state.viewer.lock();
    let Some(viewer) = v.as_mut() else {
        return Err("viewer not ready".into());
    };
    viewer.set_tone_mapping_mode(mode);
    Ok(())
}

#[tauri::command]
fn is_hdr_available(state: State<'_, Arc<ViewerState>>) -> Result<bool, String> {
    let v = state.viewer.lock();
    let Some(viewer) = v.as_ref() else {
        return Err("viewer not ready".into());
    };
    Ok(viewer.hdr_available())
}

#[tauri::command]
fn set_hdr_output(state: State<'_, Arc<ViewerState>>, enabled: bool) -> Result<(), String> {
    let mut v = state.viewer.lock();
    let Some(viewer) = v.as_mut() else {
        return Err("viewer not ready".into());
    };
    viewer.set_hdr_output(enabled);
    Ok(())
}

#[tauri::command]
fn set_mood_params(state: State<'_, Arc<ViewerState>>, args: MoodParams) -> Result<(), String> {
    let mut v = state.viewer.lock();
    let Some(viewer) = v.as_mut() else {
        return Err("viewer not ready".into());
    };
    viewer.set_mood_params(&args);
    drop(v);
    {
        let mut cf = state.current_file.lock();
        if let Some(f) = cf.as_mut() {
            f.mood = Some(mood_params_to_settings(&args));
        }
    }
    Ok(())
}

#[tauri::command]
fn set_scene_lighting(
    state: State<'_, Arc<ViewerState>>,
    args: voxelle::LightingSettings,
) -> Result<(), String> {
    let mut v = state.viewer.lock();
    let Some(viewer) = v.as_mut() else {
        return Err("viewer not ready".into());
    };
    viewer.apply_lighting_settings(&args);
    drop(v);
    {
        let mut cf = state.current_file.lock();
        if let Some(f) = cf.as_mut() {
            f.lighting = Some(args);
        }
    }
    Ok(())
}

#[tauri::command]
fn get_scene_lighting(
    state: State<'_, Arc<ViewerState>>,
) -> Result<voxelle::LightingSettings, String> {
    let g = state.current_file.lock();
    let Some(f) = g.as_ref() else {
        return Ok(voxelle::LightingSettings::default());
    };
    Ok(f.lighting.clone().unwrap_or_default())
}

#[tauri::command]
fn set_focal_length_mm(state: State<'_, Arc<ViewerState>>, mm: f32) -> Result<(), String> {
    let mm = mm.clamp(15.0, 200.0);
    let mut cam = state.camera.lock();
    if !cam.perspective {
        return Ok(());
    }
    cam.fov_y = focal_length_to_fov_y_radians(mm);
    {
        let mut cf = state.current_file.lock();
        if let Some(f) = cf.as_mut() {
            f.scene.focal_length_mm = Some(mm);
        }
    }
    Ok(())
}

#[tauri::command]
fn get_focal_length_mm(state: State<'_, Arc<ViewerState>>) -> Result<f32, String> {
    let g = state.current_file.lock();
    let Some(f) = g.as_ref() else {
        return Ok(29.0);
    };
    Ok(f.scene.focal_length_mm.unwrap_or(29.0))
}

#[tauri::command]
fn set_fly_mode(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    enabled: bool,
) -> Result<(), String> {
    *state.fly_mode.lock() = enabled;
    let mut cam = state.camera.lock();
    cam.is_fly_mode = enabled;
    if enabled {
        *state.fly_last_physics.lock() = None;
        drop(cam);
        wake_viewport_loop(&app);
    }
    Ok(())
}

#[tauri::command]
fn get_fly_mode(state: State<'_, Arc<ViewerState>>) -> Result<bool, String> {
    Ok(*state.fly_mode.lock())
}

#[tauri::command]
fn set_walk_mode(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    enabled: bool,
) -> Result<(), String> {
    if enabled {
        // Disable fly mode when entering walk mode.
        *state.fly_mode.lock() = false;
        state.camera.lock().is_fly_mode = false;
    }
    *state.walk_mode.lock() = enabled;
    let mut cam = state.camera.lock();
    cam.is_walk_mode = enabled;
    if enabled {
        // Initialize walk physics from current camera position.
        let eye = cam.target + cam.spherical.to_offset();
        let feet = glam::Vec3::new(eye.x, eye.y - camera::WALK_EYE_HEIGHT, eye.z);
        *state.walk_physics.lock() = camera::WalkPhysicsState {
            feet_pos: feet,
            vel_y: 0.0,
            on_ground: false,
        };
        *state.walk_last_physics.lock() = None;
        drop(cam);
        wake_viewport_loop(&app);
    }
    Ok(())
}

/// Check if a voxel coordinate is occupied.
#[inline]
fn walk_is_solid(vm: &AHashMap<greedy_mesh::VoxelCoord, usize>, x: i32, y: i32, z: i32) -> bool {
    vm.contains_key(&(x, y, z))
}

/// Resolve walk-mode collision against the voxel grid. Returns corrected feet position.
fn resolve_walk_collision(
    old_feet: glam::Vec3,
    mut new_feet: glam::Vec3,
    vm: &AHashMap<greedy_mesh::VoxelCoord, usize>,
    wp: &mut camera::WalkPhysicsState,
) -> glam::Vec3 {
    // --- Vertical collision (process Y first) ---
    let fc = voxel_edit::world_to_voxel(new_feet);

    // Check voxel AT feet level: are we inside a solid block?
    if walk_is_solid(vm, fc.0, fc.1, fc.2) {
        let ground_top_y = fc.1 as f32 + 0.5;
        new_feet.y = ground_top_y;
        wp.vel_y = 0.0;
        wp.on_ground = true;
    } else {
        // Check voxel directly below feet
        if walk_is_solid(vm, fc.0, fc.1 - 1, fc.2) {
            let ground_top_y = (fc.1 - 1) as f32 + 0.5;
            if new_feet.y <= ground_top_y + 0.05 {
                new_feet.y = ground_top_y;
                wp.vel_y = 0.0;
                wp.on_ground = true;
            }
        } else {
            wp.on_ground = false;
        }
    }

    // Ceiling collision: check at head height
    let head_pos = new_feet + glam::Vec3::Y * camera::WALK_EYE_HEIGHT;
    let hc = voxel_edit::world_to_voxel(head_pos);
    if walk_is_solid(vm, hc.0, hc.1, hc.2) && wp.vel_y > 0.0 {
        wp.vel_y = 0.0;
        let ceiling_bottom_y = hc.1 as f32 - 0.5;
        new_feet.y = ceiling_bottom_y - camera::WALK_EYE_HEIGHT;
    }

    // --- Horizontal collision + auto step-up ---
    // Check body voxels at the new horizontal position
    let body_low = voxel_edit::world_to_voxel(new_feet + glam::Vec3::Y * 0.1);
    let body_high = voxel_edit::world_to_voxel(new_feet + glam::Vec3::Y * 1.0);

    let blocked_low = walk_is_solid(vm, body_low.0, body_low.1, body_low.2);
    let blocked_high = walk_is_solid(vm, body_high.0, body_high.1, body_high.2);

    if blocked_low && !blocked_high {
        // Step-up candidate: blocked at feet but clear at torso
        let step_top = body_low.1 as f32 + 0.5;
        let step_height = step_top - new_feet.y;
        // Check head clearance above the step
        let clearance_ok = !walk_is_solid(vm, body_low.0, body_low.1 + 2, body_low.2);
        if clearance_ok && step_height <= camera::WALK_STEP_HEIGHT {
            new_feet.y = step_top;
            wp.vel_y = 0.0;
            wp.on_ground = true;
        } else {
            // Can't step up — try wall sliding
            new_feet = walk_slide(old_feet, new_feet, vm);
        }
    } else if blocked_low || blocked_high {
        // Full wall block — try wall sliding
        new_feet = walk_slide(old_feet, new_feet, vm);
    }

    new_feet
}

/// Wall sliding: try X-only, then Z-only, then full revert.
fn walk_slide(
    old_feet: glam::Vec3,
    new_feet: glam::Vec3,
    vm: &AHashMap<greedy_mesh::VoxelCoord, usize>,
) -> glam::Vec3 {
    // Try sliding along X only (revert Z)
    let try_x = glam::Vec3::new(new_feet.x, new_feet.y, old_feet.z);
    let bx_low = voxel_edit::world_to_voxel(try_x + glam::Vec3::Y * 0.1);
    let bx_high = voxel_edit::world_to_voxel(try_x + glam::Vec3::Y * 1.0);
    if !walk_is_solid(vm, bx_low.0, bx_low.1, bx_low.2)
        && !walk_is_solid(vm, bx_high.0, bx_high.1, bx_high.2)
    {
        return try_x;
    }

    // Try sliding along Z only (revert X)
    let try_z = glam::Vec3::new(old_feet.x, new_feet.y, new_feet.z);
    let bz_low = voxel_edit::world_to_voxel(try_z + glam::Vec3::Y * 0.1);
    let bz_high = voxel_edit::world_to_voxel(try_z + glam::Vec3::Y * 1.0);
    if !walk_is_solid(vm, bz_low.0, bz_low.1, bz_low.2)
        && !walk_is_solid(vm, bz_high.0, bz_high.1, bz_high.2)
    {
        return try_z;
    }

    // Fully blocked: revert horizontal
    glam::Vec3::new(old_feet.x, new_feet.y, old_feet.z)
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
    #[serde(default)]
    jump: bool,
}

/// WASD / shift state only. Translation integrates on the native event loop with real elapsed time.
#[tauri::command]
fn sync_fly_input(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: SyncFlyInputArgs,
) -> Result<(), String> {
    let fly = *state.fly_mode.lock();
    let walk = *state.walk_mode.lock();
    if !fly && !walk {
        return Ok(());
    }
    let scale = args.speed_scale;
    let speed_scale = if scale.is_finite() {
        scale.clamp(0.0, 1e6)
    } else {
        1.0
    };
    let has_movement = args.forward != 0.0 || args.right != 0.0 || args.up != 0.0 || args.jump;
    *state.fly_input.lock() = FlyInputState {
        forward: args.forward,
        right: args.right,
        up: args.up,
        speed_scale,
        jump: args.jump,
    };
    if has_movement {
        wake_viewport_loop(&app);
    }
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
    if !*state.fly_mode.lock() && !*state.walk_mode.lock() {
        return Ok(());
    }
    let vh = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Ok(());
        };
        let (_, h) = viewer.viewport_size();
        h as f32
    };
    let mut cam = state.camera.lock();
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
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Ok(false);
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let fg = state.current_file.lock();
    let vm = state.voxel_map.lock();
    let Some(file) = fg.as_ref() else {
        return Ok(false);
    };
    let Some(vmap) = vm.as_ref() else {
        return Ok(false);
    };
    let cam = state.camera.lock();
    Ok(voxel_edit::probe_solid_hit(file, vmap, &cam, w, h, sx, sy))
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct StrokeAnchorAtScreen {
    nx: f32,
    ny: f32,
    tool: voxel_edit::EditTool,
    #[serde(default = "default_stroke_snap_to_surface_arg")]
    stroke_snap_to_surface: bool,
}

fn default_stroke_snap_to_surface_arg() -> bool {
    true
}

/// Anchor voxel for multi-click stroke geometry (add → placement cell; remove/paint → solid under ray).
#[tauri::command]
fn voxel_stroke_anchor_coord_at_screen(
    state: State<'_, Arc<ViewerState>>,
    args: StrokeAnchorAtScreen,
) -> Result<Option<[i32; 3]>, String> {
    let (w, h) = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Ok(None);
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let fg = state.current_file.lock();
    let Some(file) = fg.as_ref() else {
        return Ok(None);
    };
    let vm = state.voxel_map.lock();
    let Some(vmap) = vm.as_ref() else {
        return Ok(None);
    };
    let cam = state.camera.lock();
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let c = voxel_edit::anchor_for_stroke_edit(
        args.tool,
        args.stroke_snap_to_surface,
        file,
        vmap,
        &cam,
        w,
        h,
        sx,
        sy,
    );
    Ok(c.map(|(x, y, z)| [x, y, z]))
}

/// Returns the surface Y (topmost voxel) at the given screen position, for the terrain hover display.
#[tauri::command]
fn terrain_surface_y_at_screen(
    state: State<'_, Arc<ViewerState>>,
    nx: f32,
    ny: f32,
) -> Result<Option<i32>, String> {
    let (w, h) = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Ok(None);
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let fg = state.current_file.lock();
    let Some(file) = fg.as_ref() else {
        return Ok(None);
    };
    let vm = state.voxel_map.lock();
    let Some(vmap) = vm.as_ref() else {
        return Ok(None);
    };
    let cam = state.camera.lock();
    let (sx, sy) = viewport_texels_from_norm(nx, ny, w, h);
    let c = voxel_edit::anchor_for_stroke_edit(
        voxel_edit::EditTool::Remove,
        true,
        file,
        vmap,
        &cam,
        w,
        h,
        sx,
        sy,
    );
    Ok(c.map(|(_, y, _)| y))
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
        PreviewMode::Add => {
            voxel_edit::preview_add_cell(file, vmap, cam, w, h, sx, sy).map(|(c, _)| c)
        }
        PreviewMode::Remove | PreviewMode::Paint | PreviewMode::Select => {
            voxel_edit::preview_remove_cell(file, vmap, cam, w, h, sx, sy).map(|(c, _)| c)
        }
        PreviewMode::Navigate | PreviewMode::Fly | PreviewMode::Squishy => {
            voxel_edit::preview_remove_cell(file, vmap, cam, w, h, sx, sy)
                .map(|(c, _)| c)
                .or_else(|| {
                    voxel_edit::preview_add_cell(file, vmap, cam, w, h, sx, sy).map(|(c, _)| c)
                })
        }
        PreviewMode::Stamp => {
            voxel_edit::preview_add_cell(file, vmap, cam, w, h, sx, sy).map(|(c, _)| c)
        }
        PreviewMode::Punch => {
            voxel_edit::preview_remove_cell(file, vmap, cam, w, h, sx, sy).map(|(c, _)| c)
        }
    }
}

fn local_accent_ping_color(state: &ViewerState) -> u32 {
    let c = state.collab.lock();
    c.roster
        .iter()
        .find(|r| r.peer_id == c.local_peer_id)
        .map(|r| r.color_rgb)
        .unwrap_or(0x66ccff)
}

fn local_accent_ping_display_name(state: &ViewerState) -> String {
    let c = state.collab.lock();
    c.roster
        .iter()
        .find(|r| r.peer_id == c.local_peer_id)
        .map(|r| {
            if r.display_name.trim().is_empty() {
                "You".to_string()
            } else {
                r.display_name.clone()
            }
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
        let v = state.viewer.lock();
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
    let mode = *state.preview_mode.lock();
    let coords = {
        let fg = state.current_file.lock();
        let vm = state.voxel_map.lock();
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
        let cam = state.camera.lock();
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
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Ok(None);
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let cam = state.camera.lock();
    Ok(voxel_edit::world_to_viewport_pixels(
        &cam, w, h, args.x, args.y, args.z,
    ))
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct PeerLabel {
    name: String,
    color_rgb: u32,
    left_pct: f32,
    top_pct: f32,
}

#[tauri::command]
fn collab_peer_labels(state: State<'_, Arc<ViewerState>>) -> Result<Vec<PeerLabel>, String> {
    let (w, h) = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Ok(vec![]);
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    if w <= 0.0 || h <= 0.0 {
        return Ok(vec![]);
    }
    let cam = state.camera.lock();
    let c = state.collab.lock();
    if !c.is_active() {
        return Ok(vec![]);
    }
    let local_id = c.local_peer_id;
    let smooth = state.smooth_presence.lock();
    let mut labels = Vec::new();
    for (pid, pr) in smooth.iter() {
        if *pid == local_id {
            continue;
        }
        let eye = collab::presence_eye(pr);
        let Some((sx, sy)) = voxel_edit::world_to_viewport_pixels(&cam, w, h, eye.x, eye.y, eye.z)
        else {
            continue;
        };
        let entry = c.roster.iter().find(|r| r.peer_id == *pid);
        let name = entry.map(|r| r.display_name.clone()).unwrap_or_default();
        let color_rgb = entry.map(|r| r.color_rgb).unwrap_or(0x888888);
        labels.push(PeerLabel {
            name,
            color_rgb,
            left_pct: (sx / w) * 100.0,
            top_pct: (sy / h) * 100.0,
        });
    }
    Ok(labels)
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
        .push(SoloUndoEntry::VoxelDeltas(deltas));
    state.solo_redo.lock().clear();
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
        .push(SoloUndoEntry::SelectionBefore(before));
    state.solo_redo.lock().clear();
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
    let stroke_on = *state.stroke_active.lock();
    if stroke_on {
        state.stroke_buffer.lock().extend(deltas.iter().copied());
        return Ok(true);
    }
    let cm = Arc::clone(&state.collab);
    let mut cb = cm.lock();
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
    *state.stroke_active.lock() = true;
    state.stroke_buffer.lock().clear();
    state.stroke_preview_union.lock().clear();
    *state.stroke_preview_last_args.lock() = None;
    state
        .stroke_preview_suppresses_hover
        .store(false, Ordering::Relaxed);
    state.sculpt_stroke_replay.lock().clear();
    // Clear spray constraint plane so it gets re-established on the first anchor of this stroke.
    *state.spray_constraint_plane.lock() = None;
    *state.wall_stroke_face_snapped.lock() = None;
    state.terrain_accum.lock().clear();
    Ok(())
}

/// Clear stroke preview GPU/state **without** starting a new stroke (`stroke_active` stays false).
/// Used when exiting cuboid/cylinder depth phase (Done / Escape / new gesture) so the next
/// `voxel_edit_at_screen` is not mistaken for an in-stroke edit.
#[tauri::command]
fn voxel_stroke_preview_reset(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
) -> Result<(), String> {
    *state.stroke_active.lock() = false;
    state.stroke_buffer.lock().clear();
    state.stroke_preview_union.lock().clear();
    *state.stroke_preview_last_args.lock() = None;
    state
        .stroke_preview_suppresses_hover
        .store(false, Ordering::Relaxed);
    state.sculpt_stroke_replay.lock().clear();
    *state.wall_stroke_face_snapped.lock() = None;
    {
        let mut v = state.viewer.lock();
        if let Some(viewer) = v.as_mut() {
            clear_preview_mesh_sync_cache(viewer, state.inner().as_ref());
        }
    }
    wake_viewport_loop(&app);
    Ok(())
}

const STROKE_PREVIEW_MAX_CELLS: usize = 25_000;

/// Empty-cells-only preview (air along stroke): dimmer fill than occupied shell.
const PREVIEW_GHOST_FILL_MUL: f32 = 0.55;
const PREVIEW_GHOST_WIRE_MUL: f32 = 0.62;
/// Paint/Remove ghosts: [`scene.wgsl`] `is_preview_ghost_*_paint_remove` — 75% transparent fill/wire.
const PREVIEW_GHOST_MAT_KIND_FILL_PAINT_REMOVE: f32 = 1.08;
const PREVIEW_GHOST_MAT_KIND_WIRE_PAINT_REMOVE: f32 = 1.72;

/// Solid + wire RGB, cube half-extent, wire `mat_kind` — hover and stroke preview cubes.
///
/// Appearance (GPU [`scene.wgsl`] `fs_preview_*`): slightly transparent fill; darker when occluded or
/// overlapping solids; Paint/Remove empty-footprint uses `mat_kind` bands for extra transparency;
/// wireframe depth-tested so it can be hidden by scene mesh.
///
/// - **Add / Paint / sculpt** (and squishy Add-style preview): `palette_rgb` packed `0xRRGGBB`.
/// - **Remove**: fixed red (ignore `palette_rgb`).
/// - **Select** single-cell preview: fixed blue in [`prepare_preview_mesh`] (not this helper).
/// - **Debug pick highlight**: fixed high-contrast red.
fn preview_tool_colors(
    tool: voxel_edit::EditTool,
    debug_pick_highlight: bool,
    palette_rgb: u32,
) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    if debug_pick_highlight {
        // Bright red fill, dark red wire — stands out for viewport cursor debug.
        return (1.0, 0.12, 0.1, 0.55, 0.0, 0.0, 0.56, 3.5);
    }
    match tool {
        voxel_edit::EditTool::Remove => (0.95, 0.28, 0.22, 0.14, 0.03, 0.03, 0.5, 2.0),
        voxel_edit::EditTool::Add | voxel_edit::EditTool::Paint => {
            let [sr, sg, sb] = voxelle::rgb24_u32_to_linear_rgb3(palette_rgb);
            let wr = (sr * 0.22).max(0.02);
            let wg = (sg * 0.22).max(0.02);
            let wb = (sb * 0.22).max(0.02);
            (sr, sg, sb, wr, wg, wb, 0.5, 2.0)
        }
    }
}

fn stroke_preview_meshes_for_union(
    tool: voxel_edit::EditTool,
    union: &AHashSet<greedy_mesh::VoxelCoord>,
    voxel_map: &AHashMap<greedy_mesh::VoxelCoord, usize>,
    file: &voxelle::VoxelleFile,
    debug_pick_highlight: bool,
    palette_rgb: u32,
    color_resolver: Option<&dyn Fn(i32, i32, i32) -> u32>,
) -> greedy_mesh::PreviewInstancedResult {
    // Occupied cells: shell only (large solid previews stay cheap). Empty footprint cells: always
    // included (full brush volume in air). Stroke commit still uses the full union in
    // [`voxel_stroke_end`].
    let mut occupied = AHashSet::with_capacity(union.len().min(65_536));
    let mut empty_only = AHashSet::with_capacity(union.len().min(65_536));
    for &c in union.iter() {
        if voxel_map.contains_key(&c) {
            occupied.insert(c);
        } else {
            empty_only.insert(c);
        }
    }
    let shell_occ = greedy_mesh::filter_voxel_set_to_shell(&occupied);
    let mut sorted: Vec<greedy_mesh::VoxelCoord> = shell_occ
        .into_iter()
        .chain(empty_only.into_iter())
        .collect();
    if sorted.is_empty() {
        return greedy_mesh::PreviewInstancedResult::empty();
    }
    sorted.sort_unstable_by_key(|&c| {
        let ghost = !voxel_map.contains_key(&c);
        (ghost, c.0, c.1, c.2)
    });
    let (sr, sg, sb, wr, wg, wb, size, wem) =
        preview_tool_colors(tool, debug_pick_highlight, palette_rgb);
    let use_per_voxel_color = color_resolver.is_some()
        && matches!(
            tool,
            voxel_edit::EditTool::Add | voxel_edit::EditTool::Paint
        );
    let n = sorted.len().min(STROKE_PREVIEW_MAX_CELLS);
    let mut solid_instances: Vec<greedy_mesh::PreviewInstance> = Vec::with_capacity(n);
    let mut wire_instances: Vec<greedy_mesh::PreviewInstance> = Vec::with_capacity(n);
    for (cx, cy, cz) in sorted.into_iter().take(STROKE_PREVIEW_MAX_CELLS) {
        let ghost = !voxel_map.contains_key(&(cx, cy, cz));
        let (base_sr, base_sg, base_sb, base_wr, base_wg, base_wb) = if use_per_voxel_color {
            if let Some(resolver) = color_resolver {
                let rgb = resolver(cx, cy, cz);
                let [r, g, b] = voxelle::rgb24_u32_to_linear_rgb3(rgb);
                let wrc = (r * 0.22).max(0.02);
                let wgc = (g * 0.22).max(0.02);
                let wbc = (b * 0.22).max(0.02);
                (r, g, b, wrc, wgc, wbc)
            } else {
                (sr, sg, sb, wr, wg, wb)
            }
        } else {
            (sr, sg, sb, wr, wg, wb)
        };
        let (srf, sgf, sbf, wrf, wgf, wbf) = if ghost {
            (
                base_sr * PREVIEW_GHOST_FILL_MUL,
                base_sg * PREVIEW_GHOST_FILL_MUL,
                base_sb * PREVIEW_GHOST_FILL_MUL,
                base_wr * PREVIEW_GHOST_WIRE_MUL,
                base_wg * PREVIEW_GHOST_WIRE_MUL,
                base_wb * PREVIEW_GHOST_WIRE_MUL,
            )
        } else {
            (base_sr, base_sg, base_sb, base_wr, base_wg, base_wb)
        };
        let ghost_pr = ghost
            && matches!(
                tool,
                voxel_edit::EditTool::Remove | voxel_edit::EditTool::Paint
            );
        let fill_mat_k = if ghost_pr {
            PREVIEW_GHOST_MAT_KIND_FILL_PAINT_REMOVE
        } else {
            1.0
        };
        let wire_mat_k = if ghost_pr {
            PREVIEW_GHOST_MAT_KIND_WIRE_PAINT_REMOVE
        } else {
            wem
        };
        let oid = voxel_map
            .get(&(cx, cy, cz))
            .map(|&vi| file.voxels[vi].object_id)
            .unwrap_or(file.active_object_id);
        let obj_m = object_world_matrix(&file.objects, oid);
        let translate =
            glam::Mat4::from_translation(glam::Vec3::new(cx as f32, cy as f32, cz as f32));
        let model = obj_m * translate;
        let cols = model.to_cols_array_2d();
        solid_instances.push(greedy_mesh::PreviewInstance {
            model_c0: cols[0],
            model_c1: cols[1],
            model_c2: cols[2],
            model_c3: cols[3],
            color: [srf, sgf, sbf],
            mat_kind: fill_mat_k,
        });
        wire_instances.push(greedy_mesh::PreviewInstance {
            model_c0: cols[0],
            model_c1: cols[1],
            model_c2: cols[2],
            model_c3: cols[3],
            color: [wrf, wgf, wbf],
            mat_kind: wire_mat_k,
        });
    }
    greedy_mesh::PreviewInstancedResult {
        solid_instances,
        wire_instances,
        cube_half: size,
        extra_solid: greedy_mesh::MeshBuffers::default(),
        extra_wire: greedy_mesh::MeshBuffers::default(),
    }
}

/// Saturated yellow corners for polygon / polygonHull placement (web `polygonPointsMaterial` parity).
fn append_polygon_vertex_marker_meshes(
    solid: &mut greedy_mesh::MeshBuffers,
    wire: &mut greedy_mesh::MeshBuffers,
    verts: &[[i32; 3]],
    vmap: &AHashMap<greedy_mesh::VoxelCoord, usize>,
    file: &voxelle::VoxelleFile,
    debug_pick_highlight: bool,
) {
    if verts.is_empty() {
        return;
    }
    let (vr, vg, vb, wr, wg, wb, size, wem) = if debug_pick_highlight {
        (1.0_f32, 0.12, 0.1, 0.55, 0.0, 0.0, 0.56, 3.5)
    } else {
        (1.0, 0.92, 0.12, 0.42, 0.4, 0.06, 0.5, 2.0)
    };
    for &[cx, cy, cz] in verts {
        let ghost = !vmap.contains_key(&(cx, cy, cz));
        let (srf, sgf, sbf, wrf, wgf, wbf) = if ghost {
            (
                vr * PREVIEW_GHOST_FILL_MUL,
                vg * PREVIEW_GHOST_FILL_MUL,
                vb * PREVIEW_GHOST_FILL_MUL,
                wr * PREVIEW_GHOST_WIRE_MUL,
                wg * PREVIEW_GHOST_WIRE_MUL,
                wb * PREVIEW_GHOST_WIRE_MUL,
            )
        } else {
            (vr, vg, vb, wr, wg, wb)
        };
        let oid = vmap
            .get(&(cx, cy, cz))
            .map(|&vi| file.voxels[vi].object_id)
            .unwrap_or(file.active_object_id);
        let mut s = greedy_mesh::preview_cube_mesh(
            cx as f32,
            cy as f32,
            cz as f32,
            size,
            [srf, sgf, sbf],
            1.0,
        );
        let mut w = greedy_mesh::preview_cube_wireframe_mesh(
            cx as f32,
            cy as f32,
            cz as f32,
            size,
            [wrf, wgf, wbf],
            wem,
        );
        let m = object_world_matrix(&file.objects, oid);
        greedy_mesh::transform_mesh_buffers(&mut s, m);
        greedy_mesh::transform_mesh_buffers(&mut w, m);
        greedy_mesh::append_mesh_buffers(solid, s);
        greedy_mesh::append_mesh_buffers(wire, w);
    }
}

/// Local-space center for the hover cube: use ray–cell **face hit** so the wireframe sits under the cursor
/// on oblique surfaces (voxel center projects elsewhere).
fn preview_single_cell_world(
    file: &voxelle::VoxelleFile,
    lx: f32,
    ly: f32,
    lz: f32,
    object_id: u32,
    sr: f32,
    sg: f32,
    sb: f32,
    wr: f32,
    wg: f32,
    wb: f32,
    size: f32,
    wem: f32,
) -> greedy_mesh::PreviewInstancedResult {
    let obj_m = object_world_matrix(&file.objects, object_id);
    let translate = glam::Mat4::from_translation(glam::Vec3::new(lx, ly, lz));
    let model = obj_m * translate;
    let cols = model.to_cols_array_2d();
    let solid_inst = greedy_mesh::PreviewInstance {
        model_c0: cols[0],
        model_c1: cols[1],
        model_c2: cols[2],
        model_c3: cols[3],
        color: [sr, sg, sb],
        mat_kind: 1.0,
    };
    let wire_inst = greedy_mesh::PreviewInstance {
        model_c0: cols[0],
        model_c1: cols[1],
        model_c2: cols[2],
        model_c3: cols[3],
        color: [wr, wg, wb],
        mat_kind: wem,
    };
    greedy_mesh::PreviewInstancedResult {
        solid_instances: vec![solid_inst],
        wire_instances: vec![wire_inst],
        cube_half: size,
        extra_solid: greedy_mesh::MeshBuffers::default(),
        extra_wire: greedy_mesh::MeshBuffers::default(),
    }
}

/// Preview-only stroke update during drag (commit on [`voxel_stroke_end`]).
#[tauri::command]
fn voxel_stroke_preview_at_screen(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: VoxelEditAtScreen,
) -> Result<(), String> {
    {
        let cm = state.collab.lock();
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
        let fg = state.current_file.lock();
        let vm = state.voxel_map.lock();
        let Some(file) = fg.as_ref() else {
            return Ok(());
        };
        let Some(vmap) = vm.as_ref() else {
            return Ok(());
        };
        let cam = state.camera.lock();
        let (w, h) = {
            let v = state.viewer.lock();
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
        let spray_cp = resolve_spray_constraint_plane(
            &state,
            &args.stroke_aux,
            args.stroke_mode,
            args.tool,
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
        );
        let targets = voxel_edit::collect_stroke_preview_targets(
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
            spray_cp,
        );
        targets
    };

    {
        let mut union = state.stroke_preview_union.lock();
        let accumulate = voxel_edit::stroke_preview_accumulates_samples(
            args.stroke_mode,
            stroke_line_start_meta,
        );
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

    *state.stroke_preview_last_args.lock() = Some(args.clone());

    let instanced = {
        let fg = state.current_file.lock();
        let vm = state.voxel_map.lock();
        let union = state.stroke_preview_union.lock();
        let Some(file) = fg.as_ref() else {
            return Ok(());
        };
        let Some(vmap) = vm.as_ref() else {
            return Ok(());
        };
        let preview_resolver_owned = if args.palette.len() > 1 {
            Some(build_color_resolver(
                args.color,
                args.palette.clone(),
                args.paint_color_distrib.clone(),
                args.stroke_seed,
            ))
        } else {
            None
        };
        let preview_resolver_ref: Option<&dyn Fn(i32, i32, i32) -> u32> = preview_resolver_owned
            .as_ref()
            .map(|f| f as &dyn Fn(i32, i32, i32) -> u32);
        let mut result = stroke_preview_meshes_for_union(
            args.tool,
            &union,
            vmap,
            file,
            false,
            args.color,
            preview_resolver_ref,
        );
        if matches!(
            args.stroke_mode,
            stroke_modes::DrawStrokeMode::Polygon | stroke_modes::DrawStrokeMode::PolygonHull
        ) && !args.stroke_aux.polygon_vertices.is_empty()
        {
            append_polygon_vertex_marker_meshes(
                &mut result.extra_solid,
                &mut result.extra_wire,
                &args.stroke_aux.polygon_vertices,
                vmap,
                file,
                false,
            );
        }
        result
    };

    {
        let mut v = state.viewer.lock();
        let Some(viewer) = v.as_mut() else {
            return Ok(());
        };
        if instanced.solid_instances.is_empty() && instanced.extra_solid.positions.is_empty() {
            clear_preview_mesh_sync_cache(viewer, state.inner().as_ref());
            state
                .stroke_preview_suppresses_hover
                .store(false, Ordering::Relaxed);
        } else {
            viewer.upload_preview_mesh_instanced(&instanced);
            viewer.preview_cache_key = None;
            *state.preview_overlay_cache_key.lock() = None;
            state
                .stroke_preview_suppresses_hover
                .store(true, Ordering::Relaxed);
        }
    }

    wake_viewport_loop(&app);
    Ok(())
}

/// Result of resolving cuboid/cylinder drag-plane geometry at a point in time.
/// Returned by [`query_cuboid_plane_geometry`] so the frontend can freeze this
/// during the depth phase and pass it back through `StrokeAux`, preventing
/// camera movement from altering the extrusion direction.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct CuboidPlaneGeoResult {
    a: [i32; 3],
    b: [i32; 3],
    plane_ax: u8,
    hit: [i32; 3],
    prev: [i32; 3],
}

/// Resolve the drag-plane geometry (anchor, far corner, face normal) for the
/// current camera state.  Call this once when entering the cuboid/cylinder
/// depth phase and pass the result back on every depth-preview call via the
/// `cuboidFrozen*` fields of `strokeAux`.
#[tauri::command]
fn query_cuboid_plane_geometry(
    state: State<'_, Arc<ViewerState>>,
    args: VoxelEditAtScreen,
) -> Option<CuboidPlaneGeoResult> {
    let fg = state.current_file.lock();
    let vm = state.voxel_map.lock();
    let file = fg.as_ref()?;
    let vmap = vm.as_ref()?;
    let cam = state.camera.lock();
    let (w, h) = {
        let v = state.viewer.lock();
        let viewer = v.as_ref()?;
        let (vw, vh) = viewer.viewport_size();
        (vw as f32, vh as f32)
    };
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let (lsx, lsy) = match (args.stroke_line_start_nx, args.stroke_line_start_ny) {
        (Some(lnx), Some(lny)) => viewport_texels_from_norm(lnx, lny, w, h),
        _ => return None,
    };
    let snap = args.stroke_aux.stroke_snap_to_surface;
    let (a, b, plane_ax, hit, prev) = stroke_modes::cuboid_drag_plane_geometry_pub(
        args.tool,
        file,
        vmap,
        &cam,
        w,
        h,
        lsx,
        lsy,
        sx,
        sy,
        args.plane_axis,
        snap,
    )?;
    Some(CuboidPlaneGeoResult {
        a: [a.0, a.1, a.2],
        b: [b.0, b.1, b.2],
        plane_ax: plane_ax as u8,
        hit: [hit.0, hit.1, hit.2],
        prev: [prev.0, prev.1, prev.2],
    })
}

#[tauri::command]
fn voxel_stroke_end(state: State<'_, Arc<ViewerState>>, app: AppHandle) -> Result<(), String> {
    *state.stroke_active.lock() = false;
    *state.extrude_ray_spine.lock() = None;
    *state.wall_stroke_face_snapped.lock() = None;
    state.terrain_accum.lock().clear();
    let had_stroke_preview = state
        .stroke_preview_suppresses_hover
        .swap(false, Ordering::Relaxed);
    let union = std::mem::take(&mut *state.stroke_preview_union.lock());
    let last_args = state.stroke_preview_last_args.lock().take();
    let buf = std::mem::take(&mut *state.stroke_buffer.lock());
    let sculpt_replay = std::mem::take(&mut *state.sculpt_stroke_replay.lock());

    if had_stroke_preview {
        let mut v = state.viewer.lock();
        if let Some(viewer) = v.as_mut() {
            clear_preview_mesh_sync_cache(viewer, state.inner().as_ref());
        }
    }

    if !sculpt_replay.is_empty() {
        // For Draw/Gouge/Extrude/Wall: commit the accumulated preview union as a single batch
        // instead of replaying frame-by-frame (which caused view-ray stacking artifacts).
        // Wall in particular would tower toward the camera when backtracking over placed voxels,
        // because replay frames detect the newly-placed wall and re-anchor on top of it.
        // Smooth/Terrain still need per-frame replay for their complex per-sample logic.
        let mode = sculpt_replay[0].sculpt_mode;
        let use_union_commit = matches!(
            mode,
            voxel_edit::SculptStrokeMode::Draw
                | voxel_edit::SculptStrokeMode::Gouge
                | voxel_edit::SculptStrokeMode::Extrude
                | voxel_edit::SculptStrokeMode::Wall
        );
        if use_union_commit && !union.is_empty() {
            let first = &sculpt_replay[0];
            let material = voxelle::MaterialId::from_str_id(&first.material);
            let color = first.color;
            let deltas = {
                let mut fg = state.current_file.lock();
                let mut vm = state.voxel_map.lock();
                let Some(file) = fg.as_mut() else {
                    return Ok(());
                };
                let Some(vmap) = vm.as_mut() else {
                    return Ok(());
                };
                voxel_edit::ensure_grid_fits_coords(file, union.iter().copied());
                let mut out: Vec<voxel_edit::VoxelEditDelta> = Vec::new();
                if mode == voxel_edit::SculptStrokeMode::Gouge {
                    for (x, y, z) in &union {
                        let Some(&idx) = vmap.get(&(*x, *y, *z)) else {
                            continue;
                        };
                        let removed = file.voxels.swap_remove(idx);
                        vmap.remove(&(*x, *y, *z));
                        if idx < file.voxels.len() {
                            let moved = &file.voxels[idx];
                            vmap.insert((moved.x, moved.y, moved.z), idx);
                        }
                        out.push(voxel_edit::VoxelEditDelta::Removed { voxel: removed });
                    }
                } else {
                    for &(x, y, z) in &union {
                        if vmap.contains_key(&(x, y, z)) {
                            continue;
                        }
                        let nv = voxelle::Voxel {
                            x,
                            y,
                            z,
                            color,
                            material,
                            object_id: file.active_object_id,
                        };
                        let idx = file.voxels.len();
                        file.voxels.push(nv);
                        vmap.insert((x, y, z), idx);
                        out.push(voxel_edit::VoxelEditDelta::Added(nv));
                    }
                }
                out
            };
            if !deltas.is_empty() {
                commit_voxel_edits(&state, &app, deltas)?;
            }
            return Ok(());
        }
        commit_sculpt_stroke_replay(&state, &app, sculpt_replay)?;
        return Ok(());
    }

    // Solid cuboid / cylinder: plane-drag preview only; depth + commit via `voxel_edit_at_screen` (web parity).
    let skip_solid_extrusion_plane_preview_commit = matches!(
        &last_args,
        Some(a)
            if matches!(
                a.stroke_mode,
                stroke_modes::DrawStrokeMode::Cuboid | stroke_modes::DrawStrokeMode::Cylinder
            ) && match a.stroke_mode {
                stroke_modes::DrawStrokeMode::Cuboid => a.stroke_aux.cuboid_depth.is_none(),
                stroke_modes::DrawStrokeMode::Cylinder => a.stroke_aux.cylinder_depth.is_none(),
                _ => false,
            }
    );

    if !union.is_empty() && !skip_solid_extrusion_plane_preview_commit {
        if let Some(args) = last_args {
            let material = voxelle::MaterialId::from_str_id(&args.material);
            let deltas = {
                let mut fg = state.current_file.lock();
                let mut vm = state.voxel_map.lock();
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
                    build_color_resolver(
                        args.color,
                        args.palette.clone(),
                        args.paint_color_distrib.clone(),
                        args.stroke_seed,
                    ),
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
    let mut cb = cm.lock();
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
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let fg = state.current_file.lock();
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };
    let vm = state.voxel_map.lock();
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let cam = state.camera.lock();
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

/// Applies a single stroke sample to the selection, using the accumulator for
/// intersect mode so that successive samples union their coords before
/// intersecting with the original (`before`) selection.
///
/// Returns the new selection size, or `None` if coords were empty and no
/// accumulator was active (caller should skip the emit).
fn apply_selection_stroke_sample(
    sel: &mut AHashSet<greedy_mesh::VoxelCoord>,
    coords: Vec<greedy_mesh::VoxelCoord>,
    mode: SelectionCombineMode,
    accum: &mut Option<AHashSet<greedy_mesh::VoxelCoord>>,
    before: &Option<AHashSet<greedy_mesh::VoxelCoord>>,
) -> Option<u32> {
    if matches!(mode, SelectionCombineMode::Intersect) {
        if let Some(accum_set) = accum.as_mut() {
            accum_set.extend(coords.iter().copied());
            if let Some(before_set) = before.as_ref() {
                *sel = before_set
                    .iter()
                    .copied()
                    .filter(|c| accum_set.contains(c))
                    .collect();
                return Some(sel.len() as u32);
            }
        }
        // No active stroke — fall through to direct merge.
    }

    // Replace mode during a stroke: accumulate all samples so the selection
    // grows as the brush moves rather than being reset to just the current sample.
    if matches!(mode, SelectionCombineMode::Replace) {
        if let Some(accum_set) = accum.as_mut() {
            accum_set.extend(coords.iter().copied());
            *sel = accum_set.iter().copied().collect();
            return Some(sel.len() as u32);
        }
        // No active stroke — fall through to direct merge.
    }

    if coords.is_empty() {
        return None;
    }

    merge_coords_into_selection(sel, coords, mode);
    Some(sel.len() as u32)
}

fn emit_selection_updated<R: Runtime>(app: &AppHandle<R>, state: &Arc<ViewerState>) {
    let has_voxels = state
        .current_file
        .lock()
        .as_ref()
        .map(|f| !f.voxels.is_empty())
        .unwrap_or(false);
    let n = {
        let s = state.selection_cells.lock();
        s.len() as u32
    };
    let has_selection = n > 0;
    let _ = app.emit_to(
        EventTarget::webview_window("main"),
        "voxelle-selection-updated",
        n,
    );
    #[cfg(desktop)]
    selection_menu_sync_enabled_for_scene(app, has_voxels, has_selection);
}

/// Snapshot for [`selection_menu_sync_enabled_for_scene`]: lock `current_file` then `selection_cells`
/// (fixed order — do not invert elsewhere) to avoid AB-BA with code that uses both.
#[cfg(desktop)]
fn scene_menu_flags(state: &ViewerState) -> (bool, bool) {
    let has_voxels = state
        .current_file
        .lock()
        .as_ref()
        .map(|f| !f.voxels.is_empty())
        .unwrap_or(false);
    let has_selection = !state.selection_cells.lock().is_empty();
    (has_voxels, has_selection)
}

/// Disables Selection menu entries when there are no voxels and/or no active selection (same rules as web).
/// Does not lock [`ViewerState`]: pass [`scene_menu_flags`] (or explicit booleans) so callers never
/// nest this under `viewer` / `current_file` guards.
#[cfg(desktop)]
fn selection_menu_sync_enabled_for_scene<R: Runtime>(
    app: &AppHandle<R>,
    has_voxels: bool,
    has_selection: bool,
) {
    let Some(menu) = app.try_state::<SelectionMenuState>() else {
        return;
    };

    let apply = |item: &tauri::menu::MenuItem<tauri::Wry>, enabled: bool| {
        let _ = item.set_enabled(enabled);
    };

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

#[derive(serde::Deserialize, Clone)]
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
    let snap = state.selection_cells.lock().clone();
    *state.selection_stroke_before.lock() = Some(snap);
    *state.selection_stroke_accum.lock() = Some(AHashSet::new());
    Ok(())
}

#[tauri::command]
fn selection_stroke_end(app: AppHandle, state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
    // NOTE: Do NOT clear selection_stroke_accum here — a fire-and-forget
    // selection_stroke_at_screen invoke may still be in flight and needs the
    // accumulator.  The accum is overwritten by the next selection_stroke_begin.
    let before = state.selection_stroke_before.lock().take();
    let Some(before) = before else {
        return Ok(());
    };
    let after = state.selection_cells.lock().clone();
    if after == before {
        return Ok(());
    }
    push_solo_selection_undo_step(state.inner(), &app, before)
}

#[tauri::command]
async fn selection_stroke_at_screen(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: SelectionStrokeAtScreen,
) -> Result<u32, String> {
    {
        let cm = state.collab.lock();
        if cm.is_client() {
            return Ok(0);
        }
    }

    let (w, h) = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let interaction = args.interaction.as_str();

    let coords: Vec<greedy_mesh::VoxelCoord> =
        if matches!(args.stroke_mode, stroke_modes::DrawStrokeMode::Fill) {
            match interaction {
                "selectCoplanar" | "selectCoplanarEmpty" => {
                    let fg = state.current_file.lock();
                    let Some(file) = fg.as_ref() else {
                        return Err("no model loaded".into());
                    };
                    let vm = state.voxel_map.lock();
                    let Some(vmap) = vm.as_ref() else {
                        return Err("voxel index not ready".into());
                    };
                    let cam = state.camera.lock();
                    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
                    match interaction {
                        "selectCoplanar" => voxel_edit::coplanar_connected_from_screen(
                            file, vmap, &cam, w, h, sx, sy,
                        )
                        .unwrap_or_default(),
                        _ => voxel_edit::coplanar_empty_connected_from_screen(
                            file, vmap, &cam, w, h, sx, sy,
                        )
                        .unwrap_or_default(),
                    }
                }
                _ => {
                    state.fill_operation_cancel.store(false, Ordering::Relaxed);
                    emit_work_progress(&app, 0.08, "Selection fill…");
                    tokio::task::yield_now().await;
                    let state_cl = Arc::clone(state.inner());
                    let app_cl = app.clone();
                    let args_cl = args.clone();
                    let res = tokio::task::spawn_blocking(move || {
                        selection_fill_flood_coords_blocking(&state_cl, &app_cl, w, h, &args_cl)
                    })
                    .await
                    .map_err(|e| e.to_string())?;
                    let coords_inner = res.map_err(|e| {
                        emit_work_progress(&app, 1.0, "");
                        e
                    })?;
                    emit_work_progress(&app, 1.0, "");
                    coords_inner
                }
            }
        } else if interaction == "selectCoplanarEmpty" {
            let fg = state.current_file.lock();
            let Some(file) = fg.as_ref() else {
                return Err("no model loaded".into());
            };
            let vm = state.voxel_map.lock();
            let Some(vmap) = vm.as_ref() else {
                return Err("voxel index not ready".into());
            };
            let cam = state.camera.lock();
            let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
            let stroke_line_start = match (args.stroke_line_start_nx, args.stroke_line_start_ny) {
                (Some(lnx), Some(lny)) => Some(viewport_texels_from_norm(lnx, lny, w, h)),
                _ => None,
            };
            let stroke_segment_prev =
                match (args.stroke_segment_prev_nx, args.stroke_segment_prev_ny) {
                    (Some(pnx), Some(pny)) => Some(viewport_texels_from_norm(pnx, pny, w, h)),
                    _ => None,
                };
            let spray_cp = resolve_spray_constraint_plane(
                &state,
                &args.stroke_aux,
                args.stroke_mode,
                voxel_edit::EditTool::Add,
                file,
                vmap,
                &cam,
                w,
                h,
                sx,
                sy,
            );
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
                spray_cp,
            );
            voxel_edit::filter_coords_coplanar_empty_from_screen(file, vmap, &cam, w, h, sx, sy, &c)
        } else {
            let fg = state.current_file.lock();
            let Some(file) = fg.as_ref() else {
                return Err("no model loaded".into());
            };
            let vm = state.voxel_map.lock();
            let Some(vmap) = vm.as_ref() else {
                return Err("voxel index not ready".into());
            };
            let cam = state.camera.lock();
            let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
            let stroke_line_start = match (args.stroke_line_start_nx, args.stroke_line_start_ny) {
                (Some(lnx), Some(lny)) => Some(viewport_texels_from_norm(lnx, lny, w, h)),
                _ => None,
            };
            let stroke_segment_prev =
                match (args.stroke_segment_prev_nx, args.stroke_segment_prev_ny) {
                    (Some(pnx), Some(pny)) => Some(viewport_texels_from_norm(pnx, pny, w, h)),
                    _ => None,
                };
            let spray_cp = resolve_spray_constraint_plane(
                &state,
                &args.stroke_aux,
                args.stroke_mode,
                voxel_edit::EditTool::Remove,
                file,
                vmap,
                &cam,
                w,
                h,
                sx,
                sy,
            );
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
                spray_cp,
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

    let mode = *state.selection_combine_mode.lock();
    let mut accum_guard = state.selection_stroke_accum.lock();
    let before_guard = state.selection_stroke_before.lock();
    let mut sel = state.selection_cells.lock();

    let result =
        apply_selection_stroke_sample(&mut sel, coords, mode, &mut accum_guard, &before_guard);

    let n = result.unwrap_or(0);
    drop(sel);
    drop(before_guard);
    drop(accum_guard);

    if result.is_some() {
        emit_selection_updated(&app, state.inner());
    }
    Ok(n)
}

#[tauri::command]
fn selection_toggle_at_screen(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: PickAtScreen,
) -> Result<bool, String> {
    let (w, h) = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let maybe_coord: Option<greedy_mesh::VoxelCoord> = {
        let fg = state.current_file.lock();
        let Some(file) = fg.as_ref() else {
            return Err("no model loaded".into());
        };
        let vm = state.voxel_map.lock();
        let Some(vmap) = vm.as_ref() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        voxel_edit::pick_solid_coord_at_screen(file, vmap, &cam, w, h, sx, sy)
    };
    let Some(c) = maybe_coord else {
        return Ok(false);
    };
    let mode = *state.selection_combine_mode.lock();
    let mut sel = state.selection_cells.lock();
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

// ── Selection gizmo projection ────────────────────────────────────────────────

#[derive(serde::Serialize, Clone, Copy)]
#[serde(rename_all = "camelCase")]
struct GizmoProj {
    sx: f32,
    sy: f32,
    in_front: bool,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct SelectionGizmoProjected {
    center_sx: f32,
    center_sy: f32,
    /// [+X, −X, +Y, −Y, +Z, −Z] move handle tips
    move_handles: [GizmoProj; 6],
    /// 3 rings × 16 samples = 48 points (ring 0 = X-axis, 1 = Y-axis, 2 = Z-axis)
    rotate_rings: Vec<GizmoProj>,
    /// Pixels per one world unit at the selection center
    px_per_world: f32,
}

fn gizmo_proj_point(cam: &OrbitCamera, w: f32, h: f32, p: glam::Vec3, in_front: bool) -> GizmoProj {
    let (sx, sy) = voxel_edit::world_to_viewport_pixels(cam, w, h, p.x, p.y, p.z)
        .unwrap_or((-99999.0, -99999.0));
    GizmoProj { sx, sy, in_front }
}

/// Squared distance from point (px,py) to segment (ax,ay)→(bx,by) in screen space.
fn dist_sq_to_segment(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let dx = bx - ax;
    let dy = by - ay;
    let len_sq = dx * dx + dy * dy;
    if len_sq == 0.0 {
        return (px - ax).powi(2) + (py - ay).powi(2);
    }
    let t = ((px - ax) * dx + (py - ay) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);
    (px - (ax + t * dx)).powi(2) + (py - (ay + t * dy)).powi(2)
}

// CSS-pixel constants for gizmo interaction (multiplied by dpr when comparing physical px).
const GIZMO_MOVE_HIT_CSS: f32 = 16.0;
const GIZMO_RING_HIT_CSS: f32 = 11.0;
const GIZMO_PX_PER_MOVE_STEP_CSS: f32 = 26.0;
const GIZMO_PX_PER_ROTATE_STEP_CSS: f32 = 65.0;
const GIZMO_RING_SAMPLES: usize = 16;

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

/// Returns the pending visual offset for the selection during a move drag, or `(0,0,0)`.
fn pending_gizmo_translate(state: &ViewerState) -> (i32, i32, i32) {
    match &*state.selection_gizmo_drag.lock() {
        SelectionGizmoDrag::Move {
            pending_dx,
            pending_dy,
            pending_dz,
            ..
        } => (*pending_dx, *pending_dy, *pending_dz),
        _ => (0, 0, 0),
    }
}

/// Compute gizmo projected positions. Shared by `get_selection_gizmo_projected`,
/// `gizmo_pointer_down`, and `gizmo_hit_test`.
fn compute_gizmo_proj(state: &ViewerState) -> Option<SelectionGizmoProjected> {
    let sel = state.selection_cells.lock();
    if sel.is_empty() {
        return None;
    }
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
    let cx = (min_x + max_x) as f32 * 0.5;
    let cy = (min_y + max_y) as f32 * 0.5;
    let cz = (min_z + max_z) as f32 * 0.5;
    let center = glam::Vec3::new(cx, cy, cz);
    let (vw, vh) = {
        let v = state.viewer.lock();
        v.as_ref()
            .map(|vw| vw.viewport_size())
            .unwrap_or((512, 512))
    };
    let w = vw as f32;
    let h = vh as f32;
    let cam = state.camera.lock();
    let (center_sx, center_sy) = voxel_edit::world_to_viewport_pixels(&cam, w, h, cx, cy, cz)?;
    let inv_view = cam.view_matrix().inverse();
    let cam_eye = glam::Vec3::new(inv_view.w_axis.x, inv_view.w_axis.y, inv_view.w_axis.z);
    let dist = (cam_eye - center).length().max(1.0);
    let arm_world = (dist * 0.13).clamp(1.5, 20.0);
    let px_per_world = voxel_edit::world_to_viewport_pixels(&cam, w, h, cx + 1.0, cy, cz)
        .map(|(sx2, sy2)| (sx2 - center_sx).hypot(sy2 - center_sy).max(0.5))
        .unwrap_or(12.0);
    let in_front_dir = |dir: glam::Vec3| -> bool { (cam_eye - center).dot(dir) > 0.0 };
    let dirs = [
        glam::Vec3::X,
        -glam::Vec3::X,
        glam::Vec3::Y,
        -glam::Vec3::Y,
        glam::Vec3::Z,
        -glam::Vec3::Z,
    ];
    let mut move_handles = [GizmoProj {
        sx: 0.0,
        sy: 0.0,
        in_front: true,
    }; 6];
    for (i, &dir) in dirs.iter().enumerate() {
        move_handles[i] = gizmo_proj_point(&cam, w, h, center + dir * arm_world, in_front_dir(dir));
    }
    let ring_radius = arm_world * 0.72;
    let ring_defs = [
        (glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z),
        (glam::Vec3::Y, glam::Vec3::X, glam::Vec3::Z),
        (glam::Vec3::Z, glam::Vec3::X, glam::Vec3::Y),
    ];
    let mut rotate_rings = Vec::with_capacity(3 * GIZMO_RING_SAMPLES);
    for (_ring_axis, u, v) in &ring_defs {
        for i in 0..GIZMO_RING_SAMPLES {
            let angle = i as f32 * 2.0 * std::f32::consts::PI / GIZMO_RING_SAMPLES as f32;
            let offset = (*u * angle.cos() + *v * angle.sin()) * ring_radius;
            let in_f = (cam_eye - center).dot(offset) > 0.0;
            rotate_rings.push(gizmo_proj_point(&cam, w, h, center + offset, in_f));
        }
    }
    Some(SelectionGizmoProjected {
        center_sx,
        center_sy,
        move_handles,
        rotate_rings,
        px_per_world,
    })
}

/// Hit-test physical-pixel point (sx, sy) against the gizmo.
/// Returns `Some(SelectionGizmoDrag)` (never `None`) on hit.
fn gizmo_hit_test_inner(
    proj: &SelectionGizmoProjected,
    sx: f32,
    sy: f32,
    dpr: f32,
) -> Option<SelectionGizmoDrag> {
    let move_hit_sq = (GIZMO_MOVE_HIT_CSS * dpr).powi(2);
    let ring_hit_sq = (GIZMO_RING_HIT_CSS * dpr).powi(2);
    // Move handles
    for (i, h) in proj.move_handles.iter().enumerate() {
        if (sx - h.sx).powi(2) + (sy - h.sy).powi(2) <= move_hit_sq {
            let adx = h.sx - proj.center_sx;
            let ady = h.sy - proj.center_sy;
            let alen = adx.hypot(ady);
            let (axis_sx, axis_sy) = if alen > 0.5 {
                (adx / alen, ady / alen)
            } else {
                (1.0, 0.0)
            };
            return Some(SelectionGizmoDrag::Move {
                axis_sx,
                axis_sy,
                world_axis: (i / 2) as u8,
                positive: i % 2 == 0,
                accum: 0.0,
                step_threshold: GIZMO_PX_PER_MOVE_STEP_CSS * dpr,
                pending_dx: 0,
                pending_dy: 0,
                pending_dz: 0,
            });
        }
    }
    // Rotation rings
    for ring in 0..3u8 {
        let start = ring as usize * GIZMO_RING_SAMPLES;
        let mut best_sq = f32::INFINITY;
        let mut best_tx = 1.0f32;
        let mut best_ty = 0.0f32;
        for i in 0..GIZMO_RING_SAMPLES {
            let p = &proj.rotate_rings[start + i];
            let next = &proj.rotate_rings[start + (i + 1) % GIZMO_RING_SAMPLES];
            let sq = dist_sq_to_segment(sx, sy, p.sx, p.sy, next.sx, next.sy);
            if sq < best_sq {
                best_sq = sq;
                let tdx = next.sx - p.sx;
                let tdy = next.sy - p.sy;
                let tlen = tdx.hypot(tdy);
                if tlen > 0.5 {
                    best_tx = tdx / tlen;
                    best_ty = tdy / tlen;
                }
            }
        }
        if best_sq <= ring_hit_sq {
            return Some(SelectionGizmoDrag::Rotate {
                ring,
                tangent_x: best_tx,
                tangent_y: best_ty,
                accum: 0.0,
                step_threshold: GIZMO_PX_PER_ROTATE_STEP_CSS * dpr,
            });
        }
    }
    None
}

#[tauri::command]
fn gizmo_pointer_down(state: State<'_, Arc<ViewerState>>, sx: f32, sy: f32, dpr: f32) -> bool {
    let Some(proj) = compute_gizmo_proj(&state) else {
        return false;
    };
    let Some(drag) = gizmo_hit_test_inner(&proj, sx, sy, dpr) else {
        return false;
    };
    *state.selection_gizmo_drag.lock() = drag;
    true
}

#[tauri::command]
fn gizmo_pointer_move(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    dcx: f32,
    dcy: f32,
) -> Result<(), String> {
    let drag = state.selection_gizmo_drag.lock().clone();
    match drag {
        SelectionGizmoDrag::None => Ok(()),
        SelectionGizmoDrag::Move {
            axis_sx,
            axis_sy,
            world_axis,
            positive,
            mut accum,
            step_threshold,
            mut pending_dx,
            mut pending_dy,
            mut pending_dz,
        } => {
            accum += dcx * axis_sx + dcy * axis_sy;
            let steps = (accum / step_threshold).trunc() as i32;
            accum -= steps as f32 * step_threshold;
            if steps != 0 {
                let magnitude = if positive { steps } else { -steps };
                if world_axis == 0 {
                    pending_dx += magnitude;
                } else if world_axis == 1 {
                    pending_dy += magnitude;
                } else {
                    pending_dz += magnitude;
                }
                // Invalidate overlay so render loop rebuilds it at the new preview position.
                *state.selection_overlay_cache_key.lock() = None;
            }
            *state.selection_gizmo_drag.lock() = SelectionGizmoDrag::Move {
                axis_sx,
                axis_sy,
                world_axis,
                positive,
                accum,
                step_threshold,
                pending_dx,
                pending_dy,
                pending_dz,
            };
            Ok(())
        }
        SelectionGizmoDrag::Rotate {
            ring,
            tangent_x,
            tangent_y,
            mut accum,
            step_threshold,
        } => {
            accum += dcx * tangent_x + dcy * tangent_y;
            let steps = (accum / step_threshold).trunc() as i32;
            accum -= steps as f32 * step_threshold;
            *state.selection_gizmo_drag.lock() = SelectionGizmoDrag::Rotate {
                ring,
                tangent_x,
                tangent_y,
                accum,
                step_threshold,
            };
            if steps == 0 {
                return Ok(());
            }
            selection_rotate_inner(state.inner(), &app, ring, steps)?;
            Ok(())
        }
    }
}

#[tauri::command]
fn gizmo_pointer_up(state: State<'_, Arc<ViewerState>>, app: AppHandle) -> Result<(), String> {
    let drag = state.selection_gizmo_drag.lock().clone();
    // Clear drag state before the translate so the overlay fingerprint (which reads pending)
    // won't double-apply the offset after selection_cells is updated.
    *state.selection_gizmo_drag.lock() = SelectionGizmoDrag::None;
    *state.selection_overlay_cache_key.lock() = None;
    if let SelectionGizmoDrag::Move {
        pending_dx,
        pending_dy,
        pending_dz,
        ..
    } = drag
    {
        if pending_dx != 0 || pending_dy != 0 || pending_dz != 0 {
            selection_translate_inner(state.inner(), &app, pending_dx, pending_dy, pending_dz)?;
        }
    }
    Ok(())
}

#[tauri::command]
fn gizmo_hit_test(state: State<'_, Arc<ViewerState>>, sx: f32, sy: f32, dpr: f32) -> bool {
    let Some(proj) = compute_gizmo_proj(&state) else {
        state.hovered_gizmo_axis.store(255, Ordering::Relaxed);
        return false;
    };
    match gizmo_hit_test_inner(&proj, sx, sy, dpr) {
        Some(SelectionGizmoDrag::Move { world_axis, .. }) => {
            state
                .hovered_gizmo_axis
                .store(world_axis, Ordering::Relaxed);
            true
        }
        Some(SelectionGizmoDrag::Rotate { ring, .. }) => {
            state.hovered_gizmo_axis.store(ring, Ordering::Relaxed);
            true
        }
        Some(SelectionGizmoDrag::None) | None => {
            state.hovered_gizmo_axis.store(255, Ordering::Relaxed);
            false
        }
    }
}

#[tauri::command]
fn get_selection_gizmo_projected(
    state: State<'_, Arc<ViewerState>>,
) -> Option<SelectionGizmoProjected> {
    compute_gizmo_proj(&state)
}

// ── Selection transform commands ─────────────────────────────────────────────

/// Push interleaved SelectionBefore + VoxelDeltas onto the solo undo stack.
/// Clears redo. No-op if in collab mode (collab hosts/clients don't have
/// selection sync, so we just apply the voxel changes without solo undo).
fn push_selection_transform_undo(
    state: &Arc<ViewerState>,
    app: &AppHandle,
    before_sel: AHashSet<greedy_mesh::VoxelCoord>,
    deltas: Vec<voxel_edit::VoxelEditDelta>,
) {
    let cm = Arc::clone(&state.collab);
    let cb = cm.lock();
    if cb.is_client() || cb.is_host() {
        return;
    }
    drop(cb);
    state
        .solo_undo
        .lock()
        .push(SoloUndoEntry::SelectionTransform {
            before: before_sel,
            deltas,
        });
    state.solo_redo.lock().clear();
    #[cfg(target_os = "macos")]
    macos_undo::register_solo_edit_completed(app, state);
}

#[tauri::command]
fn selection_mirror(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    axis: u8,
) -> Result<bool, String> {
    let t_total = Instant::now();
    let before_sel = state.selection_cells.lock().clone();
    if before_sel.is_empty() {
        return Ok(false);
    }
    let new_sel = voxel_edit::mirror_selection_coords(&before_sel, axis);
    let deltas = {
        let mut fg = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        voxel_edit::mirror_selected_voxels(file, vmap, &before_sel, axis)
    };
    *state.selection_cells.lock() = new_sel;
    if !deltas.is_empty() {
        finish_voxel_edit_gpu_deltas(
            &state,
            &deltas,
            0.0,
            t_total,
            &app,
            VoxelGpuRefreshReason::SoloEdit,
        )?;
    }
    push_selection_transform_undo(state.inner(), &app, before_sel, deltas);
    emit_selection_updated(&app, state.inner());
    Ok(true)
}

fn selection_translate_inner(
    state: &Arc<ViewerState>,
    app: &AppHandle,
    dx: i32,
    dy: i32,
    dz: i32,
) -> Result<bool, String> {
    if dx == 0 && dy == 0 && dz == 0 {
        return Ok(false);
    }
    let t_total = Instant::now();
    let before_sel = state.selection_cells.lock().clone();
    if before_sel.is_empty() {
        return Ok(false);
    }
    let deltas = {
        let mut fg = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        voxel_edit::translate_selected_voxels(file, vmap, &before_sel, dx, dy, dz)
    };
    {
        let new_sel: AHashSet<greedy_mesh::VoxelCoord> = before_sel
            .iter()
            .map(|&(x, y, z)| (x + dx, y + dy, z + dz))
            .collect();
        *state.selection_cells.lock() = new_sel;
    }
    if !deltas.is_empty() {
        finish_voxel_edit_gpu_deltas(
            state,
            &deltas,
            0.0,
            t_total,
            app,
            VoxelGpuRefreshReason::SoloEdit,
        )?;
    }
    push_selection_transform_undo(state, app, before_sel, deltas);
    emit_selection_updated(app, state);
    Ok(true)
}

#[tauri::command]
fn selection_translate(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    dx: i32,
    dy: i32,
    dz: i32,
) -> Result<bool, String> {
    selection_translate_inner(state.inner(), &app, dx, dy, dz)
}

fn selection_rotate_inner(
    state: &Arc<ViewerState>,
    app: &AppHandle,
    axis: u8,
    quarters: i32,
) -> Result<bool, String> {
    let q = quarters.rem_euclid(4);
    if q == 0 {
        return Ok(false);
    }
    let t_total = Instant::now();
    let before_sel = state.selection_cells.lock().clone();
    if before_sel.is_empty() {
        return Ok(false);
    }
    let new_sel = voxel_edit::rotate_selection_coords(&before_sel, axis, quarters);
    let deltas = {
        let mut fg = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        voxel_edit::rotate_selected_voxels(file, vmap, &before_sel, axis, quarters)
    };
    *state.selection_cells.lock() = new_sel;
    if !deltas.is_empty() {
        finish_voxel_edit_gpu_deltas(
            state,
            &deltas,
            0.0,
            t_total,
            app,
            VoxelGpuRefreshReason::SoloEdit,
        )?;
    }
    push_selection_transform_undo(state, app, before_sel, deltas);
    emit_selection_updated(app, state);
    Ok(true)
}

#[tauri::command]
fn selection_rotate(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    axis: u8,
    quarters: i32,
) -> Result<bool, String> {
    selection_rotate_inner(state.inner(), &app, axis, quarters)
}

#[tauri::command]
fn selection_scale(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    factor: f64,
) -> Result<bool, String> {
    if factor <= 0.0 || (factor - 1.0).abs() < 1e-9 {
        return Ok(false);
    }
    let t_total = Instant::now();
    let before_sel = state.selection_cells.lock().clone();
    if before_sel.is_empty() {
        return Ok(false);
    }
    let new_sel = voxel_edit::scale_selection_coords(&before_sel, factor);
    let deltas = {
        let mut fg = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        voxel_edit::scale_selected_voxels(file, vmap, &before_sel, factor)
    };
    *state.selection_cells.lock() = new_sel;
    if !deltas.is_empty() {
        finish_voxel_edit_gpu_deltas(
            &state,
            &deltas,
            0.0,
            t_total,
            &app,
            VoxelGpuRefreshReason::SoloEdit,
        )?;
    }
    push_selection_transform_undo(state.inner(), &app, before_sel, deltas);
    emit_selection_updated(&app, state.inner());
    Ok(true)
}

#[tauri::command]
fn selection_clear(app: AppHandle, state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
    state.selection_cells.lock().clear();
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
        let sel = state.selection_cells.lock();
        if sel.is_empty() {
            return Ok(0);
        }
        sel.iter().copied().collect()
    };
    let t_apply_start = Instant::now();
    let deltas = {
        let mut fg = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
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

    let stroke_on = *state.stroke_active.lock();
    let n = deltas.len() as u32;
    if stroke_on {
        state.stroke_buffer.lock().extend(deltas.iter().copied());
        return Ok(n);
    }

    let cm = Arc::clone(&state.collab);
    let mut cb = cm.lock();
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
    Ok(state.selection_cells.lock().len() as u32)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct PaintSelectionArgs {
    color: u32,
    #[serde(default)]
    palette: Vec<u32>,
    #[serde(default)]
    paint_color_distrib: Option<paint_color_distrib::PaintColorDistrib>,
    #[serde(default)]
    stroke_seed: u32,
    material: String,
}

#[tauri::command]
fn paint_selection(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: PaintSelectionArgs,
) -> Result<u32, String> {
    let t_total = Instant::now();
    let coords: AHashSet<greedy_mesh::VoxelCoord> = {
        let sel = state.selection_cells.lock();
        if sel.is_empty() {
            return Ok(0);
        }
        sel.clone()
    };
    let color_resolver = build_color_resolver(
        args.color,
        args.palette,
        args.paint_color_distrib,
        args.stroke_seed,
    );
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let t_apply_start = Instant::now();
    let deltas = {
        let mut fg = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        voxel_edit::apply_edits_to_coords(
            file,
            vmap,
            voxel_edit::EditTool::Paint,
            color_resolver,
            material,
            &coords,
        )
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
    // Invalidate the selection overlay cache so it rebuilds with the new
    // voxel colors on the next frame.  The fingerprint only covers cell
    // coordinates + mesh_refresh_generation, so a paint-only edit (same
    // cells, same count) would otherwise leave the stale overlay visible.
    // Clear both the state-level key (checked in prepare) and the viewer-
    // level key (checked in apply) so neither layer short-circuits.
    *state.selection_overlay_cache_key.lock() = None;
    {
        let mut v = state.viewer.lock();
        if let Some(viewer) = v.as_mut() {
            viewer.selection_overlay_cache_key = None;
        }
    }
    push_solo_undo_step(state.inner(), &app, deltas.clone())?;
    Ok(deltas.len() as u32)
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
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let coords: Vec<greedy_mesh::VoxelCoord> = {
        let fg = state.current_file.lock();
        let Some(file) = fg.as_ref() else {
            return Err("no model loaded".into());
        };
        let vm = state.voxel_map.lock();
        let Some(vmap) = vm.as_ref() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        let Some(v) = voxel_edit::pick_voxel_at_screen(file, vmap, &cam, w, h, sx, sy) else {
            return Ok(0);
        };
        voxel_edit::coords_matching_color(file, v.color, args.match_material, v.material)
    };
    let mode = *state.selection_combine_mode.lock();
    let mut sel = state.selection_cells.lock();
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
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let coords: Vec<greedy_mesh::VoxelCoord> = {
        let fg = state.current_file.lock();
        let Some(file) = fg.as_ref() else {
            return Err("no model loaded".into());
        };
        let vm = state.voxel_map.lock();
        let Some(vmap) = vm.as_ref() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        let Some(c) = voxel_edit::coplanar_connected_from_screen(file, vmap, &cam, w, h, sx, sy)
        else {
            return Ok(0);
        };
        c
    };
    let mode = *state.selection_combine_mode.lock();
    let mut sel = state.selection_cells.lock();
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
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let coords: Vec<greedy_mesh::VoxelCoord> = {
        let fg = state.current_file.lock();
        let Some(file) = fg.as_ref() else {
            return Err("no model loaded".into());
        };
        let vm = state.voxel_map.lock();
        let Some(vmap) = vm.as_ref() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        let Some(c) =
            voxel_edit::coplanar_empty_connected_from_screen(file, vmap, &cam, w, h, sx, sy)
        else {
            return Ok(0);
        };
        c
    };
    let mode = *state.selection_combine_mode.lock();
    let mut sel = state.selection_cells.lock();
    merge_coords_into_selection(&mut sel, coords, mode);
    let n = sel.len() as u32;
    drop(sel);
    emit_selection_updated(&app, state.inner());
    Ok(n)
}

#[tauri::command]
fn selection_select_all(app: AppHandle, state: State<'_, Arc<ViewerState>>) -> Result<u32, String> {
    let vm = state.voxel_map.lock();
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let mut sel = state.selection_cells.lock();
    sel.clear();
    sel.extend(vmap.keys().copied());
    let n = sel.len() as u32;
    drop(sel);
    emit_selection_updated(&app, state.inner());
    Ok(n)
}

#[tauri::command]
fn selection_invert(app: AppHandle, state: State<'_, Arc<ViewerState>>) -> Result<u32, String> {
    let vm = state.voxel_map.lock();
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let all: AHashSet<_> = vmap.keys().copied().collect();
    let mut sel = state.selection_cells.lock();
    let new_sel: AHashSet<_> = all.difference(&sel).copied().collect();
    *sel = new_sel;
    let n = sel.len() as u32;
    drop(sel);
    emit_selection_updated(&app, state.inner());
    Ok(n)
}

#[tauri::command]
fn selection_grow(app: AppHandle, state: State<'_, Arc<ViewerState>>) -> Result<u32, String> {
    let grid_size = {
        let fg = state.current_file.lock();
        let Some(file) = fg.as_ref() else {
            return Err("no model loaded".into());
        };
        file.grid_size.max(1)
    };
    let vm = state.voxel_map.lock();
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let mut sel = state.selection_cells.lock();
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
    let grid_size = {
        let fg = state.current_file.lock();
        let Some(file) = fg.as_ref() else {
            return Err("no model loaded".into());
        };
        file.grid_size.max(1)
    };
    let mut sel = state.selection_cells.lock();
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
    let mut sel = state.selection_cells.lock();
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
    let vm = state.voxel_map.lock();
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let mut sel = state.selection_cells.lock();
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
    let vm = state.voxel_map.lock();
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let mut sel = state.selection_cells.lock();
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
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let fg = state.current_file.lock();
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };
    let vm = state.voxel_map.lock();
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let cam = state.camera.lock();
    let mm = *state.selection_match_material.lock();
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let Some(coords) =
        voxel_edit::connected_solid_same_color_from_screen(file, vmap, &cam, w, h, sx, sy, mm)
    else {
        return Ok(0);
    };
    let mode = *state.selection_combine_mode.lock();
    let mut sel = state.selection_cells.lock();
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
    let (nx, ny) = (*state.preview_cursor.lock())
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
    *state.selection_combine_mode.lock() = mode;
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
    Ok(*state.selection_combine_mode.lock())
}

#[derive(serde::Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct VoxelFillAtScreen {
    nx: f32,
    ny: f32,
    color: u32,
    material: String,
    match_material: bool,
}

#[tauri::command]
fn voxel_fill_cancel(state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
    state.fill_operation_cancel.store(true, Ordering::Relaxed);
    Ok(())
}

fn run_voxel_fill_paint_blocking(
    state: &Arc<ViewerState>,
    app: &AppHandle,
    w: f32,
    h: f32,
    args: &VoxelFillAtScreen,
    material: voxelle::MaterialId,
) -> Result<Vec<voxel_edit::VoxelEditDelta>, String> {
    let cancel = state.fill_operation_cancel.as_ref();
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let mut fg = state.current_file.lock();
    let mut vm = state.voxel_map.lock();
    let Some(file) = fg.as_mut() else {
        return Err(String::from("no model loaded"));
    };
    let Some(vmap) = vm.as_mut() else {
        return Err(String::from("voxel index not ready"));
    };
    let cam = state.camera.lock();
    let large = voxel_edit::flood_fill_selection_region_exceeds_threshold(
        file,
        vmap,
        &cam,
        w,
        h,
        sx,
        sy,
        false,
        true,
        args.match_material,
        false,
        stroke_modes::PlaneAxis::Auto,
        voxel_edit::FILL_UNCONSTRAINED_LARGE_THRESHOLD,
        Some(cancel),
    )
    .map_err(|_| String::from("fill cancelled"))?;
    if large {
        emit_work_progress(app, 0.12, "Large fill — exploring… (Escape to cancel)");
    }
    let app_pb = app.clone();
    let mut progress_ticks: usize = 0;
    let mut on_progress = move |n: usize| {
        emit_work_progress(&app_pb, 0.25, format!("Fill — {n} cells"));
        progress_ticks = progress_ticks.wrapping_add(1);
        if progress_ticks % 4 == 0 {
            std::thread::yield_now();
        }
    };
    let fill_color = args.color;
    let o = voxel_edit::flood_fill_paint_at_screen(
        file,
        vmap,
        &cam,
        w,
        h,
        sx,
        sy,
        move |_, _, _| fill_color,
        material,
        args.match_material,
        false,
        true,
        false,
        stroke_modes::PlaneAxis::Auto,
        Some(cancel),
        &mut on_progress,
    )?;
    if o.cancelled {
        return Err(String::from("fill cancelled"));
    }
    if o.hit_absolute_cap {
        return Err(String::from(
            "fill region too large — constrain to a plane or reduce scope",
        ));
    }
    Ok(o.deltas)
}

fn run_fill_deltas_blocking(
    state: &Arc<ViewerState>,
    app: &AppHandle,
    w: f32,
    h: f32,
    args: &VoxelEditAtScreen,
    material: voxelle::MaterialId,
) -> Result<Vec<voxel_edit::VoxelEditDelta>, String> {
    let cancel = state.fill_operation_cancel.as_ref();
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let unconstrained = !args.stroke_aux.constrain_to_plane;

    let mut fg = state.current_file.lock();
    let mut vm = state.voxel_map.lock();
    let Some(file) = fg.as_mut() else {
        return Err("no model loaded".into());
    };
    let Some(vmap) = vm.as_mut() else {
        return Err("voxel index not ready".into());
    };
    let cam = state.camera.lock();

    if unconstrained {
        let large = match args.tool {
            voxel_edit::EditTool::Add => voxel_edit::flood_fill_empty_region_exceeds_threshold(
                file,
                vmap,
                &cam,
                w,
                h,
                sx,
                sy,
                args.fill_select_diagonals,
                false,
                args.plane_axis,
                voxel_edit::FILL_UNCONSTRAINED_LARGE_THRESHOLD,
                Some(cancel),
            ),
            _ => voxel_edit::flood_fill_selection_region_exceeds_threshold(
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
                false,
                args.plane_axis,
                voxel_edit::FILL_UNCONSTRAINED_LARGE_THRESHOLD,
                Some(cancel),
            ),
        }
        .map_err(|_| "fill cancelled".to_string())?;
        if large {
            emit_work_progress(app, 0.12, "Large fill — exploring… (Escape to cancel)");
        }
    }

    let app_pb = app.clone();
    let mut progress_ticks: usize = 0;
    let mut on_progress = move |n: usize| {
        emit_work_progress(&app_pb, 0.25, format!("Fill — {n} cells"));
        progress_ticks = progress_ticks.wrapping_add(1);
        if progress_ticks % 4 == 0 {
            std::thread::yield_now();
        }
    };

    let outcome = match args.tool {
        voxel_edit::EditTool::Paint => voxel_edit::flood_fill_paint_at_screen(
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
            build_color_resolver(
                args.color,
                args.palette.clone(),
                args.paint_color_distrib.clone(),
                args.stroke_seed,
            ),
            material,
            args.match_material,
            args.fill_select_diagonals,
            args.fill_respects_color,
            args.stroke_aux.constrain_to_plane,
            args.plane_axis,
            Some(cancel),
            &mut on_progress,
        )?,
        voxel_edit::EditTool::Remove => voxel_edit::flood_fill_remove_at_screen(
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
            args.stroke_aux.constrain_to_plane,
            args.plane_axis,
            Some(cancel),
            &mut on_progress,
        )?,
        voxel_edit::EditTool::Add => voxel_edit::flood_fill_empty_at_screen(
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
            args.fill_select_diagonals,
            build_color_resolver(
                args.color,
                args.palette.clone(),
                args.paint_color_distrib.clone(),
                args.stroke_seed,
            ),
            material,
            args.stroke_aux.constrain_to_plane,
            args.plane_axis,
            Some(cancel),
            &mut on_progress,
        )?,
    };

    if outcome.cancelled {
        return Err("fill cancelled".into());
    }
    if outcome.hit_absolute_cap {
        return Err("fill region too large — constrain to a plane or reduce scope".into());
    }
    Ok(outcome.deltas)
}

fn selection_fill_flood_coords_blocking(
    state: &Arc<ViewerState>,
    app: &AppHandle,
    w: f32,
    h: f32,
    args: &SelectionStrokeAtScreen,
) -> Result<Vec<greedy_mesh::VoxelCoord>, String> {
    let cancel = state.fill_operation_cancel.as_ref();
    let fg = state.current_file.lock();
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };
    let vm = state.voxel_map.lock();
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let cam = state.camera.lock();
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let interaction = args.interaction.as_str();

    let unconstrained = !args.stroke_aux.constrain_to_plane;
    if unconstrained {
        let large = voxel_edit::flood_fill_selection_region_exceeds_threshold(
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
            false,
            args.plane_axis,
            voxel_edit::FILL_UNCONSTRAINED_LARGE_THRESHOLD,
            Some(cancel),
        )
        .map_err(|_| "fill cancelled".to_string())?;
        if large {
            emit_work_progress(
                app,
                0.12,
                "Large selection fill — exploring… (Escape to cancel)",
            );
        }
    }

    let app_pb = app.clone();
    let mut progress_ticks: usize = 0;
    let mut on_progress = move |n: usize| {
        emit_work_progress(&app_pb, 0.25, format!("Selection fill — {n} cells"));
        progress_ticks = progress_ticks.wrapping_add(1);
        if progress_ticks % 4 == 0 {
            std::thread::yield_now();
        }
    };

    let o = voxel_edit::flood_fill_selection_coords_with_control(
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
        args.stroke_aux.constrain_to_plane,
        args.plane_axis,
        Some(cancel),
        &mut on_progress,
    );

    if o.cancelled {
        return Ok(Vec::new());
    }
    if o.hit_absolute_cap {
        return Err("fill region too large — constrain to a plane or reduce scope".into());
    }

    let mut c = o.coords;
    if interaction == "selectByColor" {
        if let Some(seed) = voxel_edit::pick_voxel_at_screen(file, vmap, &cam, w, h, sx, sy) {
            c = voxel_edit::filter_coords_by_seed_color(file, vmap, &c, seed, args.match_material);
        } else {
            c.clear();
        }
    }
    Ok(c)
}

#[tauri::command]
async fn voxel_fill_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: VoxelFillAtScreen,
) -> Result<bool, String> {
    let t_total = Instant::now();
    let (w, h) = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let material = voxelle::MaterialId::from_str_id(&args.material);
    state.fill_operation_cancel.store(false, Ordering::Relaxed);
    emit_work_progress(&app, 0.08, "Fill…");
    tokio::task::yield_now().await;
    let state_cl = Arc::clone(state.inner());
    let app_cl = app.clone();
    let args_cl = args.clone();
    let deltas_result = tokio::task::spawn_blocking(move || {
        run_voxel_fill_paint_blocking(&state_cl, &app_cl, w, h, &args_cl, material)
    })
    .await
    .map_err(|e| e.to_string())?;
    let deltas = deltas_result.map_err(|e| {
        emit_work_progress(&app, 1.0, "");
        e
    })?;
    if deltas.is_empty() {
        emit_work_progress(&app, 1.0, "");
        return Ok(false);
    }
    tokio::task::yield_now().await;
    finish_voxel_edit_gpu_deltas(
        &state,
        &deltas,
        0.0,
        t_total,
        &app,
        VoxelGpuRefreshReason::SoloEdit,
    )?;
    emit_work_progress(&app, 1.0, "");
    let cm = Arc::clone(&state.collab);
    let mut cb = cm.lock();
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
    let fg = state.current_file.lock();
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };
    let vm = state.voxel_map.lock();
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let sel = state.selection_cells.lock();
    let Some(clip) = voxel_edit::selection_to_clipboard(file, vmap, &sel) else {
        return Ok(false);
    };
    *state.stamp_clipboard.lock() = Some(clip);
    Ok(true)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct StampPickAtScreen {
    nx: f32,
    ny: f32,
    rot_x: f32,
    rot_y: f32,
    rot_z: f32,
    #[serde(default)]
    origin_x: i32,
    #[serde(default)]
    origin_z: i32,
}

#[tauri::command]
fn clipboard_stamp_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: StampPickAtScreen,
) -> Result<bool, String> {
    let clip = state.stamp_clipboard.lock().clone();
    let Some(clip) = clip else {
        return Ok(false);
    };
    let deltas = {
        let (w, h) = {
            let v = state.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock();
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
            args.rot_x,
            args.rot_y,
            args.rot_z,
            args.origin_x,
            args.origin_z,
        )?
    };
    commit_voxel_edits(&state, &app, deltas)
}

#[tauri::command]
fn clipboard_punch_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: StampPickAtScreen,
) -> Result<bool, String> {
    let clip = state.stamp_clipboard.lock().clone();
    let Some(clip) = clip else {
        return Ok(false);
    };
    let deltas = {
        let (w, h) = {
            let v = state.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        voxel_edit::punch_clipboard_at_screen(
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
            &clip,
            args.rot_x,
            args.rot_y,
            args.rot_z,
            args.origin_x,
            args.origin_z,
        )?
    };
    commit_voxel_edits(&state, &app, deltas)
}

/// Return the dominant axis of the face normal at the given screen position for stamp rotation.
/// Returns `[nx, ny, nz]` (one component ±1, rest 0) or `null` if no solid is hit.
#[tauri::command]
fn stamp_face_normal_at_screen(
    state: State<'_, Arc<ViewerState>>,
    args: PickAtScreen,
) -> Result<Option<[i32; 3]>, String> {
    let (w, h) = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Ok(None);
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let fg = state.current_file.lock();
    let Some(file) = fg.as_ref() else {
        return Ok(None);
    };
    let vm = state.voxel_map.lock();
    let Some(vmap) = vm.as_ref() else {
        return Ok(None);
    };
    let cam = state.camera.lock();
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let normal = voxel_edit::outward_face_normal_from_screen_ray(file, vmap, &cam, w, h, sx, sy);
    Ok(normal.map(|(x, y, z)| [x, y, z]))
}

#[tauri::command]
fn get_selection_as_stamp_entries(
    state: State<'_, Arc<ViewerState>>,
) -> Result<Vec<(i32, i32, i32, u32, String)>, String> {
    let fg = state.current_file.lock();
    let Some(file) = fg.as_ref() else {
        return Ok(vec![]);
    };
    let vm = state.voxel_map.lock();
    let Some(vmap) = vm.as_ref() else {
        return Ok(vec![]);
    };
    let sel = state.selection_cells.lock();
    let Some(clip) = voxel_edit::selection_to_clipboard(file, vmap, &sel) else {
        return Ok(vec![]);
    };
    Ok(clip
        .entries
        .into_iter()
        .map(|(x, y, z, c, m)| (x, y, z, c, m.as_str_id().to_string()))
        .collect())
}

#[derive(serde::Deserialize)]
struct StampBookLoadEntry {
    dx: i32,
    dy: i32,
    dz: i32,
    color: u32,
    material: String,
}

#[tauri::command]
fn stamp_book_load_entries(
    state: State<'_, Arc<ViewerState>>,
    entries: Vec<StampBookLoadEntry>,
) -> Result<(), String> {
    use voxelle::MaterialId;
    if entries.is_empty() {
        return Err("no entries".into());
    }
    let clip_entries: Vec<(i32, i32, i32, u32, MaterialId)> = entries
        .into_iter()
        .map(|e| {
            (
                e.dx,
                e.dy,
                e.dz,
                e.color,
                MaterialId::from_str_id(&e.material),
            )
        })
        .collect();
    *state.stamp_clipboard.lock() = Some(voxel_edit::StampClipboard {
        entries: clip_entries,
    });
    Ok(())
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
            let v = state.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        voxel_edit::sculpt_raise_at_screen(file, vmap, &cam, w, h, sx, sy, args.color, material)?
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
            let v = state.viewer.lock();
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
        let mut fg = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        let wall_poly = args.wall_polygon_vertices.as_ref().map(|v| {
            v.iter()
                .map(|a| (a[0], a[1], a[2]))
                .collect::<Vec<greedy_mesh::VoxelCoord>>()
        });
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
            args.brush_clip_bottom_half,
            line,
            seg,
            args.terrain_op,
            args.terrain_base_y,
            args.terrain_strength,
            args.terrain_smooth_radius,
            args.terrain_flatten_use_base_y,
            args.terrain_sub_voxel,
            &mut *state.terrain_accum.lock(),
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
            args.sculpt_smooth_variant,
            args.smooth_neighbor_radius,
            args.smooth_aggressiveness,
            args.smooth_laplacian_iterations,
            args.smooth_laplacian_relax_pct,
            wall_poly,
            args.extrude_profile,
            args.extrude_end_cap,
            args.extrude_taper,
            args.extrude_taper_start,
            args.extrude_taper_end,
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
    let stroke_on = *state.stroke_active.lock();
    if stroke_on {
        state.stroke_buffer.lock().extend(deltas.iter().copied());
        return Ok(true);
    }
    let cm = Arc::clone(&state.collab);
    let mut cb = cm.lock();
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
    // Fresh accumulator for replay — produces identical results to the original stroke.
    let mut replay_terrain_accum: AHashMap<(i32, i32), f32> = AHashMap::new();
    for args in replay {
        let material = voxelle::MaterialId::from_str_id(&args.material);
        let deltas = {
            let (w, h) = {
                let v = state.viewer.lock();
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
            let mut fg = state.current_file.lock();
            let mut vm = state.voxel_map.lock();
            let Some(file) = fg.as_mut() else {
                return Ok(());
            };
            let Some(vmap) = vm.as_mut() else {
                return Ok(());
            };
            let cam = state.camera.lock();
            let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
            let wall_poly = args.wall_polygon_vertices.as_ref().map(|v| {
                v.iter()
                    .map(|a| (a[0], a[1], a[2]))
                    .collect::<Vec<greedy_mesh::VoxelCoord>>()
            });
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
                args.brush_clip_bottom_half,
                line,
                seg,
                args.terrain_op,
                args.terrain_base_y,
                args.terrain_strength,
                args.terrain_smooth_radius,
                args.terrain_flatten_use_base_y,
                args.terrain_sub_voxel,
                &mut replay_terrain_accum,
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
                args.sculpt_smooth_variant,
                args.smooth_neighbor_radius,
                args.smooth_aggressiveness,
                args.smooth_laplacian_iterations,
                args.smooth_laplacian_relax_pct,
                wall_poly,
                args.extrude_profile,
                args.extrude_end_cap,
                args.extrude_taper,
                args.extrude_taper_start,
                args.extrude_taper_end,
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
        let cm = state.collab.lock();
        if cm.is_client() {
            return Ok(());
        }
    }

    let stroke_line_start_meta = match (args.stroke_line_start_nx, args.stroke_line_start_ny) {
        (Some(_), Some(_)) => Some((0.0_f32, 0.0_f32)),
        _ => None,
    };

    state.sculpt_stroke_replay.lock().push(args.clone());

    let footprint = {
        let fg = state.current_file.lock();
        let vm = state.voxel_map.lock();
        let Some(file) = fg.as_ref() else {
            return Ok(());
        };
        let Some(vmap) = vm.as_ref() else {
            return Ok(());
        };
        let cam = state.camera.lock();
        let (w, h) = {
            let v = state.viewer.lock();
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
            let wall_poly_vec: Option<Vec<greedy_mesh::VoxelCoord>> = args
                .wall_polygon_vertices
                .as_ref()
                .map(|v| v.iter().map(|a| (a[0], a[1], a[2])).collect());
            // Lock the face normal on the first drag frame so the wall orientation stays
            // constant for the entire stroke. During hover (stroke_active = false) always
            // recompute so the preview tracks the surface under the cursor.
            let stroke_on = *state.stroke_active.lock();
            let locked_face = if stroke_on {
                let mut lock = state.wall_stroke_face_snapped.lock();
                if lock.is_none() {
                    let face_out = voxel_edit::outward_face_normal_from_screen_ray(
                        file, vmap, &cam, w, h, sx, sy,
                    );
                    *lock = Some(face_out.map(voxel_edit::snap_normal_to_axis));
                }
                *lock
            } else {
                None
            };
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
                wall_poly_vec.as_deref(),
                args.spray_direction,
                args.wall_width_index,
                args.wall_height_vox,
                args.wall_lock_start_height,
                args.wall_axis_align,
                args.brush_radius,
                args.brush_falloff,
                args.brush_strength,
                args.stroke_seed,
                locked_face,
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
                args.brush_clip_bottom_half,
                args.extrude_profile,
                args.extrude_end_cap,
                args.extrude_taper,
                args.extrude_taper_start,
                args.extrude_taper_end,
            )
        }
    };

    {
        let mut union = state.stroke_preview_union.lock();
        // Sculpt Draw/Gouge/Extrude: always accumulate so the full stroke footprint is
        // available at pointer-up for single-batch commit (avoids frame-by-frame stacking).
        let sculpt_accumulate = matches!(
            args.sculpt_mode,
            voxel_edit::SculptStrokeMode::Draw
                | voxel_edit::SculptStrokeMode::Gouge
                | voxel_edit::SculptStrokeMode::Extrude
        );
        let accumulate = sculpt_accumulate
            || voxel_edit::stroke_preview_accumulates_samples(
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

    // Palette-colored preview for every sculpt mode (gouge/smooth used to use Remove red).
    let instanced = {
        let fg = state.current_file.lock();
        let vm = state.voxel_map.lock();
        let union = state.stroke_preview_union.lock();
        let Some(file) = fg.as_ref() else {
            return Ok(());
        };
        let Some(vmap) = vm.as_ref() else {
            return Ok(());
        };
        stroke_preview_meshes_for_union(
            voxel_edit::EditTool::Add,
            &union,
            vmap,
            file,
            false,
            args.color,
            None,
        )
    };

    {
        let mut v = state.viewer.lock();
        let Some(viewer) = v.as_mut() else {
            return Ok(());
        };
        if instanced.solid_instances.is_empty() {
            clear_preview_mesh_sync_cache(viewer, state.inner().as_ref());
            state
                .stroke_preview_suppresses_hover
                .store(false, Ordering::Relaxed);
        } else {
            viewer.upload_preview_mesh_instanced(&instanced);
            viewer.preview_cache_key = None;
            *state.preview_overlay_cache_key.lock() = None;
            state
                .stroke_preview_suppresses_hover
                .store(true, Ordering::Relaxed);
        }
    }

    wake_viewport_loop(&app);
    Ok(())
}

// ── Extrude ray-based preview (straight-line extrude matching web) ─────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExtrudeRayPreviewArgs {
    /// Screen position of pointer down (normalized 0..1).
    start_nx: f32,
    start_ny: f32,
    /// Screen drag delta in physical pixels (right = +dx, up = +dy).
    screen_dx: f32,
    screen_dy: f32,
    /// Direction reference mode.
    direction_ref: voxel_edit::ExtrudeDirectionRef,
    // Extrude shape params:
    color: u32,
    material: String,
    brush_radius: u32,
    #[serde(default)]
    brush_shape: voxel_edit::BrushShape,
    #[serde(default = "default_brush_strength_sculpt")]
    brush_strength: u32,
    #[serde(default)]
    brush_falloff: u32,
    #[serde(default)]
    stroke_seed: u32,
    #[serde(default)]
    extrude_profile: voxel_edit::ExtrudeProfile,
    #[serde(default)]
    extrude_end_cap: voxel_edit::ExtrudeEndCap,
    #[serde(default)]
    extrude_taper: bool,
    #[serde(default)]
    extrude_taper_start: f32,
    #[serde(default)]
    extrude_taper_end: f32,
}

#[tauri::command]
fn extrude_ray_preview(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: ExtrudeRayPreviewArgs,
) -> Result<(), String> {
    {
        let cm = state.collab.lock();
        if cm.is_client() {
            return Ok(());
        }
    }

    let (w, h) = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Ok(());
        };
        viewer.viewport_size()
    };
    let w = w as f32;
    let h = h as f32;

    // Raycast from start position to find add-position + face normal.
    let (start_sx, start_sy) = viewport_texels_from_norm(args.start_nx, args.start_ny, w, h);
    let (start_coord, face_normal) = {
        let fg = state.current_file.lock();
        let vm = state.voxel_map.lock();
        let Some(file) = fg.as_ref() else {
            return Ok(());
        };
        let Some(vmap) = vm.as_ref() else {
            return Ok(());
        };
        let cam = state.camera.lock();
        match voxel_edit::pick_extrude_start(file, vmap, &cam, w, h, start_sx, start_sy) {
            Some(v) => v,
            None => return Ok(()),
        }
    };

    // Resolve extrusion direction from screen drag + camera.
    let cam = state.camera.lock();
    let direction = voxel_edit::resolve_extrude_direction(
        args.direction_ref,
        &cam,
        args.screen_dx,
        args.screen_dy,
        face_normal,
    );
    drop(cam);

    // Compute length from drag distance (matches web: sqrt(dx²+dy²) / 6).
    let drag_dist = (args.screen_dx * args.screen_dx + args.screen_dy * args.screen_dy).sqrt();
    let length = (drag_dist / 6.0).round().max(0.0) as u32;

    // Generate straight-line spine.
    let spine = voxel_edit::get_ray_direction_path(start_coord, direction, length);

    // Compute footprint.
    let footprint = voxel_edit::extrude_ray_footprint(
        &spine,
        args.brush_radius,
        args.brush_shape,
        args.brush_strength,
        args.brush_falloff,
        args.stroke_seed,
        args.extrude_profile,
        args.extrude_end_cap,
        args.extrude_taper,
        args.extrude_taper_start,
        args.extrude_taper_end,
    );

    // Store spine for recompute.
    *state.extrude_ray_spine.lock() = Some(spine);

    // Store a synthetic sculpt replay entry so voxel_stroke_end recognizes this as an extrude
    // and commits from the preview union.
    {
        let mut replay = state.sculpt_stroke_replay.lock();
        if replay.is_empty() {
            replay.push(SculptStrokeAtScreenArgs {
                nx: args.start_nx,
                ny: args.start_ny,
                sculpt_mode: voxel_edit::SculptStrokeMode::Extrude,
                color: args.color,
                material: args.material.clone(),
                brush_radius: args.brush_radius,
                brush_shape: args.brush_shape,
                spray_density: 0.0,
                brush_clip_bottom_half: false,
                stroke_line_start_nx: None,
                stroke_line_start_ny: None,
                stroke_segment_prev_nx: None,
                stroke_segment_prev_ny: None,
                terrain_op: None,
                terrain_base_y: 0,
                terrain_strength: 50,
                terrain_smooth_radius: 0,
                terrain_flatten_use_base_y: false,
                terrain_sub_voxel: false,
                smooth_neighbor_passes: 1,
                brush_strength: args.brush_strength,
                brush_falloff: args.brush_falloff,
                stroke_seed: args.stroke_seed,
                wall_area_shape: Default::default(),
                spray_direction: Default::default(),
                wall_width_index: 0,
                wall_height_vox: 2,
                wall_lock_start_height: false,
                wall_axis_align: false,
                sculpt_smooth_variant: Default::default(),
                smooth_neighbor_radius: 0,
                smooth_aggressiveness: 100,
                smooth_laplacian_iterations: 4,
                smooth_laplacian_relax_pct: 50,
                wall_polygon_vertices: None,
                extrude_profile: args.extrude_profile,
                extrude_end_cap: args.extrude_end_cap,
                extrude_taper: args.extrude_taper,
                extrude_taper_start: args.extrude_taper_start,
                extrude_taper_end: args.extrude_taper_end,
            });
        } else {
            // Update the existing replay entry with latest extrude params.
            let entry = &mut replay[0];
            entry.extrude_profile = args.extrude_profile;
            entry.extrude_end_cap = args.extrude_end_cap;
            entry.extrude_taper = args.extrude_taper;
            entry.extrude_taper_start = args.extrude_taper_start;
            entry.extrude_taper_end = args.extrude_taper_end;
        }
    }

    // Replace preview union entirely (not accumulate — full recompute each move).
    {
        let mut union = state.stroke_preview_union.lock();
        union.clear();
        for c in &footprint {
            union.insert(*c);
        }
    }

    state
        .stroke_preview_suppresses_hover
        .store(true, Ordering::Relaxed);

    // Generate and upload preview mesh.
    let instanced = {
        let fg = state.current_file.lock();
        let vm = state.voxel_map.lock();
        let Some(file) = fg.as_ref() else {
            return Ok(());
        };
        let Some(vmap) = vm.as_ref() else {
            return Ok(());
        };
        let union = state.stroke_preview_union.lock();
        stroke_preview_meshes_for_union(
            voxel_edit::EditTool::Add,
            &union,
            vmap,
            file,
            false,
            args.color,
            None,
        )
    };

    {
        let mut v = state.viewer.lock();
        let Some(viewer) = v.as_mut() else {
            return Ok(());
        };
        if instanced.solid_instances.is_empty() {
            clear_preview_mesh_sync_cache(viewer, state.inner().as_ref());
        } else {
            viewer.upload_preview_mesh_instanced(&instanced);
            viewer.preview_cache_key = None;
            *state.preview_overlay_cache_key.lock() = None;
        }
    }

    wake_viewport_loop(&app);
    Ok(())
}

// ── Selection extrude preview ─────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SelectionExtrudePreviewArgs {
    screen_dx: f32,
    screen_dy: f32,
    direction_ref: voxel_edit::ExtrudeDirectionRef,
    color: u32,
    material: String,
}

#[tauri::command]
fn selection_extrude_preview(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: SelectionExtrudePreviewArgs,
) -> Result<(), String> {
    {
        let cm = state.collab.lock();
        if cm.is_client() {
            return Ok(());
        }
    }

    let selection: ahash::AHashSet<greedy_mesh::VoxelCoord> = state.selection_cells.lock().clone();
    if selection.is_empty() {
        return Ok(());
    }

    let (w, h) = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Ok(());
        };
        viewer.viewport_size()
    };
    let w = w as f32;
    let h = h as f32;
    let _ = (w, h); // viewport size not needed for direction resolution

    let direction = {
        let cam = state.camera.lock();
        voxel_edit::resolve_extrude_direction(
            args.direction_ref,
            &cam,
            args.screen_dx,
            args.screen_dy,
            None,
        )
    };

    let drag_dist = (args.screen_dx * args.screen_dx + args.screen_dy * args.screen_dy).sqrt();
    let length = (drag_dist / 6.0).round().max(0.0) as u32;

    let footprint = voxel_edit::extrude_selection_footprint(&selection, direction, length);

    // Replace preview union.
    {
        let mut union = state.stroke_preview_union.lock();
        union.clear();
        for c in &footprint {
            union.insert(*c);
        }
    }

    // Store a synthetic sculpt replay entry so voxel_stroke_end knows to commit from the union.
    {
        let mut replay = state.sculpt_stroke_replay.lock();
        if replay.is_empty() {
            replay.push(SculptStrokeAtScreenArgs {
                nx: 0.5,
                ny: 0.5,
                sculpt_mode: voxel_edit::SculptStrokeMode::Extrude,
                color: args.color,
                material: args.material.clone(),
                brush_radius: 0,
                brush_shape: Default::default(),
                spray_density: 0.0,
                brush_clip_bottom_half: false,
                stroke_line_start_nx: None,
                stroke_line_start_ny: None,
                stroke_segment_prev_nx: None,
                stroke_segment_prev_ny: None,
                terrain_op: None,
                terrain_base_y: 0,
                terrain_strength: 50,
                terrain_smooth_radius: 0,
                terrain_flatten_use_base_y: false,
                terrain_sub_voxel: false,
                smooth_neighbor_passes: 1,
                brush_strength: 100,
                brush_falloff: 100,
                stroke_seed: 0,
                wall_area_shape: Default::default(),
                spray_direction: Default::default(),
                wall_width_index: 0,
                wall_height_vox: 2,
                wall_lock_start_height: false,
                wall_axis_align: false,
                sculpt_smooth_variant: Default::default(),
                smooth_neighbor_radius: 0,
                smooth_aggressiveness: 100,
                smooth_laplacian_iterations: 4,
                smooth_laplacian_relax_pct: 50,
                wall_polygon_vertices: None,
                extrude_profile: Default::default(),
                extrude_end_cap: Default::default(),
                extrude_taper: false,
                extrude_taper_start: 0.0,
                extrude_taper_end: 0.0,
            });
        } else {
            let entry = &mut replay[0];
            entry.color = args.color;
            entry.material = args.material.clone();
        }
    }

    state
        .stroke_preview_suppresses_hover
        .store(true, Ordering::Relaxed);

    // Generate and upload preview mesh.
    let instanced = {
        let fg = state.current_file.lock();
        let vm = state.voxel_map.lock();
        let Some(file) = fg.as_ref() else {
            return Ok(());
        };
        let Some(vmap) = vm.as_ref() else {
            return Ok(());
        };
        let union = state.stroke_preview_union.lock();
        stroke_preview_meshes_for_union(
            voxel_edit::EditTool::Add,
            &union,
            vmap,
            file,
            false,
            args.color,
            None,
        )
    };

    {
        let mut v = state.viewer.lock();
        let Some(viewer) = v.as_mut() else {
            return Ok(());
        };
        if instanced.solid_instances.is_empty() {
            clear_preview_mesh_sync_cache(viewer, state.inner().as_ref());
        } else {
            viewer.upload_preview_mesh_instanced(&instanced);
            viewer.preview_cache_key = None;
            *state.preview_overlay_cache_key.lock() = None;
        }
    }

    wake_viewport_loop(&app);
    Ok(())
}

// ── Extrude phase: recompute preview with new settings ────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExtrudeRecomputeArgs {
    extrude_profile: voxel_edit::ExtrudeProfile,
    extrude_end_cap: voxel_edit::ExtrudeEndCap,
    extrude_taper: bool,
    #[serde(default)]
    extrude_taper_start: f32,
    #[serde(default)]
    extrude_taper_end: f32,
}

#[tauri::command]
fn extrude_recompute_preview(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: ExtrudeRecomputeArgs,
) -> Result<(), String> {
    // Use stored ray spine if available (new ray-based extrude path).
    let spine_opt = state.extrude_ray_spine.lock().clone();
    let replay = state.sculpt_stroke_replay.lock().clone();
    if replay.is_empty() {
        return Ok(());
    }
    let first = &replay[0];
    let color = first.color;

    let union: ahash::AHashSet<greedy_mesh::VoxelCoord> = if let Some(spine) = &spine_opt {
        // Ray-based extrude: recompute footprint from stored spine with new settings.
        let footprint = voxel_edit::extrude_ray_footprint(
            spine,
            first.brush_radius,
            first.brush_shape,
            first.brush_strength,
            first.brush_falloff,
            first.stroke_seed,
            args.extrude_profile,
            args.extrude_end_cap,
            args.extrude_taper,
            args.extrude_taper_start,
            args.extrude_taper_end,
        );
        footprint.into_iter().collect()
    } else {
        // Legacy freeform fallback: replay frame-by-frame.
        let mut union: ahash::AHashSet<greedy_mesh::VoxelCoord> = ahash::AHashSet::new();
        let fg = state.current_file.lock();
        let vm = state.voxel_map.lock();
        let Some(file) = fg.as_ref() else {
            return Ok(());
        };
        let Some(vmap) = vm.as_ref() else {
            return Ok(());
        };
        // Acquire viewer before camera to match the render loop's lock order
        // (viewer → camera). Inverting this order deadlocks with the render tick.
        let (w, h) = {
            let v = state.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Ok(());
            };
            viewer.viewport_size()
        };
        let cam = state.camera.lock();
        let w = w as f32;
        let h = h as f32;
        for sample in &replay {
            let (sx, sy) = viewport_texels_from_norm(sample.nx, sample.ny, w, h);
            let line = match (sample.stroke_line_start_nx, sample.stroke_line_start_ny) {
                (Some(lnx), Some(lny)) => Some(viewport_texels_from_norm(lnx, lny, w, h)),
                _ => None,
            };
            let seg = match (sample.stroke_segment_prev_nx, sample.stroke_segment_prev_ny) {
                (Some(pnx), Some(pny)) => Some(viewport_texels_from_norm(pnx, pny, w, h)),
                _ => None,
            };
            let footprint = voxel_edit::sculpt_stroke_effective_footprint(
                file,
                vmap,
                &cam,
                w,
                h,
                sx,
                sy,
                sample.sculpt_mode,
                sample.brush_radius,
                sample.brush_shape,
                sample.spray_density,
                line,
                seg,
                sample.brush_strength,
                sample.brush_falloff,
                sample.stroke_seed,
                sample.brush_clip_bottom_half,
                args.extrude_profile,
                args.extrude_end_cap,
                args.extrude_taper,
                args.extrude_taper_start,
                args.extrude_taper_end,
            );
            for c in footprint {
                union.insert(c);
            }
        }
        union
    };

    // Update stored replay args with new extrude settings so commit uses them.
    {
        let mut replay_mut = state.sculpt_stroke_replay.lock();
        for sample in replay_mut.iter_mut() {
            sample.extrude_profile = args.extrude_profile;
            sample.extrude_end_cap = args.extrude_end_cap;
            sample.extrude_taper = args.extrude_taper;
            sample.extrude_taper_start = args.extrude_taper_start;
            sample.extrude_taper_end = args.extrude_taper_end;
        }
    }

    // Replace the preview union and re-render.
    {
        let mut preview_union = state.stroke_preview_union.lock();
        preview_union.clear();
        for c in &union {
            preview_union.insert(*c);
        }
    }

    let instanced = {
        let fg = state.current_file.lock();
        let vm = state.voxel_map.lock();
        let Some(file) = fg.as_ref() else {
            return Ok(());
        };
        let Some(vmap) = vm.as_ref() else {
            return Ok(());
        };
        stroke_preview_meshes_for_union(
            voxel_edit::EditTool::Add,
            &union,
            vmap,
            file,
            false,
            color,
            None,
        )
    };

    {
        let mut v = state.viewer.lock();
        let Some(viewer) = v.as_mut() else {
            return Ok(());
        };
        if instanced.solid_instances.is_empty() {
            clear_preview_mesh_sync_cache(viewer, state.inner().as_ref());
        } else {
            viewer.upload_preview_mesh_instanced(&instanced);
            viewer.preview_cache_key = None;
            *state.preview_overlay_cache_key.lock() = None;
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
    #[serde(default = "default_rock_count")]
    count: i32,
    #[serde(default = "default_rock_cluster_radius")]
    cluster_radius: i32,
    #[serde(default)]
    sink_direction: i32,
    #[serde(default)]
    sink_amount: i32,
}

fn default_rock_size() -> i32 {
    4
}

fn default_roughness() -> f32 {
    0.4
}

fn default_rock_count() -> i32 {
    1
}

fn default_rock_cluster_radius() -> i32 {
    1
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
            let v = state.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock();
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
            args.count,
            args.cluster_radius,
            args.sink_direction,
            args.sink_amount,
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
    #[serde(default = "default_grass_radius")]
    radius: i32,
    #[serde(default = "default_grass_density")]
    density: f32,
    #[serde(default = "default_grass_height")]
    max_height: i32,
    color: u32,
    material: String,
}

fn default_grass_radius() -> i32 {
    4
}

fn default_grass_density() -> f32 {
    0.6
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
            let v = state.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock();
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
            args.radius,
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
    /// 0 = loose, 1 = taut (scales sag; web `ropeTension`).
    #[serde(default = "default_rope_tension")]
    tension: f32,
    /// Web `ropeBrushRadius` index (same mapping as sculpt brush index).
    #[serde(default = "default_rope_brush_radius_index")]
    brush_radius: u32,
    #[serde(default)]
    brush_shape: voxel_edit::BrushShape,
    color: u32,
    material: String,
    /// Web `ropeGravityDirection`: down | up | left | right | forward | back.
    #[serde(default = "default_cloth_gravity_direction")]
    gravity_direction: String,
}

fn default_rope_sag() -> f32 {
    2.5
}

fn default_rope_tension() -> f32 {
    0.5
}

fn default_rope_brush_radius_index() -> u32 {
    2
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
            let v = state.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock();
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
            args.tension,
            args.brush_radius,
            args.brush_shape,
            args.color,
            material,
            &args.gravity_direction,
        )?
    };
    commit_voxel_edits(&state, &app, deltas)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeneratorClothArgs {
    /// At least three distinct corner voxels (surface picks).
    pins: Vec<[i32; 3]>,
    #[serde(default = "default_rope_tension")]
    tension: f32,
    /// Web `ropeGravityDirection`: down | up | left | right | forward | back.
    #[serde(default = "default_cloth_gravity_direction")]
    gravity_direction: String,
    #[serde(default = "default_rope_brush_radius_index")]
    brush_radius: u32,
    #[serde(default)]
    brush_shape: voxel_edit::BrushShape,
    color: u32,
    material: String,
    /// Web `clothSimGravityPct / 100`.
    #[serde(default = "default_cloth_gravity_stiffness_scale")]
    gravity_scale: f64,
    /// Web `clothSimStiffnessPct / 100`.
    #[serde(default = "default_cloth_gravity_stiffness_scale")]
    stiffness_scale: f64,
    /// 0 = automatic iteration count from tension.
    #[serde(default)]
    cloth_iterations: u32,
    #[serde(default = "default_cloth_constraint_passes")]
    cloth_constraint_passes: u32,
}

fn default_cloth_gravity_direction() -> String {
    "down".into()
}

fn default_cloth_gravity_stiffness_scale() -> f64 {
    1.0
}

fn default_cloth_constraint_passes() -> u32 {
    2
}

#[tauri::command]
fn generator_cloth_from_pins_cmd(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: GeneratorClothArgs,
) -> Result<bool, String> {
    if args.pins.len() < 3 {
        return Err("cloth needs at least three pin points".into());
    }
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let sim = crate::generators::ClothSimOptions {
        gravity_scale: args.gravity_scale.max(0.0),
        stiffness_scale: args.stiffness_scale.clamp(0.05, 2.0),
        iterations: if args.cloth_iterations > 0 {
            Some(args.cloth_iterations.clamp(4, 96))
        } else {
            None
        },
        constraint_passes: args.cloth_constraint_passes.clamp(1, 6),
    };
    let deltas = {
        let mut fg = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        crate::generators::generator_cloth_from_pins(
            file,
            vmap,
            &args.pins,
            args.tension,
            args.gravity_direction.as_str(),
            args.brush_radius,
            args.brush_shape,
            args.color,
            material,
            sim,
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
            let v = state.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock();
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

// ── Ashlar generator ──────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeneratorAshlarArgs {
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
    #[serde(default)]
    thickness: Option<i32>,
    #[serde(default)]
    thickness_axis: Option<i32>,
}

#[tauri::command]
fn generator_ashlar_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: GeneratorAshlarArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let (w, h) = {
            let v = state.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        crate::generators::generator_ashlar_at_screen(
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
            args.thickness,
            args.thickness_axis,
        )?
    };
    commit_voxel_edits(&state, &app, deltas)
}

// ── Flora generator ───────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeneratorFloraArgs {
    nx: f32,
    ny: f32,
    #[serde(default)]
    seed: i32,
    #[serde(default = "default_flora_height")]
    height: i32,
    #[serde(default)]
    girth: i32,
    #[serde(default = "default_flora_wobble")]
    wobble: f32,
    #[serde(default = "default_flora_taper")]
    taper: f32,
    #[serde(default = "default_one_i32")]
    stem_count: i32,
    #[serde(default)]
    cluster_radius: i32,
    #[serde(default)]
    branch_count: i32,
    #[serde(default = "default_one_i32")]
    branch_depth: i32,
    #[serde(default = "default_flora_branch_start")]
    branch_start: f32,
    #[serde(default = "default_one_f32_flora")]
    branch_spread: f32,
    #[serde(default = "default_one_i32")]
    braid_strands: i32,
    #[serde(default = "default_flora_braid_twist")]
    braid_twist: f32,
    #[serde(default)]
    canopy: f32,
    color: u32,
    material: String,
}

fn default_flora_height() -> i32 {
    14
}
fn default_flora_wobble() -> f32 {
    0.12
}
fn default_flora_taper() -> f32 {
    0.12
}
fn default_one_i32() -> i32 {
    1
}
fn default_flora_branch_start() -> f32 {
    0.5
}
fn default_flora_braid_twist() -> f32 {
    0.35
}
fn default_one_f32_flora() -> f32 {
    1.0
}

#[tauri::command]
fn generator_flora_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: GeneratorFloraArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let (w, h) = {
            let v = state.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        crate::generators::generator_flora_at_screen(
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
            args.seed,
            args.height,
            args.girth,
            args.wobble,
            args.taper,
            args.stem_count,
            args.cluster_radius,
            args.branch_count,
            args.branch_depth,
            args.branch_start,
            args.branch_spread,
            args.braid_strands,
            args.braid_twist,
            args.canopy,
            args.color,
            material,
        )?
    };
    commit_voxel_edits(&state, &app, deltas)
}

// ── Roof generator ────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeneratorRoofArgs {
    pins: Vec<[i32; 3]>,
    #[serde(default = "default_roof_style")]
    style: String,
    #[serde(default = "default_roof_height")]
    height: i32,
    #[serde(default = "default_one_i32")]
    thickness: i32,
    #[serde(default)]
    shed_edge_index: i32,
    #[serde(default)]
    gable_orientation: i32,
    #[serde(default = "default_roof_break_ratio")]
    break_ratio: f32,
    #[serde(default = "default_roof_wall_height")]
    wall_height: i32,
    #[serde(default = "default_roof_parapet_height")]
    parapet_height: i32,
    #[serde(default)]
    salt_skew: f32,
    #[serde(default)]
    hollow: bool,
    color: u32,
    material: String,
}

fn default_roof_style() -> String {
    "gable".into()
}
fn default_roof_height() -> i32 {
    6
}
fn default_roof_break_ratio() -> f32 {
    0.5
}
fn default_roof_wall_height() -> i32 {
    3
}
fn default_roof_parapet_height() -> i32 {
    2
}

#[tauri::command]
fn generator_roof_from_pins_cmd(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: GeneratorRoofArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    if args.pins.len() < 3 {
        return Err("roof needs at least 3 pins".into());
    }
    let deltas = {
        let mut fg = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        crate::generators::generate_roof_from_pins(
            file,
            vmap,
            &args.pins,
            &args.style,
            args.height,
            args.thickness,
            args.shed_edge_index,
            args.gable_orientation,
            args.break_ratio,
            args.wall_height,
            args.parapet_height,
            args.salt_skew,
            args.hollow,
            args.color,
            material,
        )
    };
    commit_voxel_edits(&state, &app, deltas)
}

// ── Piscina (fish) generator ──────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeneratorPiscinaArgs {
    nx: f32,
    ny: f32,
    #[serde(default)]
    seed: i32,
    #[serde(default = "default_piscina_species")]
    species: String,
    #[serde(default = "default_piscina_length")]
    length: i32,
    #[serde(default = "default_piscina_width")]
    width_param: i32,
    #[serde(default = "default_piscina_thickness")]
    thickness: i32,
    #[serde(default)]
    spine_bend: f32,
    #[serde(default)]
    spine_s_curve: f32,
    #[serde(default = "default_piscina_fin")]
    fin_dorsal: i32,
    #[serde(default = "default_piscina_fin")]
    fin_anal: i32,
    #[serde(default = "default_piscina_fin")]
    fin_caudal: i32,
    #[serde(default = "default_piscina_fin")]
    fin_pectoral: i32,
    #[serde(default = "default_piscina_fin")]
    fin_pelvic: i32,
    #[serde(default = "default_piscina_fin")]
    fin_adipose: i32,
    #[serde(default = "default_true")]
    show_fin_dorsal: bool,
    #[serde(default = "default_true")]
    show_fin_anal: bool,
    #[serde(default = "default_true")]
    show_fin_caudal: bool,
    #[serde(default = "default_true")]
    show_fin_pectoral: bool,
    #[serde(default = "default_true")]
    show_fin_pelvic: bool,
    #[serde(default = "default_true")]
    show_fin_adipose: bool,
    #[serde(default)]
    anchor_offset_u: i32,
    #[serde(default)]
    anchor_offset_v: i32,
    color: u32,
    material: String,
}

fn default_piscina_species() -> String {
    "trout".into()
}
fn default_piscina_length() -> i32 {
    16
}
fn default_piscina_width() -> i32 {
    4
}
fn default_piscina_thickness() -> i32 {
    3
}
fn default_piscina_fin() -> i32 {
    3
}

#[tauri::command]
fn generator_piscina_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: GeneratorPiscinaArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let (w, h) = {
            let v = state.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        crate::generators::generator_piscina_at_screen(
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
            args.seed,
            &args.species,
            args.length,
            args.width_param,
            args.thickness,
            args.spine_bend,
            args.spine_s_curve,
            args.fin_dorsal,
            args.fin_anal,
            args.fin_caudal,
            args.fin_pectoral,
            args.fin_pelvic,
            args.fin_adipose,
            args.show_fin_dorsal,
            args.show_fin_anal,
            args.show_fin_caudal,
            args.show_fin_pectoral,
            args.show_fin_pelvic,
            args.show_fin_adipose,
            args.anchor_offset_u,
            args.anchor_offset_v,
            args.color,
            material,
        )?
    };
    commit_voxel_edits(&state, &app, deltas)
}

// ── Insecta (insect) generator ────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeneratorInsectaArgs {
    nx: f32,
    ny: f32,
    #[serde(default = "default_insecta_species")]
    species: String,
    #[serde(default = "default_insecta_length")]
    total_length: i32,
    #[serde(default = "default_one_f32")]
    head_ratio: f32,
    #[serde(default = "default_insecta_thorax_ratio")]
    thorax_ratio: f32,
    #[serde(default = "default_insecta_abdomen_ratio")]
    abdomen_ratio: f32,
    #[serde(default = "default_insecta_body_half_width")]
    body_half_width: i32,
    #[serde(default = "default_insecta_body_half_height")]
    body_half_height: i32,
    #[serde(default = "default_insecta_abdomen_taper")]
    abdomen_taper: f32,
    #[serde(default = "default_insecta_head_shape")]
    head_shape: i32,
    #[serde(default)]
    anchor_offset_u: i32,
    #[serde(default)]
    anchor_offset_v: i32,
    #[serde(default)]
    body_yaw: f32,
    #[serde(default)]
    body_arch: f32,
    #[serde(default = "default_insecta_antenna_length")]
    antenna_length: i32,
    #[serde(default = "default_insecta_antenna_spread")]
    antenna_spread: f32,
    #[serde(default = "default_insecta_antenna_pitch")]
    antenna_pitch: f32,
    #[serde(default)]
    antenna_root: i32,
    #[serde(default)]
    mandible_length: i32,
    #[serde(default)]
    mandible_spread: f32,
    #[serde(default)]
    mandible_forward: i32,
    #[serde(default = "default_insecta_wing_shape")]
    wing_shape: i32,
    #[serde(default = "default_true")]
    show_wing_fore: bool,
    #[serde(default = "default_insecta_wing_fore_length")]
    wing_fore_length: i32,
    #[serde(default = "default_insecta_wing_fore_width")]
    wing_fore_width: i32,
    #[serde(default = "default_insecta_wing_spread")]
    wing_fore_spread: f32,
    #[serde(default)]
    wing_fore_pitch: f32,
    #[serde(default)]
    wing_fore_offset: i32,
    #[serde(default)]
    wing_fore_forward_cant: f32,
    #[serde(default)]
    show_wing_hind: bool,
    #[serde(default = "default_insecta_wing_hind_length")]
    wing_hind_length: i32,
    #[serde(default = "default_insecta_wing_hind_width")]
    wing_hind_width: i32,
    #[serde(default = "default_insecta_wing_spread")]
    wing_hind_spread: f32,
    #[serde(default)]
    wing_hind_pitch: f32,
    #[serde(default)]
    wing_hind_offset: i32,
    color: u32,
    material: String,
}

fn default_insecta_species() -> String {
    "bee".into()
}
fn default_insecta_length() -> i32 {
    24
}
fn default_one_f32() -> f32 {
    1.0
}
fn default_insecta_thorax_ratio() -> f32 {
    1.2
}
fn default_insecta_abdomen_ratio() -> f32 {
    2.0
}
fn default_insecta_body_half_width() -> i32 {
    3
}
fn default_insecta_body_half_height() -> i32 {
    3
}
fn default_insecta_abdomen_taper() -> f32 {
    0.6
}
fn default_insecta_head_shape() -> i32 {
    60
}
fn default_insecta_antenna_length() -> i32 {
    6
}
fn default_insecta_antenna_spread() -> f32 {
    20.0
}
fn default_insecta_antenna_pitch() -> f32 {
    30.0
}
fn default_insecta_wing_shape() -> i32 {
    85
}
fn default_insecta_wing_fore_length() -> i32 {
    12
}
fn default_insecta_wing_fore_width() -> i32 {
    3
}
fn default_insecta_wing_spread() -> f32 {
    15.0
}
fn default_insecta_wing_hind_length() -> i32 {
    8
}
fn default_insecta_wing_hind_width() -> i32 {
    2
}

#[tauri::command]
fn generator_insecta_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: GeneratorInsectaArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let (w, h) = {
            let v = state.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        crate::generators::generator_insecta_at_screen(
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
            &args.species,
            args.total_length,
            args.head_ratio,
            args.thorax_ratio,
            args.abdomen_ratio,
            args.body_half_width,
            args.body_half_height,
            args.abdomen_taper,
            args.head_shape,
            args.anchor_offset_u,
            args.anchor_offset_v,
            args.body_yaw,
            args.body_arch,
            args.antenna_length,
            args.antenna_spread,
            args.antenna_pitch,
            args.antenna_root,
            args.mandible_length,
            args.mandible_spread,
            args.mandible_forward,
            args.wing_shape,
            args.show_wing_fore,
            args.wing_fore_length,
            args.wing_fore_width,
            args.wing_fore_spread,
            args.wing_fore_pitch,
            args.wing_fore_offset,
            args.wing_fore_forward_cant,
            args.show_wing_hind,
            args.wing_hind_length,
            args.wing_hind_width,
            args.wing_hind_spread,
            args.wing_hind_pitch,
            args.wing_hind_offset,
            args.color,
            material,
        )?
    };
    commit_voxel_edits(&state, &app, deltas)
}

// ── Fauna (creature) generator ────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeneratorFaunaArgs {
    nx: f32,
    ny: f32,
    #[serde(default = "default_fauna_stance")]
    stance: String,
    #[serde(default = "default_fauna_archetype")]
    archetype: String,
    #[serde(default)]
    anchor_offset_u: i32,
    #[serde(default)]
    anchor_offset_v: i32,
    #[serde(default)]
    body_yaw: f32,
    #[serde(default)]
    body_arch: f32,
    #[serde(default = "default_fauna_spine_segments")]
    spine_segments: i32,
    #[serde(default = "default_fauna_body_length")]
    body_length: i32,
    #[serde(default = "default_fauna_body_half")]
    body_half_width: i32,
    #[serde(default = "default_fauna_body_half_height")]
    body_half_height: i32,
    #[serde(default = "default_fauna_neck_length")]
    neck_length: i32,
    #[serde(default = "default_fauna_neck_half")]
    neck_half_width: i32,
    #[serde(default = "default_fauna_neck_half")]
    neck_half_height: i32,
    #[serde(default = "default_fauna_head_length")]
    head_length: i32,
    #[serde(default = "default_fauna_head_half")]
    head_half_width: i32,
    #[serde(default = "default_fauna_head_half")]
    head_half_height: i32,
    #[serde(default = "default_one_i32")]
    tail_length: i32,
    #[serde(default = "default_fauna_shoulder_offset")]
    shoulder_offset_forward: i32,
    #[serde(default = "default_fauna_hip_offset")]
    hip_offset_forward: i32,
    #[serde(default = "default_fauna_upper_length")]
    front_upper_length: i32,
    #[serde(default = "default_fauna_upper_length")]
    front_lower_length: i32,
    #[serde(default = "default_fauna_hind_upper")]
    hind_upper_length: i32,
    #[serde(default = "default_fauna_hind_upper")]
    hind_lower_length: i32,
    #[serde(default = "default_fauna_limb_targets")]
    limb_targets: [[f32; 3]; 4],
    #[serde(default = "default_fauna_limb_poles")]
    limb_poles: [[f32; 3]; 4],
    #[serde(default)]
    spine_pose_chest: [f32; 3],
    #[serde(default)]
    spine_pose_neck: [f32; 3],
    #[serde(default)]
    spine_pose_head: [f32; 3],
    #[serde(default)]
    auto_foot_placement: bool,
    color: u32,
    material: String,
}

fn default_fauna_stance() -> String {
    "quadruped".into()
}
fn default_fauna_archetype() -> String {
    "ungulate".into()
}
fn default_fauna_spine_segments() -> i32 {
    7
}
fn default_fauna_body_length() -> i32 {
    17
}
fn default_fauna_body_half() -> i32 {
    2
}
fn default_fauna_body_half_height() -> i32 {
    3
}
fn default_fauna_neck_length() -> i32 {
    8
}
fn default_fauna_neck_half() -> i32 {
    2
}
fn default_fauna_head_length() -> i32 {
    6
}
fn default_fauna_head_half() -> i32 {
    2
}
fn default_fauna_shoulder_offset() -> i32 {
    3
}
fn default_fauna_hip_offset() -> i32 {
    -3
}
fn default_fauna_upper_length() -> i32 {
    7
}
fn default_fauna_hind_upper() -> i32 {
    8
}
fn default_fauna_limb_targets() -> [[f32; 3]; 4] {
    [
        [20.0, -2.1, -19.0],
        [20.0, 2.1, -19.0],
        [-3.5, -2.2, -20.0],
        [-3.5, 2.2, -20.0],
    ]
}
fn default_fauna_limb_poles() -> [[f32; 3]; 4] {
    [
        [20.0, -2.4, 0.6],
        [20.0, 2.4, 0.6],
        [1.8, -2.8, 1.2],
        [1.8, 2.8, 1.2],
    ]
}

#[tauri::command]
fn generator_fauna_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: GeneratorFaunaArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let (w, h) = {
            let v = state.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        crate::generators::generator_fauna_at_screen(
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
            &args.stance,
            &args.archetype,
            args.anchor_offset_u,
            args.anchor_offset_v,
            args.body_yaw,
            args.body_arch,
            args.spine_segments,
            args.body_length,
            args.body_half_width,
            args.body_half_height,
            args.neck_length,
            args.neck_half_width,
            args.neck_half_height,
            args.head_length,
            args.head_half_width,
            args.head_half_height,
            args.tail_length,
            args.shoulder_offset_forward,
            args.hip_offset_forward,
            args.front_upper_length,
            args.front_lower_length,
            args.hind_upper_length,
            args.hind_lower_length,
            &args.limb_targets,
            &args.limb_poles,
            args.spine_pose_chest,
            args.spine_pose_neck,
            args.spine_pose_head,
            args.auto_foot_placement,
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
    Ok(state.squishy_session.lock().clone())
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
    let mut g = state.squishy_session.lock();
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
    let mut g = state.squishy_session.lock();
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
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let mut sg = state.squishy_session.lock();
    let fg = state.current_file.lock();
    let vm = state.voxel_map.lock();
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let cam = state.camera.lock();
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
    let mut g = state.squishy_session.lock();
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
    let mut g = state.squishy_session.lock();
    g.selected_id = args.id;
    Ok(())
}

#[tauri::command]
fn squishy_session_clear(state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
    let mut g = state.squishy_session.lock();
    g.clear();
    *state.squishy_gizmo_drag.lock() = None;
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
        let sg = state.squishy_session.lock();
        let mut fg = state.current_file.lock();
        let mut vm = state.voxel_map.lock();
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
    let mut g = state.squishy_session.lock();
    g.clear();
    *state.squishy_gizmo_drag.lock() = None;
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
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let sg = state.squishy_session.lock();
    let cam = state.camera.lock();
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    Ok(generators::pick_metaball_at_screen(&sg, &cam, w, h, sx, sy))
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
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let cam = state.camera.lock();
    let sg = state.squishy_session.lock();
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let Some(handle) = generators::pick_squishy_gizmo_handle(&sg, &cam, w, h, sx, sy) else {
        return Ok(false);
    };
    let Some(drag) = generators::squishy_gizmo_begin_drag(&sg, &cam, w, h, sx, sy, handle) else {
        return Ok(false);
    };
    drop(sg);
    drop(cam);
    *state.squishy_gizmo_drag.lock() = Some(drag);
    wake_viewport_loop(&app);
    Ok(true)
}

#[tauri::command]
fn squishy_gizmo_pointer_move(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: SquishyGizmoPointerArgs,
) -> Result<(), String> {
    let drag = state.squishy_gizmo_drag.lock().clone();
    let Some(drag) = drag else {
        return Ok(());
    };
    let (w, h) = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Ok(());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let cam = state.camera.lock();
    let mut sg = state.squishy_session.lock();
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    generators::squishy_gizmo_apply_drag(&mut sg, &cam, w, h, sx, sy, &drag);
    wake_viewport_loop(&app);
    Ok(())
}

#[tauri::command]
fn squishy_gizmo_pointer_up(state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
    *state.squishy_gizmo_drag.lock() = None;
    Ok(())
}

#[tauri::command]
async fn voxel_edit_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: VoxelEditAtScreen,
) -> Result<bool, String> {
    let t_total = Instant::now();
    let (w, h) = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };

    // Block edits if we are a guest without edit permission.
    {
        let c = state.collab.lock();
        if c.is_client() {
            let local = c.local_peer_id;
            let can_edit = c
                .roster
                .iter()
                .find(|r| r.peer_id == local)
                .map(|r| r.can_edit)
                .unwrap_or(false);
            if !can_edit {
                return Err("editing not allowed".into());
            }
        }
    }

    let material = voxelle::MaterialId::from_str_id(&args.material);

    #[cfg(desktop)]
    if !matches!(args.stroke_mode, stroke_modes::DrawStrokeMode::Fill) {
        eprintln_extrusion_stroke_checkpoint("voxel_edit begin", &args, None, None);
    }

    let (deltas, apply_edit_ms) = if matches!(args.stroke_mode, stroke_modes::DrawStrokeMode::Fill)
    {
        state.fill_operation_cancel.store(false, Ordering::Relaxed);
        emit_work_progress(&app, 0.08, "Fill…");
        tokio::task::yield_now().await;
        let state_cl = Arc::clone(state.inner());
        let app_cl = app.clone();
        let args_cl = args.clone();
        let blocking = tokio::task::spawn_blocking(move || {
            let t_apply_start = Instant::now();
            let r = run_fill_deltas_blocking(&state_cl, &app_cl, w, h, &args_cl, material);
            let apply_edit_ms = t_apply_start.elapsed().as_secs_f64() * 1000.0;
            (r, apply_edit_ms)
        })
        .await
        .map_err(|e| e.to_string())?;
        let (deltas_res, apply_edit_ms) = blocking;
        let deltas = deltas_res.map_err(|e| {
            emit_work_progress(&app, 1.0, "");
            e
        })?;
        (deltas, apply_edit_ms)
    } else {
        let t_apply_start = Instant::now();
        let deltas = {
            let mut fg = state.current_file.lock();
            let mut vm = state.voxel_map.lock();
            let Some(file) = fg.as_mut() else {
                return Err("no model loaded".into());
            };
            let Some(vmap) = vm.as_mut() else {
                return Err("voxel index not ready".into());
            };
            let cam = state.camera.lock();
            let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
            let stroke_line_start = match (args.stroke_line_start_nx, args.stroke_line_start_ny) {
                (Some(lnx), Some(lny)) => Some(viewport_texels_from_norm(lnx, lny, w, h)),
                _ => None,
            };
            let stroke_segment_prev =
                match (args.stroke_segment_prev_nx, args.stroke_segment_prev_ny) {
                    (Some(pnx), Some(pny)) => Some(viewport_texels_from_norm(pnx, pny, w, h)),
                    _ => None,
                };
            // Resolve spray constraint plane (invisible hit plane trick from web).
            let spray_cp = resolve_spray_constraint_plane(
                &state,
                &args.stroke_aux,
                args.stroke_mode,
                args.tool,
                file,
                vmap,
                &cam,
                w,
                h,
                sx,
                sy,
            );
            voxel_edit::apply_edit(
                file,
                vmap,
                &cam,
                w,
                h,
                sx,
                sy,
                args.tool,
                build_color_resolver(
                    args.color,
                    args.palette.clone(),
                    args.paint_color_distrib.clone(),
                    args.stroke_seed,
                ),
                material,
                args.brush_radius,
                args.brush_shape,
                args.spray_density,
                stroke_line_start,
                stroke_segment_prev,
                args.stroke_mode,
                args.plane_axis,
                &args.stroke_aux,
                spray_cp,
            )?
        };
        let apply_edit_ms = t_apply_start.elapsed().as_secs_f64() * 1000.0;
        #[cfg(desktop)]
        eprintln_extrusion_stroke_checkpoint(
            "voxel_edit apply_edit done",
            &args,
            Some(deltas.len()),
            Some(apply_edit_ms),
        );
        (deltas, apply_edit_ms)
    };

    if deltas.is_empty() {
        if matches!(args.stroke_mode, stroke_modes::DrawStrokeMode::Fill) {
            emit_work_progress(&app, 1.0, "");
        }
        return Ok(false);
    }

    if matches!(args.stroke_mode, stroke_modes::DrawStrokeMode::Fill) {
        tokio::task::yield_now().await;
    }
    finish_voxel_edit_gpu_deltas(
        &state,
        &deltas,
        apply_edit_ms,
        t_total,
        &app,
        VoxelGpuRefreshReason::SoloEdit,
    )?;
    #[cfg(desktop)]
    eprintln_last_edit_perf_line(state.inner().as_ref());
    if matches!(args.stroke_mode, stroke_modes::DrawStrokeMode::Fill) {
        emit_work_progress(&app, 1.0, "");
    }

    let stroke_on = *state.stroke_active.lock();
    if stroke_on {
        state.stroke_buffer.lock().extend(deltas.iter().copied());
        return Ok(true);
    }

    let cm = Arc::clone(&state.collab);
    let mut cb = cm.lock();
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
        let mut u = state.solo_undo.lock();
        u.pop()
    };
    let Some(step) = step else {
        return Ok(false);
    };
    match step {
        SoloUndoEntry::VoxelDeltas(original) => {
            let mesh_refresh: Vec<voxel_edit::VoxelEditDelta> = {
                let mut fg = state.current_file.lock();
                let mut vm = state.voxel_map.lock();
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
                .push(SoloRedoEntry::VoxelDeltas(original));
            Ok(true)
        }
        SoloUndoEntry::SelectionBefore(before) => {
            let cur = {
                let mut sel = state.selection_cells.lock();
                let cur = sel.clone();
                *sel = before;
                cur
            };
            emit_selection_updated(app, state);
            state
                .solo_redo
                .lock()
                .push(SoloRedoEntry::SelectionAfter(cur));
            Ok(true)
        }
        SoloUndoEntry::SelectionTransform { before, deltas } => {
            let mesh_refresh: Vec<voxel_edit::VoxelEditDelta> = {
                let mut fg = state.current_file.lock();
                let mut vm = state.voxel_map.lock();
                let Some(file) = fg.as_mut() else {
                    return Err("no model loaded".into());
                };
                let Some(vmap) = vm.as_mut() else {
                    return Err("voxel index not ready".into());
                };
                let mut mesh = Vec::with_capacity(deltas.len());
                for d in deltas.iter().rev() {
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
            let after = {
                let mut sel = state.selection_cells.lock();
                let after = sel.clone();
                *sel = before;
                after
            };
            emit_selection_updated(app, state);
            state
                .solo_redo
                .lock()
                .push(SoloRedoEntry::SelectionTransform { after, deltas });
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
        let mut r = state.solo_redo.lock();
        r.pop()
    };
    let Some(step) = step else {
        return Ok(false);
    };
    match step {
        SoloRedoEntry::VoxelDeltas(forward_batch) => {
            {
                let mut fg = state.current_file.lock();
                let mut vm = state.voxel_map.lock();
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
                .push(SoloUndoEntry::VoxelDeltas(forward_batch));
            Ok(true)
        }
        SoloRedoEntry::SelectionAfter(after) => {
            let cur = {
                let mut sel = state.selection_cells.lock();
                let cur = sel.clone();
                *sel = after;
                cur
            };
            emit_selection_updated(app, state);
            state
                .solo_undo
                .lock()
                .push(SoloUndoEntry::SelectionBefore(cur));
            Ok(true)
        }
        SoloRedoEntry::SelectionTransform { after, deltas } => {
            {
                let mut fg = state.current_file.lock();
                let mut vm = state.voxel_map.lock();
                let Some(file) = fg.as_mut() else {
                    return Err("no model loaded".into());
                };
                let Some(vmap) = vm.as_mut() else {
                    return Err("voxel index not ready".into());
                };
                for d in &deltas {
                    voxel_edit::apply_forward_delta(file, vmap, d)?;
                }
            }
            finish_voxel_edit_gpu_deltas(
                state,
                &deltas,
                0.0,
                t_total,
                app,
                VoxelGpuRefreshReason::Redo,
            )?;
            let before = {
                let mut sel = state.selection_cells.lock();
                let before = sel.clone();
                *sel = after;
                before
            };
            emit_selection_updated(app, state);
            state
                .solo_undo
                .lock()
                .push(SoloUndoEntry::SelectionTransform { before, deltas });
            Ok(true)
        }
    }
}

#[tauri::command]
fn voxel_undo(state: State<'_, Arc<ViewerState>>, app: AppHandle) -> Result<bool, String> {
    let cm = Arc::clone(&state.collab);
    {
        let mut c = cm.lock();
        if c.is_client() {
            if let Some(tx) = &c.client_tx {
                let _ = tx.send(serde_json::to_string(&collab::ClientToHost::Undo).unwrap());
            }
            return Ok(true);
        }
        if c.is_host() {
            // Pop undo stack, then drop the lock before GPU work.
            let original = c.host_undo.entry(collab::HOST_PEER_ID).or_default().pop();
            drop(c);
            let Some(original) = original else {
                return Ok(false);
            };
            let mesh_refresh: Vec<voxel_edit::VoxelEditDelta> = {
                let mut fg = state.current_file.lock();
                let mut vm = state.voxel_map.lock();
                let file = fg.as_mut().ok_or("no model loaded")?;
                let vmap = vm.as_mut().ok_or("voxel index not ready")?;
                let mut mesh = Vec::with_capacity(original.len());
                for d in original.iter().rev() {
                    voxel_edit::apply_inverse_delta(file, vmap, d)?;
                    mesh.push(voxel_edit::mesh_delta_after_inverse_of(d));
                }
                mesh
            };
            finish_voxel_edit_gpu_deltas(
                &state,
                &mesh_refresh,
                0.0,
                std::time::Instant::now(),
                &app,
                VoxelGpuRefreshReason::Undo,
            )?;
            // Re-acquire briefly for redo push + seq.
            let seq = {
                let mut c = cm.lock();
                c.host_redo
                    .entry(collab::HOST_PEER_ID)
                    .or_default()
                    .push(original);
                c.next_seq += 1;
                c.next_seq
            };
            collab::host_emit_edit_batch(&cm, &app, seq, collab::HOST_PEER_ID, &mesh_refresh);
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
        let mut c = cm.lock();
        if c.is_client() {
            if let Some(tx) = &c.client_tx {
                let _ = tx.send(serde_json::to_string(&collab::ClientToHost::Redo).unwrap());
            }
            return Ok(true);
        }
        if c.is_host() {
            // Pop redo stack, then drop the lock before GPU work.
            let forward = c.host_redo.entry(collab::HOST_PEER_ID).or_default().pop();
            drop(c);
            let Some(forward) = forward else {
                return Ok(false);
            };
            {
                let mut fg = state.current_file.lock();
                let mut vm = state.voxel_map.lock();
                let file = fg.as_mut().ok_or("no model loaded")?;
                let vmap = vm.as_mut().ok_or("voxel index not ready")?;
                for d in &forward {
                    voxel_edit::apply_forward_delta(file, vmap, d)?;
                }
            }
            finish_voxel_edit_gpu_deltas(
                &state,
                &forward,
                0.0,
                std::time::Instant::now(),
                &app,
                VoxelGpuRefreshReason::Redo,
            )?;
            // Re-acquire briefly for undo push + seq.
            let seq = {
                let mut c = cm.lock();
                c.host_undo
                    .entry(collab::HOST_PEER_ID)
                    .or_default()
                    .push(forward.clone());
                c.next_seq += 1;
                c.next_seq
            };
            collab::host_emit_edit_batch(&cm, &app, seq, collab::HOST_PEER_ID, &forward);
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
        let g = state.current_file.lock();
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

// ---------------------------------------------------------------------------
// Recent files list
// ---------------------------------------------------------------------------

const MAX_RECENT_FILES: usize = 10;

#[derive(serde::Serialize, serde::Deserialize)]
struct RecentFiles {
    paths: Vec<String>,
}

fn recent_files_path(app: &AppHandle) -> Result<PathBuf, String> {
    let mut p = app.path().app_data_dir().map_err(|e| e.to_string())?;
    p.push("recent_files.json");
    Ok(p)
}

fn read_recent_files(app: &AppHandle) -> Vec<String> {
    let Ok(path) = recent_files_path(app) else {
        return Vec::new();
    };
    let Ok(bytes) = std::fs::read(&path) else {
        return Vec::new();
    };
    serde_json::from_slice::<RecentFiles>(&bytes)
        .map(|r| r.paths)
        .unwrap_or_default()
}

fn persist_recent_file(app: &AppHandle, document_path: &str) {
    if !document_path.ends_with(".voxelle") {
        return;
    }
    let Ok(path) = recent_files_path(app) else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut paths = read_recent_files(app);
    // Remove if already present so we can move it to the front.
    paths.retain(|p| p != document_path);
    paths.insert(0, document_path.to_string());
    paths.truncate(MAX_RECENT_FILES);
    let data = RecentFiles { paths };
    if let Ok(s) = serde_json::to_string_pretty(&data) {
        let _ = std::fs::write(path, s);
    }
}

fn clear_recent_files(app: &AppHandle) {
    let Ok(path) = recent_files_path(app) else {
        return;
    };
    let _ = std::fs::remove_file(path);
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
    let keep = *state.autosave_keep_count.lock();
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
    let keep = *state.autosave_keep_count.lock();
    let k = (keep.max(1)) as u64;
    let idx = {
        let mut map = state.autosave_slot.lock();
        let n = map.entry(h.clone()).or_insert(0);
        let slot = (*n % k) as u32;
        *n = n.wrapping_add(1);
        slot
    };
    let mut dir = autosave_dir(app)?;
    dir.push(format!("{h}.{idx}.voxelle"));
    Ok(dir)
}

fn unsaved_autosave_anchor_path(app: &AppHandle) -> Result<PathBuf, String> {
    let mut p = app.path().app_data_dir().map_err(|e| e.to_string())?;
    p.push("unsaved_autosave_anchor.voxelle");
    Ok(p)
}

/// Logical document path for autosave keys and rotation. Saved projects use the real file path;
/// unsaved labels (e.g. `New project (…)`) use a stable app-local anchor so backups work before
/// “Save As…”.
fn autosave_document_path_for_label(app: &AppHandle, label: &str) -> Result<PathBuf, String> {
    if label.ends_with(".voxelle") {
        Ok(PathBuf::from(label))
    } else {
        unsaved_autosave_anchor_path(app)
    }
}

/// `file_label` after restoring from the unsaved-work autosave bucket (not a real on-disk project path).
const ONGOING_UNSAVED_PROJECT_LABEL: &str = "An unsaved project";

fn try_initial_autosave_after_new_project(app: &AppHandle, state: &Arc<ViewerState>, label: &str) {
    let enabled = *state.autosave_enabled.lock();
    let interval = *state.autosave_interval_secs.lock();
    if !enabled || interval == 0 {
        return;
    }
    let (collab_on, is_host) = {
        let c = state.collab.lock();
        (c.is_active(), c.is_host())
    };
    if collab_on && !is_host {
        return;
    }
    if !state.active_project.load(Ordering::Relaxed) {
        return;
    }
    let Ok(doc) = autosave_document_path_for_label(app, label) else {
        return;
    };
    let Ok(dest) = next_rotating_autosave_path(app, Arc::as_ref(state), &doc) else {
        return;
    };
    if write_voxelle_file_to_path(None, Arc::as_ref(state), &dest).is_ok() {
        *state.last_autosave.lock() = Some(Instant::now());
    }
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
    let doc_str_opt = read_last_document_path(&app);
    let anchor = unsaved_autosave_anchor_path(&app)?;
    let st = state.inner().as_ref();

    let doc_newest = doc_str_opt
        .as_ref()
        .and_then(|s| newest_autosave_path(&app, st, Path::new(s.as_str())));
    let anchor_newest = newest_autosave_path(&app, st, &anchor);

    let use_anchor_recovery = match (&doc_newest, &anchor_newest) {
        (Some(d_path), Some(a_path)) => match (file_mtime(d_path), file_mtime(a_path)) {
            (Some(dm), Some(am)) => am > dm,
            (None, Some(_)) => true,
            _ => false,
        },
        (None, Some(_)) => true,
        _ => false,
    };

    if use_anchor_recovery {
        let Some(ap) = anchor_newest else {
            return Ok(LastSessionInfo {
                last_document_path: None,
                document_basename: None,
                autosave_path: None,
                document_exists: false,
                autosave_exists: false,
                autosave_newer_than_document: false,
            });
        };
        let aex = ap.exists();
        return Ok(LastSessionInfo {
            last_document_path: Some(ONGOING_UNSAVED_PROJECT_LABEL.to_string()),
            document_basename: Some(ONGOING_UNSAVED_PROJECT_LABEL.to_string()),
            autosave_path: Some(ap.to_string_lossy().into_owned()),
            document_exists: false,
            autosave_exists: aex,
            autosave_newer_than_document: true,
        });
    }

    let Some(doc_str) = doc_str_opt else {
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
    let (autosave_str, autosave_exists, newer) = match doc_newest {
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
    *state.file_label.lock() = args.document_path.clone();
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
    let label = state.file_label.lock();
    if label.starts_with("New project") || !label.ends_with(".voxelle") {
        return Err("Use “Save As…” for new or unsaved projects.".into());
    }
    let s = label.clone();
    drop(label);
    write_voxelle_file_to_path(Some(&app), &state, Path::new(s.as_str()))?;
    persist_last_document_path(&app, s.as_str());
    persist_recent_file(&app, s.as_str());
    #[cfg(desktop)]
    if let Some(rm) = app.try_state::<RecentMenuState>() {
        rebuild_recent_submenu(&app, &rm.submenu);
    }
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
        *state_c.file_label.lock() = s.clone();
        persist_last_document_path(&app_c, &s);
        persist_recent_file(&app_c, &s);
        #[cfg(desktop)]
        if let Some(rm) = app_c.try_state::<RecentMenuState>() {
            rebuild_recent_submenu(&app_c, &rm.submenu);
        }
        emit_voxelle_loaded(&app_c, s, &state_c, false);
    });
    Ok(())
}

fn mesh_for_export(state: &Arc<ViewerState>) -> Result<greedy_mesh::MeshBuffers, String> {
    let fg = state.current_file.lock();
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };
    let rm = *state.rendering_mode.lock();
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
    palette: Vec<u32>,
    #[serde(default)]
    paint_color_distrib: Option<paint_color_distrib::PaintColorDistrib>,
    #[serde(default)]
    material: String,
    #[serde(default)]
    match_material: bool,
    #[serde(default = "default_true")]
    use_brush_preview: bool,
    #[serde(default)]
    generator_kind: Option<String>,
    #[serde(default)]
    generator_rope_first_nx: Option<f32>,
    #[serde(default)]
    generator_rope_first_ny: Option<f32>,
    #[serde(default = "default_rope_sag")]
    generator_rope_sag: f32,
    #[serde(default = "default_rope_tension")]
    generator_rope_tension: f32,
    #[serde(default = "default_cloth_gravity_direction_str")]
    generator_rope_gravity_direction: String,
    #[serde(default)]
    generator_cloth_pins: Vec<[i32; 3]>,
    #[serde(default = "default_cloth_tension_preview")]
    generator_cloth_tension: f32,
    #[serde(default = "default_cloth_gravity_direction_str")]
    generator_cloth_gravity_direction: String,
    #[serde(default = "default_one_f64")]
    generator_cloth_gravity_scale: f64,
    #[serde(default = "default_one_f64")]
    generator_cloth_stiffness_scale: f64,
    #[serde(default)]
    generator_cloth_iterations: u32,
    #[serde(default = "default_cloth_constraint_passes_u32")]
    generator_cloth_constraint_passes: u32,
    #[serde(default = "default_rock_size")]
    generator_rock_size: i32,
    #[serde(default = "default_rock_roughness")]
    generator_rock_roughness: f32,
    #[serde(default = "default_rock_seed")]
    generator_rock_seed: i32,
    #[serde(default = "default_rock_count")]
    generator_rock_count: i32,
    #[serde(default = "default_rock_cluster_radius")]
    generator_rock_cluster_radius: i32,
    #[serde(default)]
    generator_rock_sink_direction: i32,
    #[serde(default)]
    generator_rock_sink_amount: i32,
    #[serde(default = "default_grass_radius")]
    generator_grass_radius: i32,
    #[serde(default = "default_grass_density")]
    generator_grass_density: f32,
    #[serde(default = "default_grass_max_height")]
    generator_grass_max_height: i32,
    #[serde(default = "default_grass_seed")]
    generator_grass_seed: i32,
    #[serde(default)]
    generator_roof_pins: Vec<[i32; 3]>,
    #[serde(default = "default_roof_style")]
    generator_roof_style: String,
    #[serde(default = "default_roof_height")]
    generator_roof_height: i32,
    #[serde(default = "default_one_i32")]
    generator_roof_thickness: i32,
    #[serde(default = "default_roof_break_ratio")]
    generator_roof_break_ratio: f32,
    #[serde(default = "default_roof_wall_height")]
    generator_roof_wall_height: i32,
    #[serde(default = "default_roof_parapet_height")]
    generator_roof_parapet_height: i32,
    #[serde(default)]
    generator_roof_salt_skew: f32,
    #[serde(default)]
    generator_roof_hollow: bool,
    #[serde(default = "default_rock_size")]
    generator_ashlar_size: i32,
    #[serde(default = "default_ashlar_roughness")]
    generator_ashlar_roughness: f32,
    #[serde(default = "default_ashlar_seed")]
    generator_ashlar_seed: i32,
    #[serde(default = "default_ashlar_thickness")]
    generator_ashlar_thickness: i32,
    // Flora
    #[serde(default = "default_flora_seed")]
    generator_flora_seed: i32,
    #[serde(default = "default_flora_height")]
    generator_flora_height: i32,
    #[serde(default = "default_flora_girth")]
    generator_flora_girth: i32,
    #[serde(default = "default_flora_wobble")]
    generator_flora_wobble: f32,
    #[serde(default = "default_flora_taper")]
    generator_flora_taper: f32,
    #[serde(default = "default_one_i32")]
    generator_flora_stem_count: i32,
    #[serde(default)]
    generator_flora_cluster_radius: i32,
    #[serde(default = "default_flora_branch_count")]
    generator_flora_branch_count: i32,
    #[serde(default = "default_two_i32")]
    generator_flora_branch_depth: i32,
    #[serde(default = "default_flora_branch_start")]
    generator_flora_branch_start: f32,
    #[serde(default = "default_flora_branch_spread")]
    generator_flora_branch_spread: f32,
    #[serde(default)]
    generator_flora_braid_strands: i32,
    #[serde(default = "default_flora_braid_twist")]
    generator_flora_braid_twist: f32,
    #[serde(default = "default_flora_canopy")]
    generator_flora_canopy: f32,
    // Insecta
    #[serde(default = "default_insecta_species")]
    generator_insecta_species: String,
    #[serde(default = "default_insecta_total_length")]
    generator_insecta_total_length: i32,
    #[serde(default = "default_one_f32")]
    generator_insecta_head_ratio: f32,
    #[serde(default = "default_one_f32")]
    generator_insecta_thorax_ratio: f32,
    #[serde(default = "default_insecta_abdomen_ratio")]
    generator_insecta_abdomen_ratio: f32,
    #[serde(default = "default_two_i32")]
    generator_insecta_body_half_width: i32,
    #[serde(default = "default_two_i32")]
    generator_insecta_body_half_height: i32,
    #[serde(default = "default_insecta_abdomen_taper")]
    generator_insecta_abdomen_taper: f32,
    #[serde(default)]
    generator_insecta_head_shape: i32,
    #[serde(default)]
    generator_insecta_anchor_offset_u: i32,
    #[serde(default)]
    generator_insecta_anchor_offset_v: i32,
    #[serde(default)]
    generator_insecta_body_yaw: f32,
    #[serde(default)]
    generator_insecta_body_arch: f32,
    #[serde(default = "default_insecta_antenna_length")]
    generator_insecta_antenna_length: i32,
    #[serde(default = "default_insecta_antenna_spread")]
    generator_insecta_antenna_spread: f32,
    #[serde(default = "default_insecta_antenna_pitch")]
    generator_insecta_antenna_pitch: f32,
    #[serde(default = "default_one_i32")]
    generator_insecta_antenna_root: i32,
    #[serde(default = "default_two_i32")]
    generator_insecta_mandible_length: i32,
    #[serde(default = "default_insecta_mandible_spread")]
    generator_insecta_mandible_spread: f32,
    #[serde(default = "default_one_i32")]
    generator_insecta_mandible_forward: i32,
    #[serde(default)]
    generator_insecta_wing_shape: i32,
    #[serde(default = "default_true")]
    generator_insecta_show_wing_fore: bool,
    #[serde(default = "default_insecta_wing_fore_length")]
    generator_insecta_wing_fore_length: i32,
    #[serde(default = "default_four_i32")]
    generator_insecta_wing_fore_width: i32,
    #[serde(default = "default_insecta_wing_fore_spread")]
    generator_insecta_wing_fore_spread: f32,
    #[serde(default = "default_insecta_wing_fore_pitch")]
    generator_insecta_wing_fore_pitch: f32,
    #[serde(default)]
    generator_insecta_wing_fore_offset: i32,
    #[serde(default)]
    generator_insecta_wing_fore_forward_cant: f32,
    #[serde(default = "default_true")]
    generator_insecta_show_wing_hind: bool,
    #[serde(default = "default_insecta_wing_hind_length")]
    generator_insecta_wing_hind_length: i32,
    #[serde(default = "default_four_i32")]
    generator_insecta_wing_hind_width: i32,
    #[serde(default = "default_insecta_wing_hind_spread")]
    generator_insecta_wing_hind_spread: f32,
    #[serde(default = "default_insecta_wing_hind_pitch")]
    generator_insecta_wing_hind_pitch: f32,
    #[serde(default)]
    generator_insecta_wing_hind_offset: i32,
    // Fauna
    #[serde(default = "default_fauna_stance")]
    generator_fauna_stance: String,
    #[serde(default = "default_fauna_archetype")]
    generator_fauna_archetype: String,
    #[serde(default)]
    generator_fauna_anchor_offset_u: i32,
    #[serde(default)]
    generator_fauna_anchor_offset_v: i32,
    #[serde(default)]
    generator_fauna_body_yaw: f32,
    #[serde(default)]
    generator_fauna_body_arch: f32,
    #[serde(default = "default_fauna_spine_segments")]
    generator_fauna_spine_segments: i32,
    #[serde(default = "default_fauna_body_length")]
    generator_fauna_body_length: i32,
    #[serde(default = "default_two_i32")]
    generator_fauna_body_half_width: i32,
    #[serde(default = "default_two_i32")]
    generator_fauna_body_half_height: i32,
    #[serde(default = "default_three_i32")]
    generator_fauna_neck_length: i32,
    #[serde(default = "default_one_i32")]
    generator_fauna_neck_half_width: i32,
    #[serde(default = "default_one_i32")]
    generator_fauna_neck_half_height: i32,
    #[serde(default = "default_three_i32")]
    generator_fauna_head_length: i32,
    #[serde(default = "default_two_i32")]
    generator_fauna_head_half_width: i32,
    #[serde(default = "default_two_i32")]
    generator_fauna_head_half_height: i32,
    #[serde(default = "default_four_i32")]
    generator_fauna_tail_length: i32,
    #[serde(default = "default_three_i32")]
    generator_fauna_shoulder_offset_forward: i32,
    #[serde(default = "default_fauna_hip_offset_forward")]
    generator_fauna_hip_offset_forward: i32,
    #[serde(default = "default_four_i32")]
    generator_fauna_front_upper_length: i32,
    #[serde(default = "default_four_i32")]
    generator_fauna_front_lower_length: i32,
    #[serde(default = "default_four_i32")]
    generator_fauna_hind_upper_length: i32,
    #[serde(default = "default_four_i32")]
    generator_fauna_hind_lower_length: i32,
    #[serde(default = "default_true")]
    generator_fauna_auto_foot_placement: bool,
    // Piscina
    #[serde(default = "default_piscina_seed")]
    generator_piscina_seed: i32,
    #[serde(default = "default_piscina_species")]
    generator_piscina_species: String,
    #[serde(default = "default_piscina_length")]
    generator_piscina_length: i32,
    #[serde(default = "default_four_i32")]
    generator_piscina_width: i32,
    #[serde(default = "default_three_i32")]
    generator_piscina_thickness: i32,
    #[serde(default = "default_piscina_spine_bend")]
    generator_piscina_spine_bend: f32,
    #[serde(default)]
    generator_piscina_spine_s_curve: f32,
    #[serde(default = "default_four_i32")]
    generator_piscina_fin_dorsal: i32,
    #[serde(default = "default_four_i32")]
    generator_piscina_fin_anal: i32,
    #[serde(default = "default_four_i32")]
    generator_piscina_fin_caudal: i32,
    #[serde(default = "default_four_i32")]
    generator_piscina_fin_pectoral: i32,
    #[serde(default = "default_four_i32")]
    generator_piscina_fin_pelvic: i32,
    #[serde(default = "default_four_i32")]
    generator_piscina_fin_adipose: i32,
    #[serde(default = "default_true")]
    generator_piscina_show_fin_dorsal: bool,
    #[serde(default = "default_true")]
    generator_piscina_show_fin_anal: bool,
    #[serde(default = "default_true")]
    generator_piscina_show_fin_caudal: bool,
    #[serde(default = "default_true")]
    generator_piscina_show_fin_pectoral: bool,
    #[serde(default = "default_true")]
    generator_piscina_show_fin_pelvic: bool,
    #[serde(default)]
    generator_piscina_show_fin_adipose: bool,
    #[serde(default)]
    generator_piscina_anchor_offset_u: i32,
    #[serde(default)]
    generator_piscina_anchor_offset_v: i32,
    #[serde(default)]
    stamp_origin_x: i32,
    #[serde(default)]
    stamp_origin_z: i32,
}

fn default_ashlar_roughness() -> f32 {
    0.3
}
fn default_ashlar_seed() -> i32 {
    42
}
fn default_ashlar_thickness() -> i32 {
    3
}

fn default_rock_roughness() -> f32 {
    0.4
}
fn default_rock_seed() -> i32 {
    42
}
fn default_grass_max_height() -> i32 {
    3
}
fn default_grass_seed() -> i32 {
    42
}

// New defaults for bio-generator preview fields (SyncPreviewInput)
fn default_flora_seed() -> i32 {
    42
}
fn default_flora_girth() -> i32 {
    2
}
fn default_flora_branch_count() -> i32 {
    4
}
fn default_flora_branch_spread() -> f32 {
    0.5
}
fn default_flora_canopy() -> f32 {
    2.0
}
fn default_insecta_total_length() -> i32 {
    12
}
fn default_insecta_mandible_spread() -> f32 {
    0.3
}
fn default_insecta_wing_fore_spread() -> f32 {
    0.5
}
fn default_insecta_wing_fore_pitch() -> f32 {
    0.1
}
fn default_insecta_wing_hind_spread() -> f32 {
    0.6
}
fn default_insecta_wing_hind_pitch() -> f32 {
    0.2
}
fn default_fauna_hip_offset_forward() -> i32 {
    -3
}
fn default_piscina_seed() -> i32 {
    42
}
fn default_piscina_spine_bend() -> f32 {
    0.1
}
fn default_two_i32() -> i32 {
    2
}
fn default_three_i32() -> i32 {
    3
}
fn default_four_i32() -> i32 {
    4
}

fn default_cloth_tension_preview() -> f32 {
    0.5
}

fn default_cloth_gravity_direction_str() -> String {
    "down".into()
}

fn default_one_f64() -> f64 {
    1.0
}

fn default_cloth_constraint_passes_u32() -> u32 {
    2
}

#[tauri::command]
fn sync_preview_input(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: SyncPreviewInput,
) -> Result<(), String> {
    let new_mode = PreviewMode::parse(&args.mode);
    {
        let mut pm = state.preview_mode.lock();
        let changed = *pm != new_mode;
        *pm = new_mode;
        if changed {
            wake_viewport_loop(&app);
        }
    }
    {
        let mut ph = state.preview_hover.lock();
        ph.brush_radius = args.brush_radius;
        ph.brush_shape = args.brush_shape;
        ph.spray_density = args.spray_density;
        ph.stroke_mode = args.stroke_mode;
        ph.plane_axis = args.plane_axis;
        ph.stroke_aux = args.stroke_aux;
        ph.color = args.color;
        ph.palette = args.palette.clone();
        ph.paint_color_distrib = args.paint_color_distrib.clone();
        ph.material = args.material;
        ph.match_material = args.match_material;
        ph.use_brush_preview = args.use_brush_preview;
        ph.generator_kind = args.generator_kind.clone();
        ph.generator_rope_first_nx = args.generator_rope_first_nx;
        ph.generator_rope_first_ny = args.generator_rope_first_ny;
        ph.generator_rope_sag = args.generator_rope_sag;
        ph.generator_rope_tension = args.generator_rope_tension;
        ph.generator_rope_gravity_direction = args.generator_rope_gravity_direction.clone();
        ph.generator_cloth_pins
            .clone_from(&args.generator_cloth_pins);
        ph.generator_cloth_tension = args.generator_cloth_tension;
        ph.generator_cloth_gravity_direction = args.generator_cloth_gravity_direction.clone();
        ph.generator_cloth_gravity_scale = args.generator_cloth_gravity_scale;
        ph.generator_cloth_stiffness_scale = args.generator_cloth_stiffness_scale;
        ph.generator_cloth_iterations = args.generator_cloth_iterations;
        ph.generator_cloth_constraint_passes = args.generator_cloth_constraint_passes;
        ph.generator_rock_size = args.generator_rock_size;
        ph.generator_rock_roughness = args.generator_rock_roughness;
        ph.generator_rock_seed = args.generator_rock_seed;
        ph.generator_rock_count = args.generator_rock_count;
        ph.generator_rock_cluster_radius = args.generator_rock_cluster_radius;
        ph.generator_rock_sink_direction = args.generator_rock_sink_direction;
        ph.generator_rock_sink_amount = args.generator_rock_sink_amount;
        ph.generator_grass_radius = args.generator_grass_radius;
        ph.generator_grass_density = args.generator_grass_density;
        ph.generator_grass_max_height = args.generator_grass_max_height;
        ph.generator_grass_seed = args.generator_grass_seed;
        ph.generator_roof_pins = args.generator_roof_pins.clone();
        ph.generator_roof_style = args.generator_roof_style.clone();
        ph.generator_roof_height = args.generator_roof_height;
        ph.generator_roof_thickness = args.generator_roof_thickness;
        ph.generator_roof_break_ratio = args.generator_roof_break_ratio;
        ph.generator_roof_wall_height = args.generator_roof_wall_height;
        ph.generator_roof_parapet_height = args.generator_roof_parapet_height;
        ph.generator_roof_salt_skew = args.generator_roof_salt_skew;
        ph.generator_roof_hollow = args.generator_roof_hollow;
        ph.generator_ashlar_size = args.generator_ashlar_size;
        ph.generator_ashlar_roughness = args.generator_ashlar_roughness;
        ph.generator_ashlar_seed = args.generator_ashlar_seed;
        ph.generator_ashlar_thickness = args.generator_ashlar_thickness;
        // Flora
        ph.generator_flora_seed = args.generator_flora_seed;
        ph.generator_flora_height = args.generator_flora_height;
        ph.generator_flora_girth = args.generator_flora_girth;
        ph.generator_flora_wobble = args.generator_flora_wobble;
        ph.generator_flora_taper = args.generator_flora_taper;
        ph.generator_flora_stem_count = args.generator_flora_stem_count;
        ph.generator_flora_cluster_radius = args.generator_flora_cluster_radius;
        ph.generator_flora_branch_count = args.generator_flora_branch_count;
        ph.generator_flora_branch_depth = args.generator_flora_branch_depth;
        ph.generator_flora_branch_start = args.generator_flora_branch_start;
        ph.generator_flora_branch_spread = args.generator_flora_branch_spread;
        ph.generator_flora_braid_strands = args.generator_flora_braid_strands;
        ph.generator_flora_braid_twist = args.generator_flora_braid_twist;
        ph.generator_flora_canopy = args.generator_flora_canopy;
        // Insecta
        ph.generator_insecta_species = args.generator_insecta_species.clone();
        ph.generator_insecta_total_length = args.generator_insecta_total_length;
        ph.generator_insecta_head_ratio = args.generator_insecta_head_ratio;
        ph.generator_insecta_thorax_ratio = args.generator_insecta_thorax_ratio;
        ph.generator_insecta_abdomen_ratio = args.generator_insecta_abdomen_ratio;
        ph.generator_insecta_body_half_width = args.generator_insecta_body_half_width;
        ph.generator_insecta_body_half_height = args.generator_insecta_body_half_height;
        ph.generator_insecta_abdomen_taper = args.generator_insecta_abdomen_taper;
        ph.generator_insecta_head_shape = args.generator_insecta_head_shape;
        ph.generator_insecta_anchor_offset_u = args.generator_insecta_anchor_offset_u;
        ph.generator_insecta_anchor_offset_v = args.generator_insecta_anchor_offset_v;
        ph.generator_insecta_body_yaw = args.generator_insecta_body_yaw;
        ph.generator_insecta_body_arch = args.generator_insecta_body_arch;
        ph.generator_insecta_antenna_length = args.generator_insecta_antenna_length;
        ph.generator_insecta_antenna_spread = args.generator_insecta_antenna_spread;
        ph.generator_insecta_antenna_pitch = args.generator_insecta_antenna_pitch;
        ph.generator_insecta_antenna_root = args.generator_insecta_antenna_root;
        ph.generator_insecta_mandible_length = args.generator_insecta_mandible_length;
        ph.generator_insecta_mandible_spread = args.generator_insecta_mandible_spread;
        ph.generator_insecta_mandible_forward = args.generator_insecta_mandible_forward;
        ph.generator_insecta_wing_shape = args.generator_insecta_wing_shape;
        ph.generator_insecta_show_wing_fore = args.generator_insecta_show_wing_fore;
        ph.generator_insecta_wing_fore_length = args.generator_insecta_wing_fore_length;
        ph.generator_insecta_wing_fore_width = args.generator_insecta_wing_fore_width;
        ph.generator_insecta_wing_fore_spread = args.generator_insecta_wing_fore_spread;
        ph.generator_insecta_wing_fore_pitch = args.generator_insecta_wing_fore_pitch;
        ph.generator_insecta_wing_fore_offset = args.generator_insecta_wing_fore_offset;
        ph.generator_insecta_wing_fore_forward_cant = args.generator_insecta_wing_fore_forward_cant;
        ph.generator_insecta_show_wing_hind = args.generator_insecta_show_wing_hind;
        ph.generator_insecta_wing_hind_length = args.generator_insecta_wing_hind_length;
        ph.generator_insecta_wing_hind_width = args.generator_insecta_wing_hind_width;
        ph.generator_insecta_wing_hind_spread = args.generator_insecta_wing_hind_spread;
        ph.generator_insecta_wing_hind_pitch = args.generator_insecta_wing_hind_pitch;
        ph.generator_insecta_wing_hind_offset = args.generator_insecta_wing_hind_offset;
        // Fauna
        ph.generator_fauna_stance = args.generator_fauna_stance.clone();
        ph.generator_fauna_archetype = args.generator_fauna_archetype.clone();
        ph.generator_fauna_anchor_offset_u = args.generator_fauna_anchor_offset_u;
        ph.generator_fauna_anchor_offset_v = args.generator_fauna_anchor_offset_v;
        ph.generator_fauna_body_yaw = args.generator_fauna_body_yaw;
        ph.generator_fauna_body_arch = args.generator_fauna_body_arch;
        ph.generator_fauna_spine_segments = args.generator_fauna_spine_segments;
        ph.generator_fauna_body_length = args.generator_fauna_body_length;
        ph.generator_fauna_body_half_width = args.generator_fauna_body_half_width;
        ph.generator_fauna_body_half_height = args.generator_fauna_body_half_height;
        ph.generator_fauna_neck_length = args.generator_fauna_neck_length;
        ph.generator_fauna_neck_half_width = args.generator_fauna_neck_half_width;
        ph.generator_fauna_neck_half_height = args.generator_fauna_neck_half_height;
        ph.generator_fauna_head_length = args.generator_fauna_head_length;
        ph.generator_fauna_head_half_width = args.generator_fauna_head_half_width;
        ph.generator_fauna_head_half_height = args.generator_fauna_head_half_height;
        ph.generator_fauna_tail_length = args.generator_fauna_tail_length;
        ph.generator_fauna_shoulder_offset_forward = args.generator_fauna_shoulder_offset_forward;
        ph.generator_fauna_hip_offset_forward = args.generator_fauna_hip_offset_forward;
        ph.generator_fauna_front_upper_length = args.generator_fauna_front_upper_length;
        ph.generator_fauna_front_lower_length = args.generator_fauna_front_lower_length;
        ph.generator_fauna_hind_upper_length = args.generator_fauna_hind_upper_length;
        ph.generator_fauna_hind_lower_length = args.generator_fauna_hind_lower_length;
        ph.generator_fauna_auto_foot_placement = args.generator_fauna_auto_foot_placement;
        // Piscina
        ph.generator_piscina_seed = args.generator_piscina_seed;
        ph.generator_piscina_species = args.generator_piscina_species.clone();
        ph.generator_piscina_length = args.generator_piscina_length;
        ph.generator_piscina_width = args.generator_piscina_width;
        ph.generator_piscina_thickness = args.generator_piscina_thickness;
        ph.generator_piscina_spine_bend = args.generator_piscina_spine_bend;
        ph.generator_piscina_spine_s_curve = args.generator_piscina_spine_s_curve;
        ph.generator_piscina_fin_dorsal = args.generator_piscina_fin_dorsal;
        ph.generator_piscina_fin_anal = args.generator_piscina_fin_anal;
        ph.generator_piscina_fin_caudal = args.generator_piscina_fin_caudal;
        ph.generator_piscina_fin_pectoral = args.generator_piscina_fin_pectoral;
        ph.generator_piscina_fin_pelvic = args.generator_piscina_fin_pelvic;
        ph.generator_piscina_fin_adipose = args.generator_piscina_fin_adipose;
        ph.generator_piscina_show_fin_dorsal = args.generator_piscina_show_fin_dorsal;
        ph.generator_piscina_show_fin_anal = args.generator_piscina_show_fin_anal;
        ph.generator_piscina_show_fin_caudal = args.generator_piscina_show_fin_caudal;
        ph.generator_piscina_show_fin_pectoral = args.generator_piscina_show_fin_pectoral;
        ph.generator_piscina_show_fin_pelvic = args.generator_piscina_show_fin_pelvic;
        ph.generator_piscina_show_fin_adipose = args.generator_piscina_show_fin_adipose;
        ph.generator_piscina_anchor_offset_u = args.generator_piscina_anchor_offset_u;
        ph.generator_piscina_anchor_offset_v = args.generator_piscina_anchor_offset_v;
        ph.stamp_origin_x = args.stamp_origin_x;
        ph.stamp_origin_z = args.stamp_origin_z;
    }
    if args.nx < 0.0 {
        *state.preview_cursor.lock() = None;
    } else {
        *state.preview_cursor.lock() = Some((args.nx, args.ny));
    }
    Ok(())
}

/// Returns the axis index (0=X, 1=Y, 2=Z) to highlight, or 255 for none.
/// During an active drag the dragged axis stays highlighted; otherwise falls back to hover state.
fn gizmo_highlighted_axis(state: &ViewerState) -> u8 {
    match &*state.selection_gizmo_drag.lock() {
        SelectionGizmoDrag::Move { world_axis, .. } => *world_axis,
        SelectionGizmoDrag::Rotate { ring, .. } => *ring,
        SelectionGizmoDrag::None => state.hovered_gizmo_axis.load(Ordering::Relaxed),
    }
}

fn sync_gizmo_gpu(viewer: &mut WgpuViewer, state: &ViewerState, cam: &OrbitCamera) {
    let mode = *state.preview_mode.lock();
    let sel = state.selection_cells.lock();
    if sel.is_empty() || matches!(mode, PreviewMode::Stamp | PreviewMode::Punch) {
        drop(sel);
        viewer.upload_gizmo_lines(&[]);
        viewer.upload_gizmo_tris(&[]);
        viewer.upload_gizmo_delta_label(None);
        return;
    }
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

    let pending = pending_gizmo_translate(state);
    let pivot = glam::Vec3::new(
        (min_x + max_x) as f32 * 0.5 + pending.0 as f32,
        (min_y + max_y) as f32 * 0.5 + pending.1 as f32,
        (min_z + max_z) as f32 * 0.5 + pending.2 as f32,
    );
    let inv_view = cam.view_matrix().inverse();
    let cam_eye = glam::Vec3::new(inv_view.w_axis.x, inv_view.w_axis.y, inv_view.w_axis.z);
    let dist = (cam_eye - pivot).length().max(1.0);
    let arm = (dist * 0.13_f32).clamp(1.5, 20.0);

    // Axis colors in linear space (HDR target): X=red, Y=green, Z=blue
    let highlight_axis = gizmo_highlighted_axis(state);
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
        quad(&mut lv, pivot, tip, shaft_hw, col);

        // Pyramid arrowhead (4 triangles + 2-triangle base cap)
        let (u, v_ax) = perps[axis];
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

    // Rotation rings — billboard quads for each segment
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

    viewer.upload_gizmo_lines(&lv);
    viewer.upload_gizmo_tris(&tv);

    if pending != (0, 0, 0) {
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

fn sync_ping_flash(viewer: &mut WgpuViewer, state: &ViewerState, cam: &OrbitCamera) {
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
    if w > 0.0 && h > 0.0 && !f.display_name.is_empty() {
        if let Some((sx, sy)) = voxel_edit::world_to_viewport_pixels(cam, w, h, cx, cy, cz) {
            viewer.upload_ping_label(GpuPeerLabel {
                name: f.display_name.clone(),
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

fn lerp_presence(
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
fn sync_collab_peer_labels(viewer: &mut WgpuViewer, state: &ViewerState, cam: &OrbitCamera) {
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
        let Some((sx, sy)) = voxel_edit::world_to_viewport_pixels(&cam, w, h, eye.x, eye.y, eye.z)
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

fn sync_collab_peer_lines(viewer: &mut WgpuViewer, state: &ViewerState) {
    const NEAR_DIST: f32 = 0.3;
    const FAR_DIST: f32 = 12.0;
    const ASPECT: f32 = 16.0 / 9.0;
    const NEAR_ALPHA: f32 = 1.0;
    const FAR_ALPHA: f32 = 0.05;
    const SMOOTH_T: f32 = 0.12;

    let (local_id, roster, presence) = {
        let c = state.collab.lock();
        if !c.is_active() {
            viewer.clear_collab_frustum_lines();
            state.smooth_presence.lock().clear();
            return;
        }
        (c.local_peer_id, c.roster.clone(), c.presence.clone())
    };

    // Lerp smooth presence toward raw presence each frame
    let mut smooth = state.smooth_presence.lock();
    // Remove peers that left
    smooth.retain(|pid, _| presence.contains_key(pid));
    // Update / insert smoothed values
    for (&pid, raw) in &presence {
        let entry = smooth.entry(pid).or_insert(*raw);
        *entry = lerp_presence(entry, raw, SMOOTH_T);
    }

    // Lines: 24 vertices × 7 floats = 168 floats per peer
    // Tris:  4 side faces × 2 triangles × 3 verts × 7 floats = 168 floats per peer
    let peer_count = smooth.len();
    let mut line_verts: Vec<f32> = Vec::with_capacity(peer_count.saturating_mul(168));
    let mut tri_verts: Vec<f32> = Vec::with_capacity(peer_count.saturating_mul(168));
    for (&pid, pr) in smooth.iter() {
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
        let forward = (target - eye).normalize();
        // Avoid degenerate cross product when looking straight up/down
        let ref_up = if forward.y.abs() > 0.999 {
            glam::Vec3::Z
        } else {
            glam::Vec3::Y
        };
        let right = forward.cross(ref_up).normalize();
        let up = right.cross(forward);

        let (near_half_h, near_half_w, far_half_h, far_half_w) = if pr.perspective {
            let half_fov_tan = (pr.fov_y * 0.5).tan();
            let nh = NEAR_DIST * half_fov_tan;
            let fh = FAR_DIST * half_fov_tan;
            (nh, nh * ASPECT, fh, fh * ASPECT)
        } else {
            let hh = pr.ortho_half_height;
            let hw = hh * ASPECT;
            (hh, hw, hh, hw)
        };

        let near_center = eye + forward * NEAR_DIST;
        let far_center = eye + forward * FAR_DIST;

        // Near plane corners: top-left, top-right, bottom-right, bottom-left
        let ntl = near_center + up * near_half_h - right * near_half_w;
        let ntr = near_center + up * near_half_h + right * near_half_w;
        let nbr = near_center - up * near_half_h + right * near_half_w;
        let nbl = near_center - up * near_half_h - right * near_half_w;

        // Far plane corners
        let ftl = far_center + up * far_half_h - right * far_half_w;
        let ftr = far_center + up * far_half_h + right * far_half_w;
        let fbr = far_center - up * far_half_h + right * far_half_w;
        let fbl = far_center - up * far_half_h - right * far_half_w;

        // --- Wireframe edges ---
        let mut push_line = |p: glam::Vec3, a: f32| {
            line_verts.extend_from_slice(&[p.x, p.y, p.z, rf, gf, bf, a]);
        };

        // Near rectangle (4 edges)
        push_line(ntl, NEAR_ALPHA);
        push_line(ntr, NEAR_ALPHA);
        push_line(ntr, NEAR_ALPHA);
        push_line(nbr, NEAR_ALPHA);
        push_line(nbr, NEAR_ALPHA);
        push_line(nbl, NEAR_ALPHA);
        push_line(nbl, NEAR_ALPHA);
        push_line(ntl, NEAR_ALPHA);

        // Far rectangle (4 edges)
        push_line(ftl, FAR_ALPHA);
        push_line(ftr, FAR_ALPHA);
        push_line(ftr, FAR_ALPHA);
        push_line(fbr, FAR_ALPHA);
        push_line(fbr, FAR_ALPHA);
        push_line(fbl, FAR_ALPHA);
        push_line(fbl, FAR_ALPHA);
        push_line(ftl, FAR_ALPHA);

        // Connecting edges (near→far)
        push_line(ntl, NEAR_ALPHA);
        push_line(ftl, FAR_ALPHA);
        push_line(ntr, NEAR_ALPHA);
        push_line(ftr, FAR_ALPHA);
        push_line(nbr, NEAR_ALPHA);
        push_line(fbr, FAR_ALPHA);
        push_line(nbl, NEAR_ALPHA);
        push_line(fbl, FAR_ALPHA);

        // --- Filled side faces (2 triangles per quad, 4 faces) ---
        let mut push_tri = |p: glam::Vec3, a: f32| {
            tri_verts.extend_from_slice(&[p.x, p.y, p.z, rf, gf, bf, a]);
        };

        // Each side face: near_a, near_b → far_a, far_b (quad as 2 tris)
        let sides: [(glam::Vec3, glam::Vec3, glam::Vec3, glam::Vec3); 4] = [
            (ntl, ntr, ftl, ftr), // top
            (ntr, nbr, ftr, fbr), // right
            (nbr, nbl, fbr, fbl), // bottom
            (nbl, ntl, fbl, ftl), // left
        ];
        for (na, nb, fa, fb) in sides {
            // Triangle 1: na, fa, nb
            push_tri(na, NEAR_ALPHA);
            push_tri(fa, FAR_ALPHA);
            push_tri(nb, NEAR_ALPHA);
            // Triangle 2: nb, fa, fb
            push_tri(nb, NEAR_ALPHA);
            push_tri(fa, FAR_ALPHA);
            push_tri(fb, FAR_ALPHA);
        }
    }
    viewer.upload_collab_frustum_lines(&line_verts);
    viewer.upload_collab_frustum_tris(&tri_verts);
}

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

#[derive(Clone)]
enum GridBorderPrepared {
    Clear,
    Unchanged,
    Draw {
        fp: u64,
        verts: Vec<f32>,
        indices: Vec<u32>,
    },
}

fn prepare_grid_border_overlay(state: &ViewerState) -> GridBorderPrepared {
    let show = state.show_grid_borders.load(Ordering::Relaxed);
    if !show {
        return GridBorderPrepared::Clear;
    }
    let mesh_gen = state.mesh_refresh_generation.load(Ordering::Relaxed);
    let file_guard = state.current_file.lock();
    let map_guard = state.voxel_map.lock();
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
    if *state.grid_overlay_cache_key.lock() == Some(fp) {
        return GridBorderPrepared::Unchanged;
    }
    let (verts, indices) = greedy_mesh::voxel_surface_grid_line_vertices(&world);
    GridBorderPrepared::Draw { fp, verts, indices }
}

fn apply_grid_border_overlay(
    viewer: &mut WgpuViewer,
    state: &ViewerState,
    prep: GridBorderPrepared,
) {
    match prep {
        GridBorderPrepared::Clear => {
            viewer.clear_grid_border_lines();
            *state.grid_overlay_cache_key.lock() = None;
        }
        GridBorderPrepared::Unchanged => {}
        GridBorderPrepared::Draw { fp, verts, indices } => {
            if viewer.grid_border_cache_key == Some(fp) {
                return;
            }
            viewer.upload_grid_border_lines(&verts, &indices);
            viewer.grid_border_cache_key = Some(fp);
            *state.grid_overlay_cache_key.lock() = Some(fp);
        }
    }
}

#[derive(Clone)]
enum SelectionOverlayPrepared {
    Clear,
    Unchanged,
    Draw {
        fp: u64,
        solid: greedy_mesh::MeshBuffers,
        line_verts: Vec<f32>,
    },
}

fn prepare_selection_overlay(state: &ViewerState) -> SelectionOverlayPrepared {
    let sel = state.selection_cells.lock().clone();
    if sel.is_empty() {
        return SelectionOverlayPrepared::Clear;
    }
    let mesh_gen = state.mesh_refresh_generation.load(Ordering::Relaxed);
    let pending = pending_gizmo_translate(state);
    let fp = selection_overlay_cache_fingerprint(&sel, mesh_gen, pending);
    if *state.selection_overlay_cache_key.lock() == Some(fp) {
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
    let file_guard = state.current_file.lock();
    let map_guard = state.voxel_map.lock();
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

fn apply_selection_overlay(
    viewer: &mut WgpuViewer,
    state: &ViewerState,
    prep: SelectionOverlayPrepared,
) {
    match prep {
        SelectionOverlayPrepared::Clear => {
            viewer.clear_selection_overlay();
            *state.selection_overlay_cache_key.lock() = None;
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
            *state.selection_overlay_cache_key.lock() = Some(fp);
        }
    }
}

#[derive(Clone)]
enum PreviewMeshPrepared {
    Noop,
    Clear,
    Upload {
        cache_key: u64,
        instanced: greedy_mesh::PreviewInstancedResult,
    },
    /// Generator preview: lit, opaque, self-shadowing. Uses gen_preview GPU buffers.
    GenUpload {
        cache_key: u64,
        instanced: greedy_mesh::PreviewInstancedResult,
    },
}

#[inline]
fn preview_overlay_cache_key_get(state: &ViewerState) -> Option<u64> {
    *state.preview_overlay_cache_key.lock()
}

fn brush_shape_tag(s: voxel_edit::BrushShape) -> u8 {
    match s {
        voxel_edit::BrushShape::Sphere => 0,
        voxel_edit::BrushShape::Cube => 1,
        voxel_edit::BrushShape::Pyramid => 2,
        voxel_edit::BrushShape::Square => 3,
        voxel_edit::BrushShape::Circle => 4,
    }
}

fn hash_generator_rope_hover(
    sx1: f32,
    sy1: f32,
    sx2: f32,
    sy2: f32,
    tension: f32,
    gravity_dir: &str,
    brush_radius: u32,
    brush_shape: voxel_edit::BrushShape,
    color: u32,
    dbg: bool,
    mesh_gen: u64,
) -> u64 {
    let mut h = AHasher::default();
    0x52u8.hash(&mut h);
    sx1.to_bits().hash(&mut h);
    sy1.to_bits().hash(&mut h);
    sx2.to_bits().hash(&mut h);
    sy2.to_bits().hash(&mut h);
    tension.to_bits().hash(&mut h);
    gravity_dir.hash(&mut h);
    brush_radius.hash(&mut h);
    brush_shape_tag(brush_shape).hash(&mut h);
    color.hash(&mut h);
    dbg.hash(&mut h);
    mesh_gen.hash(&mut h);
    h.finish()
}

fn hash_generator_cloth_hover(
    pins: &[[i32; 3]],
    tension: f32,
    gravity_dir: &str,
    gravity_scale: f64,
    stiffness_scale: f64,
    iterations: u32,
    passes: u32,
    brush_radius: u32,
    brush_shape: voxel_edit::BrushShape,
    color: u32,
    dbg: bool,
    mesh_gen: u64,
) -> u64 {
    let mut h = AHasher::default();
    0x43u8.hash(&mut h);
    pins.len().hash(&mut h);
    for p in pins {
        p[0].hash(&mut h);
        p[1].hash(&mut h);
        p[2].hash(&mut h);
    }
    tension.to_bits().hash(&mut h);
    gravity_dir.hash(&mut h);
    gravity_scale.to_bits().hash(&mut h);
    stiffness_scale.to_bits().hash(&mut h);
    iterations.hash(&mut h);
    passes.hash(&mut h);
    brush_radius.hash(&mut h);
    brush_shape_tag(brush_shape).hash(&mut h);
    color.hash(&mut h);
    dbg.hash(&mut h);
    mesh_gen.hash(&mut h);
    h.finish()
}

fn hash_generator_rock_hover(
    sx: f32,
    sy: f32,
    size: i32,
    roughness: f32,
    seed: i32,
    color: u32,
    dbg: bool,
    mesh_gen: u64,
    count: i32,
    cluster_radius: i32,
    sink_direction: i32,
    sink_amount: i32,
) -> u64 {
    let mut h = AHasher::default();
    0x72u8.hash(&mut h); // 'r' for rock
    sx.to_bits().hash(&mut h);
    sy.to_bits().hash(&mut h);
    size.hash(&mut h);
    roughness.to_bits().hash(&mut h);
    seed.hash(&mut h);
    color.hash(&mut h);
    dbg.hash(&mut h);
    mesh_gen.hash(&mut h);
    count.hash(&mut h);
    cluster_radius.hash(&mut h);
    sink_direction.hash(&mut h);
    sink_amount.hash(&mut h);
    h.finish()
}

fn hash_generator_grass_hover(
    sx: f32,
    sy: f32,
    radius: i32,
    density: f32,
    max_height: i32,
    seed: i32,
    color: u32,
    dbg: bool,
    mesh_gen: u64,
) -> u64 {
    let mut h = AHasher::default();
    0x67u8.hash(&mut h); // 'g' for grass
    sx.to_bits().hash(&mut h);
    sy.to_bits().hash(&mut h);
    radius.hash(&mut h);
    density.to_bits().hash(&mut h);
    max_height.hash(&mut h);
    seed.hash(&mut h);
    color.hash(&mut h);
    dbg.hash(&mut h);
    mesh_gen.hash(&mut h);
    h.finish()
}

fn hash_generator_ashlar_hover(
    sx: f32,
    sy: f32,
    size: i32,
    roughness: f32,
    seed: i32,
    thickness: i32,
    color: u32,
    dbg: bool,
    mesh_gen: u64,
) -> u64 {
    let mut h = AHasher::default();
    0x61u8.hash(&mut h); // 'a' for ashlar
    sx.to_bits().hash(&mut h);
    sy.to_bits().hash(&mut h);
    size.hash(&mut h);
    roughness.to_bits().hash(&mut h);
    seed.hash(&mut h);
    thickness.hash(&mut h);
    color.hash(&mut h);
    dbg.hash(&mut h);
    mesh_gen.hash(&mut h);
    h.finish()
}

#[allow(clippy::too_many_arguments)]
fn hash_generator_flora_hover(
    sx: f32,
    sy: f32,
    seed: i32,
    height: i32,
    girth: i32,
    wobble: f32,
    taper: f32,
    stem_count: i32,
    cluster_radius: i32,
    branch_count: i32,
    branch_depth: i32,
    branch_start: f32,
    branch_spread: f32,
    braid_strands: i32,
    braid_twist: f32,
    canopy: f32,
    color: u32,
    dbg: bool,
    mesh_gen: u64,
) -> u64 {
    let mut h = AHasher::default();
    0x46u8.hash(&mut h); // 'F' for flora
    sx.to_bits().hash(&mut h);
    sy.to_bits().hash(&mut h);
    seed.hash(&mut h);
    height.hash(&mut h);
    girth.hash(&mut h);
    wobble.to_bits().hash(&mut h);
    taper.to_bits().hash(&mut h);
    stem_count.hash(&mut h);
    cluster_radius.hash(&mut h);
    branch_count.hash(&mut h);
    branch_depth.hash(&mut h);
    branch_start.to_bits().hash(&mut h);
    branch_spread.to_bits().hash(&mut h);
    braid_strands.hash(&mut h);
    braid_twist.to_bits().hash(&mut h);
    canopy.to_bits().hash(&mut h);
    color.hash(&mut h);
    dbg.hash(&mut h);
    mesh_gen.hash(&mut h);
    h.finish()
}

#[allow(clippy::too_many_arguments)]
fn hash_generator_insecta_hover(
    sx: f32,
    sy: f32,
    species: &str,
    total_length: i32,
    head_ratio: f32,
    thorax_ratio: f32,
    abdomen_ratio: f32,
    body_half_width: i32,
    body_half_height: i32,
    abdomen_taper: f32,
    head_shape: i32,
    anchor_offset_u: i32,
    anchor_offset_v: i32,
    body_yaw: f32,
    body_arch: f32,
    antenna_length: i32,
    antenna_spread: f32,
    antenna_pitch: f32,
    antenna_root: i32,
    mandible_length: i32,
    mandible_spread: f32,
    mandible_forward: i32,
    wing_shape: i32,
    show_wing_fore: bool,
    wing_fore_length: i32,
    wing_fore_width: i32,
    wing_fore_spread: f32,
    wing_fore_pitch: f32,
    wing_fore_offset: i32,
    wing_fore_forward_cant: f32,
    show_wing_hind: bool,
    wing_hind_length: i32,
    wing_hind_width: i32,
    wing_hind_spread: f32,
    wing_hind_pitch: f32,
    wing_hind_offset: i32,
    color: u32,
    dbg: bool,
    mesh_gen: u64,
) -> u64 {
    let mut h = AHasher::default();
    0x49u8.hash(&mut h); // 'I' for insecta
    sx.to_bits().hash(&mut h);
    sy.to_bits().hash(&mut h);
    species.hash(&mut h);
    total_length.hash(&mut h);
    head_ratio.to_bits().hash(&mut h);
    thorax_ratio.to_bits().hash(&mut h);
    abdomen_ratio.to_bits().hash(&mut h);
    body_half_width.hash(&mut h);
    body_half_height.hash(&mut h);
    abdomen_taper.to_bits().hash(&mut h);
    head_shape.hash(&mut h);
    anchor_offset_u.hash(&mut h);
    anchor_offset_v.hash(&mut h);
    body_yaw.to_bits().hash(&mut h);
    body_arch.to_bits().hash(&mut h);
    antenna_length.hash(&mut h);
    antenna_spread.to_bits().hash(&mut h);
    antenna_pitch.to_bits().hash(&mut h);
    antenna_root.hash(&mut h);
    mandible_length.hash(&mut h);
    mandible_spread.to_bits().hash(&mut h);
    mandible_forward.hash(&mut h);
    wing_shape.hash(&mut h);
    show_wing_fore.hash(&mut h);
    wing_fore_length.hash(&mut h);
    wing_fore_width.hash(&mut h);
    wing_fore_spread.to_bits().hash(&mut h);
    wing_fore_pitch.to_bits().hash(&mut h);
    wing_fore_offset.hash(&mut h);
    wing_fore_forward_cant.to_bits().hash(&mut h);
    show_wing_hind.hash(&mut h);
    wing_hind_length.hash(&mut h);
    wing_hind_width.hash(&mut h);
    wing_hind_spread.to_bits().hash(&mut h);
    wing_hind_pitch.to_bits().hash(&mut h);
    wing_hind_offset.hash(&mut h);
    color.hash(&mut h);
    dbg.hash(&mut h);
    mesh_gen.hash(&mut h);
    h.finish()
}

#[allow(clippy::too_many_arguments)]
fn hash_generator_fauna_hover(
    sx: f32,
    sy: f32,
    stance: &str,
    archetype: &str,
    anchor_offset_u: i32,
    anchor_offset_v: i32,
    body_yaw: f32,
    body_arch: f32,
    spine_segments: i32,
    body_length: i32,
    body_half_width: i32,
    body_half_height: i32,
    neck_length: i32,
    neck_half_width: i32,
    neck_half_height: i32,
    head_length: i32,
    head_half_width: i32,
    head_half_height: i32,
    tail_length: i32,
    shoulder_offset_forward: i32,
    hip_offset_forward: i32,
    front_upper_length: i32,
    front_lower_length: i32,
    hind_upper_length: i32,
    hind_lower_length: i32,
    auto_foot_placement: bool,
    color: u32,
    dbg: bool,
    mesh_gen: u64,
) -> u64 {
    let mut h = AHasher::default();
    0x41u8.hash(&mut h); // 'A' for fAuna
    sx.to_bits().hash(&mut h);
    sy.to_bits().hash(&mut h);
    stance.hash(&mut h);
    archetype.hash(&mut h);
    anchor_offset_u.hash(&mut h);
    anchor_offset_v.hash(&mut h);
    body_yaw.to_bits().hash(&mut h);
    body_arch.to_bits().hash(&mut h);
    spine_segments.hash(&mut h);
    body_length.hash(&mut h);
    body_half_width.hash(&mut h);
    body_half_height.hash(&mut h);
    neck_length.hash(&mut h);
    neck_half_width.hash(&mut h);
    neck_half_height.hash(&mut h);
    head_length.hash(&mut h);
    head_half_width.hash(&mut h);
    head_half_height.hash(&mut h);
    tail_length.hash(&mut h);
    shoulder_offset_forward.hash(&mut h);
    hip_offset_forward.hash(&mut h);
    front_upper_length.hash(&mut h);
    front_lower_length.hash(&mut h);
    hind_upper_length.hash(&mut h);
    hind_lower_length.hash(&mut h);
    auto_foot_placement.hash(&mut h);
    color.hash(&mut h);
    dbg.hash(&mut h);
    mesh_gen.hash(&mut h);
    h.finish()
}

#[allow(clippy::too_many_arguments)]
fn hash_generator_piscina_hover(
    sx: f32,
    sy: f32,
    seed: i32,
    species: &str,
    length: i32,
    width_param: i32,
    thickness: i32,
    spine_bend: f32,
    spine_s_curve: f32,
    fin_dorsal: i32,
    fin_anal: i32,
    fin_caudal: i32,
    fin_pectoral: i32,
    fin_pelvic: i32,
    fin_adipose: i32,
    show_fin_dorsal: bool,
    show_fin_anal: bool,
    show_fin_caudal: bool,
    show_fin_pectoral: bool,
    show_fin_pelvic: bool,
    show_fin_adipose: bool,
    anchor_offset_u: i32,
    anchor_offset_v: i32,
    color: u32,
    dbg: bool,
    mesh_gen: u64,
) -> u64 {
    let mut h = AHasher::default();
    0x50u8.hash(&mut h); // 'P' for piscina
    sx.to_bits().hash(&mut h);
    sy.to_bits().hash(&mut h);
    seed.hash(&mut h);
    species.hash(&mut h);
    length.hash(&mut h);
    width_param.hash(&mut h);
    thickness.hash(&mut h);
    spine_bend.to_bits().hash(&mut h);
    spine_s_curve.to_bits().hash(&mut h);
    fin_dorsal.hash(&mut h);
    fin_anal.hash(&mut h);
    fin_caudal.hash(&mut h);
    fin_pectoral.hash(&mut h);
    fin_pelvic.hash(&mut h);
    fin_adipose.hash(&mut h);
    show_fin_dorsal.hash(&mut h);
    show_fin_anal.hash(&mut h);
    show_fin_caudal.hash(&mut h);
    show_fin_pectoral.hash(&mut h);
    show_fin_pelvic.hash(&mut h);
    show_fin_adipose.hash(&mut h);
    anchor_offset_u.hash(&mut h);
    anchor_offset_v.hash(&mut h);
    color.hash(&mut h);
    dbg.hash(&mut h);
    mesh_gen.hash(&mut h);
    h.finish()
}

fn hash_generator_roof_hover(
    pins: &[[i32; 3]],
    style: &str,
    height: i32,
    thickness: i32,
    break_ratio: f32,
    wall_height: i32,
    parapet_height: i32,
    salt_skew: f32,
    hollow: bool,
    color: u32,
    dbg: bool,
    mesh_gen: u64,
) -> u64 {
    let mut h = AHasher::default();
    0x52u8.hash(&mut h); // 'R' for roof
    for p in pins {
        p.hash(&mut h);
    }
    style.hash(&mut h);
    height.hash(&mut h);
    thickness.hash(&mut h);
    break_ratio.to_bits().hash(&mut h);
    wall_height.hash(&mut h);
    parapet_height.hash(&mut h);
    salt_skew.to_bits().hash(&mut h);
    hollow.hash(&mut h);
    color.hash(&mut h);
    dbg.hash(&mut h);
    mesh_gen.hash(&mut h);
    h.finish()
}

fn prepare_preview_mesh(
    state: &ViewerState,
    cam: &OrbitCamera,
    viewport_w: u32,
    viewport_h: u32,
) -> PreviewMeshPrepared {
    if state
        .stroke_preview_suppresses_hover
        .load(Ordering::Relaxed)
    {
        return PreviewMeshPrepared::Noop;
    }
    let dbg = state.viewport_cursor_debug_overlay.load(Ordering::Relaxed);
    let (cursor, mode) = {
        let c = state.preview_cursor.lock();
        let m = state.preview_mode.lock();
        (*c, *m)
    };

    if matches!(mode, PreviewMode::Navigate | PreviewMode::Fly) {
        return PreviewMeshPrepared::Clear;
    }

    // Pin-based generators (cloth, roof) don't need a cursor position — run
    // them even when the mouse is outside the viewport so the preview persists.
    if cursor.is_none() && matches!(mode, PreviewMode::Add) {
        let file_guard = state.current_file.lock();
        let map_guard = state.voxel_map.lock();
        if let (Some(file), Some(vmap)) = (file_guard.as_ref(), map_guard.as_ref()) {
            let hover = state.preview_hover.lock();
            let ctx = &*hover;
            let mesh_gen = state.mesh_refresh_generation.load(Ordering::Relaxed);
            if let Some(ref gk) = ctx.generator_kind {
                match gk.as_str() {
                    "cloth" => {
                        if ctx.generator_cloth_pins.len() >= 3 {
                            let sim = crate::generators::ClothSimOptions {
                                gravity_scale: ctx.generator_cloth_gravity_scale.max(0.0),
                                stiffness_scale: ctx
                                    .generator_cloth_stiffness_scale
                                    .clamp(0.05, 2.0),
                                iterations: if ctx.generator_cloth_iterations > 0 {
                                    Some(ctx.generator_cloth_iterations.clamp(4, 96))
                                } else {
                                    None
                                },
                                constraint_passes: ctx
                                    .generator_cloth_constraint_passes
                                    .clamp(1, 6),
                            };
                            let cells = crate::generators::preview_cloth_voxels(
                                &ctx.generator_cloth_pins,
                                ctx.generator_cloth_tension,
                                ctx.generator_cloth_gravity_direction.as_str(),
                                ctx.brush_radius,
                                ctx.brush_shape,
                                &sim,
                            );
                            if !cells.is_empty() {
                                let key = hash_generator_cloth_hover(
                                    &ctx.generator_cloth_pins,
                                    ctx.generator_cloth_tension,
                                    ctx.generator_cloth_gravity_direction.as_str(),
                                    ctx.generator_cloth_gravity_scale,
                                    ctx.generator_cloth_stiffness_scale,
                                    ctx.generator_cloth_iterations,
                                    ctx.generator_cloth_constraint_passes,
                                    ctx.brush_radius,
                                    ctx.brush_shape,
                                    ctx.color,
                                    dbg,
                                    mesh_gen,
                                );
                                if preview_overlay_cache_key_get(state) == Some(key) {
                                    return PreviewMeshPrepared::Noop;
                                }
                                let set: AHashSet<_> = cells.iter().copied().collect();
                                let instanced = stroke_preview_meshes_for_union(
                                    voxel_edit::EditTool::Add,
                                    &set,
                                    vmap,
                                    file,
                                    dbg,
                                    ctx.color,
                                    None,
                                );
                                return PreviewMeshPrepared::Upload {
                                    cache_key: key,
                                    instanced,
                                };
                            }
                        }
                    }
                    "roof" => {
                        if !ctx.generator_roof_pins.is_empty() {
                            let mut instanced = if ctx.generator_roof_pins.len() >= 3 {
                                let cells = crate::generators::preview_roof_voxels(
                                    &ctx.generator_roof_pins,
                                    &ctx.generator_roof_style,
                                    ctx.generator_roof_height,
                                    ctx.generator_roof_thickness,
                                    0,
                                    0,
                                    ctx.generator_roof_break_ratio,
                                    ctx.generator_roof_wall_height,
                                    ctx.generator_roof_parapet_height,
                                    ctx.generator_roof_salt_skew,
                                    ctx.generator_roof_hollow,
                                );
                                if !cells.is_empty() {
                                    let set: AHashSet<_> = cells.iter().copied().collect();
                                    stroke_preview_meshes_for_union(
                                        voxel_edit::EditTool::Add,
                                        &set,
                                        vmap,
                                        file,
                                        dbg,
                                        ctx.color,
                                        None,
                                    )
                                } else {
                                    greedy_mesh::PreviewInstancedResult::empty()
                                }
                            } else {
                                greedy_mesh::PreviewInstancedResult::empty()
                            };
                            // Yellow markers at each pin position.
                            append_polygon_vertex_marker_meshes(
                                &mut instanced.extra_solid,
                                &mut instanced.extra_wire,
                                &ctx.generator_roof_pins,
                                vmap,
                                file,
                                dbg,
                            );
                            let key = hash_generator_roof_hover(
                                &ctx.generator_roof_pins,
                                &ctx.generator_roof_style,
                                ctx.generator_roof_height,
                                ctx.generator_roof_thickness,
                                ctx.generator_roof_break_ratio,
                                ctx.generator_roof_wall_height,
                                ctx.generator_roof_parapet_height,
                                ctx.generator_roof_salt_skew,
                                ctx.generator_roof_hollow,
                                ctx.color,
                                dbg,
                                mesh_gen,
                            );
                            if preview_overlay_cache_key_get(state) == Some(key) {
                                return PreviewMeshPrepared::Noop;
                            }
                            return PreviewMeshPrepared::Upload {
                                cache_key: key,
                                instanced,
                            };
                        }
                    }
                    _ => {}
                }
            }
        }
        return PreviewMeshPrepared::Clear;
    }

    let Some((nx, ny)) = cursor else {
        return PreviewMeshPrepared::Clear;
    };

    let file_guard = state.current_file.lock();
    let map_guard = state.voxel_map.lock();
    let Some(file) = file_guard.as_ref() else {
        return PreviewMeshPrepared::Clear;
    };
    let Some(vmap) = map_guard.as_ref() else {
        return PreviewMeshPrepared::Clear;
    };

    let w = viewport_w as f32;
    let h = viewport_h as f32;
    let (sx, sy) = viewport_texels_from_norm(nx, ny, w, h);

    if matches!(mode, PreviewMode::Squishy) {
        let hover = state.preview_hover.lock();
        let preview_radius_i = hover.brush_radius.max(2).min(64);
        let gizmo_drag = state.squishy_gizmo_drag.lock().is_some();
        let max_v = if gizmo_drag { 12_000 } else { 24_000 };

        let session_snap = state.squishy_session.lock().clone();

        let add_anchor = if session_snap.mode == generators::SquishyMode::Add {
            if session_snap.add_snap_to_surface {
                voxel_edit::preview_add_cell(file, vmap, cam, w, h, sx, sy).map(|(c, _)| c)
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
            hover.color,
        );
        if preview_overlay_cache_key_get(state) == Some(key) {
            return PreviewMeshPrepared::Noop;
        }

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
            return PreviewMeshPrepared::Clear;
        }

        let set: AHashSet<_> = coords.iter().copied().collect();
        let mut instanced = stroke_preview_meshes_for_union(
            voxel_edit::EditTool::Add,
            &set,
            vmap,
            file,
            dbg,
            hover.color,
            None,
        );

        if show_gizmo {
            generators::append_squishy_gizmo_wire(&session_snap, cam, &mut instanced.extra_wire);
        }

        if has_pick_chrome {
            generators::append_squishy_metaball_pick_rings(
                &mut instanced.extra_wire,
                &session_snap,
                add_anchor,
                preview_radius_i as i32,
                delete_hover_id,
            );
        }

        return PreviewMeshPrepared::Upload {
            cache_key: key,
            instanced,
        };
    }

    let hover = state.preview_hover.lock();
    let ctx = &*hover;

    if matches!(mode, PreviewMode::Add) {
        if let Some(ref gk) = ctx.generator_kind {
            let mesh_gen = state.mesh_refresh_generation.load(Ordering::Relaxed);
            match gk.as_str() {
                "rope" => {
                    if let (Some(n1x), Some(n1y)) =
                        (ctx.generator_rope_first_nx, ctx.generator_rope_first_ny)
                    {
                        let (sx1, sy1) = viewport_texels_from_norm(n1x, n1y, w, h);
                        let cells = crate::generators::preview_rope_voxels_between_screens(
                            file,
                            vmap,
                            cam,
                            w,
                            h,
                            sx1,
                            sy1,
                            sx,
                            sy,
                            ctx.generator_rope_tension,
                            ctx.brush_radius,
                            ctx.brush_shape,
                            &ctx.generator_rope_gravity_direction,
                        );
                        if !cells.is_empty() {
                            let key = hash_generator_rope_hover(
                                sx1,
                                sy1,
                                sx,
                                sy,
                                ctx.generator_rope_tension,
                                &ctx.generator_rope_gravity_direction,
                                ctx.brush_radius,
                                ctx.brush_shape,
                                ctx.color,
                                dbg,
                                mesh_gen,
                            );
                            if preview_overlay_cache_key_get(state) == Some(key) {
                                return PreviewMeshPrepared::Noop;
                            }
                            let set: AHashSet<_> = cells.iter().copied().collect();
                            let instanced = stroke_preview_meshes_for_union(
                                voxel_edit::EditTool::Add,
                                &set,
                                vmap,
                                file,
                                dbg,
                                ctx.color,
                                None,
                            );
                            return PreviewMeshPrepared::Upload {
                                cache_key: key,
                                instanced,
                            };
                        }
                    }
                }
                "cloth" => {
                    if ctx.generator_cloth_pins.len() >= 3 {
                        let sim = crate::generators::ClothSimOptions {
                            gravity_scale: ctx.generator_cloth_gravity_scale.max(0.0),
                            stiffness_scale: ctx.generator_cloth_stiffness_scale.clamp(0.05, 2.0),
                            iterations: if ctx.generator_cloth_iterations > 0 {
                                Some(ctx.generator_cloth_iterations.clamp(4, 96))
                            } else {
                                None
                            },
                            constraint_passes: ctx.generator_cloth_constraint_passes.clamp(1, 6),
                        };
                        let cells = crate::generators::preview_cloth_voxels(
                            &ctx.generator_cloth_pins,
                            ctx.generator_cloth_tension,
                            ctx.generator_cloth_gravity_direction.as_str(),
                            ctx.brush_radius,
                            ctx.brush_shape,
                            &sim,
                        );
                        if !cells.is_empty() {
                            let key = hash_generator_cloth_hover(
                                &ctx.generator_cloth_pins,
                                ctx.generator_cloth_tension,
                                ctx.generator_cloth_gravity_direction.as_str(),
                                ctx.generator_cloth_gravity_scale,
                                ctx.generator_cloth_stiffness_scale,
                                ctx.generator_cloth_iterations,
                                ctx.generator_cloth_constraint_passes,
                                ctx.brush_radius,
                                ctx.brush_shape,
                                ctx.color,
                                dbg,
                                mesh_gen,
                            );
                            if preview_overlay_cache_key_get(state) == Some(key) {
                                return PreviewMeshPrepared::Noop;
                            }
                            let set: AHashSet<_> = cells.iter().copied().collect();
                            let instanced = stroke_preview_meshes_for_union(
                                voxel_edit::EditTool::Add,
                                &set,
                                vmap,
                                file,
                                dbg,
                                ctx.color,
                                None,
                            );
                            return PreviewMeshPrepared::Upload {
                                cache_key: key,
                                instanced,
                            };
                        }
                    }
                }
                "rocks" => {
                    let cells = crate::generators::preview_rock_at_screen(
                        file,
                        vmap,
                        cam,
                        w,
                        h,
                        sx,
                        sy,
                        ctx.generator_rock_seed,
                        ctx.generator_rock_size,
                        ctx.generator_rock_roughness,
                        ctx.generator_rock_count,
                        ctx.generator_rock_cluster_radius,
                        ctx.generator_rock_sink_direction,
                        ctx.generator_rock_sink_amount,
                    );
                    if !cells.is_empty() {
                        let key = hash_generator_rock_hover(
                            sx,
                            sy,
                            ctx.generator_rock_size,
                            ctx.generator_rock_roughness,
                            ctx.generator_rock_seed,
                            ctx.color,
                            dbg,
                            mesh_gen,
                            ctx.generator_rock_count,
                            ctx.generator_rock_cluster_radius,
                            ctx.generator_rock_sink_direction,
                            ctx.generator_rock_sink_amount,
                        );
                        if preview_overlay_cache_key_get(state) == Some(key) {
                            return PreviewMeshPrepared::Noop;
                        }
                        const NBRS: [(i32, i32, i32); 6] = [
                            (1, 0, 0),
                            (-1, 0, 0),
                            (0, 1, 0),
                            (0, -1, 0),
                            (0, 0, 1),
                            (0, 0, -1),
                        ];
                        let set: AHashSet<_> = cells.iter().copied().collect();
                        let visible: AHashSet<_> = set
                            .iter()
                            .filter(|&&(x, y, z)| {
                                NBRS.iter()
                                    .any(|&(dx, dy, dz)| !set.contains(&(x + dx, y + dy, z + dz)))
                            })
                            .copied()
                            .collect();
                        let instanced = stroke_preview_meshes_for_union(
                            voxel_edit::EditTool::Add,
                            &visible,
                            vmap,
                            file,
                            dbg,
                            ctx.color,
                            None,
                        );
                        return PreviewMeshPrepared::GenUpload {
                            cache_key: key,
                            instanced,
                        };
                    }
                }
                "grass" => {
                    let cells = crate::generators::preview_grass_at_screen(
                        file,
                        vmap,
                        cam,
                        w,
                        h,
                        sx,
                        sy,
                        ctx.generator_grass_seed,
                        ctx.generator_grass_radius,
                        ctx.generator_grass_density,
                        ctx.generator_grass_max_height,
                    );
                    if !cells.is_empty() {
                        let key = hash_generator_grass_hover(
                            sx,
                            sy,
                            ctx.generator_grass_radius,
                            ctx.generator_grass_density,
                            ctx.generator_grass_max_height,
                            ctx.generator_grass_seed,
                            ctx.color,
                            dbg,
                            mesh_gen,
                        );
                        if preview_overlay_cache_key_get(state) == Some(key) {
                            return PreviewMeshPrepared::Noop;
                        }
                        const NBRS: [(i32, i32, i32); 6] = [
                            (1, 0, 0),
                            (-1, 0, 0),
                            (0, 1, 0),
                            (0, -1, 0),
                            (0, 0, 1),
                            (0, 0, -1),
                        ];
                        let set: AHashSet<_> = cells.iter().copied().collect();
                        let visible: AHashSet<_> = set
                            .iter()
                            .filter(|&&(x, y, z)| {
                                NBRS.iter()
                                    .any(|&(dx, dy, dz)| !set.contains(&(x + dx, y + dy, z + dz)))
                            })
                            .copied()
                            .collect();
                        let instanced = stroke_preview_meshes_for_union(
                            voxel_edit::EditTool::Add,
                            &visible,
                            vmap,
                            file,
                            dbg,
                            ctx.color,
                            None,
                        );
                        return PreviewMeshPrepared::GenUpload {
                            cache_key: key,
                            instanced,
                        };
                    }
                }
                "ashlar" => {
                    let cells = crate::generators::preview_ashlar_at_screen(
                        file,
                        vmap,
                        cam,
                        w,
                        h,
                        sx,
                        sy,
                        ctx.generator_ashlar_seed,
                        ctx.generator_ashlar_size,
                        ctx.generator_ashlar_roughness,
                        ctx.generator_ashlar_thickness,
                    );
                    if !cells.is_empty() {
                        let key = hash_generator_ashlar_hover(
                            sx,
                            sy,
                            ctx.generator_ashlar_size,
                            ctx.generator_ashlar_roughness,
                            ctx.generator_ashlar_seed,
                            ctx.generator_ashlar_thickness,
                            ctx.color,
                            dbg,
                            mesh_gen,
                        );
                        if preview_overlay_cache_key_get(state) == Some(key) {
                            return PreviewMeshPrepared::Noop;
                        }
                        const NBRS: [(i32, i32, i32); 6] = [
                            (1, 0, 0),
                            (-1, 0, 0),
                            (0, 1, 0),
                            (0, -1, 0),
                            (0, 0, 1),
                            (0, 0, -1),
                        ];
                        let set: AHashSet<_> = cells.iter().copied().collect();
                        let visible: AHashSet<_> = set
                            .iter()
                            .filter(|&&(x, y, z)| {
                                NBRS.iter()
                                    .any(|&(dx, dy, dz)| !set.contains(&(x + dx, y + dy, z + dz)))
                            })
                            .copied()
                            .collect();
                        let instanced = stroke_preview_meshes_for_union(
                            voxel_edit::EditTool::Add,
                            &visible,
                            vmap,
                            file,
                            dbg,
                            ctx.color,
                            None,
                        );
                        return PreviewMeshPrepared::GenUpload {
                            cache_key: key,
                            instanced,
                        };
                    }
                }
                "flora" => {
                    let material = voxelle::MaterialId::from_str_id(&ctx.material);
                    let cells = crate::generators::preview_flora_at_screen(
                        file,
                        vmap,
                        cam,
                        w,
                        h,
                        sx,
                        sy,
                        ctx.generator_flora_seed,
                        ctx.generator_flora_height,
                        ctx.generator_flora_girth,
                        ctx.generator_flora_wobble,
                        ctx.generator_flora_taper,
                        ctx.generator_flora_stem_count,
                        ctx.generator_flora_cluster_radius,
                        ctx.generator_flora_branch_count,
                        ctx.generator_flora_branch_depth,
                        ctx.generator_flora_branch_start,
                        ctx.generator_flora_branch_spread,
                        ctx.generator_flora_braid_strands,
                        ctx.generator_flora_braid_twist,
                        ctx.generator_flora_canopy,
                        ctx.color,
                        material,
                    );
                    if !cells.is_empty() {
                        let key = hash_generator_flora_hover(
                            sx,
                            sy,
                            ctx.generator_flora_seed,
                            ctx.generator_flora_height,
                            ctx.generator_flora_girth,
                            ctx.generator_flora_wobble,
                            ctx.generator_flora_taper,
                            ctx.generator_flora_stem_count,
                            ctx.generator_flora_cluster_radius,
                            ctx.generator_flora_branch_count,
                            ctx.generator_flora_branch_depth,
                            ctx.generator_flora_branch_start,
                            ctx.generator_flora_branch_spread,
                            ctx.generator_flora_braid_strands,
                            ctx.generator_flora_braid_twist,
                            ctx.generator_flora_canopy,
                            ctx.color,
                            dbg,
                            mesh_gen,
                        );
                        if preview_overlay_cache_key_get(state) == Some(key) {
                            return PreviewMeshPrepared::Noop;
                        }
                        const NBRS: [(i32, i32, i32); 6] = [
                            (1, 0, 0),
                            (-1, 0, 0),
                            (0, 1, 0),
                            (0, -1, 0),
                            (0, 0, 1),
                            (0, 0, -1),
                        ];
                        let color_map: AHashMap<(i32, i32, i32), u32> =
                            cells.iter().cloned().collect();
                        let visible: AHashSet<_> = color_map
                            .keys()
                            .filter(|&&(x, y, z)| {
                                NBRS.iter().any(|&(dx, dy, dz)| {
                                    !color_map.contains_key(&(x + dx, y + dy, z + dz))
                                })
                            })
                            .copied()
                            .collect();
                        let fallback = ctx.color;
                        let instanced = stroke_preview_meshes_for_union(
                            voxel_edit::EditTool::Add,
                            &visible,
                            vmap,
                            file,
                            dbg,
                            fallback,
                            Some(&|x, y, z| *color_map.get(&(x, y, z)).unwrap_or(&fallback)),
                        );
                        return PreviewMeshPrepared::GenUpload {
                            cache_key: key,
                            instanced,
                        };
                    }
                }
                "insecta" => {
                    let material = voxelle::MaterialId::from_str_id(&ctx.material);
                    let cells = crate::generators::preview_insecta_at_screen(
                        file,
                        vmap,
                        cam,
                        w,
                        h,
                        sx,
                        sy,
                        &ctx.generator_insecta_species,
                        ctx.generator_insecta_total_length,
                        ctx.generator_insecta_head_ratio,
                        ctx.generator_insecta_thorax_ratio,
                        ctx.generator_insecta_abdomen_ratio,
                        ctx.generator_insecta_body_half_width,
                        ctx.generator_insecta_body_half_height,
                        ctx.generator_insecta_abdomen_taper,
                        ctx.generator_insecta_head_shape,
                        ctx.generator_insecta_anchor_offset_u,
                        ctx.generator_insecta_anchor_offset_v,
                        ctx.generator_insecta_body_yaw,
                        ctx.generator_insecta_body_arch,
                        ctx.generator_insecta_antenna_length,
                        ctx.generator_insecta_antenna_spread,
                        ctx.generator_insecta_antenna_pitch,
                        ctx.generator_insecta_antenna_root,
                        ctx.generator_insecta_mandible_length,
                        ctx.generator_insecta_mandible_spread,
                        ctx.generator_insecta_mandible_forward,
                        ctx.generator_insecta_wing_shape,
                        ctx.generator_insecta_show_wing_fore,
                        ctx.generator_insecta_wing_fore_length,
                        ctx.generator_insecta_wing_fore_width,
                        ctx.generator_insecta_wing_fore_spread,
                        ctx.generator_insecta_wing_fore_pitch,
                        ctx.generator_insecta_wing_fore_offset,
                        ctx.generator_insecta_wing_fore_forward_cant,
                        ctx.generator_insecta_show_wing_hind,
                        ctx.generator_insecta_wing_hind_length,
                        ctx.generator_insecta_wing_hind_width,
                        ctx.generator_insecta_wing_hind_spread,
                        ctx.generator_insecta_wing_hind_pitch,
                        ctx.generator_insecta_wing_hind_offset,
                        ctx.color,
                        material,
                    );
                    if !cells.is_empty() {
                        let key = hash_generator_insecta_hover(
                            sx,
                            sy,
                            &ctx.generator_insecta_species,
                            ctx.generator_insecta_total_length,
                            ctx.generator_insecta_head_ratio,
                            ctx.generator_insecta_thorax_ratio,
                            ctx.generator_insecta_abdomen_ratio,
                            ctx.generator_insecta_body_half_width,
                            ctx.generator_insecta_body_half_height,
                            ctx.generator_insecta_abdomen_taper,
                            ctx.generator_insecta_head_shape,
                            ctx.generator_insecta_anchor_offset_u,
                            ctx.generator_insecta_anchor_offset_v,
                            ctx.generator_insecta_body_yaw,
                            ctx.generator_insecta_body_arch,
                            ctx.generator_insecta_antenna_length,
                            ctx.generator_insecta_antenna_spread,
                            ctx.generator_insecta_antenna_pitch,
                            ctx.generator_insecta_antenna_root,
                            ctx.generator_insecta_mandible_length,
                            ctx.generator_insecta_mandible_spread,
                            ctx.generator_insecta_mandible_forward,
                            ctx.generator_insecta_wing_shape,
                            ctx.generator_insecta_show_wing_fore,
                            ctx.generator_insecta_wing_fore_length,
                            ctx.generator_insecta_wing_fore_width,
                            ctx.generator_insecta_wing_fore_spread,
                            ctx.generator_insecta_wing_fore_pitch,
                            ctx.generator_insecta_wing_fore_offset,
                            ctx.generator_insecta_wing_fore_forward_cant,
                            ctx.generator_insecta_show_wing_hind,
                            ctx.generator_insecta_wing_hind_length,
                            ctx.generator_insecta_wing_hind_width,
                            ctx.generator_insecta_wing_hind_spread,
                            ctx.generator_insecta_wing_hind_pitch,
                            ctx.generator_insecta_wing_hind_offset,
                            ctx.color,
                            dbg,
                            mesh_gen,
                        );
                        if preview_overlay_cache_key_get(state) == Some(key) {
                            return PreviewMeshPrepared::Noop;
                        }
                        const NBRS: [(i32, i32, i32); 6] = [
                            (1, 0, 0),
                            (-1, 0, 0),
                            (0, 1, 0),
                            (0, -1, 0),
                            (0, 0, 1),
                            (0, 0, -1),
                        ];
                        let color_map: AHashMap<(i32, i32, i32), u32> =
                            cells.iter().cloned().collect();
                        let visible: AHashSet<_> = color_map
                            .keys()
                            .filter(|&&(x, y, z)| {
                                NBRS.iter().any(|&(dx, dy, dz)| {
                                    !color_map.contains_key(&(x + dx, y + dy, z + dz))
                                })
                            })
                            .copied()
                            .collect();
                        let fallback = ctx.color;
                        let instanced = stroke_preview_meshes_for_union(
                            voxel_edit::EditTool::Add,
                            &visible,
                            vmap,
                            file,
                            dbg,
                            fallback,
                            Some(&|x, y, z| *color_map.get(&(x, y, z)).unwrap_or(&fallback)),
                        );
                        return PreviewMeshPrepared::GenUpload {
                            cache_key: key,
                            instanced,
                        };
                    }
                }
                "fauna" => {
                    let material = voxelle::MaterialId::from_str_id(&ctx.material);
                    let cells = crate::generators::preview_fauna_at_screen(
                        file,
                        vmap,
                        cam,
                        w,
                        h,
                        sx,
                        sy,
                        &ctx.generator_fauna_stance,
                        &ctx.generator_fauna_archetype,
                        ctx.generator_fauna_anchor_offset_u,
                        ctx.generator_fauna_anchor_offset_v,
                        ctx.generator_fauna_body_yaw,
                        ctx.generator_fauna_body_arch,
                        ctx.generator_fauna_spine_segments,
                        ctx.generator_fauna_body_length,
                        ctx.generator_fauna_body_half_width,
                        ctx.generator_fauna_body_half_height,
                        ctx.generator_fauna_neck_length,
                        ctx.generator_fauna_neck_half_width,
                        ctx.generator_fauna_neck_half_height,
                        ctx.generator_fauna_head_length,
                        ctx.generator_fauna_head_half_width,
                        ctx.generator_fauna_head_half_height,
                        ctx.generator_fauna_tail_length,
                        ctx.generator_fauna_shoulder_offset_forward,
                        ctx.generator_fauna_hip_offset_forward,
                        ctx.generator_fauna_front_upper_length,
                        ctx.generator_fauna_front_lower_length,
                        ctx.generator_fauna_hind_upper_length,
                        ctx.generator_fauna_hind_lower_length,
                        ctx.generator_fauna_auto_foot_placement,
                        ctx.color,
                        material,
                    );
                    if !cells.is_empty() {
                        let key = hash_generator_fauna_hover(
                            sx,
                            sy,
                            &ctx.generator_fauna_stance,
                            &ctx.generator_fauna_archetype,
                            ctx.generator_fauna_anchor_offset_u,
                            ctx.generator_fauna_anchor_offset_v,
                            ctx.generator_fauna_body_yaw,
                            ctx.generator_fauna_body_arch,
                            ctx.generator_fauna_spine_segments,
                            ctx.generator_fauna_body_length,
                            ctx.generator_fauna_body_half_width,
                            ctx.generator_fauna_body_half_height,
                            ctx.generator_fauna_neck_length,
                            ctx.generator_fauna_neck_half_width,
                            ctx.generator_fauna_neck_half_height,
                            ctx.generator_fauna_head_length,
                            ctx.generator_fauna_head_half_width,
                            ctx.generator_fauna_head_half_height,
                            ctx.generator_fauna_tail_length,
                            ctx.generator_fauna_shoulder_offset_forward,
                            ctx.generator_fauna_hip_offset_forward,
                            ctx.generator_fauna_front_upper_length,
                            ctx.generator_fauna_front_lower_length,
                            ctx.generator_fauna_hind_upper_length,
                            ctx.generator_fauna_hind_lower_length,
                            ctx.generator_fauna_auto_foot_placement,
                            ctx.color,
                            dbg,
                            mesh_gen,
                        );
                        if preview_overlay_cache_key_get(state) == Some(key) {
                            return PreviewMeshPrepared::Noop;
                        }
                        const NBRS: [(i32, i32, i32); 6] = [
                            (1, 0, 0),
                            (-1, 0, 0),
                            (0, 1, 0),
                            (0, -1, 0),
                            (0, 0, 1),
                            (0, 0, -1),
                        ];
                        let color_map: AHashMap<(i32, i32, i32), u32> =
                            cells.iter().cloned().collect();
                        let visible: AHashSet<_> = color_map
                            .keys()
                            .filter(|&&(x, y, z)| {
                                NBRS.iter().any(|&(dx, dy, dz)| {
                                    !color_map.contains_key(&(x + dx, y + dy, z + dz))
                                })
                            })
                            .copied()
                            .collect();
                        let fallback = ctx.color;
                        let instanced = stroke_preview_meshes_for_union(
                            voxel_edit::EditTool::Add,
                            &visible,
                            vmap,
                            file,
                            dbg,
                            fallback,
                            Some(&|x, y, z| *color_map.get(&(x, y, z)).unwrap_or(&fallback)),
                        );
                        return PreviewMeshPrepared::GenUpload {
                            cache_key: key,
                            instanced,
                        };
                    }
                }
                "piscina" => {
                    let material = voxelle::MaterialId::from_str_id(&ctx.material);
                    let cells = crate::generators::preview_piscina_at_screen(
                        file,
                        vmap,
                        cam,
                        w,
                        h,
                        sx,
                        sy,
                        ctx.generator_piscina_seed,
                        &ctx.generator_piscina_species,
                        ctx.generator_piscina_length,
                        ctx.generator_piscina_width,
                        ctx.generator_piscina_thickness,
                        ctx.generator_piscina_spine_bend,
                        ctx.generator_piscina_spine_s_curve,
                        ctx.generator_piscina_fin_dorsal,
                        ctx.generator_piscina_fin_anal,
                        ctx.generator_piscina_fin_caudal,
                        ctx.generator_piscina_fin_pectoral,
                        ctx.generator_piscina_fin_pelvic,
                        ctx.generator_piscina_fin_adipose,
                        ctx.generator_piscina_show_fin_dorsal,
                        ctx.generator_piscina_show_fin_anal,
                        ctx.generator_piscina_show_fin_caudal,
                        ctx.generator_piscina_show_fin_pectoral,
                        ctx.generator_piscina_show_fin_pelvic,
                        ctx.generator_piscina_show_fin_adipose,
                        ctx.generator_piscina_anchor_offset_u,
                        ctx.generator_piscina_anchor_offset_v,
                        ctx.color,
                        material,
                    );
                    if !cells.is_empty() {
                        let key = hash_generator_piscina_hover(
                            sx,
                            sy,
                            ctx.generator_piscina_seed,
                            &ctx.generator_piscina_species,
                            ctx.generator_piscina_length,
                            ctx.generator_piscina_width,
                            ctx.generator_piscina_thickness,
                            ctx.generator_piscina_spine_bend,
                            ctx.generator_piscina_spine_s_curve,
                            ctx.generator_piscina_fin_dorsal,
                            ctx.generator_piscina_fin_anal,
                            ctx.generator_piscina_fin_caudal,
                            ctx.generator_piscina_fin_pectoral,
                            ctx.generator_piscina_fin_pelvic,
                            ctx.generator_piscina_fin_adipose,
                            ctx.generator_piscina_show_fin_dorsal,
                            ctx.generator_piscina_show_fin_anal,
                            ctx.generator_piscina_show_fin_caudal,
                            ctx.generator_piscina_show_fin_pectoral,
                            ctx.generator_piscina_show_fin_pelvic,
                            ctx.generator_piscina_show_fin_adipose,
                            ctx.generator_piscina_anchor_offset_u,
                            ctx.generator_piscina_anchor_offset_v,
                            ctx.color,
                            dbg,
                            mesh_gen,
                        );
                        if preview_overlay_cache_key_get(state) == Some(key) {
                            return PreviewMeshPrepared::Noop;
                        }
                        const NBRS: [(i32, i32, i32); 6] = [
                            (1, 0, 0),
                            (-1, 0, 0),
                            (0, 1, 0),
                            (0, -1, 0),
                            (0, 0, 1),
                            (0, 0, -1),
                        ];
                        let color_map: AHashMap<(i32, i32, i32), u32> =
                            cells.iter().cloned().collect();
                        let visible: AHashSet<_> = color_map
                            .keys()
                            .filter(|&&(x, y, z)| {
                                NBRS.iter().any(|&(dx, dy, dz)| {
                                    !color_map.contains_key(&(x + dx, y + dy, z + dz))
                                })
                            })
                            .copied()
                            .collect();
                        let fallback = ctx.color;
                        let instanced = stroke_preview_meshes_for_union(
                            voxel_edit::EditTool::Add,
                            &visible,
                            vmap,
                            file,
                            dbg,
                            fallback,
                            Some(&|x, y, z| *color_map.get(&(x, y, z)).unwrap_or(&fallback)),
                        );
                        return PreviewMeshPrepared::GenUpload {
                            cache_key: key,
                            instanced,
                        };
                    }
                }
                "roof" => {
                    if !ctx.generator_roof_pins.is_empty() {
                        let mut instanced = if ctx.generator_roof_pins.len() >= 3 {
                            let cells = crate::generators::preview_roof_voxels(
                                &ctx.generator_roof_pins,
                                &ctx.generator_roof_style,
                                ctx.generator_roof_height,
                                ctx.generator_roof_thickness,
                                0, // shed_edge_index
                                0, // gable_orientation
                                ctx.generator_roof_break_ratio,
                                ctx.generator_roof_wall_height,
                                ctx.generator_roof_parapet_height,
                                ctx.generator_roof_salt_skew,
                                ctx.generator_roof_hollow,
                            );
                            if !cells.is_empty() {
                                let set: AHashSet<_> = cells.iter().copied().collect();
                                stroke_preview_meshes_for_union(
                                    voxel_edit::EditTool::Add,
                                    &set,
                                    vmap,
                                    file,
                                    dbg,
                                    ctx.color,
                                    None,
                                )
                            } else {
                                greedy_mesh::PreviewInstancedResult::empty()
                            }
                        } else {
                            greedy_mesh::PreviewInstancedResult::empty()
                        };
                        // Yellow markers at each pin position.
                        append_polygon_vertex_marker_meshes(
                            &mut instanced.extra_solid,
                            &mut instanced.extra_wire,
                            &ctx.generator_roof_pins,
                            vmap,
                            file,
                            dbg,
                        );
                        let key = hash_generator_roof_hover(
                            &ctx.generator_roof_pins,
                            &ctx.generator_roof_style,
                            ctx.generator_roof_height,
                            ctx.generator_roof_thickness,
                            ctx.generator_roof_break_ratio,
                            ctx.generator_roof_wall_height,
                            ctx.generator_roof_parapet_height,
                            ctx.generator_roof_salt_skew,
                            ctx.generator_roof_hollow,
                            ctx.color,
                            dbg,
                            mesh_gen,
                        );
                        if preview_overlay_cache_key_get(state) == Some(key) {
                            return PreviewMeshPrepared::Noop;
                        }
                        return PreviewMeshPrepared::Upload {
                            cache_key: key,
                            instanced,
                        };
                    }
                }
                _ => {}
            }
        }
    }

    if matches!(mode, PreviewMode::Select) {
        let poly_placing = matches!(
            ctx.stroke_mode,
            stroke_modes::DrawStrokeMode::Polygon | stroke_modes::DrawStrokeMode::PolygonHull
        ) && !ctx.stroke_aux.polygon_vertices.is_empty()
            && ctx.use_brush_preview;
        if poly_placing {
            let material = voxelle::MaterialId::from_str_id(&ctx.material);
            let spray_cp = *state.spray_constraint_plane.lock();
            let targets = voxel_edit::collect_stroke_preview_targets(
                file,
                vmap,
                cam,
                w,
                h,
                sx,
                sy,
                voxel_edit::EditTool::Remove,
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
                spray_cp,
            );
            let key = hash_brush_hover_targets(mode, ctx, &targets, vmap, dbg);
            if preview_overlay_cache_key_get(state) == Some(key) {
                return PreviewMeshPrepared::Noop;
            }
            let set: AHashSet<_> = targets.iter().copied().collect();
            let mut instanced = if targets.is_empty() {
                greedy_mesh::PreviewInstancedResult::empty()
            } else {
                stroke_preview_meshes_for_union(
                    voxel_edit::EditTool::Remove,
                    &set,
                    vmap,
                    file,
                    dbg,
                    ctx.color,
                    None,
                )
            };
            append_polygon_vertex_marker_meshes(
                &mut instanced.extra_solid,
                &mut instanced.extra_wire,
                &ctx.stroke_aux.polygon_vertices,
                vmap,
                file,
                dbg,
            );
            if instanced.solid_instances.is_empty() && instanced.extra_solid.positions.is_empty() {
                return PreviewMeshPrepared::Clear;
            }
            return PreviewMeshPrepared::Upload {
                cache_key: key,
                instanced,
            };
        }
        let key_cell = voxel_edit::preview_remove_cell(file, vmap, cam, w, h, sx, sy);
        let key = match key_cell {
            Some(((cx, cy, cz), oid)) => {
                hash_single_cell_preview(mode, cx, cy, cz, 3, dbg, 0, oid, sx, sy)
            }
            None => hash_preview_miss(mode, dbg),
        };
        if preview_overlay_cache_key_get(state) == Some(key) {
            return PreviewMeshPrepared::Noop;
        }
        if let Some(((cx, cy, cz), oid)) = key_cell {
            let (sr, sg, sb, wr, wg, wb, size, wem) = if dbg {
                (1.0f32, 0.12, 0.1, 0.55, 0.0, 0.0, 0.56f32, 3.5f32)
            } else {
                // Fixed blue for selection hover — not the active palette.
                (0.35, 0.55, 0.98, 0.05, 0.08, 0.2, 0.5, 2.0)
            };
            // Grid-snap: render at integer cell center (same as brush preview)
            // instead of the face-hit float, so the highlight locks to the voxel.
            let instanced = preview_single_cell_world(
                file, cx as f32, cy as f32, cz as f32, oid, sr, sg, sb, wr, wg, wb, size, wem,
            );
            return PreviewMeshPrepared::Upload {
                cache_key: key,
                instanced,
            };
        }
        return PreviewMeshPrepared::Clear;
    }

    if matches!(mode, PreviewMode::Stamp | PreviewMode::Punch) {
        let clip = state.stamp_clipboard.lock().clone();
        let Some(clip) = clip else {
            return PreviewMeshPrepared::Clear;
        };
        if clip.entries.is_empty() {
            return PreviewMeshPrepared::Clear;
        }
        let anchor = if matches!(mode, PreviewMode::Stamp) {
            // Stamp places at the empty cell in front of the first solid.
            voxel_edit::preview_add_cell(file, vmap, cam, w, h, sx, sy).map(|(c, _)| c)
        } else {
            // Punch removes starting at the hit solid cell.
            voxel_edit::preview_remove_cell(file, vmap, cam, w, h, sx, sy).map(|(c, _)| c)
        };
        let (origin_x, origin_z) = (ctx.stamp_origin_x, ctx.stamp_origin_z);
        let (off_x, off_z) =
            voxel_edit::stamp_origin_offsets_pub(&clip.entries, origin_x, origin_z);
        let key = {
            let mut hasher = AHasher::default();
            mode.hash(&mut hasher);
            anchor.hash(&mut hasher);
            for &(dx, dy, dz, color, _mat) in &clip.entries {
                dx.hash(&mut hasher);
                dy.hash(&mut hasher);
                dz.hash(&mut hasher);
                color.hash(&mut hasher);
            }
            origin_x.hash(&mut hasher);
            origin_z.hash(&mut hasher);
            dbg.hash(&mut hasher);
            hasher.finish()
        };
        if preview_overlay_cache_key_get(state) == Some(key) {
            return PreviewMeshPrepared::Noop;
        }
        let Some((ax, ay, az)) = anchor else {
            return PreviewMeshPrepared::Clear;
        };
        let tool = if matches!(mode, PreviewMode::Stamp) {
            voxel_edit::EditTool::Add
        } else {
            voxel_edit::EditTool::Remove
        };
        // Build coord→color map for stamp so each ghost voxel shows its source color.
        let color_map: AHashMap<greedy_mesh::VoxelCoord, u32> = clip
            .entries
            .iter()
            .map(|&(dx, dy, dz, src_color, _)| {
                ((ax + dx - off_x, ay + dy, az + dz - off_z), src_color)
            })
            .collect();
        let cells: AHashSet<greedy_mesh::VoxelCoord> = color_map.keys().copied().collect();
        let color_resolver =
            |x: i32, y: i32, z: i32| color_map.get(&(x, y, z)).copied().unwrap_or(ctx.color);
        let instanced = stroke_preview_meshes_for_union(
            tool,
            &cells,
            vmap,
            file,
            dbg,
            ctx.color,
            if matches!(mode, PreviewMode::Stamp) {
                Some(&color_resolver as &dyn Fn(i32, i32, i32) -> u32)
            } else {
                None
            },
        );
        if instanced.solid_instances.is_empty() && instanced.extra_solid.positions.is_empty() {
            return PreviewMeshPrepared::Clear;
        }
        return PreviewMeshPrepared::Upload {
            cache_key: key,
            instanced,
        };
    }

    let tool = match mode {
        PreviewMode::Add => voxel_edit::EditTool::Add,
        PreviewMode::Remove => voxel_edit::EditTool::Remove,
        PreviewMode::Paint => voxel_edit::EditTool::Paint,
        PreviewMode::Navigate
        | PreviewMode::Fly
        | PreviewMode::Select
        | PreviewMode::Squishy
        | PreviewMode::Stamp
        | PreviewMode::Punch => {
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
            Some(((cx, cy, cz), oid)) => {
                hash_single_cell_preview(mode, cx, cy, cz, mode_tag, dbg, ctx.color, oid, sx, sy)
            }
            None => hash_preview_miss(mode, dbg),
        };
        if preview_overlay_cache_key_get(state) == Some(key) {
            return PreviewMeshPrepared::Noop;
        }
        match key_cell {
            Some(((cx, cy, cz), oid)) => {
                let (sr, sg, sb, wr, wg, wb, size, wem) = preview_tool_colors(tool, dbg, ctx.color);
                // Grid-snap: render at integer cell center so the preview locks
                // to the voxel grid instead of the floating-point face-hit.
                let instanced = preview_single_cell_world(
                    file, cx as f32, cy as f32, cz as f32, oid, sr, sg, sb, wr, wg, wb, size, wem,
                );
                return PreviewMeshPrepared::Upload {
                    cache_key: key,
                    instanced,
                };
            }
            None => return PreviewMeshPrepared::Clear,
        }
    }

    let material = voxelle::MaterialId::from_str_id(&ctx.material);
    let spray_cp = *state.spray_constraint_plane.lock();
    let targets = voxel_edit::collect_stroke_preview_targets(
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
        spray_cp,
    );
    let key = hash_brush_hover_targets(mode, ctx, &targets, vmap, dbg);
    if preview_overlay_cache_key_get(state) == Some(key) {
        return PreviewMeshPrepared::Noop;
    }
    let poly_corners = matches!(
        ctx.stroke_mode,
        stroke_modes::DrawStrokeMode::Polygon | stroke_modes::DrawStrokeMode::PolygonHull
    ) && !ctx.stroke_aux.polygon_vertices.is_empty();
    if targets.is_empty() && !poly_corners {
        return PreviewMeshPrepared::Clear;
    }
    let set: AHashSet<_> = targets.iter().copied().collect();
    let hover_resolver_owned = if ctx.palette.len() > 1 {
        Some(build_color_resolver(
            ctx.color,
            ctx.palette.clone(),
            ctx.paint_color_distrib.clone(),
            0, // fixed seed for consistent hover preview
        ))
    } else {
        None
    };
    let hover_resolver_ref: Option<&dyn Fn(i32, i32, i32) -> u32> = hover_resolver_owned
        .as_ref()
        .map(|f| f as &dyn Fn(i32, i32, i32) -> u32);
    let mut instanced = if targets.is_empty() {
        greedy_mesh::PreviewInstancedResult::empty()
    } else {
        stroke_preview_meshes_for_union(tool, &set, vmap, file, dbg, ctx.color, hover_resolver_ref)
    };
    if poly_corners {
        append_polygon_vertex_marker_meshes(
            &mut instanced.extra_solid,
            &mut instanced.extra_wire,
            &ctx.stroke_aux.polygon_vertices,
            vmap,
            file,
            dbg,
        );
    }
    if instanced.solid_instances.is_empty() && instanced.extra_solid.positions.is_empty() {
        PreviewMeshPrepared::Clear
    } else {
        PreviewMeshPrepared::Upload {
            cache_key: key,
            instanced,
        }
    }
}

fn clear_preview_mesh_sync_cache(viewer: &mut WgpuViewer, state: &ViewerState) {
    viewer.clear_preview_mesh();
    *state.preview_overlay_cache_key.lock() = None;
}

fn apply_preview_mesh(viewer: &mut WgpuViewer, state: &ViewerState, prep: PreviewMeshPrepared) {
    match prep {
        PreviewMeshPrepared::Noop => {}
        PreviewMeshPrepared::Clear => {
            clear_preview_mesh_sync_cache(viewer, state);
        }
        PreviewMeshPrepared::Upload {
            cache_key,
            instanced,
        } => {
            viewer.upload_preview_mesh_instanced(&instanced);
            viewer.preview_cache_key = Some(cache_key);
            *state.preview_overlay_cache_key.lock() = Some(cache_key);
        }
        PreviewMeshPrepared::GenUpload {
            cache_key,
            instanced,
        } => {
            viewer.upload_gen_preview_mesh_instanced(&instanced);
            viewer.preview_cache_key = Some(cache_key);
            *state.preview_overlay_cache_key.lock() = Some(cache_key);
        }
    }
}

/// Non-blocking `pick_file` — `blocking_pick_file` stalls the wry event loop and freezes the
/// window (spinner) on macOS while the sheet is open.
fn open_voxelle_file_dialog(app: AppHandle, state: Arc<ViewerState>) {
    let state = Arc::clone(&state);
    let is_guest = state.collab.lock().is_client();

    if is_guest {
        // Warn the guest that opening a file will disconnect them from the session.
        let app_confirm = app.clone();
        let state_confirm = Arc::clone(&state);
        app.dialog()
            .message("Opening a file will disconnect you from the current collaboration session.")
            .title("Leave session?")
            .kind(MessageDialogKind::Warning)
            .buttons(MessageDialogButtons::OkCancelCustom(
                "Open File".into(),
                "Cancel".into(),
            ))
            .show(move |confirmed| {
                if confirmed {
                    leave_collab_guest(&state_confirm, &app_confirm);
                    show_file_picker(app_confirm, state_confirm);
                }
            });
    } else {
        show_file_picker(app, state);
    }
}

/// Disconnect a guest from the current collab session.
fn leave_collab_guest(state: &Arc<ViewerState>, app: &AppHandle) {
    let mut c = state.collab.lock();
    if c.is_client() {
        if let Some(tx) = &c.client_tx {
            let msg = serde_json::to_string(&collab::ClientToHost::Leave).unwrap();
            let _ = tx.send(msg);
        }
        c.leave();
        drop(c);
        *state.ping_flash.lock() = None;
        let _ = app.emit("collab-ended", "You left the collaboration session.");
    }
}

fn show_file_picker(app: AppHandle, state: Arc<ViewerState>) {
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
        *state.file_label.lock() = label.clone();
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

/// Native menu handles for [`CheckMenuItem`] sync (match material, debug overlay) and
/// selection/voxel-dependent enable state (mirrors web `MenuBar` disabled rules).
#[cfg(desktop)]
/// Holds the "Open Recent" submenu so it can be rebuilt when the list changes.
pub struct RecentMenuState {
    pub submenu: tauri::menu::Submenu<tauri::Wry>,
}

/// Rebuild the contents of the "Open Recent" submenu from disk.
#[cfg(desktop)]
fn rebuild_recent_submenu(app: &AppHandle, submenu: &tauri::menu::Submenu<tauri::Wry>) {
    use tauri::menu::{MenuItem, PredefinedMenuItem};
    // Clear existing items.
    while submenu.items().map(|v| v.len()).unwrap_or(0) > 0 {
        let _ = submenu.remove_at(0);
    }
    let recent = read_recent_files(app);
    if recent.is_empty() {
        let empty = MenuItem::with_id(
            app,
            "recent_none",
            "No Recent Projects",
            false,
            None::<&str>,
        );
        if let Ok(item) = empty {
            let _ = submenu.append(&item);
        }
    } else {
        for (i, path) in recent.iter().enumerate() {
            // Show just the filename, with the full path as the menu ID.
            let display = std::path::Path::new(path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| path.clone());
            let id = format!("recent_file_{i}");
            let item = MenuItem::with_id(app, &id, &display, true, None::<&str>);
            if let Ok(item) = item {
                let _ = submenu.append(&item);
            }
        }
        let sep = PredefinedMenuItem::separator(app);
        if let Ok(sep) = sep {
            let _ = submenu.append(&sep);
        }
        let clear = MenuItem::with_id(app, "recent_clear", "Clear Recent", true, None::<&str>);
        if let Ok(item) = clear {
            let _ = submenu.append(&item);
        }
    }
}

pub struct SelectionMenuState {
    pub match_material: tauri::menu::CheckMenuItem<tauri::Wry>,
    pub viewport_cursor_debug: tauri::menu::CheckMenuItem<tauri::Wry>,
    pub view_show_borders: tauri::menu::CheckMenuItem<tauri::Wry>,
    pub view_hide_ui: tauri::menu::CheckMenuItem<tauri::Wry>,
    pub render_greedy: tauri::menu::CheckMenuItem<tauri::Wry>,
    pub render_marching: tauri::menu::CheckMenuItem<tauri::Wry>,
    pub render_dual: tauri::menu::CheckMenuItem<tauri::Wry>,
    pub render_ray: tauri::menu::CheckMenuItem<tauri::Wry>,
    pub ortho_toggle: tauri::menu::CheckMenuItem<tauri::Wry>,
    pub sel_all: tauri::menu::MenuItem<tauri::Wry>,
    pub sel_by_color: tauri::menu::MenuItem<tauri::Wry>,
    pub sel_connected: tauri::menu::MenuItem<tauri::Wry>,
    pub sel_coplanar: tauri::menu::MenuItem<tauri::Wry>,
    pub sel_coplanar_empty: tauri::menu::MenuItem<tauri::Wry>,
    pub sel_grow: tauri::menu::MenuItem<tauri::Wry>,
    pub sel_shrink: tauri::menu::MenuItem<tauri::Wry>,
    pub sel_invert: tauri::menu::MenuItem<tauri::Wry>,
    pub sel_deselect_all: tauri::menu::MenuItem<tauri::Wry>,
    pub sel_deselect_inner: tauri::menu::MenuItem<tauri::Wry>,
    pub sel_deselect_voxels: tauri::menu::MenuItem<tauri::Wry>,
    pub sel_deselect_empty: tauri::menu::MenuItem<tauri::Wry>,
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
fn install_app_menu(app: &AppHandle) -> tauri::Result<(SelectionMenuState, RecentMenuState)> {
    use tauri::menu::{CheckMenuItem, Menu, MenuItem, MenuItemKind, PredefinedMenuItem, Submenu};

    let menu = Menu::default(app)?;
    let about_item = PredefinedMenuItem::about(app, None, Some(vd_about_metadata(app)?))?;
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
    let export_glb_item =
        MenuItem::with_id(app, "menu_export_glb", "Export GLB…", true, None::<&str>)?;
    let open_recent_submenu = Submenu::with_id(app, "open_recent_submenu", "Open Recent", true)?;
    rebuild_recent_submenu(app, &open_recent_submenu);
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
    let collab_join_item =
        MenuItem::with_id(app, "menu_collab_join", "Join Session…", true, None::<&str>)?;
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
    let debug_raytrace_bench = MenuItem::with_id(
        app,
        "debug_raytrace_benchmark",
        "Ray trace benchmark (50 frames)",
        true,
        None::<&str>,
    )?;
    let debug_clear_autosaves_item = MenuItem::with_id(
        app,
        "debug_clear_autosaves",
        "Clear autosaves and session…",
        true,
        None::<&str>,
    )?;
    let debug_test_crash = MenuItem::with_id(
        app,
        "debug_test_crash",
        "Test crash report…",
        true,
        None::<&str>,
    )?;
    let debug_menu = Submenu::with_items(
        app,
        "Debug",
        true,
        &[
            &debug_viewport_cursor,
            &debug_copy_perf,
            &debug_raytrace_bench,
            &debug_clear_autosaves_item,
            &debug_test_crash,
        ],
    )?;
    let sep = PredefinedMenuItem::separator(app)?;
    let current_mode = *app.state::<Arc<ViewerState>>().rendering_mode.lock();
    let view_render_greedy = CheckMenuItem::with_id(
        app,
        "view_render_greedy",
        "Blocky",
        true,
        matches!(current_mode, RenderingMode::Greedy),
        None::<&str>,
    )?;
    let view_render_marching = CheckMenuItem::with_id(
        app,
        "view_render_marching",
        "Smooth",
        true,
        matches!(current_mode, RenderingMode::MarchingCubes),
        None::<&str>,
    )?;
    let view_render_dual = CheckMenuItem::with_id(
        app,
        "view_render_dual",
        "Crisp",
        true,
        matches!(current_mode, RenderingMode::DualContour),
        None::<&str>,
    )?;
    let view_render_ray = CheckMenuItem::with_id(
        app,
        "menu_view_render_ray",
        "Ray Tracing",
        true,
        matches!(current_mode, RenderingMode::Ray),
        None::<&str>,
    )?;
    let rendering_submenu = Submenu::with_items(
        app,
        "Rendering",
        true,
        &[
            &view_render_greedy,
            &view_render_marching,
            &view_render_dual,
            &view_render_ray,
        ],
    )?;
    let is_ortho = !app.state::<Arc<ViewerState>>().camera.lock().perspective;
    let ortho_view_item = CheckMenuItem::with_id(
        app,
        "menu_view_ortho",
        "Orthographic",
        true,
        is_ortho,
        None::<&str>,
    )?;
    let sep_view_extras = PredefinedMenuItem::separator(app)?;
    let view_show_borders = CheckMenuItem::with_id(
        app,
        "menu_view_show_borders",
        "Show borders",
        true,
        false,
        None::<&str>,
    )?;
    let view_hide_ui = CheckMenuItem::with_id(
        app,
        "menu_view_hide_ui",
        "Hide UI",
        true,
        false,
        None::<&str>,
    )?;
    let sep_view_stamp = PredefinedMenuItem::separator(app)?;
    let view_stamp_book = MenuItem::with_id(
        app,
        "menu_view_stamp_book",
        "Stamp book…",
        true,
        None::<&str>,
    )?;
    let sep_before_chat = PredefinedMenuItem::separator(app)?;

    let mut file_inserted = false;
    let mut edit_inserted = false;
    let mut view_inserted = false;
    for item in menu.items()? {
        if let MenuItemKind::Submenu(sub) = item {
            let text = sub.text()?;
            #[cfg(target_os = "macos")]
            if text == app.package_info().name.clone() {
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
                    &open_recent_submenu,
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
                sub.append(&ortho_view_item)?;
                sub.append(&sep_view_extras)?;
                sub.append(&view_show_borders)?;
                sub.append(&view_hide_ui)?;
                sub.append(&sep_view_stamp)?;
                sub.append(&view_stamp_book)?;
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
                &ortho_view_item,
                &view_render_ray,
                &sep_view_extras,
                &view_show_borders,
                &view_hide_ui,
                &sep_view_stamp,
                &view_stamp_book,
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
    let voxel_hollow =
        MenuItem::with_id(app, "menu_voxel_hollow", "Hollow out", true, None::<&str>)?;
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
    let voxel_mirror_hdr =
        MenuItem::with_id(app, "menu_voxel_mirror_hdr", "Mirror", false, None::<&str>)?;
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
    let menu_sel_deselect_all = MenuItem::with_id(
        app,
        "menu_sel_deselect_all",
        "Deselect All",
        true,
        None::<&str>,
    )?;
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
    let menu_sel_mode_add = MenuItem::with_id(
        app,
        "menu_sel_mode_add",
        "Add to Selection",
        true,
        None::<&str>,
    )?;
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

    place_voxelle_custom_top_level_menus(&menu, &selection_submenu, &voxels_submenu, &debug_menu)?;
    menu.set_as_app_menu()?;
    Ok((
        SelectionMenuState {
            match_material: menu_sel_match_material.clone(),
            viewport_cursor_debug: debug_viewport_cursor.clone(),
            view_show_borders: view_show_borders.clone(),
            view_hide_ui: view_hide_ui.clone(),
            render_greedy: view_render_greedy.clone(),
            render_marching: view_render_marching.clone(),
            render_dual: view_render_dual.clone(),
            render_ray: view_render_ray.clone(),
            ortho_toggle: ortho_view_item.clone(),
            sel_all: menu_sel_all.clone(),
            sel_by_color: menu_sel_by_color.clone(),
            sel_connected: menu_sel_connected.clone(),
            sel_coplanar: menu_sel_coplanar.clone(),
            sel_coplanar_empty: menu_sel_coplanar_empty.clone(),
            sel_grow: menu_sel_grow.clone(),
            sel_shrink: menu_sel_shrink.clone(),
            sel_invert: menu_sel_invert.clone(),
            sel_deselect_all: menu_sel_deselect_all.clone(),
            sel_deselect_inner: menu_sel_deselect_inner.clone(),
            sel_deselect_voxels: menu_sel_deselect_voxels.clone(),
            sel_deselect_empty: menu_sel_deselect_empty.clone(),
        },
        RecentMenuState {
            submenu: open_recent_submenu,
        },
    ))
}

#[cfg(desktop)]
fn performance_report_text(state: &ViewerState) -> String {
    let fps = state.fps.lock().last_fps;
    let file_label = state.file_label.lock().clone();
    let (vw, vh, idx_count, vtx_buf_verts) = state
        .viewer
        .lock()
        .as_ref()
        .map(|viewer| {
            let (vw, vh) = viewer.viewport_size();
            (
                vw,
                vh,
                viewer.opaque_index_count(),
                viewer.opaque_vertex_buffer_vertices(),
            )
        })
        .unwrap_or((0, 0, 0, 0));
    let (voxel_n, grid_size) = state
        .current_file
        .lock()
        .as_ref()
        .map(|f| (f.voxels.len(), f.grid_size))
        .unwrap_or((0, 0));
    let unix_s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let edit_block = state
        .last_edit_perf
        .lock()
        .clone()
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

/// stderr one-liners for cuboid/cylinder depth commits — visible in `tauri dev` when the webview is wedged.
#[cfg(desktop)]
fn eprintln_extrusion_stroke_checkpoint(
    label: &str,
    args: &VoxelEditAtScreen,
    deltas_len: Option<usize>,
    apply_ms: Option<f64>,
) {
    if !matches!(
        args.stroke_mode,
        stroke_modes::DrawStrokeMode::Cuboid | stroke_modes::DrawStrokeMode::Cylinder
    ) {
        return;
    }
    if let (Some(n), Some(ms)) = (deltas_len, apply_ms) {
        eprintln!(
            "[voxelle] {label} stroke={:?} deltas={n} apply_ms={ms:.1}",
            args.stroke_mode
        );
    } else {
        eprintln!(
            "[voxelle] {label} stroke={:?} cuboid_depth={:?} cylinder_depth={:?}",
            args.stroke_mode, args.stroke_aux.cuboid_depth, args.stroke_aux.cylinder_depth
        );
    }
}

#[cfg(desktop)]
fn eprintln_last_edit_perf_line(state: &ViewerState) {
    if let Some(e) = state.last_edit_perf.lock().clone() {
        eprintln!(
            "[voxelle] voxel_edit GPU refresh total_ms={:.1} mesh_ms={:.1} route={} apply_ms={:.1}",
            e.total_ms, e.mesh_ms, e.mesh_route, e.apply_edit_ms
        );
    }
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

/// The start-screen logo, embedded at compile time.
static START_SCREEN_LOGO: &[u8] = include_bytes!("../Logo.voxelle");

/// Bundled mascot models.  Key strings are used in the `mascot_load_embedded` command.
static MASCOT_SEAGULL: &[u8] = include_bytes!("../mascots/Seagull.voxelle");

/// Loads bundled `Logo.voxelle` for the cold-start screen (no `voxelle-load-start`, empty `file_label`).
#[tauri::command]
fn load_start_screen_logo(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
) -> Result<(), String> {
    *state.file_label.lock() = String::new();
    spawn_decode_and_mesh_from_bytes(
        Arc::clone(&*state),
        app,
        START_SCREEN_LOGO,
        String::new(),
        true,
    );
    Ok(())
}

// ── Mascot commands (start-screen floating voxel models) ─────────────────────

/// Load a `.voxelle` file as a mascot model.
/// `path` should be a full filesystem path (the frontend resolves bundled assets
/// via Tauri's resource path API). `id` is a caller-chosen integer key (0–3).
#[tauri::command]
fn mascot_load(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    id: u32,
    path: String,
) -> Result<(), String> {
    let state = Arc::clone(&*state);
    let app_err = app.clone();
    std::thread::Builder::new()
        .name("mascot-load".into())
        .spawn(move || {
            let bytes = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    let _ = app_err.emit("mascot-load-error", format!("id={id}: {e}"));
                    return;
                }
            };
            let file = match decode_payload(&bytes) {
                Ok(f) => f,
                Err(e) => {
                    let _ = app_err.emit("mascot-load-error", format!("id={id}: {e}"));
                    return;
                }
            };
            let (mesh, bounds) = greedy_mesh::build_greedy_mesh(&file.voxels, &file.objects);
            let state_up = Arc::clone(&state);
            let _ = app_err.run_on_main_thread(move || {
                let mut v = state_up.viewer.lock();
                if let Some(viewer) = v.as_mut() {
                    viewer.load_mascot_mesh(id, &mesh, bounds);
                }
            });
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Set the viewport-relative screen rect for a mascot (physical pixels).
#[tauri::command]
fn mascot_set_screen_rect(
    state: State<'_, Arc<ViewerState>>,
    id: u32,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
) -> Result<(), String> {
    let mut v = state.viewer.lock();
    if let Some(viewer) = v.as_mut() {
        viewer.set_mascot_screen_rect(id, x, y, w, h);
    }
    Ok(())
}

/// Load a bundled (compile-time embedded) mascot by name.
/// Supported names: "seagull"
#[tauri::command]
fn mascot_load_embedded(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    id: u32,
    name: String,
) -> Result<(), String> {
    let bytes: &'static [u8] = match name.as_str() {
        "seagull" => MASCOT_SEAGULL,
        other => return Err(format!("unknown mascot: {other}")),
    };
    let state = Arc::clone(&*state);
    let app_err = app.clone();
    std::thread::Builder::new()
        .name("mascot-load-embedded".into())
        .spawn(move || {
            let file = match decode_payload(bytes) {
                Ok(f) => f,
                Err(e) => {
                    let _ = app_err.emit("mascot-load-error", format!("id={id}: {e}"));
                    return;
                }
            };
            let (mesh, bounds) =
                greedy_mesh::build_greedy_mesh(&file.voxels, &file.objects);
            let app_main = app_err.clone();
            let _ = app_err.run_on_main_thread(move || {
                let mut v = state.viewer.lock();
                if let Some(viewer) = v.as_mut() {
                    viewer.load_mascot_mesh(id, &mesh, bounds);
                    drop(v);
                    let _ = app_main.emit("mascot-loaded", id);
                    wake_viewport_loop(&app_main);
                } else {
                    log::warn!("mascot_load_embedded: viewer was None when uploading mascot id={id}, mesh not uploaded");
                }
            });
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Show or hide a mascot.
#[tauri::command]
fn mascot_set_visible(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    id: u32,
    visible: bool,
) -> Result<(), String> {
    let mut v = state.viewer.lock();
    if let Some(viewer) = v.as_mut() {
        viewer.set_mascot_visible(id, visible);
    }
    drop(v);
    wake_viewport_loop(&app);
    Ok(())
}

/// Show (or replace) a speech bubble.
/// `rx`, `ry`, `rw`, `rh` — bubble rect in viewport-relative physical pixels.
/// `tx`, `ty` — tail tip in viewport-relative physical pixels (anchor point toward subject).
/// `pages` — ordered list of text strings; click advances through them.
#[tauri::command]
fn speech_bubble_show(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    id: u32,
    pages: Vec<String>,
    rx: f32,
    ry: f32,
    rw: f32,
    rh: f32,
    tx: f32,
    ty: f32,
) -> Result<(), String> {
    let mut v = state.viewer.lock();
    if let Some(viewer) = v.as_mut() {
        viewer.show_speech_bubble(id, pages, [rx, ry, rw, rh], [tx, ty]);
    }
    drop(v);
    wake_viewport_loop(&app);
    Ok(())
}

/// Register a click on bubble `id`.
/// Advances to the next page, or begins a shake-then-dismiss sequence on the last page.
/// Emits `"speech-bubble-dismissed"` with `id` when the bubble finally closes.
#[tauri::command]
fn speech_bubble_click(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    id: u32,
) -> Result<(), String> {
    let mut v = state.viewer.lock();
    let changed = if let Some(viewer) = v.as_mut() {
        viewer.click_speech_bubble(id)
    } else {
        false
    };
    drop(v);
    if changed {
        wake_viewport_loop(&app);
    }
    Ok(())
}

/// Immediately dismiss a speech bubble without the shake animation.
/// Emits `"speech-bubble-dismissed"` with `id`.
#[tauri::command]
fn speech_bubble_dismiss(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    id: u32,
) -> Result<(), String> {
    let mut v = state.viewer.lock();
    if let Some(viewer) = v.as_mut() {
        viewer.dismiss_speech_bubble(id);
    }
    drop(v);
    let _ = app.emit("speech-bubble-dismissed", id);
    wake_viewport_loop(&app);
    Ok(())
}

/// Move an existing bubble to a new screen rect + tail tip without resetting its page or state.
/// Used to keep bubbles anchored to their subject after a window resize.
#[tauri::command]
fn speech_bubble_reposition(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    id: u32,
    rx: f32,
    ry: f32,
    rw: f32,
    rh: f32,
    tx: f32,
    ty: f32,
) -> Result<(), String> {
    let mut v = state.viewer.lock();
    if let Some(viewer) = v.as_mut() {
        viewer.reposition_speech_bubble(id, [rx, ry, rw, rh], [tx, ty]);
    }
    drop(v);
    wake_viewport_loop(&app);
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
    *state.file_label.lock() = path.clone();
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
fn collab_cancel_join(state: State<'_, Arc<ViewerState>>) {
    let token = state.collab.lock().join_cancel.take();
    if let Some(t) = token {
        t.cancel();
    }
}

#[tauri::command]
fn collab_leave(state: State<'_, Arc<ViewerState>>, app: AppHandle) -> Result<(), String> {
    let (was_host, was_client, upnp_port) = {
        let mut c = state.collab.lock();
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

#[tauri::command]
fn collab_local_peer_id(state: State<'_, Arc<ViewerState>>) -> u32 {
    state.collab.lock().local_peer_id
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
    let mut c = state.collab.lock();
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
            let _ = tx.send(msg);
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
fn collab_snap_camera(state: State<'_, Arc<ViewerState>>, peer_id: u32) -> Result<(), String> {
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

#[tauri::command]
fn collab_send_chat(
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
        );
        c = state.collab.lock();
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
            let _ = tx.send(tokio_tungstenite::tungstenite::Message::Text(json));
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
        enabled: *state.autosave_enabled.lock(),
        interval_secs: *state.autosave_interval_secs.lock(),
        keep_count: *state.autosave_keep_count.lock(),
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
    *state.autosave_enabled.lock() = args.enabled;
    *state.autosave_interval_secs.lock() = args.interval_secs;
    let k = args.keep_count.max(1).min(64);
    *state.autosave_keep_count.lock() = k;
    Ok(())
}

fn clear_autosaves_and_session(app: &AppHandle) -> Result<(), String> {
    let dir = autosave_dir(app)?;
    let mut deleted = 0u32;
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("voxelle") {
                if std::fs::remove_file(&path).is_ok() {
                    deleted += 1;
                }
            }
        }
    }
    if let Ok(session_path) = session_state_path(app) {
        let _ = std::fs::remove_file(&session_path);
    }
    log::info!(
        "debug_clear_autosaves: deleted {deleted} autosave file(s) and cleared last_session.json"
    );
    Ok(())
}

#[tauri::command]
fn debug_clear_autosaves(app: AppHandle) -> Result<(), String> {
    clear_autosaves_and_session(&app)
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
    *state.file_label.lock() = label.clone();
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
    let _ =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_filter))
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
        load_generation: AtomicU64::new(0),
        chunk_mesh_inbox: Mutex::new(VecDeque::new()),
        collab_edit_inbox: Mutex::new(VecDeque::new()),
        deferred_spatial_cache: Mutex::new(None),
        voxel_edit_stats_cache: Mutex::new(None),
        solo_undo: Mutex::new(Vec::new()),
        solo_redo: Mutex::new(Vec::new()),
        stroke_active: Mutex::new(false),
        stroke_buffer: Mutex::new(Vec::new()),
        stroke_preview_union: Mutex::new(AHashSet::new()),
        stroke_preview_last_args: Mutex::new(None),
        stroke_preview_suppresses_hover: AtomicBool::new(false),
        sculpt_stroke_replay: Mutex::new(Vec::new()),
        extrude_ray_spine: Mutex::new(None),
        collab: Arc::new(Mutex::new(collab::CollabRuntime::default())),
        smooth_presence: Mutex::new(HashMap::new()),
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
        walk_mode: Mutex::new(false),
        walk_physics: Mutex::new(camera::WalkPhysicsState::default()),
        walk_last_physics: Mutex::new(None),
        selection_cells: Mutex::new(AHashSet::new()),
        selection_stroke_before: Mutex::new(None),
        selection_stroke_accum: Mutex::new(None),
        selection_combine_mode: Mutex::new(SelectionCombineMode::Replace),
        selection_match_material: Mutex::new(false),
        stamp_clipboard: Mutex::new(None),
        squishy_session: Mutex::new(generators::SquishySession::new()),
        squishy_gizmo_drag: Mutex::new(None),
        selection_gizmo_drag: Mutex::new(SelectionGizmoDrag::None),
        start_screen_logo_transparent: std::sync::atomic::AtomicBool::new(true),
        start_screen_light: std::sync::atomic::AtomicBool::new(false),
        viewport_cursor_debug_overlay: AtomicBool::new(false),
        show_grid_borders: AtomicBool::new(false),
        hovered_gizmo_axis: AtomicU8::new(255),
        grid_overlay_cache_key: Mutex::new(None),
        selection_overlay_cache_key: Mutex::new(None),
        preview_overlay_cache_key: Mutex::new(None),
        fill_operation_cancel: Arc::new(AtomicBool::new(false)),
        spray_constraint_plane: Mutex::new(None),
        wall_stroke_face_snapped: Mutex::new(None),
        terrain_accum: Mutex::new(AHashMap::new()),
    });
    let vs = viewer_state.clone();

    #[cfg(all(desktop, unix))]
    {
        let st = viewer_state.clone();
        let _ = std::thread::Builder::new()
            .name("voxelle-sigusr1-perf".into())
            .spawn(move || {
                use signal_hook::consts::SIGUSR1;
                use signal_hook::iterator::Signals;
                let Ok(mut signals) = Signals::new([SIGUSR1]) else {
                    return;
                };
                for _ in signals.forever() {
                    let text = performance_report_text(st.as_ref());
                    eprintln!(
                        "--- voxelle SIGUSR1 performance dump (paste for bugs) ---\n{text}\n--- end dump ---"
                    );
                }
            });
    }

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
                if save_voxelle(app.clone(), state).is_err() {
                    let state: State<'_, Arc<ViewerState>> = app.state();
                    let _ = save_voxelle_as(state, app.clone());
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
                eprintln!(
                    "--- Debug → Copy performance info (terminal backup) ---\n{}\n--- end ---",
                    performance_report_text(state.inner())
                );
                if let Err(e) = copy_performance_data_to_clipboard(state.inner()) {
                    eprintln!("copy performance data: {e}");
                }
            } else if event.id() == "debug_raytrace_benchmark" {
                let state = app.state::<Arc<ViewerState>>();
                let result = state.viewer.lock().as_mut().map(|viewer| viewer.run_raytrace_benchmark(50));
                if let Some(result) = result {
                    eprintln!(
                        "[raytrace bench] {}×{}  {} frames  avg {:.1} ms  σ {:.1}  p50 {:.1}  p95 {:.1}  p99 {:.1}  max {:.1}  {:.1} Mpix/s",
                        result.viewport_width, result.viewport_height,
                        result.frame_count,
                        result.avg_ms, result.stddev_ms,
                        result.p50_ms, result.p95_ms, result.p99_ms, result.max_ms,
                        result.mpix_per_sec,
                    );
                    let _ = app.emit_to(
                        EventTarget::webview_window("main"),
                        "voxelle-debug-raytrace-benchmark",
                        &result,
                    );
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
            } else if event.id() == "debug_clear_autosaves" {
                let _ = clear_autosaves_and_session(&app);
            } else if event.id() == "debug_test_crash" {
                panic!("Test crash triggered from Debug menu");
            } else if event.id() == "view_render_greedy"
                || event.id() == "view_render_marching"
                || event.id() == "view_render_dual"
                || event.id() == "menu_view_render_ray"
            {
                let (mode, label) = match event.id().0.as_ref() {
                    "view_render_greedy" => (RenderingMode::Greedy, "greedy"),
                    "view_render_marching" => (RenderingMode::MarchingCubes, "marchingCubes"),
                    "view_render_dual" => (RenderingMode::DualContour, "dualContour"),
                    _ => (RenderingMode::Ray, "ray"),
                };
                let state = app.state::<Arc<ViewerState>>();
                let _ = apply_rendering_mode(&state, &app, mode);
                wake_viewport_loop(&app);
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-rendering-mode-changed",
                    label,
                );
                // Enforce radio-button style: exactly one checked at a time.
                #[cfg(desktop)]
                if let Some(sel) = app.try_state::<SelectionMenuState>() {
                    let _ = sel.render_greedy.set_checked(matches!(mode, RenderingMode::Greedy));
                    let _ = sel.render_marching.set_checked(matches!(mode, RenderingMode::MarchingCubes));
                    let _ = sel.render_dual.set_checked(matches!(mode, RenderingMode::DualContour));
                    let _ = sel.render_ray.set_checked(matches!(mode, RenderingMode::Ray));
                }
            } else if event.id() == "menu_view_ortho" {
                #[cfg(desktop)]
                if let Some(sel) = app.try_state::<SelectionMenuState>() {
                    if let Ok(checked) = sel.ortho_toggle.is_checked() {
                        let state = app.state::<Arc<ViewerState>>();
                        let _ = apply_orthographic(&state, checked);
                        wake_viewport_loop(&app);
                    }
                }
            } else if event.id() == "menu_view_show_borders" {
                #[cfg(desktop)]
                if let Some(sel) = app.try_state::<SelectionMenuState>() {
                    if let Ok(checked) = sel.view_show_borders.is_checked() {
                        let state = app.state::<Arc<ViewerState>>();
                        state
                            .show_grid_borders
                            .store(checked, Ordering::Relaxed);
                        let _ = app.emit_to(
                            EventTarget::webview_window("main"),
                            "voxelle-show-grid-borders",
                            checked,
                        );
                        wake_viewport_loop(&app);
                    }
                }
            } else if event.id() == "menu_view_hide_ui" {
                #[cfg(desktop)]
                if let Some(sel) = app.try_state::<SelectionMenuState>() {
                    if let Ok(checked) = sel.view_hide_ui.is_checked() {
                        let _ = app.emit_to(
                            EventTarget::webview_window("main"),
                            "voxelle-hide-ui",
                            checked,
                        );
                    }
                }
            } else if event.id() == "menu_view_stamp_book" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-menu-stamp-book",
                    (),
                );
            } else if event.id() == "menu_voxel_mirror_x" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_mirror(state, app.clone(), 0);
            } else if event.id() == "menu_voxel_mirror_y" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_mirror(state, app.clone(), 1);
            } else if event.id() == "menu_voxel_mirror_z" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_mirror(state, app.clone(), 2);
            } else if event.id() == "menu_voxel_rotate" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-menu-rotate-selection",
                    (),
                );
            } else if event.id() == "menu_voxel_scale" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-menu-scale-selection",
                    (),
                );
            } else if event.id() == "menu_voxel_hide_selected"
                || event.id() == "menu_voxel_unhide_all"
                || event.id() == "menu_voxel_hollow"
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
                        *state.selection_match_material.lock() = checked;
                        let _ = app.emit_to(
                            EventTarget::webview_window("main"),
                            "voxelle-menu-match-material",
                            checked,
                        );
                    }
                }
            } else if event.id() == "recent_clear" {
                clear_recent_files(app);
                #[cfg(desktop)]
                if let Some(rm) = app.try_state::<RecentMenuState>() {
                    rebuild_recent_submenu(app, &rm.submenu);
                }
            } else if event.id().0.starts_with("recent_file_") {
                let id_str = event.id().0.to_string();
                if let Some(idx_str) = id_str.strip_prefix("recent_file_") {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        let recent = read_recent_files(app);
                        if let Some(path_str) = recent.get(idx) {
                            let path = PathBuf::from(path_str);
                            if path.exists() {
                                let state = app.state::<Arc<ViewerState>>();
                                let label = path.to_string_lossy().to_string();
                                *state.file_label.lock() = label.clone();
                                let _ = app.emit("voxelle-load-start", label);
                                spawn_decode_and_mesh(
                                    state.inner().clone(),
                                    app.clone(),
                                    path,
                                );
                            } else {
                                let _ = app.emit(
                                    "voxelle-load-error",
                                    format!("File not found: {path_str}"),
                                );
                            }
                        }
                    }
                }
            }
        })
        .setup(move |app| {
            #[cfg(desktop)]
            {
                let (selection_menu_state, recent_menu_state) = install_app_menu(app.handle())?;
                app.manage(selection_menu_state);
                app.manage(recent_menu_state);
                let (has_voxels, has_selection) = scene_menu_flags(vs.as_ref());
                selection_menu_sync_enabled_for_scene(app.handle(), has_voxels, has_selection);
            }

            let window = app.get_webview_window("main").expect("main window");
            #[cfg(target_os = "macos")]
            if let Err(e) = macos_titlebar::apply_transparent_titlebar(&window) {
                eprintln!("macos_titlebar: {e}");
            }
            if headless_server_port.is_some() {
                let _ = window.hide();
            }
            let viewer = {
                let w = window.clone();
                tauri::async_runtime::block_on(async move { WgpuViewer::new(w).await })
                    .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?
            };
            // Do not resize to `inner_size()` here: the 3D view matches the `.viewport` div (below
            // toolbar / beside sidebar), not the full window. Wrong dimensions break screen→world
            // raycasts until the frontend sends `viewer_resize`.
            *vs.viewer.lock() = Some(viewer);

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
            terrain_surface_y_at_screen,
            ping_cursor_pick,
            world_to_viewport_pixels,
            collab_peer_labels,
            sync_preview_input,
            voxel_stroke_begin,
            voxel_stroke_preview_reset,
            voxel_stroke_preview_at_screen,
            query_cuboid_plane_geometry,
            voxel_stroke_end,
            voxel_pick_color_at_screen,
            voxel_edit_at_screen,
            voxel_fill_cancel,
            voxel_undo,
            voxel_redo,
            save_voxelle,
            save_voxelle_as,
            collab_host_start,
            collab_join,
            collab_cancel_join,
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
            debug_clear_autosaves,
            get_rendering_mode,
            set_rendering_mode,
            set_raytrace_mode,
            benchmark_raytrace,
            get_orthographic,
            set_orthographic,
            get_show_grid_borders,
            view_menu_sync_show_borders,
            view_menu_sync_hide_ui,
            selection_menu_sync_match_material,
            debug_menu_sync_viewport_cursor_overlay,
            set_soft_shadows,
            set_soft_sunshafts,
            set_emission_lighting,
            set_tone_mapping,
            is_hdr_available,
            set_hdr_output,
            set_mood_params,
            set_scene_lighting,
            get_scene_lighting,
            set_focal_length_mm,
            get_focal_length_mm,
            set_fly_mode,
            get_fly_mode,
            set_walk_mode,
            sync_fly_input,
            camera_fly_look,
            selection_toggle_at_screen,
            get_selection_gizmo_projected,
            gizmo_pointer_down,
            gizmo_pointer_move,
            gizmo_pointer_up,
            gizmo_hit_test,
            set_gizmo_on_top,
            selection_translate,
            selection_rotate,
            selection_scale,
            selection_mirror,
            selection_clear,
            selection_delete_selected_voxels,
            selection_get_count,
            paint_selection,
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
            stamp_face_normal_at_screen,
            get_selection_as_stamp_entries,
            stamp_book_load_entries,
            voxel_sculpt_raise_at_screen,
            voxel_sculpt_stroke_at_screen,
            voxel_sculpt_stroke_preview_at_screen,
            extrude_ray_preview,
            selection_extrude_preview,
            extrude_recompute_preview,
            generator_rocks_at_screen,
            generator_grass_at_screen,
            generator_rope_at_screen,
            generator_cloth_from_pins_cmd,
            generator_ashlar_at_screen,
            generator_flora_at_screen,
            generator_roof_from_pins_cmd,
            generator_piscina_at_screen,
            generator_insecta_at_screen,
            generator_fauna_at_screen,
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
            mascot_load,
            mascot_load_embedded,
            mascot_set_screen_rect,
            mascot_set_visible,
            speech_bubble_show,
            speech_bubble_click,
            speech_bubble_dismiss,
            speech_bubble_reposition,
        ])
        .build(tauri::generate_context!())
        .expect("error building app")
        .run(move |app, event| {
            if let RunEvent::MainEventsCleared = event {
                let app_wake = app.clone();
                let state = app.state::<Arc<ViewerState>>();
                {
                    let mut cam = state.camera.lock();
                    cam.update_damping();
                }
                // Fly WASD: integrate here with wall-clock dt between native iterations (not webview RAF).
                if *state.fly_mode.lock() {
                    let now = Instant::now();
                    let dt = {
                        let mut last = state.fly_last_physics.lock();
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
                    let input = *state.fly_input.lock();
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
                        let mut cam = state.camera.lock();
                        cam.fly_move(
                            input.forward,
                            input.right,
                            input.up,
                            dt,
                            SPEED * scale,
                        );
                    }
                }
                // Walk mode physics: gravity, collision, jumping.
                if *state.walk_mode.lock() {
                    let now = Instant::now();
                    let dt = {
                        let mut last = state.walk_last_physics.lock();
                        match *last {
                            None => {
                                *last = Some(now);
                                0.0
                            }
                            Some(t) => {
                                let d = (now - t).as_secs_f32();
                                *last = Some(now);
                                d.clamp(0.0, 0.05)
                            }
                        }
                    };
                    if dt > 0.0 {
                        let input = *state.fly_input.lock();
                        let scale = if input.speed_scale.is_finite() {
                            input.speed_scale.clamp(0.0, 1e6)
                        } else {
                            1.0
                        };

                        let mut wp = state.walk_physics.lock();
                        let h_delta = {
                            let cam = state.camera.lock();
                            cam.walk_horizontal_delta(
                                input.forward,
                                input.right,
                                dt,
                                camera::WALK_MOVE_SPEED * scale,
                            )
                        };

                        // Gravity
                        if !wp.on_ground {
                            wp.vel_y += camera::WALK_GRAVITY * dt;
                        }

                        // Jump
                        if input.jump && wp.on_ground {
                            wp.vel_y = camera::WALK_JUMP_VEL;
                            wp.on_ground = false;
                        }

                        // Candidate position
                        let mut new_feet = wp.feet_pos + h_delta + glam::Vec3::Y * (wp.vel_y * dt);

                        // Collision against voxel_map
                        {
                            let vm_guard = state.voxel_map.lock();
                            if let Some(ref vm) = *vm_guard {
                                new_feet = resolve_walk_collision(wp.feet_pos, new_feet, vm, &mut wp);
                            }
                        }

                        // Void floor safety
                        if new_feet.y < -100.0 {
                            new_feet.y = -100.0;
                            wp.vel_y = 0.0;
                            wp.on_ground = true;
                        }

                        wp.feet_pos = new_feet;
                        let mut cam = state.camera.lock();
                        cam.walk_set_eye_from_feet(new_feet, camera::WALK_EYE_HEIGHT);
                    }
                }
                // Prepare overlays without holding the viewer mutex so `current_file` can be locked
                // while IPC may be waiting on `viewer` + `camera` (see `finish_voxel_edit_gpu_deltas`).
                let frame_prep = {
                    let wh = {
                        let v = state.viewer.lock();
                        v.as_ref().map(|viewer| viewer.viewport_size())
                    };
                    match wh {
                        Some((viewport_w, viewport_h)) => {
                            let cam_snap = state.camera.lock().clone();
                            let grid_p = prepare_grid_border_overlay(Arc::as_ref(&state));
                            let sel_p = prepare_selection_overlay(Arc::as_ref(&state));
                            let prev_p = prepare_preview_mesh(
                                Arc::as_ref(&state),
                                &cam_snap,
                                viewport_w,
                                viewport_h,
                            );
                            Some((grid_p, sel_p, prev_p))
                        }
                        None => None,
                    }
                };
                // Drain collab edit inbox — apply queued guest edits/undo/redo
                // on the main thread, before we hold the viewer lock for the frame.
                {
                    let items: Vec<collab::CollabInboxItem> =
                        state.collab_edit_inbox.lock().drain(..).collect();
                    collab::process_inbox_items_batched(app, &state, &state.collab, items);
                }
                let mut v = state.viewer.lock();
                if let Some(viewer) = v.as_mut() {
                    let cam = state.camera.lock();
                    viewer.update_uniforms(&cam);
                    if let Some((grid_p, sel_p, prev_p)) = frame_prep {
                        apply_grid_border_overlay(viewer, Arc::as_ref(&state), grid_p);
                        apply_selection_overlay(viewer, Arc::as_ref(&state), sel_p);
                        apply_preview_mesh(viewer, Arc::as_ref(&state), prev_p);
                    }
                    sync_collab_peer_lines(viewer, Arc::as_ref(&state));
                    sync_collab_peer_labels(viewer, Arc::as_ref(&state), &cam);
                    sync_ping_flash(viewer, Arc::as_ref(&state), &cam);
                    sync_gizmo_gpu(viewer, Arc::as_ref(&state), &cam);
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
                    // Progressive loading: move chunks from background-thread inbox to viewer queue.
                    {
                        let mut inbox = state.chunk_mesh_inbox.lock();
                        if !inbox.is_empty() {
                            viewer.enqueue_chunk_uploads(&mut inbox);
                        }
                    }
                    // Drip-feed queued mesh chunks to GPU each frame.
                    if viewer.has_pending_chunk_uploads() {
                        viewer.drain_pending_chunk_uploads(std::time::Duration::from_millis(4));
                    }
                    // Once all chunks are uploaded, apply deferred spatial cache for editing.
                    if !viewer.has_pending_chunk_uploads() && !viewer.has_spatial_mesh_cache() {
                        let mut deferred = state.deferred_spatial_cache.lock();
                        if let Some(cache) = deferred.take() {
                            viewer.set_spatial_mesh_cache(cache);
                        }
                    }
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

                    // Drain speech bubbles that completed their shake-dismiss animation.
                    // app.emit is non-blocking; holding viewer lock here is safe.
                    for id in viewer.pending_dismissed_bubble_ids.drain(..) {
                        let _ = app.emit("speech-bubble-dismissed", id);
                    }

                    let enabled = *state.autosave_enabled.lock();
                    let interval = *state.autosave_interval_secs.lock();
                    let (collab_on, is_host) = {
                        let c = state.collab.lock();
                        (c.is_active(), c.is_host())
                    };
                    if enabled
                        && interval > 0
                        && (!collab_on || is_host)
                        && state.active_project.load(Ordering::Relaxed)
                    {
                        let label = state.file_label.lock().clone();
                        if !label.is_empty() {
                            if let Ok(doc) = autosave_document_path_for_label(&app, &label) {
                                let now = Instant::now();
                                let last = state.last_autosave.lock();
                                let do_save = last
                                    .map(|t| now.duration_since(t).as_secs() >= interval)
                                    .unwrap_or(true);
                                if do_save {
                                    drop(last);
                                    if let Ok(dest) =
                                        next_rotating_autosave_path(&app, Arc::as_ref(&state), &doc)
                                    {
                                        if write_voxelle_file_to_path(None, &state, &dest).is_ok() {
                                            *state.last_autosave.lock() = Some(now);
                                        }
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
                // Raytrace mode: accumulation requires one sample per frame, so spin continuously.
                let rt_active = v.as_ref().map_or(false, |viewer| viewer.raytrace_enabled);
                let mascots_active = v
                    .as_ref()
                    .map_or(false, |viewer| viewer.any_mascot_visible());
                let bubbles_active = v
                    .as_ref()
                    .map_or(false, |viewer| viewer.has_visible_speech_bubbles());
                drop(v);
                let fly_on = *state.fly_mode.lock();
                let walk_on = *state.walk_mode.lock();
                let has_fly_movement = if fly_on || walk_on {
                    let input = *state.fly_input.lock();
                    input.forward != 0.0 || input.right != 0.0 || input.up != 0.0 || input.jump
                } else {
                    false
                };
                // Walk mode always spins (gravity may be in progress even with no input).
                let needs_next = state.camera.lock().needs_redraw()
                    || fly_on
                    || walk_on
                    || has_fly_movement
                    || rt_active
                    || mascots_active
                    || bubbles_active;
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
        load_generation: AtomicU64::new(0),
        chunk_mesh_inbox: Mutex::new(VecDeque::new()),
        collab_edit_inbox: Mutex::new(VecDeque::new()),
        deferred_spatial_cache: Mutex::new(None),
        voxel_edit_stats_cache: Mutex::new(None),
        solo_undo: Mutex::new(Vec::new()),
        solo_redo: Mutex::new(Vec::new()),
        stroke_active: Mutex::new(false),
        stroke_buffer: Mutex::new(Vec::new()),
        stroke_preview_union: Mutex::new(AHashSet::new()),
        stroke_preview_last_args: Mutex::new(None),
        stroke_preview_suppresses_hover: AtomicBool::new(false),
        sculpt_stroke_replay: Mutex::new(Vec::new()),
        extrude_ray_spine: Mutex::new(None),
        collab: Arc::new(Mutex::new(collab::CollabRuntime::default())),
        smooth_presence: Mutex::new(HashMap::new()),
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
        walk_mode: Mutex::new(false),
        walk_physics: Mutex::new(camera::WalkPhysicsState::default()),
        walk_last_physics: Mutex::new(None),
        selection_cells: Mutex::new(AHashSet::new()),
        selection_stroke_before: Mutex::new(None),
        selection_stroke_accum: Mutex::new(None),
        selection_combine_mode: Mutex::new(SelectionCombineMode::Replace),
        selection_match_material: Mutex::new(false),
        stamp_clipboard: Mutex::new(None),
        squishy_session: Mutex::new(generators::SquishySession::new()),
        squishy_gizmo_drag: Mutex::new(None),
        selection_gizmo_drag: Mutex::new(SelectionGizmoDrag::None),
        start_screen_logo_transparent: std::sync::atomic::AtomicBool::new(true),
        start_screen_light: std::sync::atomic::AtomicBool::new(false),
        viewport_cursor_debug_overlay: AtomicBool::new(false),
        show_grid_borders: AtomicBool::new(false),
        hovered_gizmo_axis: AtomicU8::new(255),
        grid_overlay_cache_key: Mutex::new(None),
        selection_overlay_cache_key: Mutex::new(None),
        preview_overlay_cache_key: Mutex::new(None),
        fill_operation_cancel: Arc::new(AtomicBool::new(false)),
        spray_constraint_plane: Mutex::new(None),
        wall_stroke_face_snapped: Mutex::new(None),
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

    // ── merge_coords_into_selection ─────────────────────────────────

    #[test]
    fn merge_replace_clears_and_sets() {
        let mut sel: AHashSet<greedy_mesh::VoxelCoord> =
            [(0, 0, 0), (1, 1, 1)].into_iter().collect();
        merge_coords_into_selection(&mut sel, vec![(2, 2, 2)], SelectionCombineMode::Replace);
        assert_eq!(sel, [(2, 2, 2)].into_iter().collect());
    }

    #[test]
    fn merge_add_unions() {
        let mut sel: AHashSet<greedy_mesh::VoxelCoord> = [(0, 0, 0)].into_iter().collect();
        merge_coords_into_selection(&mut sel, vec![(1, 1, 1)], SelectionCombineMode::Add);
        assert_eq!(sel, [(0, 0, 0), (1, 1, 1)].into_iter().collect());
    }

    #[test]
    fn merge_subtract_removes() {
        let mut sel: AHashSet<greedy_mesh::VoxelCoord> =
            [(0, 0, 0), (1, 1, 1)].into_iter().collect();
        merge_coords_into_selection(&mut sel, vec![(1, 1, 1)], SelectionCombineMode::Subtract);
        assert_eq!(sel, [(0, 0, 0)].into_iter().collect());
    }

    #[test]
    fn merge_intersect_keeps_overlap() {
        let mut sel: AHashSet<greedy_mesh::VoxelCoord> =
            [(0, 0, 0), (1, 1, 1), (2, 2, 2)].into_iter().collect();
        merge_coords_into_selection(
            &mut sel,
            vec![(1, 1, 1), (2, 2, 2), (3, 3, 3)],
            SelectionCombineMode::Intersect,
        );
        assert_eq!(sel, [(1, 1, 1), (2, 2, 2)].into_iter().collect());
    }

    // ── apply_selection_stroke_sample (accumulator) ─────────────────

    /// Simulates a full stroke: begin → N samples → verify selection.
    /// With intersect mode, successive samples should union their coords
    /// against the original `before` snapshot rather than shrinking.
    #[test]
    fn intersect_stroke_accumulates_across_samples() {
        // Original selection: A B C D
        let a = (0, 0, 0);
        let b = (1, 0, 0);
        let c = (2, 0, 0);
        let d = (3, 0, 0);
        let before: AHashSet<greedy_mesh::VoxelCoord> = [a, b, c, d].into_iter().collect();

        // stroke_begin: snapshot before, create empty accumulator
        let mut sel = before.clone();
        let mut accum: Option<AHashSet<greedy_mesh::VoxelCoord>> = Some(AHashSet::new());
        let before_snap = Some(before);

        // Sample 1: spray hits only A
        let r = apply_selection_stroke_sample(
            &mut sel,
            vec![a],
            SelectionCombineMode::Intersect,
            &mut accum,
            &before_snap,
        );
        assert!(r.is_some());
        assert_eq!(sel, [a].into_iter().collect());
        // Accum should contain A
        assert!(accum.as_ref().unwrap().contains(&a));

        // Sample 2: spray hits only C
        let r = apply_selection_stroke_sample(
            &mut sel,
            vec![c],
            SelectionCombineMode::Intersect,
            &mut accum,
            &before_snap,
        );
        assert!(r.is_some());
        // Selection should be before ∩ {A, C} = {A, C}
        assert_eq!(sel, [a, c].into_iter().collect());

        // Sample 3: spray hits D and B
        let r = apply_selection_stroke_sample(
            &mut sel,
            vec![d, b],
            SelectionCombineMode::Intersect,
            &mut accum,
            &before_snap,
        );
        assert!(r.is_some());
        // Selection should be before ∩ {A, B, C, D} = {A, B, C, D}
        assert_eq!(sel, [a, b, c, d].into_iter().collect());
    }

    /// Without an accumulator (no active stroke), intersect should work
    /// directly on the current selection (single-click fallthrough).
    #[test]
    fn intersect_no_stroke_falls_through_to_direct_merge() {
        let a = (0, 0, 0);
        let b = (1, 0, 0);
        let c = (2, 0, 0);
        let mut sel: AHashSet<greedy_mesh::VoxelCoord> = [a, b, c].into_iter().collect();
        let mut accum: Option<AHashSet<greedy_mesh::VoxelCoord>> = None;
        let before: Option<AHashSet<greedy_mesh::VoxelCoord>> = None;

        let r = apply_selection_stroke_sample(
            &mut sel,
            vec![b],
            SelectionCombineMode::Intersect,
            &mut accum,
            &before,
        );
        assert!(r.is_some());
        assert_eq!(sel, [b].into_iter().collect());
    }

    /// Empty coords with active accumulator should still recompute
    /// selection (accum unchanged, but selection is re-derived).
    #[test]
    fn intersect_empty_sample_preserves_accum_state() {
        let a = (0, 0, 0);
        let b = (1, 0, 0);
        let before: AHashSet<greedy_mesh::VoxelCoord> = [a, b].into_iter().collect();
        let mut sel = before.clone();
        let mut accum: Option<AHashSet<greedy_mesh::VoxelCoord>> = Some([a].into_iter().collect());
        let before_snap = Some(before);

        // Empty sample — accum stays {A}, sel = before ∩ {A} = {A}
        let r = apply_selection_stroke_sample(
            &mut sel,
            vec![],
            SelectionCombineMode::Intersect,
            &mut accum,
            &before_snap,
        );
        assert!(r.is_some());
        assert_eq!(sel, [a].into_iter().collect());
    }

    /// Non-intersect modes should ignore the accumulator entirely.
    #[test]
    fn add_mode_ignores_accumulator() {
        let a = (0, 0, 0);
        let b = (1, 0, 0);
        let mut sel: AHashSet<greedy_mesh::VoxelCoord> = [a].into_iter().collect();
        let mut accum: Option<AHashSet<greedy_mesh::VoxelCoord>> = Some(AHashSet::new());
        let before = Some([a].into_iter().collect());

        let r = apply_selection_stroke_sample(
            &mut sel,
            vec![b],
            SelectionCombineMode::Add,
            &mut accum,
            &before,
        );
        assert!(r.is_some());
        assert_eq!(sel, [a, b].into_iter().collect());
    }
}
