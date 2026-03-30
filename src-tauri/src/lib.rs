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
/// Voxel format / types (public for `cargo bench` and tests).
pub mod voxelle;

use camera::OrbitCamera;
use gpu_brick::{BrickCellWrite, GpuVoxelBrick};
use render::{PreparedOpaqueUpload, WgpuViewer};
use std::any::Any;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, EventTarget, Manager, RunEvent, State};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

use std::collections::HashMap;
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

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum PreviewMode {
    #[default]
    Navigate,
    Add,
    Remove,
}

impl PreviewMode {
    fn parse(s: &str) -> Self {
        match s {
            "add" => Self::Add,
            "remove" => Self::Remove,
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

/// Millisecond timings for the last successful voxel add/remove (`voxel_edit_at_screen`).
#[derive(Clone, Debug)]
pub struct EditPerfBreakdown {
    pub apply_edit_ms: f64,
    /// Bounds scan + brick patch args (second `current_file` read, `mesh_bounds_from_voxels`, etc.).
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
    /// `cpu_chunked_incremental`: CPU greedy per dirty chunk.
    pub mesh_greedy_ms: f64,
    /// `cpu_chunked_incremental`: interleaved mesh → `wgpu` chunk buffers.
    pub mesh_chunk_buffers_ms: f64,
    /// Full [`WgpuViewer::upload_cpu_mesh_chunked_full`] inside remesh (origin drift).
    pub mesh_full_chunked_rebuild_ms: f64,
    /// Non-incremental [`WgpuViewer::rebuild_mesh_gpu_greedy`] wall time (includes internal CPU fallback).
    pub mesh_pipeline_ms: f64,
    pub preview_clear_ms: f64,
}

pub struct ViewerState {
    pub viewer: Mutex<Option<WgpuViewer>>,
    pub camera: Mutex<OrbitCamera>,
    pub file_label: Mutex<String>,
    /// Latest loaded model for CPU-side edits (add/remove voxels).
    pub current_file: Mutex<Option<voxelle::VoxelleFile>>,
    /// Spatial index: coord → index in `current_file.voxels` (kept in sync; used for raycasts + O(1) remove).
    pub voxel_map: Mutex<Option<HashMap<greedy_mesh::VoxelCoord, usize>>>,
    /// Latest pointer position in physical pixels (for hover preview; updated from UI, read each frame).
    pub preview_cursor: Mutex<Option<(f32, f32)>>,
    pub(crate) preview_mode: Mutex<PreviewMode>,
    pub rendering_mode: Mutex<RenderingMode>,
    fps: Mutex<FpsCounter>,
    /// Last successful edit timings (updated each time `voxel_edit_at_screen` applies a change).
    pub last_edit_perf: Mutex<Option<EditPerfBreakdown>>,
    /// Last scene AABB used for lighting/brick; drives incremental bounds on edit when possible.
    pub last_scene_bounds: Mutex<Option<greedy_mesh::MeshBounds>>,
    pub edit_undo: Mutex<Vec<voxel_edit::VoxelEditDelta>>,
    pub edit_redo: Mutex<Vec<voxel_edit::VoxelEditDelta>>,
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
}

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ViewportPixelSize {
    width: u32,
    height: u32,
}

/// Last known `.viewport` size in physical pixels (matches projection / picking).
#[tauri::command]
fn get_viewport_pixel_size(state: State<'_, Arc<ViewerState>>) -> Result<ViewportPixelSize, String> {
    let v = state.viewer.lock().map_err(|e| e.to_string())?;
    let Some(viewer) = v.as_ref() else {
        return Err("viewer not ready".into());
    };
    let (w, h) = viewer.viewport_size();
    Ok(ViewportPixelSize { width: w, height: h })
}

#[tauri::command]
fn viewer_resize(
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
    }
    Ok(())
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct PointerEvent {
    kind: String,
    x: f32,
    y: f32,
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

    let mut cam = state.camera.lock().map_err(|e| e.to_string())?;

    match ev.kind.as_str() {
        "down" | "move" => {
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
        "up" => {}
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
    let mut cam = state.camera.lock().map_err(|e| e.to_string())?;
    // Same `deltaY` semantics as the browser / Three.js `onMouseWheel`.
    cam.dolly_delta(ev.delta_y);
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

pub(crate) fn emit_load_progress(app: &AppHandle, fraction: f32, phase: impl Into<String>) {
    let _ = app.emit(
        "voxelle-load-progress",
        LoadProgressPayload {
            fraction: fraction.clamp(0.0, 1.0),
            phase: phase.into(),
        },
    );
}

/// Status bar progress for save, heavy mesh refresh, undo/redo (webview `voxelle-work-progress`).
pub(crate) fn emit_work_progress(app: &AppHandle, fraction: f32, phase: impl Into<String>) {
    let _ = app.emit(
        "voxelle-work-progress",
        LoadProgressPayload {
            fraction: fraction.clamp(0.0, 1.0),
            phase: phase.into(),
        },
    );
}

/// When armed, [`Drop`] emits 100% work progress so the status bar clears after `?` early returns too.
pub(crate) struct WorkProgressGuard<'a> {
    app: &'a AppHandle,
    armed: bool,
}

impl<'a> WorkProgressGuard<'a> {
    pub fn new(app: &'a AppHandle) -> Self {
        Self { app, armed: false }
    }

    pub fn arm(&mut self) {
        self.armed = true;
    }
}

impl Drop for WorkProgressGuard<'_> {
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
pub(crate) fn prepare_load_scene_cpu(
    grid_size: i32,
    voxels: &[voxelle::Voxel],
    objects: &[voxelle::SceneObject],
    mode: RenderingMode,
    app: Option<&AppHandle>,
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
    } else if voxels.len() >= greedy_mesh::CHUNKED_CPU_MESH_MIN_VOXELS
        && voxels
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
            voxels,
            greedy_mesh::SPATIAL_CHUNK_SIZE,
            |chunk_t| {
                let g = LOAD_P_MESH_START + chunk_t * mesh_span;
                let pct = (chunk_t * 100.0).min(100.0) as u32;
                emit(
                    g,
                    &format!("Building mesh ({pct}%)"),
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
                                let res = apply_mesh_and_camera(&state_c, &app_mesh, file_c, prepared);
                                let _ = done_tx.send(res);
                            });
                            return match done_rx.recv() {
                                Ok(Ok(())) => Ok(()),
                                Ok(Err(e)) => Err(e),
                                Err(_) => Err("main thread disconnected".into()),
                            };
                        }

                        run_v3_mesh_on_main(&state, &app, file, prepared)?;
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
                    let _ = app.emit("voxelle-loaded", label);
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
        let res = apply_mesh_and_camera(&state_c, &app_mesh, file, prepared);
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
    spawn_decode_and_mesh_with_label(state, app, path, label);
}

fn spawn_decode_and_mesh_with_label(
    state: Arc<ViewerState>,
    app: AppHandle,
    read_from: PathBuf,
    file_label: String,
) {
    let app_spawn_err = app.clone();
    match std::thread::Builder::new()
        .name("voxelle-load".into())
        .spawn(move || {
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
                            run_v3_mesh_on_main(&state, &app, file, prepared)?;
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
                        let r = apply_mesh_and_camera(&state_c, &app_emit, file, prepared);
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
                    if label.ends_with(".voxelle") {
                        persist_last_document_path(&app, &label);
                    }
                    let _ = app.emit("voxelle-loaded", label);
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

pub(crate) fn apply_mesh_and_camera(
    state: &Arc<ViewerState>,
    app: &AppHandle,
    file: voxelle::VoxelleFile,
    prepared: PreparedLoadScene,
) -> Result<(), String> {
    emit_load_progress(app, 0.76, "Uploading scene to GPU…");
    let PreparedLoadScene {
        bounds,
        brick,
        opaque,
    } = prepared;
    let fl = file.scene.focal_length_mm.unwrap_or(29.0);
    let orthographic = file.scene.orthographic;
    let voxel_map = greedy_mesh::voxel_map_indices(&file.voxels);
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
    *state.last_scene_bounds.lock().map_err(|e| e.to_string())? = Some(bounds);
    if let Ok(mut u) = state.edit_undo.lock() {
        u.clear();
    }
    if let Ok(mut r) = state.edit_redo.lock() {
        r.clear();
    }
    #[cfg(target_os = "macos")]
    macos_undo::clear_all(app);
    collab::broadcast_snapshot_to_guests(state);
    emit_load_progress(app, 0.97, "Finishing…");
    emit_load_progress(app, 1.0, "");
    Ok(())
}

fn scene_bounds_for_edit(
    _state: &ViewerState,
    file: &voxelle::VoxelleFile,
    _delta: &voxel_edit::VoxelEditDelta,
) -> Result<greedy_mesh::MeshBounds, String> {
    if file.voxels.is_empty() {
        return Ok(greedy_mesh::mesh_bounds_for_cube_side(file.grid_size));
    }
    Ok(
        greedy_mesh::mesh_bounds_from_voxels_world(&file.voxels, &file.objects)
            .or_else(|| greedy_mesh::mesh_bounds_from_voxels(&file.voxels))
            .unwrap_or_else(|| greedy_mesh::mesh_bounds_for_cube_side(file.grid_size)),
    )
}

/// GPU upload + mesh rebuild after a voxel change (shared by edit, undo, redo, collab).
pub(crate) fn finish_voxel_edit_gpu(
    state: &Arc<ViewerState>,
    delta: &voxel_edit::VoxelEditDelta,
    apply_edit_ms: f64,
    t_total: Instant,
    app: &AppHandle,
    reason: VoxelGpuRefreshReason,
) -> Result<(), String> {
    let t_prep_start = Instant::now();
    let fg = state.current_file.lock().map_err(|e| e.to_string())?;
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };

    let bounds = scene_bounds_for_edit(&**state, file, delta)?;

    let brick_patch = if file.voxels.is_empty() {
        None
    } else {
        Some(match delta {
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
        })
    };
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
    viewer.upload_scene_data(bounds, &file.voxels, brick_patch);
    let brick_ms = t_brick.elapsed().as_secs_f64() * 1000.0;

    if show_work {
        emit_work_progress(app, 0.38, "Rebuilding mesh…");
    }

    let t_mesh = Instant::now();
    let mut mesh_voxel_map_ms = 0.0;
    let mut mesh_buckets_ms = 0.0;
    let mut mesh_greedy_ms = 0.0;
    let mut mesh_chunk_buffers_ms = 0.0;
    let mut mesh_full_chunked_rebuild_ms = 0.0;
    let mut mesh_pipeline_ms = 0.0;

    if file.voxels.is_empty() {
        viewer.upload_mesh(&greedy_mesh::MeshBuffers::default());
        viewer.last_mesh_route = "clear".to_string();
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
    } else {
        let origin_new = greedy_mesh::voxel_aabb_min_int(&file.voxels).unwrap();
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
            && file.voxels.len() >= greedy_mesh::CHUNKED_CPU_MESH_MIN_VOXELS
            && viewer.chunk_grid_origin == origin_iv;

        if use_incremental {
            let t_cache = Instant::now();
            viewer.apply_spatial_cache_edit(delta);
            mesh_voxel_map_ms = t_cache.elapsed().as_secs_f64() * 1000.0;
            let cell = match delta {
                voxel_edit::VoxelEditDelta::Added(v) => (v.x, v.y, v.z),
                voxel_edit::VoxelEditDelta::Removed { voxel } => (voxel.x, voxel.y, voxel.z),
            };
            let center = greedy_mesh::chunk_key_from_world(
                cell.0,
                cell.1,
                cell.2,
                origin_new,
                greedy_mesh::SPATIAL_CHUNK_SIZE,
            );
            let dirty = greedy_mesh::dirty_chunk_keys_3x3(center);
            let (ok, rperf) = viewer.remesh_opaque_chunks(&dirty, &file.voxels);
            mesh_buckets_ms = rperf.buckets_ms;
            mesh_greedy_ms = rperf.greedy_ms;
            mesh_chunk_buffers_ms = rperf.chunk_buffers_ms;
            mesh_full_chunked_rebuild_ms = rperf.full_chunked_rebuild_ms;
            if ok {
                viewer.last_mesh_route = "cpu_chunked_incremental".to_string();
            }
        } else {
            let t_pipe = Instant::now();
            let _ = viewer.rebuild_mesh_gpu_greedy(&file.voxels, &file.objects);
            mesh_pipeline_ms = t_pipe.elapsed().as_secs_f64() * 1000.0;
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
        mesh_chunk_buffers_ms,
        mesh_full_chunked_rebuild_ms,
        mesh_pipeline_ms,
        preview_clear_ms,
    });

    *state.last_scene_bounds.lock().map_err(|e| e.to_string())? = Some(bounds);

    Ok(())
}

/// Rebuild opaque mesh from current voxels + [`RenderingMode`] (after switching view mode in the UI).
pub(crate) fn refresh_opaque_mesh(
    state: &Arc<ViewerState>,
    app: Option<&AppHandle>,
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
    let mut wp: Option<WorkProgressGuard> = None;
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
        let _ = viewer.rebuild_mesh_gpu_greedy(&file.voxels, &file.objects);
    }
    drop(wp);
    Ok(())
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
    let mut fg = state.current_file.lock().map_err(|e| e.to_string())?;
    let Some(file) = fg.as_mut() else {
        return Err("no model loaded".into());
    };
    let Some(obj) = file.objects.iter_mut().find(|o| o.id == id) else {
        return Err("unknown object".into());
    };
    obj.visible = visible;
    refresh_opaque_mesh(state.inner(), Some(&app))?;
    Ok(())
}

#[tauri::command]
fn create_scene_object(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    name: String,
) -> Result<u32, String> {
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
struct PickAtScreen {
    x: f32,
    y: f32,
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
        file, vmap, &cam, w, h, args.x, args.y,
    ))
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct VoxelEditAtScreen {
    x: f32,
    y: f32,
    /// `true` = place voxel on face under cursor; `false` = remove hit voxel.
    add: bool,
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
    let edit_delta = {
        let mut fg = state.current_file.lock().map_err(|e| e.to_string())?;
        let mut vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.camera.lock().map_err(|e| e.to_string())?;
        voxel_edit::apply_edit(file, vmap, &cam, w, h, args.x, args.y, args.add)?
    };
    let apply_edit_ms = t_apply_start.elapsed().as_secs_f64() * 1000.0;

    let Some(delta) = edit_delta else {
        return Ok(false);
    };

    finish_voxel_edit_gpu(
        &state,
        &delta,
        apply_edit_ms,
        t_total,
        &app,
        VoxelGpuRefreshReason::SoloEdit,
    )?;

    let cm = Arc::clone(&state.collab);
    let mut cb = cm.lock().map_err(|e| e.to_string())?;
    if cb.is_client() {
        if let Some(tx) = &cb.client_tx {
            let msg = serde_json::to_string(&collab::ClientToHost::Edit { delta }).unwrap();
            let _ = tx.send(msg);
        }
    } else if cb.is_host() {
        cb.next_seq += 1;
        let seq = cb.next_seq;
        cb.host_undo
            .entry(collab::HOST_PEER_ID)
            .or_default()
            .push(delta);
        cb.host_redo.remove(&collab::HOST_PEER_ID);
        drop(cb);
        collab::host_emit_edit(&cm, &app, seq, collab::HOST_PEER_ID, delta);
    } else {
        drop(cb);
        state
            .edit_undo
            .lock()
            .map_err(|e| e.to_string())?
            .push(delta);
        state.edit_redo.lock().map_err(|e| e.to_string())?.clear();
        #[cfg(target_os = "macos")]
        macos_undo::register_solo_edit_completed(&app, &state);
    }

    Ok(true)
}

/// Solo (non-collab) undo: pop `edit_undo`, apply inverse, GPU refresh, push `edit_redo`.
pub(crate) fn perform_solo_voxel_undo(
    state: &Arc<ViewerState>,
    app: &AppHandle,
) -> Result<bool, String> {
    let t_total = Instant::now();
    let original = {
        let mut u = state.edit_undo.lock().map_err(|e| e.to_string())?;
        u.pop()
    };
    let Some(original) = original else {
        return Ok(false);
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
                    .ok_or_else(|| "undo: voxel missing".to_string())?;
                voxel_edit::VoxelEditDelta::Removed { voxel: v }
            }
            voxel_edit::VoxelEditDelta::Removed { voxel } => {
                voxel_edit::push_voxel_known(file, vmap, voxel);
                voxel_edit::VoxelEditDelta::Added(voxel)
            }
        }
    };
    finish_voxel_edit_gpu(
        state,
        &mesh_delta,
        0.0,
        t_total,
        app,
        VoxelGpuRefreshReason::Undo,
    )?;
    state
        .edit_redo
        .lock()
        .map_err(|e| e.to_string())?
        .push(original);
    Ok(true)
}

/// Solo redo: pop `edit_redo`, re-apply, GPU refresh, push `edit_undo`.
pub(crate) fn perform_solo_voxel_redo(
    state: &Arc<ViewerState>,
    app: &AppHandle,
) -> Result<bool, String> {
    let t_total = Instant::now();
    let forward = {
        let mut r = state.edit_redo.lock().map_err(|e| e.to_string())?;
        r.pop()
    };
    let Some(forward) = forward else {
        return Ok(false);
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
                    .ok_or_else(|| "redo: voxel missing".to_string())?;
                voxel_edit::VoxelEditDelta::Removed { voxel }
            }
        }
    };
    finish_voxel_edit_gpu(
        state,
        &mesh_delta,
        0.0,
        t_total,
        app,
        VoxelGpuRefreshReason::Redo,
    )?;
    state
        .edit_undo
        .lock()
        .map_err(|e| e.to_string())?
        .push(forward);
    Ok(true)
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
            collab::host_emit_edit(&cm, &app, seq, collab::HOST_PEER_ID, d);
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
            collab::host_emit_edit(&cm, &app, seq, collab::HOST_PEER_ID, d);
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
        let _ = app_c.emit("voxelle-loaded", s);
    });
    Ok(())
}

/// Lightweight: only stores cursor + mode for the next frame’s GPU preview (no mesh work on IPC thread).
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SyncPreviewInput {
    x: f32,
    y: f32,
    mode: String,
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
    if args.x < 0.0 {
        *state.preview_cursor.lock().map_err(|e| e.to_string())? = None;
    } else {
        *state.preview_cursor.lock().map_err(|e| e.to_string())? = Some((args.x, args.y));
    }
    Ok(())
}

fn sync_ping_flash(viewer: &mut WgpuViewer, state: &ViewerState) {
    let snap = state.ping_flash.lock().ok().and_then(|g| *g);
    let Some(f) = snap else {
        viewer.clear_ping_mesh();
        return;
    };
    if std::time::Instant::now() > f.until {
        *state.ping_flash.lock().unwrap() = None;
        viewer.clear_ping_mesh();
        return;
    }
    let key = (f.x, f.y, f.z, f.color_rgb);
    if viewer.ping_mesh_key == Some(key) {
        return;
    }
    viewer.ping_mesh_key = Some(key);
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

fn refresh_preview_mesh(viewer: &mut WgpuViewer, state: &ViewerState, cam: &OrbitCamera) {
    let (cursor, mode) = {
        let c = state.preview_cursor.lock().unwrap();
        let m = state.preview_mode.lock().unwrap();
        (*c, *m)
    };

    if matches!(mode, PreviewMode::Navigate) {
        viewer.clear_preview_mesh();
        return;
    }

    let Some((sx, sy)) = cursor else {
        viewer.clear_preview_mesh();
        return;
    };
    if sx < 0.0 || sy < 0.0 {
        viewer.clear_preview_mesh();
        return;
    }

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
    let key = match mode {
        PreviewMode::Add => voxel_edit::preview_add_cell(file, vmap, cam, w, h, sx, sy)
            .map(|(x, y, z)| (x, y, z, 0u8)),
        PreviewMode::Remove => voxel_edit::preview_remove_cell(file, vmap, cam, w, h, sx, sy)
            .map(|(x, y, z)| (x, y, z, 1u8)),
        PreviewMode::Navigate => None,
    };

    if key == viewer.preview_cache_key {
        return;
    }
    viewer.preview_cache_key = key;

    match key {
        Some((cx, cy, cz, 0)) => {
            let solid = greedy_mesh::preview_cube_mesh(
                cx as f32,
                cy as f32,
                cz as f32,
                0.5,
                [0.25, 0.92, 0.4],
                1.0,
            );
            let wire = greedy_mesh::preview_cube_wireframe_mesh(
                cx as f32,
                cy as f32,
                cz as f32,
                0.5,
                [0.02, 0.09, 0.05],
                2.0,
            );
            viewer.upload_preview_mesh(&solid, &wire);
        }
        Some((cx, cy, cz, 1)) => {
            let solid = greedy_mesh::preview_cube_mesh(
                cx as f32,
                cy as f32,
                cz as f32,
                0.53,
                [0.95, 0.28, 0.22],
                1.0,
            );
            let wire = greedy_mesh::preview_cube_wireframe_mesh(
                cx as f32,
                cy as f32,
                cz as f32,
                0.53,
                [0.14, 0.03, 0.03],
                2.0,
            );
            viewer.upload_preview_mesh(&solid, &wire);
        }
        None | Some(_) => {
            viewer.clear_preview_mesh();
        }
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
        website_label: Some("GitHub — source & issues".into()),
        comments: Some("Collaborative voxel editing on the desktop.".into()),
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
            "Collaborative voxel editing on the desktop.\n\nSource & issues:\n{GITHUB_VD}"
        ));
    }
    Ok(m)
}

#[cfg(desktop)]
fn install_app_menu(app: &AppHandle) -> tauri::Result<()> {
    use tauri::menu::{Menu, MenuItem, MenuItemKind, PredefinedMenuItem, Submenu};

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
    let debug_copy_perf = MenuItem::with_id(
        app,
        "debug_copy_performance",
        "Copy Performance Data to Clipboard",
        true,
        None::<&str>,
    )?;
    let debug_menu = Submenu::with_items(app, "Debug", true, &[&debug_copy_perf])?;
    let sep = PredefinedMenuItem::separator(app)?;
    let view_render_greedy = MenuItem::with_id(app, "view_render_greedy", "Blocky (Greedy)", true, None::<&str>)?;
    let view_render_marching = MenuItem::with_id(
        app,
        "view_render_marching",
        "Smooth (Marching Cubes)",
        true,
        None::<&str>,
    )?;
    let view_render_dual = MenuItem::with_id(
        app,
        "view_render_dual",
        "Smooth (Dual Contour)",
        true,
        None::<&str>,
    )?;
    let rendering_submenu = Submenu::with_items(
        app,
        "Rendering",
        true,
        &[&view_render_greedy, &view_render_marching, &view_render_dual],
    )?;
    let ortho_view_item = MenuItem::with_id(app, "menu_view_ortho", "Orthographic", true, None::<&str>)?;
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
                sub.prepend_items(&[&new_item, &open_item, &save_item, &save_as_item, &sep])?;
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
            &[&rendering_submenu, &ortho_view_item, &sep_before_chat, &chat_panel_item],
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

    menu.append(&debug_menu)?;
    menu.set_as_app_menu()?;
    Ok(())
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
                 \tprepare (bounds + brick patch): {:.2}\n\
                 \tviewer lock wait: {:.2}\n\
                 \tbrick (upload_scene_data): {:.2}\n\
                 \tmesh (total): {:.2}\n\
                 \t  spatial cache delta: {:.2}\n\
                 \t  spatial cache cold init: {:.2}\n\
                 \t  greedy (dirty chunks): {:.2}\n\
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

#[tauri::command]
fn load_voxelle_path(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    path: String,
) -> Result<(), String> {
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
        drop(c);
        collab::record_ping_flash_colored(Arc::as_ref(&*state), x, y, z, host_color);
        c = state.collab.lock().map_err(|e| e.to_string())?;
        let ping = collab::HostToClient::Ping {
            peer_id: collab::HOST_PEER_ID,
            x,
            y,
            z,
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
        rendering_mode: Mutex::new(RenderingMode::Greedy),
        fps: Mutex::new(FpsCounter {
            period_start: None,
            accum_frames: 0,
            last_fps: 0,
        }),
        last_edit_perf: Mutex::new(None),
        last_scene_bounds: Mutex::new(None),
        edit_undo: Mutex::new(Vec::new()),
        edit_redo: Mutex::new(Vec::new()),
        collab: Arc::new(std::sync::Mutex::new(collab::CollabRuntime::default())),
        ping_flash: Mutex::new(None),
        autosave_interval_secs: Mutex::new(120),
        last_autosave: Mutex::new(None),
        autosave_enabled: Mutex::new(true),
        autosave_keep_count: Mutex::new(5),
        autosave_slot: Mutex::new(HashMap::new()),
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
            } else if event.id() == "menu_view_ortho" {
                let state = app.state::<Arc<ViewerState>>();
                let new_o = state
                    .camera
                    .lock()
                    .map(|c| c.perspective)
                    .unwrap_or(true);
                let _ = apply_orthographic(&state, new_o);
                wake_viewport_loop(&app);
            }
        })
        .setup(move |app| {
            #[cfg(desktop)]
            install_app_menu(app.handle())?;

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
            viewport_pointer,
            viewport_wheel,
            open_voxelle_dialog,
            confirm_app_update_dialog,
            load_voxelle_path,
            load_voxelle_recovery,
            get_last_session_info,
            create_new_project,
            voxel_pick_probe,
            sync_preview_input,
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
            set_tone_mapping,
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
                let mut v = state.viewer.lock().unwrap();
                if let Some(viewer) = v.as_mut() {
                    let cam = state.camera.lock().unwrap();
                    viewer.update_uniforms(&cam);
                    refresh_preview_mesh(viewer, Arc::as_ref(&state), &cam);
                    sync_collab_peer_lines(viewer, Arc::as_ref(&state));
                    sync_ping_flash(viewer, Arc::as_ref(&state));
                    let sz_before = viewer.surface_size;
                    let _ = viewer.render();
                    let (vw, vh) = viewer.viewport_size();
                    if viewer.surface_size != sz_before {
                        let _ = app.emit_to(
                            EventTarget::webview_window("main"),
                            "viewport-pixel-size",
                            ViewportPixelSize {
                                width: vw,
                                height: vh,
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
                    if enabled && interval > 0 && (!collab_on || is_host) {
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
                let needs_next = state.camera.lock().unwrap().needs_redraw();
                if needs_next {
                    tauri::async_runtime::spawn(async move {
                        let _ = app_wake.run_on_main_thread(|| {});
                    });
                }
            }
        });
}
