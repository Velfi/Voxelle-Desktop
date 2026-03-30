mod camera;
mod gpu_brick;
/// Greedy CPU meshing (public for `cargo bench`).
pub mod greedy_mesh;
mod render;
mod render_constants;
mod voxel_edit;
/// Voxel format / types (public for `cargo bench` and tests).
pub mod voxelle;

use camera::OrbitCamera;
use gpu_brick::BrickCellWrite;
use render::WgpuViewer;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager, RunEvent, State};
use tauri_plugin_dialog::DialogExt;

use std::collections::HashMap;

use voxelle::{decode_payload, focal_length_to_fov_y_radians, start_shape::StartShape};

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
    fps: Mutex<FpsCounter>,
    /// Last successful edit timings (updated each time `voxel_edit_at_screen` applies a change).
    pub last_edit_perf: Mutex<Option<EditPerfBreakdown>>,
    /// Last scene AABB used for lighting/brick; drives incremental bounds on edit when possible.
    pub last_scene_bounds: Mutex<Option<greedy_mesh::MeshBounds>>,
}

#[tauri::command]
fn viewer_resize(state: State<'_, Arc<ViewerState>>, width: u32, height: u32) -> Result<(), String> {
    let mut g = state.viewer.lock().map_err(|e| e.to_string())?;
    if let Some(v) = g.as_mut() {
        v.resize(width, height);
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
fn viewport_pointer(state: State<'_, Arc<ViewerState>>, ev: PointerEvent) -> Result<(), String> {
    // Read size without holding `camera` — the run loop locks `viewer` then `camera`; taking
    // `camera` then `viewer` here deadlocks with the render tick and freezes orbit input.
    let (vw, vh) = {
        let v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_ref() else {
            return Ok(());
        };
        let w = viewer.size.0 as f32;
        let h = viewer.size.1 as f32;
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
    Ok(())
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct WheelEvent {
    delta_x: f32,
    delta_y: f32,
}

#[tauri::command]
fn viewport_wheel(state: State<'_, Arc<ViewerState>>, ev: WheelEvent) -> Result<(), String> {
    let mut cam = state.camera.lock().map_err(|e| e.to_string())?;
    // Same `deltaY` semantics as the browser / Three.js `onMouseWheel`.
    cam.dolly_delta(ev.delta_y);
    Ok(())
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
    std::thread::Builder::new()
        .name("voxelle-new-project".into())
        .spawn(move || {
            let _ = app.emit("voxelle-load-progress", 0.05f32);

            let mesh_result: Result<(), String> = (|| {
                let size = grid_size as i32;
                let voxels = voxelle::start_shape::voxels_for_start_shape(size, shape)?;
                let file = voxelle::VoxelleFile {
                    version: 3,
                    grid_size: size,
                    scene: Default::default(),
                    voxels,
                };

                if file.voxels.is_empty() {
                    let _ = app.emit("voxelle-load-progress", 0.85f32);
                    let (done_tx, done_rx) = std::sync::mpsc::channel();
                    let app_c = app.clone();
                    let state_c = Arc::clone(&state);
                    let file_c = file.clone();
                    let _ = app_c.run_on_main_thread(move || {
                        let res = apply_mesh_and_camera(&state_c, file_c);
                        let _ = done_tx.send(res);
                    });
                    return match done_rx.recv() {
                        Ok(Ok(())) => Ok(()),
                        Ok(Err(e)) => Err(e),
                        Err(_) => Err("main thread disconnected".into()),
                    };
                }

                run_v3_mesh_on_main(&state, &app, &file)?;
                Ok(())
            })();

            let app_emit = app.clone();
            let _ = app.run_on_main_thread(move || {
                match mesh_result {
                    Ok(()) => {
                        let _ = app_emit.emit("voxelle-loaded", label);
                    }
                    Err(e) => {
                        let _ = app_emit.emit("voxelle-load-error", e);
                    }
                }
            });
        })
        .ok();
}

enum DecodeMeshOutcome {
    /// Single upload on the main thread (BSON / small payloads, non-v3).
    ApplyOnce { file: voxelle::VoxelleFile },
    /// v3 with voxels: mesh already applied inside `run_v3_mesh_on_main`.
    Done,
}

fn run_v3_mesh_on_main(state: &Arc<ViewerState>, app: &AppHandle, file: &voxelle::VoxelleFile) -> Result<(), String> {
    let _ = app.emit("voxelle-load-progress", 0.55f32);
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let app_c = app.clone();
    let state_c = Arc::clone(state);
    let file_c = file.clone();
    let _ = app_c.run_on_main_thread(move || {
        let res = apply_mesh_and_camera(&state_c, file_c);
        let _ = done_tx.send(res);
    });
    match done_rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err("main thread disconnected".into()),
    }
    let _ = app.emit("voxelle-load-progress", 0.85f32);
    Ok(())
}

fn spawn_decode_and_mesh(state: Arc<ViewerState>, app: AppHandle, path: std::path::PathBuf) {
    std::thread::Builder::new()
        .name("voxelle-load".into())
        .spawn(move || {
            let label = path.to_string_lossy().to_string();
            let _ = app.emit("voxelle-load-progress", 0.05f32);

            let mesh_result: Result<DecodeMeshOutcome, String> = (|| {
                let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
                let _ = app.emit("voxelle-load-progress", 0.2f32);
                let file = decode_payload(&bytes).map_err(|e| e.to_string())?;

                if file.version == 3 && !file.voxels.is_empty() {
                    let _ = app.emit("voxelle-load-progress", 0.45f32);
                    run_v3_mesh_on_main(&state, &app, &file)?;
                    return Ok(DecodeMeshOutcome::Done);
                }

                let _ = app.emit("voxelle-load-progress", 0.45f32);
                let _ = app.emit("voxelle-load-progress", 0.85f32);
                Ok(DecodeMeshOutcome::ApplyOnce { file })
            })();

            let app_emit = app.clone();
            let _ = app.run_on_main_thread(move || {
                match mesh_result {
                    Ok(DecodeMeshOutcome::ApplyOnce { file }) => {
                        if let Err(e) = apply_mesh_and_camera(&state, file) {
                            let _ = app_emit.emit("voxelle-load-error", e);
                        } else {
                            let _ = app_emit.emit("voxelle-loaded", label);
                        }
                    }
                    Ok(DecodeMeshOutcome::Done) => {
                        let _ = app_emit.emit("voxelle-loaded", label);
                    }
                    Err(e) => {
                        let _ = app_emit.emit("voxelle-load-error", e);
                    }
                }
            });
        })
        .ok();
}

fn apply_mesh_and_camera(state: &Arc<ViewerState>, file: voxelle::VoxelleFile) -> Result<(), String> {
    let bounds = if file.voxels.is_empty() {
        greedy_mesh::mesh_bounds_for_cube_side(file.grid_size)
    } else {
        greedy_mesh::mesh_bounds_from_voxels(&file.voxels)
            .unwrap_or_else(|| greedy_mesh::mesh_bounds_for_cube_side(file.grid_size))
    };
    {
        let mut cf = state.current_file.lock().map_err(|e| e.to_string())?;
        let mut vm = state.voxel_map.lock().map_err(|e| e.to_string())?;
        *cf = Some(file.clone());
        *vm = Some(greedy_mesh::voxel_map_indices(&file.voxels));
    }
    let mut v = state.viewer.lock().map_err(|e| e.to_string())?;
    let Some(viewer) = v.as_mut() else {
        return Err("viewer not ready".into());
    };
    viewer.upload_scene_data(bounds, &file.voxels, None);
    if file.voxels.is_empty() {
        viewer.upload_mesh(&greedy_mesh::MeshBuffers::default());
    } else if viewer.rebuild_mesh_gpu_greedy(&file.voxels).is_err() {
        viewer.cpu_mesh_fallback(&file.voxels);
    }
    viewer.clear_preview_mesh();

    let mut cam = state.camera.lock().map_err(|e| e.to_string())?;
    let fl = file.scene.focal_length_mm.unwrap_or(29.0);
    cam.fov_y = focal_length_to_fov_y_radians(fl);
    cam.perspective = !file.scene.orthographic;
    if file.scene.orthographic {
        let r = bounds.radius().max(1.0);
        cam.ortho_half_height = r * 1.1;
    }

    let center = bounds.center();
    let r = bounds.radius().max(1.0);
    let (w, h) = viewer.size;
    cam.fit_sphere(center, r, w as f32, h as f32);
    *state.last_scene_bounds.lock().map_err(|e| e.to_string())? = Some(bounds);
    Ok(())
}

fn scene_bounds_for_edit(
    state: &ViewerState,
    file: &voxelle::VoxelleFile,
    delta: &voxel_edit::VoxelEditDelta,
) -> Result<greedy_mesh::MeshBounds, String> {
    if file.voxels.is_empty() {
        return Ok(greedy_mesh::mesh_bounds_for_cube_side(file.grid_size));
    }
    let last = state
        .last_scene_bounds
        .lock()
        .map_err(|e| e.to_string())?
        .clone();
    let Some(prev) = last else {
        return Ok(
            greedy_mesh::mesh_bounds_from_voxels(&file.voxels)
                .unwrap_or_else(|| greedy_mesh::mesh_bounds_for_cube_side(file.grid_size)),
        );
    };
    match delta {
        voxel_edit::VoxelEditDelta::Added(v) => Ok(greedy_mesh::mesh_bounds_expand_with_voxel(&prev, v)),
        voxel_edit::VoxelEditDelta::Removed { x, y, z } => {
            if greedy_mesh::mesh_bounds_remove_is_strict_interior(&prev, *x, *y, *z) {
                Ok(prev)
            } else {
                Ok(
                    greedy_mesh::mesh_bounds_from_voxels(&file.voxels)
                        .unwrap_or_else(|| greedy_mesh::mesh_bounds_for_cube_side(file.grid_size)),
                )
            }
        }
    }
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct PickAtScreen {
    x: f32,
    y: f32,
}

/// Whether the camera ray from this screen point hits solid geometry (voxel) — used to choose camera vs edit.
#[tauri::command]
fn voxel_pick_probe(state: State<'_, Arc<ViewerState>>, args: PickAtScreen) -> Result<bool, String> {
    let (w, h) = {
        let v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_ref() else {
            return Ok(false);
        };
        (viewer.size.0 as f32, viewer.size.1 as f32)
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
    Ok(voxel_edit::probe_solid_hit(file, vmap, &cam, w, h, args.x, args.y))
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
fn voxel_edit_at_screen(state: State<'_, Arc<ViewerState>>, args: VoxelEditAtScreen) -> Result<bool, String> {
    let t_total = Instant::now();
    let (w, h) = {
        let v = state.viewer.lock().map_err(|e| e.to_string())?;
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        (viewer.size.0 as f32, viewer.size.1 as f32)
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

    let t_prep_start = Instant::now();
    let fg = state.current_file.lock().map_err(|e| e.to_string())?;
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };

    let bounds = scene_bounds_for_edit(&**state, file, &delta)?;

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
            voxel_edit::VoxelEditDelta::Removed { x, y, z } => BrickCellWrite {
                x,
                y,
                z,
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

    let t_brick = Instant::now();
    viewer.upload_scene_data(bounds, &file.voxels, brick_patch);
    let brick_ms = t_brick.elapsed().as_secs_f64() * 1000.0;

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
    } else {
        let origin_new = greedy_mesh::voxel_aabb_min_int(&file.voxels).unwrap();
        let origin_iv = glam::IVec3::new(origin_new.0, origin_new.1, origin_new.2);
        let use_incremental = viewer.opaque_chunked
            && file.voxels.len() >= greedy_mesh::CHUNKED_CPU_MESH_MIN_VOXELS
            && viewer.chunk_grid_origin == origin_iv;

        if use_incremental {
            let t_cache = Instant::now();
            viewer.apply_spatial_cache_edit(&delta);
            mesh_voxel_map_ms = t_cache.elapsed().as_secs_f64() * 1000.0;
            let cell = match delta {
                voxel_edit::VoxelEditDelta::Added(v) => (v.x, v.y, v.z),
                voxel_edit::VoxelEditDelta::Removed { x, y, z } => (x, y, z),
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
            let _ = viewer.rebuild_mesh_gpu_greedy(&file.voxels);
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

    Ok(true)
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
fn sync_preview_input(state: State<'_, Arc<ViewerState>>, args: SyncPreviewInput) -> Result<(), String> {
    if args.x < 0.0 {
        *state.preview_cursor.lock().map_err(|e| e.to_string())? = None;
    } else {
        *state.preview_cursor.lock().map_err(|e| e.to_string())? = Some((args.x, args.y));
    }
    *state.preview_mode.lock().map_err(|e| e.to_string())? = PreviewMode::parse(&args.mode);
    Ok(())
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

    let (w, h) = (viewer.size.0 as f32, viewer.size.1 as f32);
    let key = match mode {
        PreviewMode::Add => {
            voxel_edit::preview_add_cell(file, vmap, cam, w, h, sx, sy).map(|(x, y, z)| (x, y, z, 0u8))
        }
        PreviewMode::Remove => {
            voxel_edit::preview_remove_cell(file, vmap, cam, w, h, sx, sy).map(|(x, y, z)| (x, y, z, 1u8))
        }
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
    let mut builder = app
        .dialog()
        .file()
        .add_filter("Voxelle", &["voxelle"]);
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
fn install_app_menu(app: &AppHandle) -> tauri::Result<()> {
    use tauri::menu::{Menu, MenuItem, MenuItemKind, PredefinedMenuItem, Submenu};

    let menu = Menu::default(app)?;
    let new_item = MenuItem::with_id(app, "new_project", "New Project…", true, None::<&str>)?;
    let open_item = MenuItem::with_id(
        app,
        "open_voxelle",
        "Open…",
        true,
        Some("CommandOrCtrl+O"),
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

    let mut inserted = false;
    for item in menu.items()? {
        if let MenuItemKind::Submenu(sub) = item {
            if sub.text()? == "File" {
                sub.prepend_items(&[&new_item, &open_item, &sep])?;
                inserted = true;
                break;
            }
        }
    }

    if !inserted {
        let close = PredefinedMenuItem::close_window(app, None)?;
        let file_menu = Submenu::with_items(app, "File", true, &[&new_item, &open_item, &sep, &close])?;
        menu.prepend(&file_menu)?;
    }

    menu.append(&debug_menu)?;
    menu.set_as_app_menu()?;
    Ok(())
}

#[cfg(desktop)]
fn performance_report_text(state: &ViewerState) -> String {
    let fps = state
        .fps
        .lock()
        .map(|c| c.last_fps)
        .unwrap_or(0);
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
                (
                    viewer.size.0,
                    viewer.size.1,
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
        .and_then(|g| {
            g.as_ref()
                .map(|f| (f.voxels.len(), f.grid_size))
        })
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

#[tauri::command]
fn open_voxelle_dialog(state: State<'_, Arc<ViewerState>>, app: AppHandle) -> Result<(), String> {
    open_voxelle_file_dialog(app, Arc::clone(&*state));
    Ok(())
}

#[tauri::command]
fn load_voxelle_path(state: State<'_, Arc<ViewerState>>, app: AppHandle, path: String) -> Result<(), String> {
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
fn create_new_project(state: State<'_, Arc<ViewerState>>, app: AppHandle, args: NewProjectArgs) -> Result<(), String> {
    let grid_size = args.grid_size.clamp(1, MAX_GRID_SIZE);
    let shape_l = start_shape_label(args.shape);
    let label = format!("New project ({grid_size}³, {shape_l})");
    *state.file_label.lock().map_err(|e| e.to_string())? = label.clone();
    let _ = app.emit("voxelle-load-start", label);
    spawn_new_project(Arc::clone(&*state), app, grid_size, args.shape);
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let viewer_state = Arc::new(ViewerState {
        viewer: Mutex::new(None),
        camera: Mutex::new(OrbitCamera::new()),
        file_label: Mutex::new(String::new()),
        current_file: Mutex::new(None),
        voxel_map: Mutex::new(None),
        preview_cursor: Mutex::new(None),
        preview_mode: Mutex::new(PreviewMode::Navigate),
        fps: Mutex::new(FpsCounter {
            period_start: None,
            accum_frames: 0,
            last_fps: 0,
        }),
        last_edit_perf: Mutex::new(None),
        last_scene_bounds: Mutex::new(None),
    });
    let vs = viewer_state.clone();

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(viewer_state.clone())
        .on_menu_event(|app, event| {
            if event.id() == "open_voxelle" {
                let state = app.state::<Arc<ViewerState>>();
                open_voxelle_file_dialog(app.clone(), state.inner().clone());
            } else if event.id() == "new_project" {
                let _ = app.emit("voxelle-open-new-project", ());
            } else if event.id() == "debug_copy_performance" {
                let state = app.state::<Arc<ViewerState>>();
                if let Err(e) = copy_performance_data_to_clipboard(state.inner()) {
                    eprintln!("copy performance data: {e}");
                }
            }
        })
        .setup(move |app| {
            #[cfg(desktop)]
            install_app_menu(app.handle())?;

            let window = app.get_webview_window("main").expect("main window");
            let w = window.clone();
            let mut viewer = tauri::async_runtime::block_on(async move { WgpuViewer::new(w).await })
                .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
            if let Ok(sz) = window.inner_size() {
                viewer.resize(sz.width, sz.height);
            }
            *vs.viewer.lock().unwrap() = Some(viewer);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            viewer_resize,
            viewport_pointer,
            viewport_wheel,
            open_voxelle_dialog,
            load_voxelle_path,
            create_new_project,
            voxel_pick_probe,
            sync_preview_input,
            voxel_edit_at_screen
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
                    let _ = viewer.render();
                    sample_fps_and_emit(app, &state.fps);
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
