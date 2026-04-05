//! File loading / unloading pipeline.
//!
//! Extracted from `lib.rs` — contains the CPU mesh-prep, GPU upload, project
//! create/open/unload helpers, and the background decode-and-mesh machinery.

use crate::collab;
use crate::gpu_brick::GpuVoxelBrick;
use crate::greedy_mesh;
#[cfg(target_os = "macos")]
use crate::macos_undo;
#[cfg(desktop)]
use crate::native_menu::{rebuild_recent_submenu, RecentMenuState};
use crate::render::{MoodParams, PreparedOpaqueUpload};
use crate::smooth_mesh;
use crate::state::*;
use crate::voxelle;
use crate::voxelle::{decode_payload, focal_length_to_fov_y_radians, start_shape::StartShape};
use crate::{
    clear_preview_mesh_sync_cache, mood_settings_to_params, persist_last_document_path,
    persist_recent_file, try_initial_autosave_after_new_project,
};
#[cfg(desktop)]
use crate::{scene_menu_flags, selection_menu_sync_enabled_for_scene};

use std::any::Any;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use ahash::AHashSet;
use tauri::{AppHandle, Emitter, Manager, Runtime};

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

/// Build chunk meshes from a [`SpatialMeshCache`] and push each to `state.gpu.chunk_mesh_inbox`.
/// When done, deposits the cache into `state.gpu.deferred_spatial_cache`.
/// Respects `load_gen` — bails if a newer load has started.
pub(crate) fn stream_chunk_meshes_to_inbox(
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
        state.gpu.chunk_mesh_inbox.lock().push_back((key, mesh));
    });

    if is_load_stale(state, load_gen) {
        log::info!(target: "voxelle_load", "stream_chunk_meshes_to_inbox: cancelled (stale)");
        return;
    }

    // Hand the cache to the viewer (main thread will pick it up).
    *state.gpu.deferred_spatial_cache.lock() = Some(cache);
    log::info!(
        target: "voxelle_load",
        "stream_chunk_meshes_to_inbox: done {total} chunks {:?}",
        t.elapsed()
    );
}

/// Clears the loaded model, GPU meshes, and editing state. Must run on the main thread (GPU + AppKit undo).
pub(crate) fn unload_current_project<R: Runtime>(
    state: &Arc<ViewerState>,
    app: &AppHandle<R>,
) -> Result<(), String> {
    let mode = *state.gpu.rendering_mode.lock();
    let objects = voxelle::default_scene_objects();
    let prepared =
        prepare_load_scene_cpu::<R>(crate::MAX_GRID_SIZE as i32, &[], &objects, mode, None)?;
    {
        let mut cf = state.file.current_file.lock();
        let mut vm = state.file.voxel_map.lock();
        *cf = None;
        *vm = None;
    }
    state.cam.active_project.store(false, Ordering::Release);
    let mut v = state.gpu.viewer.lock();
    let Some(viewer) = v.as_mut() else {
        return Err("viewer not ready".into());
    };
    viewer.upload_scene_data_from_brick(prepared.bounds, prepared.brick);
    viewer.upload_prepared_opaque(prepared.opaque);
    clear_preview_mesh_sync_cache(viewer, state.as_ref());
    viewer.clear_selection_overlay();
    *state.gpu.selection_overlay_cache_key.lock() = None;
    viewer.clear_grid_border_lines();
    *state.gpu.grid_overlay_cache_key.lock() = None;
    viewer.clear_collab_peer_lines();
    viewer.clear_ping_mesh();
    viewer.set_mood_params(&MoodParams::default());
    viewer.speech_bubbles.clear();
    // Restore start-screen rendering state so the logo and mascots draw again.
    viewer.set_start_screen_transparent(true);
    if let Some(logo) = viewer.logo_overlay.as_mut() {
        logo.visible = true;
    }
    drop(v);
    state
        .gpu
        .start_screen_logo_transparent
        .store(true, Ordering::Release);

    *state.gpu.last_scene_bounds.lock() = Some(prepared.bounds);
    *state.gpu.voxel_edit_stats_cache.lock() = None;
    *state.gpu.last_edit_perf.lock() = None;
    state
        .gpu
        .mesh_refresh_generation
        .fetch_add(1, Ordering::Release);

    state.file.solo_undo.lock().clear();
    state.file.solo_redo.lock().clear();
    #[cfg(target_os = "macos")]
    macos_undo::clear_all(app);

    *state.selection.selection_cells.lock() = AHashSet::default();
    *state.selection.selection_stroke_before.lock() = None;
    *state.selection.selection_stroke_accum.lock() = None;
    *state.selection.selection_combine_mode.lock() = SelectionCombineMode::default();
    *state.selection.stamp_clipboard.lock() = None;
    *state.file.stroke_buffer.lock() = Vec::new();
    *state.file.stroke_preview_union.lock() = AHashSet::default();
    *state.file.stroke_preview_last_args.lock() = None;
    state
        .file
        .stroke_preview_suppresses_hover
        .store(false, Ordering::Release);
    *state.file.sculpt_stroke_replay.lock() = Vec::new();
    *state.file.stroke_active.lock() = false;
    *state.ping_flash.lock() = None;
    *state.preview.preview_cursor.lock() = None;

    state.gizmos.squishy_session.lock().clear();
    state.gizmos.bone_session.lock().clear();

    log::info!(target: "voxelle_load", "unload_current_project: done");
    #[cfg(desktop)]
    selection_menu_sync_enabled_for_scene(app, false, false, false);
    Ok(())
}

pub(crate) fn run_unload_on_main_thread<R: Runtime>(
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

pub(crate) fn load_thread_panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        return (*s).to_string();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    "load thread panicked (details on stderr)".to_string()
}

pub(crate) fn start_shape_label(shape: StartShape) -> &'static str {
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

pub(crate) fn spawn_new_project(
    state: Arc<ViewerState>,
    app: AppHandle,
    grid_size: u32,
    shape: StartShape,
) {
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
                .gpu
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

                        let mode = *state.gpu.rendering_mode.lock();
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
                                let res =
                                    apply_mesh_and_camera(&state_c, &app_mesh, file_c, prepared);
                                let _ = done_tx.send(res);
                            });
                            return match done_rx.recv() {
                                Ok(Ok(())) => Ok(()),
                                Ok(Err(e)) => Err(e),
                                Err(_) => Err("main thread disconnected".into()),
                            };
                        }

                        run_v3_mesh_on_main(&state, &app, file, prepared, load_gen)?;
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
                    emit_voxelle_loaded(&app, label.clone(), &state);
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

pub(crate) enum DecodeMeshOutcome {
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

pub(crate) fn run_v3_mesh_on_main(
    state: &Arc<ViewerState>,
    app: &AppHandle,
    file: voxelle::VoxelleFile,
    prepared: PreparedLoadScene,
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

/// Bump the load generation counter and return the new value.
/// Every load entry point should call this so that older in-flight loads can detect they are stale.
pub(crate) fn next_load_generation(state: &ViewerState) -> u64 {
    state.gpu.load_generation.fetch_add(1, Ordering::SeqCst) + 1
}

/// Returns true when a newer load has started since `gen` was issued.
pub(crate) fn is_load_stale(state: &ViewerState, gen: u64) -> bool {
    state.gpu.load_generation.load(Ordering::SeqCst) != gen
}

pub(crate) fn spawn_decode_and_mesh(state: Arc<ViewerState>, app: AppHandle, path: PathBuf) {
    let label = path.to_string_lossy().to_string();
    spawn_decode_and_mesh_with_label(state, app, path, label);
}

pub(crate) fn spawn_decode_and_mesh_with_label(
    state: Arc<ViewerState>,
    app: AppHandle,
    read_from: PathBuf,
    file_label: String,
) {
    // Bump the generation BEFORE spawning the read thread so that a slow disk read
    // that started earlier cannot "win" the generation race over a faster subsequent load.
    let load_gen = next_load_generation(&state);
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
                    spawn_decode_and_mesh_inner(state, app, bytes, file_label, load_gen);
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

pub(crate) fn spawn_decode_and_mesh_inner(
    state: Arc<ViewerState>,
    app: AppHandle,
    preloaded_bytes: Vec<u8>,
    file_label: String,
    load_gen: u64,
) {
    let app_spawn_err = app.clone();
    match std::thread::Builder::new()
        .name("voxelle-load".into())
        .spawn(move || {
            // Check before unloading: if a newer load has already started (e.g. a faster
            // re-open whose disk read completed while ours was still in flight), bail out
            // rather than wiping the scene the newer load already applied.
            if is_load_stale(&state, load_gen) {
                log::info!(target: "voxelle_load", "load cancelled (stale before unload)");
                return;
            }
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
                        let mode = *state.gpu.rendering_mode.lock();
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
                            run_v3_mesh_on_main(&state, &app, file, prepared, load_gen)?;
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
                        let r = apply_mesh_and_camera(&state_c, &app_emit, file, prepared);
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
                    if label.ends_with(".voxelle") {
                        persist_last_document_path(&app, &label);
                        persist_recent_file(&app, &label);
                        #[cfg(desktop)]
                        if let Some(rm) = app.try_state::<RecentMenuState>() {
                            rebuild_recent_submenu(&app, &rm.submenu);
                        }
                    }
                    emit_voxelle_loaded(&app, label, &state);
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
        let mut cf = state.file.current_file.lock();
        let mut vm = state.file.voxel_map.lock();
        *cf = Some(file);
        *vm = Some(voxel_map);
    }
    let mut v = state.gpu.viewer.lock();
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

    let mut cam = state.cam.camera.lock();
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
    // Hide the logo overlay now that a real project is loaded.
    if let Some(logo) = viewer.logo_overlay.as_mut() {
        logo.visible = false;
    }
    drop(v);
    *state.gpu.last_scene_bounds.lock() = Some(bounds);
    *state.gpu.voxel_edit_stats_cache.lock() = voxel_edit_stats_cache;
    state.file.solo_undo.lock().clear();
    state.file.solo_redo.lock().clear();
    #[cfg(target_os = "macos")]
    macos_undo::clear_all(app);
    collab::broadcast_snapshot_to_guests(state);
    state.cam.active_project.store(true, Ordering::Release);
    emit_load_progress(app, 0.97, "Finishing…");
    emit_load_progress(app, 1.0, "");
    #[cfg(desktop)]
    {
        let (has_project, has_voxels, has_selection) = scene_menu_flags(state.as_ref());
        selection_menu_sync_enabled_for_scene(app, has_project, has_voxels, has_selection);
    }
    Ok(())
}

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct VoxelleLoadedEvent {
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mood: Option<voxelle::MoodSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lighting: Option<voxelle::LightingSettings>,
}

pub(crate) fn emit_voxelle_loaded<R: Runtime>(
    app: &AppHandle<R>,
    path: String,
    state: &ViewerState,
) {
    state
        .gpu
        .start_screen_logo_transparent
        .store(false, Ordering::Release);
    let (mood, lighting) = match state.file.current_file.lock().as_ref() {
        Some(f) => (f.mood.clone(), f.lighting.clone()),
        None => (None, None),
    };
    let _ = app.emit(
        "voxelle-loaded",
        VoxelleLoadedEvent {
            path,
            mood,
            lighting,
        },
    );
}
