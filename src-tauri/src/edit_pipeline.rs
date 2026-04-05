//! Voxel edit pipeline — GPU upload + mesh rebuild helpers extracted from `lib.rs`.

use crate::*;

/// Show status progress for voxel GPU refresh when the scene is large or a full mesh rebuild is required.
pub(crate) fn work_progress_for_voxel_refresh(
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

pub(crate) fn scene_bounds_for_edit(
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
        let guard = state.gpu.last_scene_bounds.lock();
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
        let fg = state.file.current_file.lock();
        let Some(file) = fg.as_ref() else {
            return Err("no model loaded".into());
        };
        scene_bounds_for_edits(state.as_ref(), file, deltas)?
    };

    let prepare_ms = t_prep_start.elapsed().as_secs_f64() * 1000.0;

    let t_lock_start = Instant::now();
    let mut v = state.gpu.viewer.lock();
    let viewer_lock_wait_ms = t_lock_start.elapsed().as_secs_f64() * 1000.0;

    let mut fg = state.file.current_file.lock();
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };

    let rm = *state.gpu.rendering_mode.lock();
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
    v = state.gpu.viewer.lock();
    fg = state.file.current_file.lock();
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
        *state.gpu.voxel_edit_stats_cache.lock() = None;
    } else if rm.uses_smooth_surface() {
        let nv = file.voxels.len();
        if nv >= OFF_THREAD_SMOOTH_MESH_MIN_VOXELS {
            let voxels = file.voxels.clone();
            let rm_copy = rm;
            let token = state.gpu.mesh_refresh_generation.fetch_add(1, Ordering::SeqCst) + 1;
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
            v = state.gpu.viewer.lock();
            let Some(viewer) = v.as_mut() else {
                return Err("viewer not ready".into());
            };
            fg = state.file.current_file.lock();
            let Some(file) = fg.as_ref() else {
                return Err("no model loaded".into());
            };
            let mut mesh = if state.gpu.mesh_refresh_generation.load(Ordering::SeqCst) == token {
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
            *state.gpu.voxel_edit_stats_cache.lock() =
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
            *state.gpu.voxel_edit_stats_cache.lock() =
                Some(voxel_aabb_min_and_single_object_one_pass(&file.voxels));
        }
    } else {
        let cached_stats = *state.gpu.voxel_edit_stats_cache.lock();
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
                    let mut v2 = state.gpu.viewer.lock();
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
                v = state.gpu.viewer.lock();
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
                let token = state.gpu.mesh_refresh_generation.fetch_add(1, Ordering::SeqCst) + 1;
                drop(fg);
                drop(v);
                let prepared_result =
                    off_thread_prepare_greedy_rebuild(app, grid_size, voxels, objects);
                v = state.gpu.viewer.lock();
                let Some(viewer) = v.as_mut() else {
                    return Err("viewer not ready".into());
                };
                fg = state.file.current_file.lock();
                let Some(file) = fg.as_ref() else {
                    return Err("no model loaded".into());
                };
                let t_pipe = Instant::now();
                match prepared_result {
                    Ok(prepared) => {
                        if state.gpu.mesh_refresh_generation.load(Ordering::SeqCst) != token {
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
        *state.gpu.voxel_edit_stats_cache.lock() = Some(voxel_stats);
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
    *state.gpu.last_edit_perf.lock() = Some(EditPerfBreakdown {
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

    *state.gpu.last_scene_bounds.lock() = Some(bounds);

    drop(v);

    #[cfg(desktop)]
    {
        let (has_project, has_voxels, has_selection) = scene_menu_flags(state.as_ref());
        selection_menu_sync_enabled_for_scene(app, has_project, has_voxels, has_selection);
    }
    Ok(())
}

/// Rebuild opaque mesh from current voxels + [`RenderingMode`] (after switching view mode in the UI).
pub(crate) fn refresh_opaque_mesh<R: Runtime>(
    state: &Arc<ViewerState>,
    app: Option<&AppHandle<R>>,
) -> Result<(), String> {
    let rm = *state.gpu.rendering_mode.lock();
    let mut v = state.gpu.viewer.lock();
    let Some(viewer) = v.as_mut() else {
        return Err("viewer not ready".into());
    };
    let fg = state.file.current_file.lock();
    let Some(file) = fg.as_ref() else {
        drop(fg);
        drop(v);
        #[cfg(desktop)]
        if let Some(a) = app {
            let (has_project, has_voxels, has_selection) = scene_menu_flags(state.as_ref());
            selection_menu_sync_enabled_for_scene(a, has_project, has_voxels, has_selection);
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
        *state.gpu.voxel_edit_stats_cache.lock() = None;
        drop(wp);
        drop(fg);
        drop(v);
        #[cfg(desktop)]
        if let Some(a) = app {
            let (has_project, has_voxels, has_selection) = scene_menu_flags(state.as_ref());
            selection_menu_sync_enabled_for_scene(a, has_project, has_voxels, has_selection);
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
                *state.gpu.last_scene_bounds.lock() = Some(b);
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
                *state.gpu.last_scene_bounds.lock() = Some(b);
            }
        }
    }
    *state.gpu.voxel_edit_stats_cache.lock() =
        Some(voxel_aabb_min_and_single_object_one_pass(&file.voxels));
    drop(wp);
    drop(fg);
    drop(v);
    #[cfg(desktop)]
    if let Some(a) = app {
        let (has_project, has_voxels, has_selection) = scene_menu_flags(state.as_ref());
        selection_menu_sync_enabled_for_scene(a, has_project, has_voxels, has_selection);
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
pub(crate) fn schedule_opaque_mesh_refresh(state: &Arc<ViewerState>, app: &AppHandle) {
    let token = state.gpu.mesh_refresh_generation.fetch_add(1, Ordering::SeqCst) + 1;
    let state_c = Arc::clone(state);
    let app = app.clone();
    let file = (*state.file.current_file.lock()).clone();
    let Some(file) = file else {
        return;
    };
    let rm = *state.gpu.rendering_mode.lock();
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
                    move || state_check.gpu.mesh_refresh_generation.load(Ordering::Relaxed) != token
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
                if state_c.gpu.mesh_refresh_generation.load(Ordering::SeqCst) != token {
                    return;
                }
                let mut vl = Some(state_c.gpu.viewer.lock());
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
                        *state_c.gpu.last_scene_bounds.lock() = Some(bounds);
                    }
                    OpaqueRefreshWork::Greedy(prepared) => {
                        match viewer.apply_prepared_greedy_rebuild(prepared) {
                            Ok(b) => {
                                viewer.set_scene_bounds(b);
                                *state_c.gpu.last_scene_bounds.lock() = Some(b);
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
                                *state_c.gpu.last_scene_bounds.lock() = Some(b);
                            }
                        }
                    }
                }
                if file_snapshot.voxels.is_empty() {
                    *state_c.gpu.voxel_edit_stats_cache.lock() = None;
                } else {
                    *state_c.gpu.voxel_edit_stats_cache.lock() = Some(
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

/// Wake the winit/tauri main loop so the next `MainEventsCleared` runs (projection / mesh / preview refresh).
pub(crate) fn wake_viewport_loop(app: &AppHandle) {
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let _ = app.run_on_main_thread(|| {});
    });
}
