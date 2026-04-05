use crate::*;

// ── Default value functions for serde ────────────────────────────────────────

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
pub(crate) struct SculptStrokeAtScreenArgs {
    pub(crate) nx: f32,
    pub(crate) ny: f32,
    pub(crate) sculpt_mode: voxel_edit::SculptStrokeMode,
    pub(crate) color: u32,
    pub(crate) material: String,
    pub(crate) brush_radius: u32,
    pub(crate) brush_shape: voxel_edit::BrushShape,
    #[serde(default)]
    pub(crate) spray_density: f32,
    #[serde(default)]
    pub(crate) brush_clip_bottom_half: bool,
    #[serde(default)]
    pub(crate) stroke_line_start_nx: Option<f32>,
    #[serde(default)]
    pub(crate) stroke_line_start_ny: Option<f32>,
    #[serde(default)]
    pub(crate) stroke_segment_prev_nx: Option<f32>,
    #[serde(default)]
    pub(crate) stroke_segment_prev_ny: Option<f32>,
    #[serde(default)]
    pub(crate) terrain_op: Option<voxel_edit::TerrainSculptOp>,
    #[serde(default)]
    pub(crate) terrain_base_y: i32,
    #[serde(default = "default_terrain_strength_sculpt")]
    pub(crate) terrain_strength: i32,
    #[serde(default)]
    pub(crate) terrain_smooth_radius: i32,
    #[serde(default)]
    pub(crate) terrain_flatten_use_base_y: bool,
    #[serde(default)]
    pub(crate) terrain_sub_voxel: bool,
    #[serde(default = "default_smooth_passes_sculpt")]
    pub(crate) smooth_neighbor_passes: u32,
    #[serde(default = "default_brush_strength_sculpt")]
    pub(crate) brush_strength: u32,
    #[serde(default)]
    pub(crate) brush_falloff: u32,
    #[serde(default)]
    pub(crate) stroke_seed: u32,
    #[serde(default)]
    pub(crate) wall_area_shape: voxel_edit::WallAreaShape,
    #[serde(default)]
    pub(crate) spray_direction: voxel_edit::SprayDirection,
    #[serde(default)]
    pub(crate) wall_width_index: u32,
    #[serde(default = "default_wall_height_vox_sculpt")]
    pub(crate) wall_height_vox: u32,
    #[serde(default)]
    pub(crate) wall_lock_start_height: bool,
    #[serde(default)]
    pub(crate) wall_axis_align: bool,
    #[serde(default)]
    pub(crate) sculpt_smooth_variant: crate::sculpt_mesh_smooth::SculptSmoothVariant,
    #[serde(default)]
    pub(crate) smooth_neighbor_radius: u32,
    #[serde(default = "default_smooth_aggressiveness_sculpt")]
    pub(crate) smooth_aggressiveness: u32,
    #[serde(default = "default_laplacian_iterations_sculpt")]
    pub(crate) smooth_laplacian_iterations: u32,
    #[serde(default = "default_laplacian_relax_sculpt")]
    pub(crate) smooth_laplacian_relax_pct: u32,
    #[serde(default)]
    pub(crate) wall_polygon_vertices: Option<Vec<[i32; 3]>>,
    #[serde(default)]
    pub(crate) extrude_profile: voxel_edit::ExtrudeProfile,
    #[serde(default)]
    pub(crate) extrude_end_cap: voxel_edit::ExtrudeEndCap,
    #[serde(default)]
    pub(crate) extrude_taper: bool,
    #[serde(default)]
    pub(crate) extrude_taper_start: f32,
    #[serde(default)]
    pub(crate) extrude_taper_end: f32,
    /// Screen-space position used to sample the face normal for Draw mode.
    /// When set, the normal is locked to the initial click rather than the current cursor.
    #[serde(default)]
    pub(crate) draw_normal_nx: Option<f32>,
    #[serde(default)]
    pub(crate) draw_normal_ny: Option<f32>,
}

// ── Sculpt raise ─────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SculptRaiseArgs {
    nx: f32,
    ny: f32,
    color: u32,
    material: String,
}

#[tauri::command]
pub(crate) fn voxel_sculpt_raise_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: SculptRaiseArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let (w, h) = {
            let v = state.gpu.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.file.current_file.lock();
        let mut vm = state.file.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.cam.camera.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        voxel_edit::sculpt_raise_at_screen(file, vmap, &cam, w, h, sx, sy, args.color, material)?
    };
    commit_voxel_edits(&state, &app, deltas)
}

// ── Sculpt stroke ────────────────────────────────────────────────────────────

#[tauri::command]
pub(crate) fn voxel_sculpt_stroke_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: SculptStrokeAtScreenArgs,
) -> Result<bool, String> {
    let t_total = Instant::now();
    let t_apply_start = Instant::now();
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let (w, h) = {
            let v = state.gpu.viewer.lock();
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
        let draw_normal_pos = match (args.draw_normal_nx, args.draw_normal_ny) {
            (Some(dnx), Some(dny)) => Some(viewport_texels_from_norm(dnx, dny, w, h)),
            _ => None,
        };
        let mut fg = state.file.current_file.lock();
        let mut vm = state.file.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.cam.camera.lock();
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
            &mut state.file.terrain_accum.lock(),
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
            draw_normal_pos,
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
    let stroke_on = *state.file.stroke_active.lock();
    if stroke_on {
        state.file.stroke_buffer.lock().extend(deltas.iter().copied());
        return Ok(true);
    }
    let cm = Arc::clone(&state.collab);
    let mut cb = cm.lock();
    if cb.is_client() {
        if let Some(tx) = &cb.client_tx {
            let _ = tx.try_send(collab::ClientOutgoing::Binary(
                collab::encode_client_edit_binary(&deltas),
            ));
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

// ── Sculpt stroke replay helper ──────────────────────────────────────────────

pub(crate) fn commit_sculpt_stroke_replay(
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
                let v = state.gpu.viewer.lock();
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
            let draw_normal_pos = match (args.draw_normal_nx, args.draw_normal_ny) {
                (Some(dnx), Some(dny)) => Some(viewport_texels_from_norm(dnx, dny, w, h)),
                _ => None,
            };
            let mut fg = state.file.current_file.lock();
            let mut vm = state.file.voxel_map.lock();
            let Some(file) = fg.as_mut() else {
                return Ok(());
            };
            let Some(vmap) = vm.as_mut() else {
                return Ok(());
            };
            let cam = state.cam.camera.lock();
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
                draw_normal_pos,
            )?
        };
        all_deltas.extend(deltas);
    }
    if !all_deltas.is_empty() {
        commit_voxel_edits(state, app, all_deltas)?;
    }
    Ok(())
}

// ── Sculpt stroke preview ────────────────────────────────────────────────────

#[tauri::command]
pub(crate) fn voxel_sculpt_stroke_preview_at_screen(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: SculptStrokeAtScreenArgs,
) -> Result<(), String> {
    {
        let cm = state.collab.lock();
        if cm.is_client() && !cm.client_can_edit() {
            return Ok(());
        }
    }

    let stroke_line_start_meta = match (args.stroke_line_start_nx, args.stroke_line_start_ny) {
        (Some(_), Some(_)) => Some((0.0_f32, 0.0_f32)),
        _ => None,
    };

    state.file.sculpt_stroke_replay.lock().push(args.clone());

    let footprint = {
        let fg = state.file.current_file.lock();
        let vm = state.file.voxel_map.lock();
        let Some(file) = fg.as_ref() else {
            return Ok(());
        };
        let Some(vmap) = vm.as_ref() else {
            return Ok(());
        };
        let cam = state.cam.camera.lock();
        let (w, h) = {
            let v = state.gpu.viewer.lock();
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
        let draw_normal_pos = match (args.draw_normal_nx, args.draw_normal_ny) {
            (Some(dnx), Some(dny)) => Some(viewport_texels_from_norm(dnx, dny, w, h)),
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
            let stroke_on = *state.file.stroke_active.lock();
            let locked_face = if stroke_on {
                let mut lock = state.file.wall_stroke_face_snapped.lock();
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
                draw_normal_pos,
            )
        }
    };

    {
        let mut union = state.file.stroke_preview_union.lock();
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
        let fg = state.file.current_file.lock();
        let vm = state.file.voxel_map.lock();
        let union = state.file.stroke_preview_union.lock();
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
        let mut v = state.gpu.viewer.lock();
        let Some(viewer) = v.as_mut() else {
            return Ok(());
        };
        if instanced.solid_instances.is_empty() {
            clear_preview_mesh_sync_cache(viewer, state.inner().as_ref());
            state
                .file.stroke_preview_suppresses_hover
                .store(false, Ordering::Relaxed);
        } else {
            viewer.upload_preview_mesh_instanced(&instanced);
            viewer.preview_cache_key = None;
            *state.gpu.preview_overlay_cache_key.lock() = None;
            state
                .file.stroke_preview_suppresses_hover
                .store(true, Ordering::Relaxed);
        }
    }

    wake_viewport_loop(&app);
    Ok(())
}

// ── Extrude ray-based preview (straight-line extrude matching web) ───────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ExtrudeRayPreviewArgs {
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
pub(crate) fn extrude_ray_preview(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: ExtrudeRayPreviewArgs,
) -> Result<(), String> {
    {
        let cm = state.collab.lock();
        if cm.is_client() && !cm.client_can_edit() {
            return Ok(());
        }
    }

    let (w, h) = {
        let v = state.gpu.viewer.lock();
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
        let fg = state.file.current_file.lock();
        let vm = state.file.voxel_map.lock();
        let Some(file) = fg.as_ref() else {
            return Ok(());
        };
        let Some(vmap) = vm.as_ref() else {
            return Ok(());
        };
        let cam = state.cam.camera.lock();
        match voxel_edit::pick_extrude_start(file, vmap, &cam, w, h, start_sx, start_sy) {
            Some(v) => v,
            None => return Ok(()),
        }
    };

    // Resolve extrusion direction from screen drag + camera.
    let cam = state.cam.camera.lock();
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
    *state.gizmos.extrude_ray_spine.lock() = Some(spine);

    // Store a synthetic sculpt replay entry so voxel_stroke_end recognizes this as an extrude
    // and commits from the preview union.
    {
        let mut replay = state.file.sculpt_stroke_replay.lock();
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
                draw_normal_nx: None,
                draw_normal_ny: None,
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
        let mut union = state.file.stroke_preview_union.lock();
        union.clear();
        for c in &footprint {
            union.insert(*c);
        }
    }

    state
        .file.stroke_preview_suppresses_hover
        .store(true, Ordering::Relaxed);

    // Generate and upload preview mesh.
    let instanced = {
        let fg = state.file.current_file.lock();
        let vm = state.file.voxel_map.lock();
        let Some(file) = fg.as_ref() else {
            return Ok(());
        };
        let Some(vmap) = vm.as_ref() else {
            return Ok(());
        };
        let union = state.file.stroke_preview_union.lock();
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
        let mut v = state.gpu.viewer.lock();
        let Some(viewer) = v.as_mut() else {
            return Ok(());
        };
        if instanced.solid_instances.is_empty() {
            clear_preview_mesh_sync_cache(viewer, state.inner().as_ref());
        } else {
            viewer.upload_preview_mesh_instanced(&instanced);
            viewer.preview_cache_key = None;
            *state.gpu.preview_overlay_cache_key.lock() = None;
        }
    }

    wake_viewport_loop(&app);
    Ok(())
}

// ── Selection extrude preview ────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SelectionExtrudePreviewArgs {
    screen_dx: f32,
    screen_dy: f32,
    direction_ref: voxel_edit::ExtrudeDirectionRef,
    color: u32,
    material: String,
}

#[tauri::command]
pub(crate) fn selection_extrude_preview(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: SelectionExtrudePreviewArgs,
) -> Result<(), String> {
    {
        let cm = state.collab.lock();
        if cm.is_client() && !cm.client_can_edit() {
            return Ok(());
        }
    }

    let selection: ahash::AHashSet<greedy_mesh::VoxelCoord> = state.selection.selection_cells.lock().clone();
    if selection.is_empty() {
        return Ok(());
    }

    let (w, h) = {
        let v = state.gpu.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Ok(());
        };
        viewer.viewport_size()
    };
    let w = w as f32;
    let h = h as f32;
    let _ = (w, h); // viewport size not needed for direction resolution

    let direction = {
        let cam = state.cam.camera.lock();
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
        let mut union = state.file.stroke_preview_union.lock();
        union.clear();
        for c in &footprint {
            union.insert(*c);
        }
    }

    // Store a synthetic sculpt replay entry so voxel_stroke_end knows to commit from the union.
    {
        let mut replay = state.file.sculpt_stroke_replay.lock();
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
                draw_normal_nx: None,
                draw_normal_ny: None,
            });
        } else {
            let entry = &mut replay[0];
            entry.color = args.color;
            entry.material = args.material.clone();
        }
    }

    state
        .file.stroke_preview_suppresses_hover
        .store(true, Ordering::Relaxed);

    // Generate and upload preview mesh.
    let instanced = {
        let fg = state.file.current_file.lock();
        let vm = state.file.voxel_map.lock();
        let Some(file) = fg.as_ref() else {
            return Ok(());
        };
        let Some(vmap) = vm.as_ref() else {
            return Ok(());
        };
        let union = state.file.stroke_preview_union.lock();
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
        let mut v = state.gpu.viewer.lock();
        let Some(viewer) = v.as_mut() else {
            return Ok(());
        };
        if instanced.solid_instances.is_empty() {
            clear_preview_mesh_sync_cache(viewer, state.inner().as_ref());
        } else {
            viewer.upload_preview_mesh_instanced(&instanced);
            viewer.preview_cache_key = None;
            *state.gpu.preview_overlay_cache_key.lock() = None;
        }
    }

    wake_viewport_loop(&app);
    Ok(())
}

// ── Extrude phase: recompute preview with new settings ──────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ExtrudeRecomputeArgs {
    extrude_profile: voxel_edit::ExtrudeProfile,
    extrude_end_cap: voxel_edit::ExtrudeEndCap,
    extrude_taper: bool,
    #[serde(default)]
    extrude_taper_start: f32,
    #[serde(default)]
    extrude_taper_end: f32,
}

#[tauri::command]
pub(crate) fn extrude_recompute_preview(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: ExtrudeRecomputeArgs,
) -> Result<(), String> {
    // Use stored ray spine if available (new ray-based extrude path).
    let spine_opt = state.gizmos.extrude_ray_spine.lock().clone();
    let replay = state.file.sculpt_stroke_replay.lock().clone();
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
        let fg = state.file.current_file.lock();
        let vm = state.file.voxel_map.lock();
        let Some(file) = fg.as_ref() else {
            return Ok(());
        };
        let Some(vmap) = vm.as_ref() else {
            return Ok(());
        };
        // Acquire viewer before camera to match the render loop's lock order
        // (viewer → camera). Inverting this order deadlocks with the render tick.
        let (w, h) = {
            let v = state.gpu.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Ok(());
            };
            viewer.viewport_size()
        };
        let cam = state.cam.camera.lock();
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
            let draw_normal_pos = match (sample.draw_normal_nx, sample.draw_normal_ny) {
                (Some(dnx), Some(dny)) => Some(viewport_texels_from_norm(dnx, dny, w, h)),
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
                draw_normal_pos,
            );
            for c in footprint {
                union.insert(c);
            }
        }
        union
    };

    // Update stored replay args with new extrude settings so commit uses them.
    {
        let mut replay_mut = state.file.sculpt_stroke_replay.lock();
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
        let mut preview_union = state.file.stroke_preview_union.lock();
        preview_union.clear();
        for c in &union {
            preview_union.insert(*c);
        }
    }

    let instanced = {
        let fg = state.file.current_file.lock();
        let vm = state.file.voxel_map.lock();
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
        let mut v = state.gpu.viewer.lock();
        let Some(viewer) = v.as_mut() else {
            return Ok(());
        };
        if instanced.solid_instances.is_empty() {
            clear_preview_mesh_sync_cache(viewer, state.inner().as_ref());
        } else {
            viewer.upload_preview_mesh_instanced(&instanced);
            viewer.preview_cache_key = None;
            *state.gpu.preview_overlay_cache_key.lock() = None;
        }
    }

    wake_viewport_loop(&app);
    Ok(())
}
