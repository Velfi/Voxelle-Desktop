use crate::*;

// ── Arg structs ──────────────────────────────────────────────────────────────

#[derive(serde::Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SelectionStrokeAtScreen {
    pub(crate) nx: f32,
    pub(crate) ny: f32,
    pub(crate) brush_radius: u32,
    pub(crate) brush_shape: voxel_edit::BrushShape,
    #[serde(default)]
    pub(crate) spray_density: f32,
    #[serde(default)]
    pub(crate) stroke_line_start_nx: Option<f32>,
    #[serde(default)]
    pub(crate) stroke_line_start_ny: Option<f32>,
    #[serde(default)]
    pub(crate) stroke_segment_prev_nx: Option<f32>,
    #[serde(default)]
    pub(crate) stroke_segment_prev_ny: Option<f32>,
    #[serde(default)]
    pub(crate) stroke_mode: stroke_modes::DrawStrokeMode,
    #[serde(default)]
    pub(crate) plane_axis: stroke_modes::PlaneAxis,
    #[serde(default)]
    pub(crate) stroke_aux: stroke_modes::StrokeAux,
    #[serde(default)]
    pub(crate) fill_select_diagonals: bool,
    #[serde(default = "default_fill_respects_color")]
    pub(crate) fill_respects_color: bool,
    #[serde(default)]
    pub(crate) match_material: bool,
    /// `select` | `selectByColor` | `selectCoplanar` | `selectCoplanarEmpty`
    #[serde(default)]
    pub(crate) interaction: String,
    /// When set, overrides the global selection_combine_mode for this stroke (e.g. shift-key → add).
    #[serde(default)]
    pub(crate) combine_mode_override: Option<SelectionCombineMode>,
    /// When `true`, skip the large-fill confirmation gate (user already confirmed).
    #[serde(default)]
    pub(crate) confirmed: bool,
}

fn default_fill_respects_color() -> bool {
    true
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PaintSelectionArgs {
    color: u32,
    #[serde(default)]
    palette: Vec<u32>,
    #[serde(default)]
    paint_color_distrib: Option<paint_color_distrib::PaintColorDistrib>,
    #[serde(default)]
    stroke_seed: u32,
    material: String,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SelectByColorArgs {
    nx: f32,
    ny: f32,
    match_material: bool,
    #[serde(default)]
    combine_mode_override: Option<SelectionCombineMode>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SelectCoplanarArgs {
    nx: f32,
    ny: f32,
    #[serde(default)]
    combine_mode_override: Option<SelectionCombineMode>,
}

// ── Selection helpers ────────────────────────────────────────────────────────

pub(crate) fn merge_coords_into_selection(
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
pub(crate) fn apply_selection_stroke_sample(
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

pub(crate) fn emit_selection_updated<R: Runtime>(app: &AppHandle<R>, state: &Arc<ViewerState>) {
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

// ── Selection gizmo projection ────────────────────────────────────────────────

#[derive(serde::Serialize, Clone, Copy)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GizmoProj {
    sx: f32,
    sy: f32,
    in_front: bool,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SelectionGizmoProjected {
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


/// Compute gizmo projected positions. Shared by `get_selection_gizmo_projected`,
/// `gizmo_pointer_down`, and `gizmo_hit_test`.
fn compute_gizmo_proj(state: &ViewerState) -> Option<SelectionGizmoProjected> {
    let gen_center = *state.generator_gizmo_center.lock();
    let (cx, cy, cz) = if let Some([gx, gy, gz]) = gen_center {
        (gx, gy, gz)
    } else {
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
            if x < min_x { min_x = x; }
            if x > max_x { max_x = x; }
            if y < min_y { min_y = y; }
            if y > max_y { max_y = y; }
            if z < min_z { min_z = z; }
            if z > max_z { max_z = z; }
        }
        drop(sel);
        (
            (min_x + max_x) as f32 * 0.5,
            (min_y + max_y) as f32 * 0.5,
            (min_z + max_z) as f32 * 0.5,
        )
    };
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

// ── Selection transform helpers ─────────────────────────────────────────────

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
    {
        let mut stack = state.solo_undo.lock();
        stack.push(SoloUndoEntry::SelectionTransform {
            before: before_sel,
            deltas,
        });
        crate::commands::edit::enforce_solo_undo_cap(&mut stack);
    }
    state.solo_redo.lock().clear();
    #[cfg(target_os = "macos")]
    macos_undo::register_solo_edit_completed(app, state);
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
        if large && !args.confirmed {
            return Err("confirm_large_fill".to_string());
        }
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
        if progress_ticks.is_multiple_of(4) {
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

// ── Extrude gizmo preview helper ─────────────────────────────────────────────

/// Shared logic: update stroke_preview_union + preview mesh for axis-locked extrude.
fn extrude_gizmo_preview_inner(
    state: &Arc<ViewerState>,
    app: &AppHandle,
    world_axis: u8,
    depth: i32,
    color: u32,
    material: &str,
) -> Result<(), String> {
    let selection: ahash::AHashSet<greedy_mesh::VoxelCoord> =
        state.selection_cells.lock().clone();
    if selection.is_empty() {
        return Ok(());
    }
    if depth == 0 {
        state.stroke_preview_union.lock().clear();
        let mut v = state.viewer.lock();
        if let Some(viewer) = v.as_mut() {
            clear_preview_mesh_sync_cache(viewer, state.as_ref());
        }
        return Ok(());
    }
    let dir_vec = match world_axis {
        0 => glam::Vec3::X,
        1 => glam::Vec3::Y,
        _ => glam::Vec3::Z,
    };
    let direction = if depth > 0 { dir_vec } else { -dir_vec };
    let length = depth.unsigned_abs();
    let footprint = voxel_edit::extrude_selection_footprint(&selection, direction, length);
    {
        let mut union = state.stroke_preview_union.lock();
        union.clear();
        for c in &footprint {
            union.insert(*c);
        }
    }
    {
        let mut replay = state.sculpt_stroke_replay.lock();
        if replay.is_empty() {
            replay.push(SculptStrokeAtScreenArgs {
                nx: 0.5,
                ny: 0.5,
                sculpt_mode: voxel_edit::SculptStrokeMode::Extrude,
                color,
                material: material.to_string(),
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
            entry.color = color;
            entry.material = material.to_string();
        }
    }
    state
        .stroke_preview_suppresses_hover
        .store(true, Ordering::Relaxed);
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
            clear_preview_mesh_sync_cache(viewer, state.as_ref());
        } else {
            viewer.upload_preview_mesh_instanced(&instanced);
            viewer.preview_cache_key = None;
            *state.preview_overlay_cache_key.lock() = None;
        }
    }
    wake_viewport_loop(app);
    Ok(())
}

// ── Tauri commands ───────────────────────────────────────────────────────────

#[tauri::command]
pub(crate) fn selection_stroke_begin(state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
    let snap = state.selection_cells.lock().clone();
    *state.selection_stroke_before.lock() = Some(snap);
    *state.selection_stroke_accum.lock() = Some(AHashSet::new());
    Ok(())
}

#[tauri::command]
pub(crate) fn selection_stroke_end(app: AppHandle, state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
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
pub(crate) async fn selection_stroke_at_screen(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: SelectionStrokeAtScreen,
) -> Result<u32, String> {
    {
        let cm = state.collab.lock();
        if cm.is_client() && !cm.client_can_edit() {
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
                    let coords_inner = res.inspect_err(|_e| {
                        emit_work_progress(&app, 1.0, "");
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

    let mode = args
        .combine_mode_override
        .unwrap_or_else(|| *state.selection_combine_mode.lock());
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
pub(crate) fn selection_toggle_at_screen(
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

#[tauri::command]
pub(crate) fn gizmo_pointer_down(state: State<'_, Arc<ViewerState>>, sx: f32, sy: f32, dpr: f32) -> bool {
    let Some(proj) = compute_gizmo_proj(&state) else {
        return false;
    };
    // Check scale ring first (generator_gizmo_ring_radius).
    if let Some(radius) = *state.generator_gizmo_ring_radius.lock() {
        let ring_hit = GIZMO_RING_HIT_CSS * dpr;
        let dx = sx - proj.center_sx;
        let dy = sy - proj.center_sy;
        let cursor_dist = dx.hypot(dy);
        // The ring is at `radius` world units; project to screen pixels.
        let ring_screen_r = radius * proj.px_per_world;
        if (cursor_dist - ring_screen_r).abs() <= ring_hit {
            *state.selection_gizmo_drag.lock() = SelectionGizmoDrag::Scale {
                center_sx: proj.center_sx,
                center_sy: proj.center_sy,
                start_dist: cursor_dist,
                start_radius: radius,
            };
            return true;
        }
    }
    let Some(drag) = gizmo_hit_test_inner(&proj, sx, sy, dpr) else {
        return false;
    };
    *state.selection_gizmo_drag.lock() = drag;
    true
}

#[tauri::command]
pub(crate) fn gizmo_pointer_move(
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
                // When the generator gizmo is active (shape settings phase), also
                // invalidate the preview overlay cache so the shape preview mesh
                // rebuilds at the new offset, and wake the frame loop.
                if state.generator_gizmo_center.lock().is_some() {
                    *state.preview_overlay_cache_key.lock() = None;
                    crate::edit_pipeline::wake_viewport_loop(&app);
                }
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
            // When the generator gizmo is active (shape settings phase),
            // emit a rotation event for the frontend instead of rotating
            // the selection. Each step = 15° of shape rotation.
            if state.generator_gizmo_center.lock().is_some() {
                let degrees = steps * 15;
                let _ = app.emit("generator-gizmo-rotated", (ring, degrees));
                *state.preview_overlay_cache_key.lock() = None;
                crate::edit_pipeline::wake_viewport_loop(&app);
                return Ok(());
            }
            selection_rotate_inner(state.inner(), &app, ring, steps)?;
            Ok(())
        }
        SelectionGizmoDrag::Scale { center_sx, center_sy, start_dist, mut start_radius } => {
            // Map horizontal pixel drag to radius change.
            // px_per_world converts world units to screen pixels.
            let proj = compute_gizmo_proj(&state);
            let ppw = proj.map(|p| p.px_per_world).unwrap_or(10.0);
            let delta_world = dcx / ppw.max(0.1);
            start_radius = (start_radius + delta_world).clamp(0.5, 64.0);
            // Update the ring radius state and emit event.
            *state.generator_gizmo_ring_radius.lock() = Some(start_radius);
            let _ = app.emit("generator-gizmo-scaled", start_radius);
            *state.preview_overlay_cache_key.lock() = None;
            crate::edit_pipeline::wake_viewport_loop(&app);
            *state.selection_gizmo_drag.lock() = SelectionGizmoDrag::Scale {
                center_sx, center_sy, start_dist, start_radius,
            };
            Ok(())
        }
    }
}

#[tauri::command]
pub(crate) fn gizmo_pointer_up(state: State<'_, Arc<ViewerState>>, app: AppHandle) -> Result<(), String> {
    let drag = state.selection_gizmo_drag.lock().clone();
    // Clear drag state before the translate so the overlay fingerprint (which reads pending)
    // won't double-apply the offset after selection_cells is updated.
    *state.selection_gizmo_drag.lock() = SelectionGizmoDrag::None;
    *state.selection_overlay_cache_key.lock() = None;
    match drag {
        SelectionGizmoDrag::Move {
            pending_dx,
            pending_dy,
            pending_dz,
            ..
        } => {
            if pending_dx != 0 || pending_dy != 0 || pending_dz != 0 {
                // If generator gizmo override is active, update the override center
                // and emit an event instead of translating the selection.
                let mut gen = state.generator_gizmo_center.lock();
                if let Some(ref mut center) = *gen {
                    center[0] += pending_dx as f32;
                    center[1] += pending_dy as f32;
                    center[2] += pending_dz as f32;
                    let _ = app.emit("generator-gizmo-moved", [center[0], center[1], center[2]]);
                    return Ok(());
                }
                drop(gen);
                selection_translate_inner(state.inner(), &app, pending_dx, pending_dy, pending_dz)?;
            }
        }
        SelectionGizmoDrag::Rotate { .. } => {
            // When generator gizmo is active, rotation was applied incrementally
            // during the drag. Just refresh the preview to ensure final state is shown.
            if state.generator_gizmo_center.lock().is_some() {
                *state.preview_overlay_cache_key.lock() = None;
                crate::edit_pipeline::wake_viewport_loop(&app);
            }
        }
        SelectionGizmoDrag::Scale { .. } => {
            // Scale was applied incrementally during drag via events.
        }
        SelectionGizmoDrag::None => {}
    }
    Ok(())
}

#[tauri::command]
pub(crate) fn gizmo_hit_test(state: State<'_, Arc<ViewerState>>, sx: f32, sy: f32, dpr: f32) -> bool {
    let Some(proj) = compute_gizmo_proj(&state) else {
        state.hovered_gizmo_axis.store(255, Ordering::Relaxed);
        return false;
    };
    // Check scale ring hover.
    if let Some(radius) = *state.generator_gizmo_ring_radius.lock() {
        let ring_hit = GIZMO_RING_HIT_CSS * dpr;
        let dx = sx - proj.center_sx;
        let dy = sy - proj.center_sy;
        let cursor_dist = dx.hypot(dy);
        let ring_screen_r = radius * proj.px_per_world;
        if (cursor_dist - ring_screen_r).abs() <= ring_hit {
            state.hovered_gizmo_axis.store(255, Ordering::Relaxed);
            return true;
        }
    }
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
        Some(SelectionGizmoDrag::Scale { .. }) => {
            state.hovered_gizmo_axis.store(255, Ordering::Relaxed);
            true
        }
        Some(SelectionGizmoDrag::None) | None => {
            state.hovered_gizmo_axis.store(255, Ordering::Relaxed);
            false
        }
    }
}

#[tauri::command]
pub(crate) fn get_selection_gizmo_projected(
    state: State<'_, Arc<ViewerState>>,
) -> Option<SelectionGizmoProjected> {
    compute_gizmo_proj(&state)
}

#[tauri::command]
pub(crate) fn extrude_gizmo_pointer_down(
    state: State<'_, Arc<ViewerState>>,
    sx: f32,
    sy: f32,
    dpr: f32,
) -> bool {
    let Some(proj) = compute_gizmo_proj(&state) else {
        return false;
    };
    let move_hit_sq = (GIZMO_MOVE_HIT_CSS * dpr).powi(2);
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
            let base = *state.extrude_gizmo_base_depth.lock();
            *state.extrude_gizmo_drag.lock() = ExtrudeGizmoDrag::Drag {
                axis_sx,
                axis_sy,
                world_axis: (i / 2) as u8,
                positive: i % 2 == 0,
                accum: 0.0,
                step_threshold: GIZMO_PX_PER_MOVE_STEP_CSS * dpr,
                depth: base,
            };
            return true;
        }
    }
    false
}

#[tauri::command]
pub(crate) fn extrude_gizmo_pointer_move(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    dcx: f32,
    dcy: f32,
    color: u32,
    material: String,
) -> Result<(), String> {
    let drag = state.extrude_gizmo_drag.lock().clone();
    let ExtrudeGizmoDrag::Drag {
        axis_sx,
        axis_sy,
        world_axis,
        positive,
        mut accum,
        step_threshold,
        mut depth,
    } = drag
    else {
        return Ok(());
    };
    accum += dcx * axis_sx + dcy * axis_sy;
    let steps = (accum / step_threshold).trunc() as i32;
    accum -= steps as f32 * step_threshold;
    let magnitude = if positive { steps } else { -steps };
    depth += magnitude;
    *state.extrude_gizmo_drag.lock() = ExtrudeGizmoDrag::Drag {
        axis_sx,
        axis_sy,
        world_axis,
        positive,
        accum,
        step_threshold,
        depth,
    };
    if magnitude != 0 {
        extrude_gizmo_preview_inner(state.inner(), &app, world_axis, depth, color, &material)?;
    }
    Ok(())
}

#[tauri::command]
pub(crate) fn extrude_gizmo_pointer_up(state: State<'_, Arc<ViewerState>>) {
    let drag = std::mem::take(&mut *state.extrude_gizmo_drag.lock());
    if let ExtrudeGizmoDrag::Drag { depth, .. } = drag {
        *state.extrude_gizmo_base_depth.lock() = depth;
    }
}

#[tauri::command]
pub(crate) fn extrude_gizmo_hit_test(
    state: State<'_, Arc<ViewerState>>,
    sx: f32,
    sy: f32,
    dpr: f32,
) -> bool {
    let Some(proj) = compute_gizmo_proj(&state) else {
        state.hovered_extrude_axis.store(255, Ordering::Relaxed);
        return false;
    };
    let move_hit_sq = (GIZMO_MOVE_HIT_CSS * dpr).powi(2);
    for (i, h) in proj.move_handles.iter().enumerate() {
        if (sx - h.sx).powi(2) + (sy - h.sy).powi(2) <= move_hit_sq {
            state
                .hovered_extrude_axis
                .store((i / 2) as u8, Ordering::Relaxed);
            return true;
        }
    }
    state.hovered_extrude_axis.store(255, Ordering::Relaxed);
    false
}

#[tauri::command]
pub(crate) fn selection_mirror(
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

#[tauri::command]
pub(crate) fn selection_translate(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    dx: i32,
    dy: i32,
    dz: i32,
) -> Result<bool, String> {
    selection_translate_inner(state.inner(), &app, dx, dy, dz)
}

#[tauri::command]
pub(crate) fn selection_rotate(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    axis: u8,
    quarters: i32,
) -> Result<bool, String> {
    selection_rotate_inner(state.inner(), &app, axis, quarters)
}

#[tauri::command]
pub(crate) fn selection_scale(
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
pub(crate) fn selection_clear(app: AppHandle, state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
    state.selection_cells.lock().clear();
    emit_selection_updated(&app, state.inner());
    Ok(())
}

#[tauri::command]
pub(crate) fn selection_delete_selected_voxels(
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
            let _ = tx.try_send(collab::ClientOutgoing::Binary(collab::encode_client_edit_binary(&deltas)));
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
pub(crate) fn selection_get_count(state: State<'_, Arc<ViewerState>>) -> Result<u32, String> {
    Ok(state.selection_cells.lock().len() as u32)
}

#[tauri::command]
pub(crate) fn paint_selection(
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

#[tauri::command]
pub(crate) fn selection_add_by_color_at_screen(
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
    let mode = args
        .combine_mode_override
        .unwrap_or_else(|| *state.selection_combine_mode.lock());
    let mut sel = state.selection_cells.lock();
    merge_coords_into_selection(&mut sel, coords, mode);
    let n = sel.len() as u32;
    drop(sel);
    emit_selection_updated(&app, state.inner());
    Ok(n)
}

#[tauri::command]
pub(crate) fn selection_add_coplanar_at_screen(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: SelectCoplanarArgs,
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
    let mode = args
        .combine_mode_override
        .unwrap_or_else(|| *state.selection_combine_mode.lock());
    let mut sel = state.selection_cells.lock();
    merge_coords_into_selection(&mut sel, coords, mode);
    let n = sel.len() as u32;
    drop(sel);
    emit_selection_updated(&app, state.inner());
    Ok(n)
}

#[tauri::command]
pub(crate) fn selection_add_coplanar_empty_at_screen(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: SelectCoplanarArgs,
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
    let mode = args
        .combine_mode_override
        .unwrap_or_else(|| *state.selection_combine_mode.lock());
    let mut sel = state.selection_cells.lock();
    merge_coords_into_selection(&mut sel, coords, mode);
    let n = sel.len() as u32;
    drop(sel);
    emit_selection_updated(&app, state.inner());
    Ok(n)
}

#[tauri::command]
pub(crate) fn selection_select_all(app: AppHandle, state: State<'_, Arc<ViewerState>>) -> Result<u32, String> {
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
pub(crate) fn selection_invert(app: AppHandle, state: State<'_, Arc<ViewerState>>) -> Result<u32, String> {
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
pub(crate) fn selection_grow(app: AppHandle, state: State<'_, Arc<ViewerState>>) -> Result<u32, String> {
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
pub(crate) fn selection_shrink(app: AppHandle, state: State<'_, Arc<ViewerState>>) -> Result<u32, String> {
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
pub(crate) fn selection_deselect_inner_voxels(
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
pub(crate) fn selection_retain_empty_only(
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
pub(crate) fn selection_retain_solid_only(
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

#[tauri::command]
pub(crate) fn selection_add_connected_at_screen(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: PickAtScreen,
) -> Result<u32, String> {
    let n = run_selection_add_connected(state.inner(), args)?;
    emit_selection_updated(&app, state.inner());
    Ok(n)
}

#[tauri::command]
pub(crate) fn selection_add_connected_at_cursor(
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
pub(crate) fn selection_set_combine_mode(
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
pub(crate) fn get_selection_combine_mode(
    state: State<'_, Arc<ViewerState>>,
) -> Result<SelectionCombineMode, String> {
    Ok(*state.selection_combine_mode.lock())
}
