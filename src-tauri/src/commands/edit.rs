use crate::*;

// ── Arg structs ──���──────────────────────────────────────────────────────────

#[derive(serde::Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct VoxelEditAtScreen {
    pub(crate) nx: f32,
    pub(crate) ny: f32,
    pub(crate) tool: voxel_edit::EditTool,
    pub(crate) color: u32,
    /// Multi-color palette (when non-empty and len > 1, overrides `color`).
    #[serde(default)]
    pub(crate) palette: Vec<u32>,
    /// Color distribution mode + params; used only when `palette.len() > 1`.
    #[serde(default)]
    pub(crate) paint_color_distrib: Option<paint_color_distrib::PaintColorDistrib>,
    /// Deterministic seed for the current stroke (for randomSingle / preview consistency).
    #[serde(default)]
    pub(crate) stroke_seed: u32,
    pub(crate) material: String,
    pub(crate) brush_radius: u32,
    pub(crate) brush_shape: voxel_edit::BrushShape,
    /// 0 = full brush; (0,1] = deterministic spray thinning.
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
    /// When `stroke_mode` is fill + paint: match material as well as color.
    #[serde(default)]
    pub(crate) match_material: bool,
    /// Solid/empty flood adjacency (fill stroke); mirrors selection fill.
    #[serde(default)]
    pub(crate) fill_select_diagonals: bool,
    #[serde(default = "default_fill_respects_color")]
    pub(crate) fill_respects_color: bool,
    /// Symmetry bitmask: bit 0 = X, bit 1 = Y, bit 2 = Z.  0 = no mirroring.
    #[serde(default)]
    pub(crate) mirror_axes: u8,
    /// When `true`, skip the large-fill confirmation gate (user already confirmed).
    #[serde(default)]
    pub(crate) confirmed: bool,
}

pub(crate) fn default_fill_respects_color() -> bool {
    true
}

/// Build a per-voxel color resolver from the palette + distribution args.
/// Falls back to `color_single` when palette has 0 or 1 entry.
pub(crate) fn build_color_resolver(
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

// ── Stroke anchor ────────��──────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct StrokeAnchorAtScreen {
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
pub(crate) fn voxel_stroke_anchor_coord_at_screen(
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

// ── Undo helpers ──────────��─────────────────────────────────────────────────

pub(crate) fn push_solo_undo_step(
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

pub(crate) fn push_solo_selection_undo_step(
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

pub(crate) fn commit_voxel_edits(
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
            let _ = tx.try_send(msg);
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

// ── Stroke begin / end / preview ────────────────────────────────────────────

#[tauri::command]
pub(crate) fn voxel_stroke_begin(state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
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
pub(crate) fn voxel_stroke_preview_reset(
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
    *state.extrude_gizmo_base_depth.lock() = 0;
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
pub(crate) fn preview_tool_colors(
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

pub(crate) fn stroke_preview_meshes_for_union(
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
pub(crate) fn append_polygon_vertex_marker_meshes(
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
pub(crate) fn preview_single_cell_world(
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

// ── Stroke preview at screen ───────────���────────────────────────────────────

/// Preview-only stroke update during drag (commit on [`voxel_stroke_end`]).
#[tauri::command]
pub(crate) fn voxel_stroke_preview_at_screen(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: VoxelEditAtScreen,
) -> Result<(), String> {
    {
        let cm = state.collab.lock();
        if cm.is_client() && !cm.client_can_edit() {
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
        let mut targets = voxel_edit::collect_stroke_preview_targets(
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
        voxel_edit::extend_with_mirror_targets(&mut targets, args.mirror_axes);
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

// ── Cuboid plane geometry query ─────────────────────────────────────────────

/// Result of resolving cuboid/cylinder drag-plane geometry at a point in time.
/// Returned by [`query_cuboid_plane_geometry`] so the frontend can freeze this
/// during the depth phase and pass it back through `StrokeAux`, preventing
/// camera movement from altering the extrusion direction.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CuboidPlaneGeoResult {
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
pub(crate) fn query_cuboid_plane_geometry(
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

// ── Stroke end ─────────���────────────────────────────────────────────────────

#[tauri::command]
pub(crate) fn voxel_stroke_end(state: State<'_, Arc<ViewerState>>, app: AppHandle) -> Result<(), String> {
    *state.stroke_active.lock() = false;
    *state.extrude_ray_spine.lock() = None;
    *state.wall_stroke_face_snapped.lock() = None;
    state.terrain_accum.lock().clear();
    *state.extrude_gizmo_base_depth.lock() = 0;
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
            let _ = tx.try_send(msg);
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

// ── Pick color ──────────────────────────────────────────────────────────────

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct VoxelPickColorResult {
    color: u32,
    material: String,
}

#[tauri::command]
pub(crate) fn voxel_pick_color_at_screen(
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

// ── Fill commands ───────────────────────────────────────────────────────────

#[derive(serde::Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct VoxelFillAtScreen {
    nx: f32,
    ny: f32,
    color: u32,
    material: String,
    match_material: bool,
    /// When `true`, skip the large-fill confirmation gate (user already confirmed).
    #[serde(default)]
    confirmed: bool,
}

#[tauri::command]
pub(crate) fn voxel_fill_cancel(state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
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
    if large && !args.confirmed {
        return Err(String::from("confirm_large_fill"));
    }
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
        if large && !args.confirmed {
            return Err("confirm_large_fill".to_string());
        }
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

#[tauri::command]
pub(crate) async fn voxel_fill_at_screen(
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
            let _ = tx.try_send(msg);
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

// ── Clipboard / stamp commands ─────���────────────────────────────────────────

#[tauri::command]
pub(crate) fn clipboard_copy_selection(state: State<'_, Arc<ViewerState>>) -> Result<bool, String> {
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
pub(crate) struct StampPickAtScreen {
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
pub(crate) fn clipboard_stamp_at_screen(
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
pub(crate) fn clipboard_punch_at_screen(
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
pub(crate) fn stamp_face_normal_at_screen(
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
pub(crate) fn get_selection_as_stamp_entries(
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
pub(crate) struct StampBookLoadEntry {
    dx: i32,
    dy: i32,
    dz: i32,
    color: u32,
    material: String,
}

#[tauri::command]
pub(crate) fn stamp_book_load_entries(
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

// ── Main edit command ───────────────────────────────────────────────────────

#[tauri::command]
pub(crate) async fn voxel_edit_at_screen(
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
        if c.is_client() && !c.client_can_edit() {
            return Err("editing not allowed".into());
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
        emit_work_progress(&app, 0.08, "Fill���");
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
                args.mirror_axes,
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
            let _ = tx.try_send(msg);
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

// ── Solo undo / redo ────────────────────────────────────────────────────────

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

// ── Undo / redo commands ────────��───────────────────────────────────────────

#[tauri::command]
pub(crate) fn voxel_undo(state: State<'_, Arc<ViewerState>>, app: AppHandle) -> Result<bool, String> {
    let cm = Arc::clone(&state.collab);
    {
        let mut c = cm.lock();
        if c.is_client() {
            if let Some(tx) = &c.client_tx {
                let _ = tx.try_send(serde_json::to_string(&collab::ClientToHost::Undo).unwrap());
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
pub(crate) fn voxel_redo(state: State<'_, Arc<ViewerState>>, app: AppHandle) -> Result<bool, String> {
    let cm = Arc::clone(&state.collab);
    {
        let mut c = cm.lock();
        if c.is_client() {
            if let Some(tx) = &c.client_tx {
                let _ = tx.try_send(serde_json::to_string(&collab::ClientToHost::Redo).unwrap());
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

// ── Debug helpers (desktop only) ────────────────────────────────────────────

/// stderr one-liners for cuboid/cylinder depth commits — visible in `tauri dev` when the webview is wedged.
#[cfg(desktop)]
pub(crate) fn eprintln_extrusion_stroke_checkpoint(
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
pub(crate) fn eprintln_last_edit_perf_line(state: &ViewerState) {
    if let Some(e) = state.last_edit_perf.lock().clone() {
        eprintln!(
            "[voxelle] voxel_edit GPU refresh total_ms={:.1} mesh_ms={:.1} route={} apply_ms={:.1}",
            e.total_ms, e.mesh_ms, e.mesh_route, e.apply_edit_ms
        );
    }
}
