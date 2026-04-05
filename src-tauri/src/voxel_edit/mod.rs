//! Screen-space ray → grid traversal for add/remove voxel editing.

pub mod brush;
pub mod fill;
pub mod raycasting;

use crate::camera::OrbitCamera;
use crate::greedy_mesh::VoxelCoord;
use crate::sculpt_mesh_smooth::{
    apply_sculpt_smooth_majority_pass, apply_sculpt_smooth_mesh_laplacian, SculptSmoothVariant,
};
use crate::stroke_modes::{stroke_anchor_centers_with_mode, DrawStrokeMode, PlaneAxis, StrokeAux};
use crate::voxelle::{MaterialId, Voxel, VoxelleFile};
use ahash::{AHashMap, AHashSet};
use glam::Vec3;
use std::collections::HashSet;

// Re-export sub-module public items at the voxel_edit level so callers don't need to
// know which sub-module they live in.
pub use brush::{
    brush_offset_cells, brush_offset_cells_for_size, extrude_ray_footprint,
    extrude_selection_footprint, face_normal_to_axis, get_ray_direction_path,
    resolve_extrude_direction, snap_normal_to_axis, spray_direction_vector, BrushShape,
    ExtrudeDirectionRef, ExtrudeEndCap, ExtrudeProfile, SprayDirection,
};
pub use fill::{
    connected_solid_same_color_from_screen, coplanar_connected_from_screen,
    coplanar_empty_connected_from_screen, filter_coords_by_seed_color,
    filter_coords_coplanar_empty_from_screen, filter_coords_coplanar_solid_from_screen,
    flood_fill_empty_at_screen, flood_fill_empty_region_exceeds_threshold,
    flood_fill_paint_at_screen, flood_fill_remove_at_screen, flood_fill_selection_coords,
    flood_fill_selection_coords_with_control, flood_fill_selection_region_exceeds_threshold,
    neighbors_6, remove_voxel_at_coord, FillCoordOutcome, FloodFillEditOutcome,
    FILL_ABSOLUTE_MAX_CELLS, FILL_BFS_CANCEL_CHECK_INTERVAL, FILL_BFS_PROGRESS_INTERVAL,
    FILL_THRESHOLD_PROBE_CANCEL_INTERVAL, FILL_UNCONSTRAINED_LARGE_THRESHOLD,
};
pub use raycasting::{
    anchor_on_plane, constrain_plane_normal, local_ray_entry_on_voxel_cell,
    outward_face_normal_from_screen_ray, pick_extrude_start, pick_solid_coord_at_screen,
    pick_voxel_at_screen, preview_add_cell, preview_remove_cell, probe_solid_hit,
    screen_to_world_ray, world_ray_entry_on_voxel_cell, world_to_viewport_pixels,
};
pub(crate) use raycasting::{
    anchor_for_stroke_edit, ray_first_solid, ray_first_solid_scene, voxel_line_dda,
};

// ── Grid utilities (used by all sub-modules via super::) ─────────────────────

/// Web parity with `MAX_GRID_SIZE` in `store/core.ts`: symmetric grid about origin, capped for safety.
pub const MAX_GRID_SIZE: i32 = 65536;

/// Inclusive min/max voxel indices for a centered `grid_size³` grid (same as `mesh_bounds_for_cube_side`).
pub fn grid_valid_range(grid_size: i32) -> (i32, i32) {
    let gs = grid_size.max(1);
    let start = -gs / 2;
    let end = start + gs;
    (start, end - 1)
}

#[inline]
pub fn in_grid(x: i32, y: i32, z: i32, grid_size: i32) -> bool {
    let (lo, hi) = grid_valid_range(grid_size);
    x >= lo && x <= hi && y >= lo && y <= hi && z >= lo && z <= hi
}

#[inline]
pub(crate) fn world_to_voxel(p: Vec3) -> (i32, i32, i32) {
    (
        (p.x + 0.5).floor() as i32,
        (p.y + 0.5).floor() as i32,
        (p.z + 0.5).floor() as i32,
    )
}

#[inline]
fn min_grid_size_for_max_abs(max_abs: i32) -> i32 {
    (2 * (max_abs + 1)).max(1)
}

/// Ray/pick volume: at least the file's declared grid and enough extent to include all voxels plus one
/// shell layer (so "add in front of face" works when looking at the outer boundary).
pub fn effective_ray_grid_size(file: &VoxelleFile) -> i32 {
    let base = file.grid_size.max(1);
    let mut max_a = 0i32;
    for v in &file.voxels {
        max_a = max_a.max(v.x.abs()).max(v.y.abs()).max(v.z.abs());
    }
    let content_slack = 2 * (max_a + 2);
    base.max(content_slack).min(MAX_GRID_SIZE)
}

pub(crate) fn min_grid_size_for_centers_offsets(
    centers: &[(i32, i32, i32)],
    offsets: &[(i32, i32, i32)],
) -> i32 {
    let mut max_abs = 0i32;
    for &(cx, cy, cz) in centers {
        for &(dx, dy, dz) in offsets {
            let x = cx + dx;
            let y = cy + dy;
            let z = cz + dz;
            max_abs = max_abs.max(x.abs()).max(y.abs()).max(z.abs());
        }
    }
    min_grid_size_for_max_abs(max_abs)
}

fn min_grid_size_for_coords(coords: &[(i32, i32, i32)]) -> i32 {
    let mut max_abs = 0i32;
    for &(x, y, z) in coords {
        max_abs = max_abs.max(x.abs()).max(y.abs()).max(z.abs());
    }
    min_grid_size_for_max_abs(max_abs)
}

/// Grow `file.grid_size` so centered bounds fit all stroke cells (web `ensureGridFitsPositions`).
pub(crate) fn ensure_grid_fits_centers_offsets(
    file: &mut VoxelleFile,
    centers: &[(i32, i32, i32)],
    offsets: &[(i32, i32, i32)],
) {
    let need = min_grid_size_for_centers_offsets(centers, offsets);
    if need > file.grid_size.max(1) {
        file.grid_size = need.min(MAX_GRID_SIZE);
    }
}

pub(crate) fn ensure_grid_fits_coords(
    file: &mut VoxelleFile,
    coords: impl Iterator<Item = (i32, i32, i32)>,
) {
    let mut max_abs = 0i32;
    for (x, y, z) in coords {
        max_abs = max_abs.max(x.abs()).max(y.abs()).max(z.abs());
    }
    let need = min_grid_size_for_max_abs(max_abs);
    if need > file.grid_size.max(1) {
        file.grid_size = need.min(MAX_GRID_SIZE);
    }
}

#[inline]
pub(crate) fn ensure_grid_fits_coord(file: &mut VoxelleFile, x: i32, y: i32, z: i32) {
    let max_abs = x.abs().max(y.abs()).max(z.abs());
    let need = min_grid_size_for_max_abs(max_abs);
    if need > file.grid_size.max(1) {
        file.grid_size = need.min(MAX_GRID_SIZE);
    }
}

/// For preview / sampling without mutating `file`: pretend grid is large enough for stroke geometry.
pub(crate) fn stroke_clip_grid_size(
    file: &VoxelleFile,
    centers: &[(i32, i32, i32)],
    offsets: &[(i32, i32, i32)],
) -> i32 {
    let need = min_grid_size_for_centers_offsets(centers, offsets);
    file.grid_size.max(1).max(need).min(MAX_GRID_SIZE)
}

// ── Core types ───────────────────────────────────────────────────────────────

/// Result of a successful edit for GPU incremental brick updates.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum VoxelEditDelta {
    Added(Voxel),
    Removed {
        voxel: Voxel,
    },
    /// Recolor / material change; `before` and `after` share the same cell.
    Painted {
        before: Voxel,
        after: Voxel,
    },
}

impl VoxelEditDelta {
    /// Coordinate of the affected voxel.
    pub fn coord(&self) -> (i32, i32, i32) {
        match self {
            Self::Added(v) | Self::Removed { voxel: v } => (v.x, v.y, v.z),
            Self::Painted { after, .. } => (after.x, after.y, after.z),
        }
    }

    /// Clone this delta but move it to a different coordinate.
    pub fn with_coord(&self, x: i32, y: i32, z: i32) -> Self {
        match self {
            Self::Added(v) => Self::Added(Voxel { x, y, z, ..*v }),
            Self::Removed { voxel } => Self::Removed {
                voxel: Voxel { x, y, z, ..*voxel },
            },
            Self::Painted { before, after } => Self::Painted {
                before: Voxel { x, y, z, ..*before },
                after: Voxel { x, y, z, ..*after },
            },
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum EditTool {
    Add,
    Remove,
    Paint,
}

/// Matches web [`SculptMode`](digital-garden) for stroke-based sculpting (excluding rope/cloth generators).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SculptStrokeMode {
    Draw,
    Smooth,
    Gouge,
    Wall,
    Terrain,
    #[serde(alias = "branch")]
    Extrude,
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TerrainSculptOp {
    Raise,
    Lower,
    Smooth,
    Flatten,
    Erode,
}

/// Web `WallAreaShape` — circle/polygon need extra pointer flows; brush uses freehand spine.
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum WallAreaShape {
    #[default]
    Brush,
    Circle,
    Polygon,
}

// ── Symmetry / mirror helpers ────────────────────────────────────────────────

/// Flip a coordinate tuple by an axis-flip mask (bit 0 = X, bit 1 = Y, bit 2 = Z).
#[inline]
pub fn flip_coord(x: i32, y: i32, z: i32, flip_mask: u8) -> (i32, i32, i32) {
    (
        if flip_mask & 1 != 0 { -x } else { x },
        if flip_mask & 2 != 0 { -y } else { y },
        if flip_mask & 4 != 0 { -z } else { z },
    )
}

/// Extends `targets` with mirrored copies for each enabled axis subset in `mirror_axes`
/// (bit 0 = X, bit 1 = Y, bit 2 = Z).  Used by hover-preview to show the symmetric footprint.
pub fn extend_with_mirror_targets(targets: &mut Vec<VoxelCoord>, mirror_axes: u8) {
    if mirror_axes == 0 {
        return;
    }
    let original: Vec<VoxelCoord> = targets.clone();
    let mut seen: HashSet<VoxelCoord> = original.iter().copied().collect();
    for flip_mask in 1u8..=7u8 {
        if flip_mask & mirror_axes != flip_mask {
            continue;
        }
        for &(x, y, z) in &original {
            let m = flip_coord(x, y, z, flip_mask);
            if seen.insert(m) {
                targets.push(m);
            }
        }
    }
}

/// Like [`extend_with_mirror_targets`] but for `(VoxelCoord, u32)` pairs (coord + color).
pub fn extend_with_mirror_targets_colored(targets: &mut Vec<(VoxelCoord, u32)>, mirror_axes: u8) {
    if mirror_axes == 0 {
        return;
    }
    let original: Vec<(VoxelCoord, u32)> = targets.clone();
    let mut seen: HashSet<VoxelCoord> = original.iter().map(|&(c, _)| c).collect();
    for flip_mask in 1u8..=7u8 {
        if flip_mask & mirror_axes != flip_mask {
            continue;
        }
        for &((x, y, z), color) in &original {
            let m = flip_coord(x, y, z, flip_mask);
            if seen.insert(m) {
                targets.push((m, color));
            }
        }
    }
}

// ── Core edit operations ─────────────────────────────────────────────────────

/// Apply add / remove / paint with optional brush; returns all atomic deltas (may be empty).
///
/// `spray_density`: `0` = full brush; `(0, 1]` thins voxels deterministically per cell.
/// `stroke_line_start`: when `Some`, brush samples along the 3D line between anchors at
/// pointer-down and `(sx, sy)` (Stroke / line mode).
/// `stroke_segment_prev`: when `stroke_line_start` is `None` and this is `Some`, samples along
/// the segment from the previous screen position to `(sx, sy)` (Brush path / web spray-style drag).
/// `mirror_axes`: symmetry bitmask (bit 0 = X, bit 1 = Y, bit 2 = Z); 0 = no mirroring.
#[allow(clippy::too_many_arguments)]
pub fn apply_edit(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    tool: EditTool,
    color_resolver: impl Fn(i32, i32, i32) -> u32,
    material: MaterialId,
    brush_radius: u32,
    brush_shape: BrushShape,
    spray_density: f32,
    stroke_line_start: Option<(f32, f32)>,
    stroke_segment_prev: Option<(f32, f32)>,
    stroke_mode: DrawStrokeMode,
    plane_axis: PlaneAxis,
    stroke_aux: &StrokeAux,
    spray_constraint_plane: Option<(Vec3, Vec3)>,
    mirror_axes: u8,
) -> Result<Vec<VoxelEditDelta>, String> {
    let brush_radius = brush::brush_radius_for_area_polygon_stroke(stroke_mode, brush_radius);
    let clip_half = brush::brush_clip_half_normal_from_screen(
        stroke_aux.brush_clip_bottom_half,
        file,
        voxel_map,
        camera,
        width,
        height,
        sx,
        sy,
    );
    // Spray mode: use scatter-based stamps (web parity) instead of density thinning.
    let is_spray_scatter = stroke_mode == DrawStrokeMode::Spray
        && (stroke_aux.spray_scatter > 0 || stroke_aux.spray_size_range);
    let effective_shape = if stroke_mode == DrawStrokeMode::Spray {
        stroke_aux.spray_brush_shape.unwrap_or(brush_shape)
    } else {
        brush_shape
    };
    let offsets = brush::brush_offset_cells(effective_shape, brush_radius, clip_half, None);
    let spray = spray_density.clamp(0.0, 1.0);
    let centers = stroke_anchor_centers_with_mode(
        stroke_mode,
        plane_axis,
        stroke_aux,
        tool,
        file,
        voxel_map,
        camera,
        width,
        height,
        sx,
        sy,
        brush_radius,
        stroke_line_start,
        stroke_segment_prev,
        spray_constraint_plane,
    );
    let centers = brush::adjust_add_centers_for_surface_snap_brush(
        centers,
        tool,
        effective_shape,
        brush_radius,
        stroke_aux,
        file,
        voxel_map,
        camera,
        width,
        height,
        sx,
        sy,
    );
    if centers.is_empty() {
        return Ok(Vec::new());
    }

    ensure_grid_fits_centers_offsets(file, &centers, &offsets);
    let grid_size = file.grid_size.max(1);

    let mut seen: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut out: Vec<VoxelEditDelta> = Vec::new();

    // Helper: resolve brush offsets per center for scatter spray (variable radius + scatter offset).
    let scatter = stroke_aux.spray_scatter;
    let size_range = stroke_aux.spray_size_range;
    let rmin = stroke_aux.spray_radius_min;
    let rmax = stroke_aux.spray_radius_max;

    match tool {
        EditTool::Add => {
            for (cx, cy, cz) in &centers {
                let (scx, scy, scz, cur_offsets);
                if is_spray_scatter {
                    let ox = brush::spray_scatter_offset((*cx, *cy, *cz), scatter, 0);
                    let oy = brush::spray_scatter_offset((*cx, *cy, *cz), scatter, 1);
                    let oz = brush::spray_scatter_offset((*cx, *cy, *cz), scatter, 2);
                    scx = cx + ox;
                    scy = cy + oy;
                    scz = cz + oz;
                    if size_range && rmax > rmin {
                        let r = brush::spray_random_radius((*cx, *cy, *cz), rmin, rmax);
                        cur_offsets = brush::brush_offset_cells(effective_shape, r, clip_half, None);
                    } else {
                        cur_offsets = Vec::new();
                    }
                } else {
                    scx = *cx;
                    scy = *cy;
                    scz = *cz;
                    cur_offsets = Vec::new();
                }
                let use_offsets = if !cur_offsets.is_empty() {
                    &cur_offsets
                } else {
                    &offsets
                };
                for (dx, dy, dz) in use_offsets {
                    let x = scx + dx;
                    let y = scy + dy;
                    let z = scz + dz;
                    if !in_grid(x, y, z, grid_size) {
                        continue;
                    }
                    if !is_spray_scatter && !brush::spray_passes((x, y, z), spray) {
                        continue;
                    }
                    if !seen.insert((x, y, z)) {
                        continue;
                    }
                    if voxel_map.contains_key(&(x, y, z)) {
                        continue;
                    }
                    let nv = Voxel {
                        x,
                        y,
                        z,
                        color: color_resolver(x, y, z),
                        material,
                        object_id: file.active_object_id,
                    };
                    let idx = file.voxels.len();
                    file.voxels.push(nv);
                    voxel_map.insert((x, y, z), idx);
                    out.push(VoxelEditDelta::Added(nv));
                }
            }
        }
        EditTool::Remove => {
            for (hx, hy, hz) in &centers {
                let (scx, scy, scz, cur_offsets);
                if is_spray_scatter {
                    let ox = brush::spray_scatter_offset((*hx, *hy, *hz), scatter, 0);
                    let oy = brush::spray_scatter_offset((*hx, *hy, *hz), scatter, 1);
                    let oz = brush::spray_scatter_offset((*hx, *hy, *hz), scatter, 2);
                    scx = hx + ox;
                    scy = hy + oy;
                    scz = hz + oz;
                    if size_range && rmax > rmin {
                        let r = brush::spray_random_radius((*hx, *hy, *hz), rmin, rmax);
                        cur_offsets = brush::brush_offset_cells(effective_shape, r, clip_half, None);
                    } else {
                        cur_offsets = Vec::new();
                    }
                } else {
                    scx = *hx;
                    scy = *hy;
                    scz = *hz;
                    cur_offsets = Vec::new();
                }
                let use_offsets = if !cur_offsets.is_empty() {
                    &cur_offsets
                } else {
                    &offsets
                };
                for (dx, dy, dz) in use_offsets {
                    let x = scx + dx;
                    let y = scy + dy;
                    let z = scz + dz;
                    if !is_spray_scatter && !brush::spray_passes((x, y, z), spray) {
                        continue;
                    }
                    if !seen.insert((x, y, z)) {
                        continue;
                    }
                    let Some(&remove_idx) = voxel_map.get(&(x, y, z)) else {
                        continue;
                    };
                    let removed_voxel = file.voxels[remove_idx];
                    let last = file.voxels.len() - 1;
                    if remove_idx != last {
                        file.voxels.swap(remove_idx, last);
                        let moved = file.voxels[remove_idx];
                        voxel_map.insert((moved.x, moved.y, moved.z), remove_idx);
                    }
                    file.voxels.pop();
                    voxel_map.remove(&(x, y, z));
                    out.push(VoxelEditDelta::Removed {
                        voxel: removed_voxel,
                    });
                }
            }
        }
        EditTool::Paint => {
            for (hx, hy, hz) in &centers {
                let (scx, scy, scz, cur_offsets);
                if is_spray_scatter {
                    let ox = brush::spray_scatter_offset((*hx, *hy, *hz), scatter, 0);
                    let oy = brush::spray_scatter_offset((*hx, *hy, *hz), scatter, 1);
                    let oz = brush::spray_scatter_offset((*hx, *hy, *hz), scatter, 2);
                    scx = hx + ox;
                    scy = hy + oy;
                    scz = hz + oz;
                    if size_range && rmax > rmin {
                        let r = brush::spray_random_radius((*hx, *hy, *hz), rmin, rmax);
                        cur_offsets = brush::brush_offset_cells(effective_shape, r, clip_half, None);
                    } else {
                        cur_offsets = Vec::new();
                    }
                } else {
                    scx = *hx;
                    scy = *hy;
                    scz = *hz;
                    cur_offsets = Vec::new();
                }
                let use_offsets = if !cur_offsets.is_empty() {
                    &cur_offsets
                } else {
                    &offsets
                };
                for (dx, dy, dz) in use_offsets {
                    let x = scx + dx;
                    let y = scy + dy;
                    let z = scz + dz;
                    if !is_spray_scatter && !brush::spray_passes((x, y, z), spray) {
                        continue;
                    }
                    if !seen.insert((x, y, z)) {
                        continue;
                    }
                    let Some(&idx) = voxel_map.get(&(x, y, z)) else {
                        continue;
                    };
                    let before = file.voxels[idx];
                    let resolved_color = color_resolver(x, y, z);
                    if before.color == resolved_color && before.material == material {
                        continue;
                    }
                    let after = Voxel {
                        color: resolved_color,
                        material,
                        ..before
                    };
                    file.voxels[idx] = after;
                    out.push(VoxelEditDelta::Painted { before, after });
                }
            }
        }
    }

    // Symmetry pass: replicate every delta across the enabled mirror axes.
    if mirror_axes != 0 {
        let base_deltas = out.clone();
        let mut mirror_seen: HashSet<(i32, i32, i32)> = HashSet::new();
        for flip_mask in 1u8..=7u8 {
            if flip_mask & mirror_axes != flip_mask {
                continue;
            }
            match tool {
                EditTool::Add => {
                    for delta in &base_deltas {
                        let VoxelEditDelta::Added(v) = delta else {
                            continue;
                        };
                        let (mx, my, mz) = flip_coord(v.x, v.y, v.z, flip_mask);
                        if !in_grid(mx, my, mz, grid_size) {
                            continue;
                        }
                        if !mirror_seen.insert((mx, my, mz)) {
                            continue;
                        }
                        if voxel_map.contains_key(&(mx, my, mz)) {
                            continue;
                        }
                        let nv = Voxel {
                            x: mx,
                            y: my,
                            z: mz,
                            color: color_resolver(mx, my, mz),
                            material,
                            object_id: file.active_object_id,
                        };
                        let idx = file.voxels.len();
                        file.voxels.push(nv);
                        voxel_map.insert((mx, my, mz), idx);
                        out.push(VoxelEditDelta::Added(nv));
                    }
                }
                EditTool::Remove => {
                    for delta in &base_deltas {
                        let VoxelEditDelta::Removed { voxel: v } = delta else {
                            continue;
                        };
                        let (mx, my, mz) = flip_coord(v.x, v.y, v.z, flip_mask);
                        if !mirror_seen.insert((mx, my, mz)) {
                            continue;
                        }
                        let Some(&remove_idx) = voxel_map.get(&(mx, my, mz)) else {
                            continue;
                        };
                        let removed_voxel = file.voxels[remove_idx];
                        let last = file.voxels.len() - 1;
                        if remove_idx != last {
                            file.voxels.swap(remove_idx, last);
                            let moved = file.voxels[remove_idx];
                            voxel_map.insert((moved.x, moved.y, moved.z), remove_idx);
                        }
                        file.voxels.pop();
                        voxel_map.remove(&(mx, my, mz));
                        out.push(VoxelEditDelta::Removed {
                            voxel: removed_voxel,
                        });
                    }
                }
                EditTool::Paint => {
                    for delta in &base_deltas {
                        let VoxelEditDelta::Painted { after: v, .. } = delta else {
                            continue;
                        };
                        let (mx, my, mz) = flip_coord(v.x, v.y, v.z, flip_mask);
                        if !mirror_seen.insert((mx, my, mz)) {
                            continue;
                        }
                        let Some(&idx) = voxel_map.get(&(mx, my, mz)) else {
                            continue;
                        };
                        let before = file.voxels[idx];
                        let resolved_color = color_resolver(mx, my, mz);
                        if before.color == resolved_color && before.material == material {
                            continue;
                        }
                        let after = Voxel {
                            color: resolved_color,
                            material,
                            ..before
                        };
                        file.voxels[idx] = after;
                        out.push(VoxelEditDelta::Painted { before, after });
                    }
                }
            }
        }
    }

    Ok(out)
}

/// Whether drag samples for stroke preview should union across moves (`true`) or replace each frame (`false`).
pub fn stroke_preview_accumulates_samples(
    stroke_mode: DrawStrokeMode,
    stroke_line_start: Option<(f32, f32)>,
) -> bool {
    match stroke_mode {
        // Area shapes from drag origin → current: each preview is already the full footprint; unioning
        // would keep cells from earlier corners (e.g. a diagonal pass), inflating the minimum size.
        DrawStrokeMode::Plane
        | DrawStrokeMode::Circle
        | DrawStrokeMode::Cuboid
        | DrawStrokeMode::Cylinder => stroke_line_start.is_none(),
        DrawStrokeMode::Spray => true,
        DrawStrokeMode::Precise => false,
        DrawStrokeMode::Line => stroke_line_start.is_none(),
        _ => false,
    }
}

/// Voxel coordinates one `apply_edit` call would affect (same rules; no mutation).
///
/// For Paint, no-op cells removed. For Remove, empty footprint cells removed (preview may include
/// them for meshing only).
///
/// Not all call sites use this (preview uses [`collect_stroke_preview_targets`] only); kept for
/// parity with [`apply_edit`] and for tests / tooling.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn collect_stroke_edit_targets(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    tool: EditTool,
    color: u32,
    material: MaterialId,
    brush_radius: u32,
    brush_shape: BrushShape,
    spray_density: f32,
    stroke_line_start: Option<(f32, f32)>,
    stroke_segment_prev: Option<(f32, f32)>,
    stroke_mode: DrawStrokeMode,
    plane_axis: PlaneAxis,
    stroke_aux: &StrokeAux,
    spray_constraint_plane: Option<(Vec3, Vec3)>,
) -> Vec<VoxelCoord> {
    let mut out = collect_stroke_preview_targets(
        file,
        voxel_map,
        camera,
        width,
        height,
        sx,
        sy,
        tool,
        color,
        material,
        brush_radius,
        brush_shape,
        spray_density,
        stroke_line_start,
        stroke_segment_prev,
        stroke_mode,
        plane_axis,
        stroke_aux,
        spray_constraint_plane,
    );
    if matches!(tool, EditTool::Remove) {
        out.retain(|c| voxel_map.contains_key(c));
    }
    if matches!(tool, EditTool::Paint) {
        out.retain(|&(x, y, z)| {
            let Some(&idx) = voxel_map.get(&(x, y, z)) else {
                return false;
            };
            let before = file.voxels[idx];
            before.color != color || before.material != material
        });
    }
    out
}

/// Geometric brush footprint for hover / stroke **preview** meshes only.
///
/// Same centers/offsets/spray as [`collect_stroke_edit_targets`], except **Fill** uses a single
/// seed cell (no brush expansion), matching the click commit. **Remove** and **Paint** include
/// every in-grid cell in the footprint (occupied and empty) so the full brush shape is visible in
/// air; [`collect_stroke_edit_targets`] still applies Paint no-op filtering for commits.
#[allow(clippy::too_many_arguments)]
pub fn collect_stroke_preview_targets(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    tool: EditTool,
    _color: u32,
    _material: MaterialId,
    brush_radius: u32,
    brush_shape: BrushShape,
    spray_density: f32,
    stroke_line_start: Option<(f32, f32)>,
    stroke_segment_prev: Option<(f32, f32)>,
    stroke_mode: DrawStrokeMode,
    plane_axis: PlaneAxis,
    stroke_aux: &StrokeAux,
    spray_constraint_plane: Option<(Vec3, Vec3)>,
) -> Vec<VoxelCoord> {
    // Fill commits a single seed voxel; brush footprint does not apply. `Fill` returns no centers
    // from `stroke_anchor_centers_with_mode`, so without this branch hover preview would be empty.
    if stroke_mode == DrawStrokeMode::Fill {
        let snap = stroke_aux.stroke_snap_to_surface;
        let Some((cx, cy, cz)) = raycasting::anchor_for_stroke_edit(
            tool,
            snap,
            file,
            voxel_map,
            camera,
            width,
            height,
            sx,
            sy,
        ) else {
            return Vec::new();
        };
        let ma = cx.abs().max(cy.abs()).max(cz.abs());
        let grid_size = file
            .grid_size
            .max(1)
            .max(min_grid_size_for_max_abs(ma))
            .min(MAX_GRID_SIZE);
        if !in_grid(cx, cy, cz, grid_size) {
            return Vec::new();
        }
        return match tool {
            EditTool::Add => {
                if voxel_map.contains_key(&(cx, cy, cz)) {
                    Vec::new()
                } else {
                    vec![(cx, cy, cz)]
                }
            }
            EditTool::Remove | EditTool::Paint => vec![(cx, cy, cz)],
        };
    }
    let brush_radius = brush::brush_radius_for_area_polygon_stroke(stroke_mode, brush_radius);
    let clip_half = brush::brush_clip_half_normal_from_screen(
        stroke_aux.brush_clip_bottom_half,
        file,
        voxel_map,
        camera,
        width,
        height,
        sx,
        sy,
    );
    let is_spray_scatter = stroke_mode == DrawStrokeMode::Spray
        && (stroke_aux.spray_scatter > 0 || stroke_aux.spray_size_range);
    let effective_shape = if stroke_mode == DrawStrokeMode::Spray {
        stroke_aux.spray_brush_shape.unwrap_or(brush_shape)
    } else {
        brush_shape
    };
    let offsets = brush::brush_offset_cells(effective_shape, brush_radius, clip_half, None);
    let spray = spray_density.clamp(0.0, 1.0);
    let centers = stroke_anchor_centers_with_mode(
        stroke_mode,
        plane_axis,
        stroke_aux,
        tool,
        file,
        voxel_map,
        camera,
        width,
        height,
        sx,
        sy,
        brush_radius,
        stroke_line_start,
        stroke_segment_prev,
        spray_constraint_plane,
    );
    let centers = brush::adjust_add_centers_for_surface_snap_brush(
        centers,
        tool,
        effective_shape,
        brush_radius,
        stroke_aux,
        file,
        voxel_map,
        camera,
        width,
        height,
        sx,
        sy,
    );
    if centers.is_empty() {
        return Vec::new();
    }
    let grid_size = stroke_clip_grid_size(file, &centers, &offsets);
    let scatter = stroke_aux.spray_scatter;
    let size_range = stroke_aux.spray_size_range;
    let rmin = stroke_aux.spray_radius_min;
    let rmax = stroke_aux.spray_radius_max;
    let mut seen: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut out: Vec<VoxelCoord> = Vec::new();

    match tool {
        EditTool::Add => {
            for (cx, cy, cz) in &centers {
                let (scx, scy, scz, cur_offsets);
                if is_spray_scatter {
                    let ox = brush::spray_scatter_offset((*cx, *cy, *cz), scatter, 0);
                    let oy = brush::spray_scatter_offset((*cx, *cy, *cz), scatter, 1);
                    let oz = brush::spray_scatter_offset((*cx, *cy, *cz), scatter, 2);
                    scx = cx + ox;
                    scy = cy + oy;
                    scz = cz + oz;
                    if size_range && rmax > rmin {
                        let r = brush::spray_random_radius((*cx, *cy, *cz), rmin, rmax);
                        cur_offsets = brush::brush_offset_cells(effective_shape, r, clip_half, None);
                    } else {
                        cur_offsets = Vec::new();
                    }
                } else {
                    scx = *cx;
                    scy = *cy;
                    scz = *cz;
                    cur_offsets = Vec::new();
                }
                let use_offsets = if !cur_offsets.is_empty() {
                    &cur_offsets
                } else {
                    &offsets
                };
                for (dx, dy, dz) in use_offsets {
                    let x = scx + dx;
                    let y = scy + dy;
                    let z = scz + dz;
                    if !in_grid(x, y, z, grid_size) {
                        continue;
                    }
                    if !is_spray_scatter && !brush::spray_passes((x, y, z), spray) {
                        continue;
                    }
                    if !seen.insert((x, y, z)) {
                        continue;
                    }
                    if voxel_map.contains_key(&(x, y, z)) {
                        continue;
                    }
                    out.push((x, y, z));
                }
            }
        }
        EditTool::Remove | EditTool::Paint => {
            for (hx, hy, hz) in &centers {
                let (scx, scy, scz, cur_offsets);
                if is_spray_scatter {
                    let ox = brush::spray_scatter_offset((*hx, *hy, *hz), scatter, 0);
                    let oy = brush::spray_scatter_offset((*hx, *hy, *hz), scatter, 1);
                    let oz = brush::spray_scatter_offset((*hx, *hy, *hz), scatter, 2);
                    scx = hx + ox;
                    scy = hy + oy;
                    scz = hz + oz;
                    if size_range && rmax > rmin {
                        let r = brush::spray_random_radius((*hx, *hy, *hz), rmin, rmax);
                        cur_offsets = brush::brush_offset_cells(effective_shape, r, clip_half, None);
                    } else {
                        cur_offsets = Vec::new();
                    }
                } else {
                    scx = *hx;
                    scy = *hy;
                    scz = *hz;
                    cur_offsets = Vec::new();
                }
                let use_offsets = if !cur_offsets.is_empty() {
                    &cur_offsets
                } else {
                    &offsets
                };
                for (dx, dy, dz) in use_offsets {
                    let x = scx + dx;
                    let y = scy + dy;
                    let z = scz + dz;
                    if !in_grid(x, y, z, grid_size) {
                        continue;
                    }
                    if !is_spray_scatter && !brush::spray_passes((x, y, z), spray) {
                        continue;
                    }
                    if !seen.insert((x, y, z)) {
                        continue;
                    }
                    out.push((x, y, z));
                }
            }
        }
    }
    out
}

/// Apply add/remove/paint to a precomputed union of target cells (one undo step).
pub fn apply_edits_to_coords(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    tool: EditTool,
    color_resolver: impl Fn(i32, i32, i32) -> u32,
    material: MaterialId,
    coords: &AHashSet<VoxelCoord>,
) -> Vec<VoxelEditDelta> {
    ensure_grid_fits_coords(file, coords.iter().copied());
    let grid_size = file.grid_size.max(1);
    let mut out: Vec<VoxelEditDelta> = Vec::new();

    match tool {
        EditTool::Add => {
            for &(x, y, z) in coords {
                if !in_grid(x, y, z, grid_size) {
                    continue;
                }
                if voxel_map.contains_key(&(x, y, z)) {
                    continue;
                }
                let nv = Voxel {
                    x,
                    y,
                    z,
                    color: color_resolver(x, y, z),
                    material,
                    object_id: file.active_object_id,
                };
                let idx = file.voxels.len();
                file.voxels.push(nv);
                voxel_map.insert((x, y, z), idx);
                out.push(VoxelEditDelta::Added(nv));
            }
        }
        EditTool::Remove => {
            for &(x, y, z) in coords {
                let Some(&remove_idx) = voxel_map.get(&(x, y, z)) else {
                    continue;
                };
                let removed_voxel = file.voxels[remove_idx];
                let last = file.voxels.len() - 1;
                if remove_idx != last {
                    file.voxels.swap(remove_idx, last);
                    let moved = file.voxels[remove_idx];
                    voxel_map.insert((moved.x, moved.y, moved.z), remove_idx);
                }
                file.voxels.pop();
                voxel_map.remove(&(x, y, z));
                out.push(VoxelEditDelta::Removed {
                    voxel: removed_voxel,
                });
            }
        }
        EditTool::Paint => {
            for &(x, y, z) in coords {
                let Some(&idx) = voxel_map.get(&(x, y, z)) else {
                    continue;
                };
                let before = file.voxels[idx];
                let resolved_color = color_resolver(x, y, z);
                if before.color == resolved_color && before.material == material {
                    continue;
                }
                let after = Voxel {
                    color: resolved_color,
                    material,
                    ..before
                };
                file.voxels[idx] = after;
                out.push(VoxelEditDelta::Painted { before, after });
            }
        }
    }
    out
}

/// Solid voxel coordinates along a selection stroke (web parity: same geometry as remove/paint,
/// keeping only cells that exist — same as `apply_edit` remove sampling without mutating voxels).
#[allow(clippy::too_many_arguments)]
pub fn selection_stroke_sample_coords(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    brush_radius: u32,
    brush_shape: BrushShape,
    spray_density: f32,
    stroke_line_start: Option<(f32, f32)>,
    stroke_segment_prev: Option<(f32, f32)>,
    stroke_mode: DrawStrokeMode,
    plane_axis: PlaneAxis,
    stroke_aux: &StrokeAux,
    spray_constraint_plane: Option<(Vec3, Vec3)>,
) -> Vec<VoxelCoord> {
    let brush_radius = brush::brush_radius_for_area_polygon_stroke(stroke_mode, brush_radius);
    let clip_half = brush::brush_clip_half_normal_from_screen(
        stroke_aux.brush_clip_bottom_half,
        file,
        voxel_map,
        camera,
        width,
        height,
        sx,
        sy,
    );
    let is_spray_scatter = stroke_mode == DrawStrokeMode::Spray
        && (stroke_aux.spray_scatter > 0 || stroke_aux.spray_size_range);
    let effective_shape = if stroke_mode == DrawStrokeMode::Spray {
        stroke_aux.spray_brush_shape.unwrap_or(brush_shape)
    } else {
        brush_shape
    };
    let offsets = brush::brush_offset_cells(effective_shape, brush_radius, clip_half, None);
    let spray = spray_density.clamp(0.0, 1.0);
    let tool = EditTool::Remove;
    let centers = stroke_anchor_centers_with_mode(
        stroke_mode,
        plane_axis,
        stroke_aux,
        tool,
        file,
        voxel_map,
        camera,
        width,
        height,
        sx,
        sy,
        brush_radius,
        stroke_line_start,
        stroke_segment_prev,
        spray_constraint_plane,
    );
    if centers.is_empty() {
        return Vec::new();
    }
    let grid_size = stroke_clip_grid_size(file, &centers, &offsets);

    let scatter = stroke_aux.spray_scatter;
    let size_range = stroke_aux.spray_size_range;
    let rmin = stroke_aux.spray_radius_min;
    let rmax = stroke_aux.spray_radius_max;

    let mut seen: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut out: Vec<VoxelCoord> = Vec::new();

    for (hx, hy, hz) in &centers {
        let (scx, scy, scz, cur_offsets);
        if is_spray_scatter {
            let ox = brush::spray_scatter_offset((*hx, *hy, *hz), scatter, 0);
            let oy = brush::spray_scatter_offset((*hx, *hy, *hz), scatter, 1);
            let oz = brush::spray_scatter_offset((*hx, *hy, *hz), scatter, 2);
            scx = hx + ox;
            scy = hy + oy;
            scz = hz + oz;
            if size_range && rmax > rmin {
                let r = brush::spray_random_radius((*hx, *hy, *hz), rmin, rmax);
                cur_offsets = brush::brush_offset_cells(effective_shape, r, clip_half, None);
            } else {
                cur_offsets = Vec::new();
            }
        } else {
            scx = *hx;
            scy = *hy;
            scz = *hz;
            cur_offsets = Vec::new();
        }
        let use_offsets = if !cur_offsets.is_empty() {
            &cur_offsets
        } else {
            &offsets
        };
        for (dx, dy, dz) in use_offsets {
            let x = scx + dx;
            let y = scy + dy;
            let z = scz + dz;
            if !in_grid(x, y, z, grid_size) {
                continue;
            }
            if !is_spray_scatter && !brush::spray_passes((x, y, z), spray) {
                continue;
            }
            if !seen.insert((x, y, z)) {
                continue;
            }
            if voxel_map.contains_key(&(x, y, z)) {
                out.push((x, y, z));
            }
        }
    }
    out
}

/// Empty-cell stroke samples (e.g. coplanar void selection) — `EditTool::Add` anchors, keep only empty cells.
#[allow(clippy::too_many_arguments)]
pub fn selection_stroke_sample_empty_coords(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    brush_radius: u32,
    brush_shape: BrushShape,
    spray_density: f32,
    stroke_line_start: Option<(f32, f32)>,
    stroke_segment_prev: Option<(f32, f32)>,
    stroke_mode: DrawStrokeMode,
    plane_axis: PlaneAxis,
    stroke_aux: &StrokeAux,
    spray_constraint_plane: Option<(Vec3, Vec3)>,
) -> Vec<VoxelCoord> {
    let brush_radius = brush::brush_radius_for_area_polygon_stroke(stroke_mode, brush_radius);
    let clip_half = brush::brush_clip_half_normal_from_screen(
        stroke_aux.brush_clip_bottom_half,
        file,
        voxel_map,
        camera,
        width,
        height,
        sx,
        sy,
    );
    let is_spray_scatter = stroke_mode == DrawStrokeMode::Spray
        && (stroke_aux.spray_scatter > 0 || stroke_aux.spray_size_range);
    let effective_shape = if stroke_mode == DrawStrokeMode::Spray {
        stroke_aux.spray_brush_shape.unwrap_or(brush_shape)
    } else {
        brush_shape
    };
    let offsets = brush::brush_offset_cells(effective_shape, brush_radius, clip_half, None);
    let spray = spray_density.clamp(0.0, 1.0);
    let tool = EditTool::Add;
    let centers = stroke_anchor_centers_with_mode(
        stroke_mode,
        plane_axis,
        stroke_aux,
        tool,
        file,
        voxel_map,
        camera,
        width,
        height,
        sx,
        sy,
        brush_radius,
        stroke_line_start,
        stroke_segment_prev,
        spray_constraint_plane,
    );
    if centers.is_empty() {
        return Vec::new();
    }
    let grid_size = stroke_clip_grid_size(file, &centers, &offsets);

    let scatter = stroke_aux.spray_scatter;
    let size_range = stroke_aux.spray_size_range;
    let rmin = stroke_aux.spray_radius_min;
    let rmax = stroke_aux.spray_radius_max;

    let mut seen: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut out: Vec<VoxelCoord> = Vec::new();

    for (cx, cy, cz) in &centers {
        let (scx, scy, scz, cur_offsets);
        if is_spray_scatter {
            let ox = brush::spray_scatter_offset((*cx, *cy, *cz), scatter, 0);
            let oy = brush::spray_scatter_offset((*cx, *cy, *cz), scatter, 1);
            let oz = brush::spray_scatter_offset((*cx, *cy, *cz), scatter, 2);
            scx = cx + ox;
            scy = cy + oy;
            scz = cz + oz;
            if size_range && rmax > rmin {
                let r = brush::spray_random_radius((*cx, *cy, *cz), rmin, rmax);
                cur_offsets = brush::brush_offset_cells(effective_shape, r, clip_half, None);
            } else {
                cur_offsets = Vec::new();
            }
        } else {
            scx = *cx;
            scy = *cy;
            scz = *cz;
            cur_offsets = Vec::new();
        }
        let use_offsets = if !cur_offsets.is_empty() {
            &cur_offsets
        } else {
            &offsets
        };
        for (dx, dy, dz) in use_offsets {
            let x = scx + dx;
            let y = scy + dy;
            let z = scz + dz;
            if !in_grid(x, y, z, grid_size) {
                continue;
            }
            if !is_spray_scatter && !brush::spray_passes((x, y, z), spray) {
                continue;
            }
            if !seen.insert((x, y, z)) {
                continue;
            }
            if !voxel_map.contains_key(&(x, y, z)) {
                out.push((x, y, z));
            }
        }
    }
    out
}

// ── Delta application / undo-redo ────────────────────────────────────────────

pub fn apply_forward_delta(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    delta: &VoxelEditDelta,
) -> Result<(), String> {
    match *delta {
        VoxelEditDelta::Added(v) => {
            push_voxel_known(file, voxel_map, v);
        }
        VoxelEditDelta::Removed { voxel } => {
            remove_voxel_at(file, voxel_map, (voxel.x, voxel.y, voxel.z))
                .ok_or_else(|| "apply_forward: remove".to_string())?;
        }
        VoxelEditDelta::Painted { before: _, after } => {
            let Some(&idx) = voxel_map.get(&(after.x, after.y, after.z)) else {
                return Err("apply_forward: paint".into());
            };
            if file.voxels[idx].x != after.x
                || file.voxels[idx].y != after.y
                || file.voxels[idx].z != after.z
            {
                return Err("apply_forward: paint idx".into());
            }
            file.voxels[idx] = after;
        }
    }
    Ok(())
}

pub fn apply_inverse_delta(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    delta: &VoxelEditDelta,
) -> Result<(), String> {
    match *delta {
        VoxelEditDelta::Added(v) => {
            remove_voxel_at(file, voxel_map, (v.x, v.y, v.z))
                .ok_or_else(|| "inverse: add".to_string())?;
        }
        VoxelEditDelta::Removed { voxel } => {
            push_voxel_known(file, voxel_map, voxel);
        }
        VoxelEditDelta::Painted { before, after: _ } => {
            let Some(&idx) = voxel_map.get(&(before.x, before.y, before.z)) else {
                return Err("inverse: paint".into());
            };
            file.voxels[idx] = before;
        }
    }
    Ok(())
}

/// Spatial/GPU delta after `apply_inverse_delta(forward)` (file now matches pre-forward state for that op).
pub fn mesh_delta_after_inverse_of(forward: &VoxelEditDelta) -> VoxelEditDelta {
    match *forward {
        VoxelEditDelta::Added(v) => VoxelEditDelta::Removed { voxel: v },
        VoxelEditDelta::Removed { voxel } => VoxelEditDelta::Added(voxel),
        VoxelEditDelta::Painted { before, after } => VoxelEditDelta::Painted {
            before: after,
            after: before,
        },
    }
}

/// Swap-remove a voxel at `coord`. Returns the removed voxel if present.
pub fn remove_voxel_at(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    coord: VoxelCoord,
) -> Option<Voxel> {
    let remove_idx = *voxel_map.get(&coord)?;
    let removed_voxel = file.voxels[remove_idx];
    let last = file.voxels.len() - 1;
    if remove_idx != last {
        file.voxels.swap(remove_idx, last);
        let moved = file.voxels[remove_idx];
        voxel_map.insert((moved.x, moved.y, moved.z), remove_idx);
    }
    file.voxels.pop();
    voxel_map.remove(&coord);
    Some(removed_voxel)
}

/// Remove solid voxels at the given coordinates. Skips empty cells.
/// Web parity: `deleteSelectedVoxels` — selection set is unchanged.
pub fn remove_voxels_at_coords(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    coords: impl IntoIterator<Item = VoxelCoord>,
) -> Vec<VoxelEditDelta> {
    let mut out = Vec::new();
    for c in coords {
        if let Some(v) = remove_voxel_at(file, voxel_map, c) {
            out.push(VoxelEditDelta::Removed { voxel: v });
        }
    }
    out
}

pub fn push_voxel_known(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    v: Voxel,
) {
    let idx = file.voxels.len();
    file.voxels.push(v);
    voxel_map.insert((v.x, v.y, v.z), idx);
}

// ── Clipboard / stamp ────────────────────────────────────────────────────────

/// Relative offsets + appearance (matches web `VoxelleClipboard` semantics: bbox min at anchor).
#[derive(Clone, Debug)]
pub struct StampClipboard {
    pub entries: Vec<(i32, i32, i32, u32, MaterialId)>,
}

/// All voxel cells matching `color`; optionally also `material`.
pub fn coords_matching_color(
    file: &VoxelleFile,
    color: u32,
    match_material: bool,
    material: MaterialId,
) -> Vec<VoxelCoord> {
    file.voxels
        .iter()
        .filter(|v| v.color == color && (!match_material || v.material == material))
        .map(|v| (v.x, v.y, v.z))
        .collect()
}

fn rotate_x(x: f64, y: f64, z: f64, a: f64) -> (f64, f64, f64) {
    let (s, c) = a.sin_cos();
    (x, c * y - s * z, s * y + c * z)
}

fn rotate_y(x: f64, y: f64, z: f64, a: f64) -> (f64, f64, f64) {
    let (s, c) = a.sin_cos();
    (c * x + s * z, y, -s * x + c * z)
}

fn rotate_z(x: f64, y: f64, z: f64, a: f64) -> (f64, f64, f64) {
    let (s, c) = a.sin_cos();
    (c * x - s * y, s * x + c * y, z)
}

/// Rotate stamp entries around their bounding-box center by Euler XYZ angles (degrees).
fn apply_stamp_rotation(
    entries: &[(i32, i32, i32, u32, MaterialId)],
    rot_x_deg: f32,
    rot_y_deg: f32,
    rot_z_deg: f32,
) -> Vec<(i32, i32, i32, u32, MaterialId)> {
    if rot_x_deg == 0.0 && rot_y_deg == 0.0 && rot_z_deg == 0.0 {
        return entries.to_vec();
    }
    let mut min_x = i32::MAX;
    let mut max_x = i32::MIN;
    let mut min_y = i32::MAX;
    let mut max_y = i32::MIN;
    let mut min_z = i32::MAX;
    let mut max_z = i32::MIN;
    for &(dx, dy, dz, _, _) in entries {
        min_x = min_x.min(dx);
        max_x = max_x.max(dx);
        min_y = min_y.min(dy);
        max_y = max_y.max(dy);
        min_z = min_z.min(dz);
        max_z = max_z.max(dz);
    }
    let cx = (min_x + max_x) as f64 / 2.0;
    let cy = (min_y + max_y) as f64 / 2.0;
    let cz = (min_z + max_z) as f64 / 2.0;
    let rx = rot_x_deg as f64 * std::f64::consts::PI / 180.0;
    let ry = rot_y_deg as f64 * std::f64::consts::PI / 180.0;
    let rz = rot_z_deg as f64 * std::f64::consts::PI / 180.0;
    entries
        .iter()
        .map(|&(dx, dy, dz, color, mat)| {
            let (mut x, mut y, mut z) = (dx as f64 - cx, dy as f64 - cy, dz as f64 - cz);
            (x, y, z) = rotate_x(x, y, z, rx);
            (x, y, z) = rotate_y(x, y, z, ry);
            (x, y, z) = rotate_z(x, y, z, rz);
            let rdx = (x + cx).round() as i32;
            let rdy = (y + cy).round() as i32;
            let rdz = (z + cz).round() as i32;
            (rdx, rdy, rdz, color, mat)
        })
        .collect()
}

/// Compute X/Z anchor offsets for stamp origin placement.
/// `origin_x` / `origin_z`: 0 = min edge, 1 = center, 2 = max edge.
/// Returns `(off_x, off_z)` to subtract from each entry's dx/dz before adding the anchor.
pub fn stamp_origin_offsets_pub(
    entries: &[(i32, i32, i32, u32, MaterialId)],
    origin_x: i32,
    origin_z: i32,
) -> (i32, i32) {
    stamp_origin_offsets(entries, origin_x, origin_z)
}

fn stamp_origin_offsets(
    entries: &[(i32, i32, i32, u32, MaterialId)],
    origin_x: i32,
    origin_z: i32,
) -> (i32, i32) {
    let min_x = entries.iter().map(|e| e.0).min().unwrap_or(0);
    let max_x = entries.iter().map(|e| e.0).max().unwrap_or(0);
    let min_z = entries.iter().map(|e| e.2).min().unwrap_or(0);
    let max_z = entries.iter().map(|e| e.2).max().unwrap_or(0);
    let off_x = match origin_x {
        1 => (min_x + max_x) / 2,
        2 => max_x,
        _ => min_x,
    };
    let off_z = match origin_z {
        1 => (min_z + max_z) / 2,
        2 => max_z,
        _ => min_z,
    };
    (off_x, off_z)
}

/// Stamp pattern at the add-tool anchor (empty cell in front of first solid).
pub fn stamp_clipboard_at_screen(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    clip: &StampClipboard,
    rot_x: f32,
    rot_y: f32,
    rot_z: f32,
    origin_x: i32,
    origin_z: i32,
) -> Result<Vec<VoxelEditDelta>, String> {
    let grid_size = effective_ray_grid_size(file);
    let (origin, dir) = raycasting::screen_to_world_ray(camera, width, height, sx, sy);
    let Some((_, prev, _oid)) =
        raycasting::ray_first_solid_scene(origin, dir, file, voxel_map, grid_size)
    else {
        return Ok(Vec::new());
    };
    let Some((ax, ay, az)) = prev else {
        return Ok(Vec::new());
    };
    let rotated = apply_stamp_rotation(&clip.entries, rot_x, rot_y, rot_z);
    let (off_x, off_z) = stamp_origin_offsets(&rotated, origin_x, origin_z);
    ensure_grid_fits_coords(
        file,
        rotated
            .iter()
            .map(|e| (ax + e.0 - off_x, ay + e.1, az + e.2 - off_z)),
    );
    let grid_size = file.grid_size.max(1);
    let mut seen: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut out: Vec<VoxelEditDelta> = Vec::new();
    for &(dx, dy, dz, src_color, src_mat) in &rotated {
        let x = ax + dx - off_x;
        let y = ay + dy;
        let z = az + dz - off_z;
        if !in_grid(x, y, z, grid_size) {
            continue;
        }
        if !seen.insert((x, y, z)) {
            continue;
        }
        if voxel_map.contains_key(&(x, y, z)) {
            continue;
        }
        let nv = Voxel {
            x,
            y,
            z,
            color: src_color,
            material: src_mat,
            object_id: file.active_object_id,
        };
        let idx = file.voxels.len();
        file.voxels.push(nv);
        voxel_map.insert((x, y, z), idx);
        out.push(VoxelEditDelta::Added(nv));
    }
    Ok(out)
}

/// Remove voxels in the clipboard shape; hit cell is the origin (same as bbox min when copied).
pub fn punch_clipboard_at_screen(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    clip: &StampClipboard,
    rot_x: f32,
    rot_y: f32,
    rot_z: f32,
    origin_x: i32,
    origin_z: i32,
) -> Result<Vec<VoxelEditDelta>, String> {
    let grid_size = effective_ray_grid_size(file);
    let (origin, dir) = raycasting::screen_to_world_ray(camera, width, height, sx, sy);
    let Some((hit, _, _oid)) =
        raycasting::ray_first_solid_scene(origin, dir, file, voxel_map, grid_size)
    else {
        return Ok(Vec::new());
    };
    let (hx, hy, hz) = hit;
    let rotated = apply_stamp_rotation(&clip.entries, rot_x, rot_y, rot_z);
    let (off_x, off_z) = stamp_origin_offsets(&rotated, origin_x, origin_z);
    let mut seen: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut out: Vec<VoxelEditDelta> = Vec::new();
    for &(dx, dy, dz, _, _) in &rotated {
        let x = hx + dx - off_x;
        let y = hy + dy;
        let z = hz + dz - off_z;
        if !seen.insert((x, y, z)) {
            continue;
        }
        let Some(&remove_idx) = voxel_map.get(&(x, y, z)) else {
            continue;
        };
        let removed_voxel = file.voxels[remove_idx];
        let last = file.voxels.len() - 1;
        if remove_idx != last {
            file.voxels.swap(remove_idx, last);
            let moved = file.voxels[remove_idx];
            voxel_map.insert((moved.x, moved.y, moved.z), remove_idx);
        }
        file.voxels.pop();
        voxel_map.remove(&(x, y, z));
        out.push(VoxelEditDelta::Removed {
            voxel: removed_voxel,
        });
    }
    Ok(out)
}

pub fn selection_to_clipboard(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    selection: &AHashSet<VoxelCoord>,
) -> Option<StampClipboard> {
    if selection.is_empty() {
        return None;
    }
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut min_z = i32::MAX;
    for &(x, y, z) in selection.iter() {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        min_z = min_z.min(z);
    }
    let mut entries = Vec::new();
    for &(x, y, z) in selection.iter() {
        let Some(&idx) = voxel_map.get(&(x, y, z)) else {
            continue;
        };
        let v = file.voxels[idx];
        entries.push((x - min_x, y - min_y, z - min_z, v.color, v.material));
    }
    if entries.is_empty() {
        return None;
    }
    Some(StampClipboard { entries })
}

/// One-voxel "raise" sculpt: add above the top face of the ray-hit solid (if empty).
pub fn sculpt_raise_at_screen(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    color: u32,
    material: MaterialId,
) -> Result<Vec<VoxelEditDelta>, String> {
    let grid_size = effective_ray_grid_size(file);
    let (origin, dir) = raycasting::screen_to_world_ray(camera, width, height, sx, sy);
    let Some((hit, _, _oid)) =
        raycasting::ray_first_solid_scene(origin, dir, file, voxel_map, grid_size)
    else {
        return Ok(Vec::new());
    };
    let (hx, hy, hz) = hit;
    let x = hx;
    let y = hy + 1;
    let z = hz;
    ensure_grid_fits_coord(file, x, y, z);
    if voxel_map.contains_key(&(x, y, z)) {
        return Ok(Vec::new());
    }
    let nv = Voxel {
        x,
        y,
        z,
        color,
        material,
        object_id: file.active_object_id,
    };
    let idx = file.voxels.len();
    file.voxels.push(nv);
    voxel_map.insert((x, y, z), idx);
    Ok(vec![VoxelEditDelta::Added(nv)])
}

// ── Sculpt strokes ────────────────────────────────────────────────────────────

fn smoothstep01(t: f32) -> f32 {
    let x = t.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// XZ distance falloff to nearest spine sample (column centers), matching web terrain brush.
fn terrain_brush_falloff(x: i32, z: i32, spine: &[(i32, i32, i32)], brush_radius_vox: f32) -> f32 {
    let mut d_min = f32::INFINITY;
    for (px, _, pz) in spine {
        let d = (((x - px).pow(2) + (z - pz).pow(2)) as f32).sqrt();
        if d < d_min {
            d_min = d;
        }
    }
    if !d_min.is_finite() {
        return 0.0;
    }
    let r = brush_radius_vox.max(0.25) + 0.25;
    let u = (d_min / r).clamp(0.0, 1.0);
    1.0 - smoothstep01(u)
}

fn column_top_bottom(
    voxel_map: &AHashMap<VoxelCoord, usize>,
    grid_size: i32,
    x: i32,
    z: i32,
) -> Option<(i32, i32)> {
    let (y_lo, y_hi) = grid_valid_range(grid_size);
    let mut max_y: Option<i32> = None;
    for y in (y_lo..=y_hi).rev() {
        if voxel_map.contains_key(&(x, y, z)) {
            max_y = Some(y);
            break;
        }
    }
    let max_y = max_y?;
    let mut min_y = y_lo;
    for y in y_lo..=max_y {
        if voxel_map.contains_key(&(x, y, z)) {
            min_y = y;
            break;
        }
    }
    Some((min_y, max_y))
}

fn column_max_y(
    voxel_map: &AHashMap<VoxelCoord, usize>,
    grid_size: i32,
    x: i32,
    z: i32,
) -> Option<i32> {
    let (y_lo, y_hi) = grid_valid_range(grid_size);
    (y_lo..=y_hi)
        .rev()
        .find(|&y| voxel_map.contains_key(&(x, y, z)))
}

/// Particle-based hydraulic erosion on the local heightfield patch.
///
/// Runs `strength * 4` particles. Each particle flows downhill, eroding material and depositing
/// sediment. Results are written into `new_heights` for the inner `cols` only.
fn apply_terrain_erode(
    col_meta: &[(i32, i32, i32, i32, Voxel)],
    cols: &[(i32, i32)],
    spine: &[(i32, i32, i32)],
    brush_r_vox: f32,
    strength: i32,
    stroke_seed: u32,
    grid_size: i32,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    base_y: i32,
    new_heights: &mut [i32],
) {
    const ERODE_K: f32 = 0.35;
    const DEPOSIT_K: f32 = 0.25;
    const MAX_STEPS: i32 = 32;

    // Build working height buffer from col_meta
    let mut heights: AHashMap<(i32, i32), f32> = AHashMap::with_capacity(col_meta.len() + 32);
    let mut y_fills: AHashMap<(i32, i32), i32> = AHashMap::with_capacity(col_meta.len() + 32);
    for meta in col_meta {
        heights.insert((meta.0, meta.1), meta.3 as f32);
        y_fills.insert((meta.0, meta.1), meta.2);
    }

    // Populate 1-cell margin neighbours as read-only gradient helpers
    let col_set: AHashSet<(i32, i32)> = cols.iter().copied().collect();
    for &(x, z) in cols {
        for (dx, dz) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
            let nx = x + dx;
            let nz = z + dz;
            if col_set.contains(&(nx, nz)) || !in_grid(nx, base_y, nz, grid_size) {
                continue;
            }
            heights.entry((nx, nz)).or_insert_with(|| {
                column_max_y(voxel_map, grid_size, nx, nz).unwrap_or(base_y - 1) as f32
            });
            y_fills.entry((nx, nz)).or_insert(base_y);
        }
    }

    // Precompute brush falloff for inner columns
    let mut falloffs: AHashMap<(i32, i32), f32> = AHashMap::with_capacity(cols.len());
    for &(x, z) in cols {
        falloffs.insert((x, z), terrain_brush_falloff(x, z, spine, brush_r_vox));
    }

    let n_particles = ((strength * 4) as u32).clamp(1, 512);
    let col_count = cols.len();
    if col_count == 0 {
        return;
    }

    for p in 0..n_particles {
        // Inline Mulberry32 RNG, seeded deterministically per-particle
        let mut rng_state = stroke_seed.wrapping_add(p.wrapping_mul(2_654_435_761));
        let mut rng_next = || -> f32 {
            rng_state = rng_state.wrapping_add(0x6D2B79F5);
            let mut t = (rng_state as u64).wrapping_mul((rng_state ^ (rng_state >> 15)) as u64);
            t = (t & 0xFFFF_FFFF) ^ (t >> 16);
            (t as u32 as f32) / (u32::MAX as f32)
        };

        // Random start position within the brush footprint
        let idx = (rng_next() * col_count as f32) as usize % col_count;
        let (mut px, mut pz) = cols[idx];
        let mut sediment: f32 = 0.0;

        for _ in 0..MAX_STEPS {
            let cur_h = *heights.get(&(px, pz)).unwrap_or(&(base_y as f32 - 1.0));
            let cur_y_fill = *y_fills.get(&(px, pz)).unwrap_or(&base_y);
            let falloff_t = *falloffs.get(&(px, pz)).unwrap_or(&0.0);

            // Find steepest descent among 4 neighbours
            let mut best_h = cur_h;
            let mut best_next: Option<(i32, i32)> = None;
            for (dx, dz) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let nx = px + dx;
                let nz = pz + dz;
                if let Some(&nh) = heights.get(&(nx, nz)) {
                    if nh < best_h {
                        best_h = nh;
                        best_next = Some((nx, nz));
                    }
                }
            }

            let slope = (cur_h - best_h).max(0.0);

            // Erode from current position
            let e = (ERODE_K * slope * falloff_t)
                .min(cur_h - (cur_y_fill as f32 - 1.0))
                .max(0.0);
            if let Some(h) = heights.get_mut(&(px, pz)) {
                *h -= e;
            }
            sediment += e;

            // Deposit some sediment here if slope is gentle
            let deposit = (sediment * DEPOSIT_K * (1.0 - slope.min(1.0))).max(0.0);
            if col_set.contains(&(px, pz)) {
                if let Some(h) = heights.get_mut(&(px, pz)) {
                    *h += deposit;
                }
                sediment -= deposit;
            }

            // Move downhill, or settle if no lower neighbour found
            match best_next {
                Some(np) => {
                    px = np.0;
                    pz = np.1;
                }
                None => break,
            }
        }

        // Deposit all remaining sediment at final resting position
        if col_set.contains(&(px, pz)) {
            if let Some(h) = heights.get_mut(&(px, pz)) {
                *h += sediment;
            }
        }
    }

    // Write results back — inner columns only
    let (_, y_hi) = grid_valid_range(grid_size);
    for (i, &(x, z)) in cols.iter().enumerate() {
        let y_fill = col_meta[i].2;
        let h = heights.get(&(x, z)).copied().unwrap_or(y_fill as f32 - 1.0);
        new_heights[i] = (h.round() as i32).clamp(y_fill - 1, y_hi);
    }
}

/// Heightfield terrain: columns listed in `cols` are rebuilt from `y_fill` through target height.
fn apply_terrain_sculpt(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    grid_size: i32,
    cols: &[(i32, i32)],
    col_meta: &[(i32, i32, i32, i32, Voxel)], // x, z, y_fill, old_max, template voxel appearance
    new_heights: &[i32],
) -> Vec<VoxelEditDelta> {
    let mut out: Vec<VoxelEditDelta> = Vec::new();
    let (y_lo, y_hi) = grid_valid_range(grid_size);
    for (i, &(x, z)) in cols.iter().enumerate() {
        let h = new_heights[i];
        let (y_fill, _old_max, template) = {
            let m = &col_meta[i];
            (m.2, m.3, m.4)
        };
        for y in y_lo..=y_hi {
            let want = h >= y_fill && y >= y_fill && y <= h;
            let had = voxel_map.contains_key(&(x, y, z));
            if had && !want {
                let Some(&remove_idx) = voxel_map.get(&(x, y, z)) else {
                    continue;
                };
                let removed_voxel = file.voxels[remove_idx];
                let last = file.voxels.len() - 1;
                if remove_idx != last {
                    file.voxels.swap(remove_idx, last);
                    let moved = file.voxels[remove_idx];
                    voxel_map.insert((moved.x, moved.y, moved.z), remove_idx);
                }
                file.voxels.pop();
                voxel_map.remove(&(x, y, z));
                out.push(VoxelEditDelta::Removed {
                    voxel: removed_voxel,
                });
            } else if want {
                let key = (x, y, z);
                if let Some(&idx) = voxel_map.get(&key) {
                    let before = file.voxels[idx];
                    if before.color != template.color || before.material != template.material {
                        let after = Voxel {
                            color: template.color,
                            material: template.material,
                            ..before
                        };
                        file.voxels[idx] = after;
                        out.push(VoxelEditDelta::Painted { before, after });
                    }
                } else {
                    let nv = Voxel {
                        x,
                        y,
                        z,
                        color: template.color,
                        material: template.material,
                        object_id: file.active_object_id,
                    };
                    let idx = file.voxels.len();
                    file.voxels.push(nv);
                    voxel_map.insert(key, idx);
                    out.push(VoxelEditDelta::Added(nv));
                }
            }
        }
    }
    out
}

/// Brush footprint for one sculpt stroke sample (spine + brush offsets + spray), no edits.
pub fn collect_sculpt_stroke_footprint(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    mode: SculptStrokeMode,
    brush_radius: u32,
    brush_shape: BrushShape,
    spray_density: f32,
    stroke_line_start: Option<(f32, f32)>,
    stroke_segment_prev: Option<(f32, f32)>,
    brush_clip_bottom_half: bool,
    // For Draw mode: screen position of the initial click, used to lock the face normal
    // so it doesn't change as the cursor moves over different faces during drag.
    draw_normal_pos: Option<(f32, f32)>,
) -> Vec<VoxelCoord> {
    // For Draw mode, all normal lookups use the locked click position so brush orientation
    // stays constant across the whole stroke regardless of where the cursor moves.
    let (normal_sx, normal_sy) = if mode == SculptStrokeMode::Draw {
        draw_normal_pos.unwrap_or((sx, sy))
    } else {
        (sx, sy)
    };
    let clip_half = brush::brush_clip_half_normal_from_screen(
        brush_clip_bottom_half,
        file,
        voxel_map,
        camera,
        width,
        height,
        normal_sx,
        normal_sy,
    );
    // For 2D shapes (Square/Circle), determine the face-normal axis so offsets stay in the tangent plane.
    let face_axis = if matches!(brush_shape, BrushShape::Square | BrushShape::Circle) {
        raycasting::outward_face_normal_from_screen_ray(file, voxel_map, camera, width, height, normal_sx, normal_sy)
            .map(brush::face_normal_to_axis)
    } else {
        None
    };
    let offsets = brush::brush_offset_cells(brush_shape, brush_radius, clip_half, face_axis);
    let spray = spray_density.clamp(0.0, 1.0);

    let mut spine = brush::stroke_anchor_centers_sculpt(
        mode,
        file,
        voxel_map,
        camera,
        width,
        height,
        sx,
        sy,
        stroke_line_start,
        stroke_segment_prev,
    );
    if spine.is_empty() {
        return Vec::new();
    }
    // Draw/Extrude spine sits on the solid surface (EditTool::Remove). Nudge it one step
    // along the outward face normal so the brush footprint layers on top of existing voxels.
    // Draw uses normal_sx/normal_sy (locked click position) so the direction stays constant.
    if matches!(mode, SculptStrokeMode::Draw | SculptStrokeMode::Extrude) {
        let (nsx, nsy) = if mode == SculptStrokeMode::Draw { (normal_sx, normal_sy) } else { (sx, sy) };
        if let Some(n) =
            raycasting::outward_face_normal_from_screen_ray(file, voxel_map, camera, width, height, nsx, nsy)
        {
            for c in spine.iter_mut() {
                c.0 += n.0;
                c.1 += n.1;
                c.2 += n.2;
            }
        }
    }
    let grid_size = stroke_clip_grid_size(file, &spine, &offsets);

    let mut seen: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut footprint: Vec<VoxelCoord> = Vec::new();
    for (cx, cy, cz) in &spine {
        for (dx, dy, dz) in &offsets {
            let x = cx + dx;
            let y = cy + dy;
            let z = cz + dz;
            if !in_grid(x, y, z, grid_size) {
                continue;
            }
            if !brush::spray_passes((x, y, z), spray) {
                continue;
            }
            if seen.insert((x, y, z)) {
                footprint.push((x, y, z));
            }
        }
    }
    footprint
}

/// Brush footprint after web-style strength / falloff thinning. Terrain skips thinning (column ops).
/// For Extrude with cylinder profile, uses the cylinder/capsule/taper geometry instead of brush offsets.
pub fn sculpt_stroke_effective_footprint(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    mode: SculptStrokeMode,
    brush_radius: u32,
    brush_shape: BrushShape,
    spray_density: f32,
    stroke_line_start: Option<(f32, f32)>,
    stroke_segment_prev: Option<(f32, f32)>,
    brush_strength: u32,
    brush_falloff: u32,
    stroke_seed: u32,
    brush_clip_bottom_half: bool,
    extrude_profile: ExtrudeProfile,
    extrude_end_cap: ExtrudeEndCap,
    extrude_taper: bool,
    extrude_taper_start: f32,
    extrude_taper_end: f32,
    draw_normal_pos: Option<(f32, f32)>,
) -> Vec<VoxelCoord> {
    // Extrude + cylinder: use dedicated geometry instead of generic brush offsets
    if mode == SculptStrokeMode::Extrude && extrude_profile == ExtrudeProfile::Cylinder {
        let spine = brush::stroke_anchor_centers_sculpt(
            mode,
            file,
            voxel_map,
            camera,
            width,
            height,
            sx,
            sy,
            stroke_line_start,
            stroke_segment_prev,
        );
        if spine.is_empty() {
            return Vec::new();
        }
        let r = (brush_radius + 1) as f32 / 2.0;
        let footprint = if extrude_taper {
            let start_r = extrude_taper_start.max(0.0);
            let end_r = extrude_taper_end.max(0.0);
            brush::extrude_tapered_cylinder_footprint(&spine, start_r, end_r, extrude_end_cap)
        } else {
            brush::extrude_uniform_cylinder_footprint(&spine, r, extrude_end_cap)
        };
        return brush::filter_sculpt_footprint_stochastic(
            footprint,
            &spine,
            brush_radius,
            brush_falloff,
            brush_strength,
            stroke_seed,
        );
    }

    let footprint = collect_sculpt_stroke_footprint(
        file,
        voxel_map,
        camera,
        width,
        height,
        sx,
        sy,
        mode,
        brush_radius,
        brush_shape,
        spray_density,
        stroke_line_start,
        stroke_segment_prev,
        brush_clip_bottom_half,
        draw_normal_pos,
    );
    if footprint.is_empty() {
        return footprint;
    }
    if matches!(mode, SculptStrokeMode::Terrain) {
        return footprint;
    }
    let spine = brush::stroke_anchor_centers_sculpt(
        mode,
        file,
        voxel_map,
        camera,
        width,
        height,
        sx,
        sy,
        stroke_line_start,
        stroke_segment_prev,
    );
    brush::filter_sculpt_footprint_stochastic(
        footprint,
        &spine,
        brush_radius,
        brush_falloff,
        brush_strength,
        stroke_seed,
    )
}

fn polygon_closed_outline(points: &[VoxelCoord]) -> Vec<VoxelCoord> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![points[0]];
    }
    if n == 2 {
        return raycasting::voxel_line_dda(points[0], points[1]);
    }
    let mut seen: HashSet<VoxelCoord> = HashSet::new();
    let mut out = Vec::new();
    for i in 0..n {
        let a = points[i];
        let b = points[(i + 1) % n];
        for p in raycasting::voxel_line_dda(a, b) {
            if seen.insert(p) {
                out.push(p);
            }
        }
    }
    out
}

fn axis_aligned_circle_filled_disk(
    center: VoxelCoord,
    edge: VoxelCoord,
    face_n: (i32, i32, i32),
) -> Vec<VoxelCoord> {
    let ax = face_n.0.abs();
    let ay = face_n.1.abs();
    let az = face_n.2.abs();
    let fixed_axis = if ax >= ay && ax >= az {
        0usize
    } else if ay >= az {
        1usize
    } else {
        2usize
    };

    let (cu, cv, eu, ev) = match fixed_axis {
        0 => (center.1, center.2, edge.1, edge.2),
        1 => (center.0, center.2, edge.0, edge.2),
        _ => (center.0, center.1, edge.0, edge.1),
    };
    let du = eu - cu;
    let dv = ev - cv;
    let r_sq = du * du + dv * dv;
    if r_sq == 0 {
        return match fixed_axis {
            0 => vec![(center.0, cu, cv)],
            1 => vec![(cu, center.1, cv)],
            _ => vec![(cu, cv, center.2)],
        };
    }
    let ru = (r_sq as f64).sqrt().ceil() as i32;
    let mut filled = Vec::new();
    for u in (cu - ru)..=(cu + ru) {
        for v in (cv - ru)..=(cv + ru) {
            let ddu = u - cu;
            let ddv = v - cv;
            if ddu * ddu + ddv * ddv <= r_sq {
                let p = match fixed_axis {
                    0 => (center.0, u, v),
                    1 => (u, center.1, v),
                    _ => (u, v, center.2),
                };
                filled.push(p);
            }
        }
    }
    filled
}

fn wall_circle_disk_spine(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx_center: f32,
    sy_center: f32,
    sx_edge: f32,
    sy_edge: f32,
) -> Option<Vec<VoxelCoord>> {
    let grid_size = effective_ray_grid_size(file);
    let (o1, d1) = raycasting::screen_to_world_ray(camera, width, height, sx_center, sy_center);
    let (o2, d2) = raycasting::screen_to_world_ray(camera, width, height, sx_edge, sy_edge);
    let (hit1, _, _) = raycasting::ray_first_solid_scene(o1, d1, file, voxel_map, grid_size)?;
    let (hit2, _, _) = raycasting::ray_first_solid_scene(o2, d2, file, voxel_map, grid_size)?;
    let n = raycasting::outward_face_normal_from_screen_ray(
        file, voxel_map, camera, width, height, sx_edge, sy_edge,
    )?;
    Some(axis_aligned_circle_filled_disk(hit1, hit2, n))
}

/// Web `thickenPathForStroke` wall path + strength/falloff.
pub fn compute_wall_sculpt_footprint(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    stroke_line_start: Option<(f32, f32)>,
    stroke_segment_prev: Option<(f32, f32)>,
    wall_area_shape: WallAreaShape,
    wall_polygon_vertices: Option<&[VoxelCoord]>,
    spray_direction: SprayDirection,
    wall_width_index: u32,
    wall_height_vox: u32,
    wall_lock_start_height: bool,
    wall_axis_align: bool,
    brush_radius: u32,
    brush_falloff: u32,
    brush_strength: u32,
    stroke_seed: u32,
    // Pre-locked face normal from the start of the stroke. `Some(v)` overrides the per-frame
    // ray cast so the wall orientation stays constant across the entire drag.
    locked_face_snapped: Option<Option<(i32, i32, i32)>>,
) -> Vec<VoxelCoord> {
    let mut spine = match wall_area_shape {
        WallAreaShape::Polygon => {
            if let Some(corners) = wall_polygon_vertices {
                if corners.len() >= 2 {
                    let o = polygon_closed_outline(corners);
                    if o.is_empty() {
                        brush::stroke_anchor_centers_sculpt(
                            SculptStrokeMode::Wall,
                            file,
                            voxel_map,
                            camera,
                            width,
                            height,
                            sx,
                            sy,
                            stroke_line_start,
                            stroke_segment_prev,
                        )
                    } else {
                        o
                    }
                } else {
                    brush::stroke_anchor_centers_sculpt(
                        SculptStrokeMode::Wall,
                        file,
                        voxel_map,
                        camera,
                        width,
                        height,
                        sx,
                        sy,
                        stroke_line_start,
                        stroke_segment_prev,
                    )
                }
            } else {
                brush::stroke_anchor_centers_sculpt(
                    SculptStrokeMode::Wall,
                    file,
                    voxel_map,
                    camera,
                    width,
                    height,
                    sx,
                    sy,
                    stroke_line_start,
                    stroke_segment_prev,
                )
            }
        }
        WallAreaShape::Circle => {
            if let Some((lsx, lsy)) = stroke_line_start {
                if let Some(disk) =
                    wall_circle_disk_spine(file, voxel_map, camera, width, height, lsx, lsy, sx, sy)
                {
                    if disk.is_empty() {
                        brush::stroke_anchor_centers_sculpt(
                            SculptStrokeMode::Wall,
                            file,
                            voxel_map,
                            camera,
                            width,
                            height,
                            sx,
                            sy,
                            stroke_line_start,
                            stroke_segment_prev,
                        )
                    } else {
                        disk
                    }
                } else {
                    brush::stroke_anchor_centers_sculpt(
                        SculptStrokeMode::Wall,
                        file,
                        voxel_map,
                        camera,
                        width,
                        height,
                        sx,
                        sy,
                        stroke_line_start,
                        stroke_segment_prev,
                    )
                }
            } else {
                brush::stroke_anchor_centers_sculpt(
                    SculptStrokeMode::Wall,
                    file,
                    voxel_map,
                    camera,
                    width,
                    height,
                    sx,
                    sy,
                    stroke_line_start,
                    stroke_segment_prev,
                )
            }
        }
        WallAreaShape::Brush => brush::stroke_anchor_centers_sculpt(
            SculptStrokeMode::Wall,
            file,
            voxel_map,
            camera,
            width,
            height,
            sx,
            sy,
            stroke_line_start,
            stroke_segment_prev,
        ),
    };
    if spine.is_empty() {
        return Vec::new();
    }

    if wall_axis_align && spine.len() >= 2 {
        let a = spine[0];
        let b = spine[spine.len() - 1];
        spine = raycasting::voxel_line_dda(a, b);
    }

    let face_snapped = match locked_face_snapped {
        Some(v) => v,
        None => {
            let face_out =
                raycasting::outward_face_normal_from_screen_ray(file, voxel_map, camera, width, height, sx, sy);
            face_out.map(brush::snap_normal_to_axis)
        }
    };

    if wall_lock_start_height {
        if let Some(axis) = brush::wall_lock_axis(spray_direction, face_snapped) {
            let fixed = match axis {
                0 => spine[0].0,
                1 => spine[0].1,
                _ => spine[0].2,
            };
            for p in spine.iter_mut() {
                match axis {
                    0 => p.0 = fixed,
                    1 => p.1 = fixed,
                    _ => p.2 = fixed,
                }
            }
        }
    }

    let spine_for_weights = spine.clone();

    let wparam = if wall_width_index == 0 {
        0u32
    } else {
        wall_width_index.saturating_add(1)
    };

    let dir_vec = brush::spray_direction_vector(spray_direction, face_snapped);
    let dir_for_plane = dir_vec.unwrap_or((0, 1, 0));
    let plane_normal_axis = if dir_for_plane.0 != 0 {
        0usize
    } else if dir_for_plane.1 != 0 {
        1usize
    } else {
        2usize
    };

    let mut base_positions: Vec<(i32, i32, i32)> = spine;

    if wparam == 0 {
        // path only
    } else if wparam == 1 {
        let perp = brush::perpendicular_step_thick(dir_for_plane);
        let mut seen: HashSet<(i32, i32, i32)> = base_positions.iter().copied().collect();
        let mut extra = base_positions.clone();
        for &(px, py, pz) in &base_positions {
            let p = (px + perp.0, py + perp.1, pz + perp.2);
            if seen.insert(p) {
                extra.push(p);
            }
        }
        base_positions = extra;
    } else {
        let r = (wparam - 1) as f32 * 0.5;
        base_positions = brush::thicken_path_in_plane_wall(&base_positions, r, plane_normal_axis);
    }

    let h = wall_height_vox.max(2) as i32;
    let mut out = if let Some(dv) = dir_vec {
        brush::directional_streak_wall(&base_positions, dv, h)
    } else {
        base_positions
    };

    let grid_size = file
        .grid_size
        .max(1)
        .max(min_grid_size_for_coords(&out))
        .min(MAX_GRID_SIZE);
    out.retain(|&(x, y, z)| in_grid(x, y, z, grid_size));

    brush::filter_sculpt_footprint_stochastic(
        out,
        &spine_for_weights,
        brush_radius,
        brush_falloff,
        brush_strength,
        stroke_seed,
    )
}

/// Stroke-based sculpt: draw / gouge / smooth / wall / extrude behave like add/remove/smooth;
/// terrain uses column heightfield ops (web `applyTerrainStroke`).
pub fn apply_sculpt_stroke(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    mode: SculptStrokeMode,
    color: u32,
    material: MaterialId,
    brush_radius: u32,
    brush_shape: BrushShape,
    spray_density: f32,
    brush_clip_bottom_half: bool,
    stroke_line_start: Option<(f32, f32)>,
    stroke_segment_prev: Option<(f32, f32)>,
    terrain_op: Option<TerrainSculptOp>,
    terrain_base_y: i32,
    terrain_strength: i32,
    terrain_smooth_radius: i32,
    terrain_flatten_use_base_y: bool,
    terrain_sub_voxel: bool,
    terrain_accum: &mut AHashMap<(i32, i32), f32>,
    smooth_neighbor_passes: u32,
    brush_strength: u32,
    brush_falloff: u32,
    stroke_seed: u32,
    wall_area_shape: WallAreaShape,
    spray_direction: SprayDirection,
    wall_width_index: u32,
    wall_height_vox: u32,
    wall_lock_start_height: bool,
    wall_axis_align: bool,
    sculpt_smooth_variant: SculptSmoothVariant,
    smooth_neighbor_radius: u32,
    smooth_aggressiveness: u32,
    smooth_laplacian_iterations: u32,
    smooth_laplacian_relax_pct: u32,
    wall_polygon_vertices: Option<Vec<VoxelCoord>>,
    extrude_profile: ExtrudeProfile,
    extrude_end_cap: ExtrudeEndCap,
    extrude_taper: bool,
    extrude_taper_start: f32,
    extrude_taper_end: f32,
    draw_normal_pos: Option<(f32, f32)>,
) -> Result<Vec<VoxelEditDelta>, String> {
    let footprint = if mode == SculptStrokeMode::Wall {
        compute_wall_sculpt_footprint(
            file,
            voxel_map,
            camera,
            width,
            height,
            sx,
            sy,
            stroke_line_start,
            stroke_segment_prev,
            wall_area_shape,
            wall_polygon_vertices.as_deref(),
            spray_direction,
            wall_width_index,
            wall_height_vox,
            wall_lock_start_height,
            wall_axis_align,
            brush_radius,
            brush_falloff,
            brush_strength,
            stroke_seed,
            None,
        )
    } else {
        sculpt_stroke_effective_footprint(
            file,
            voxel_map,
            camera,
            width,
            height,
            sx,
            sy,
            mode,
            brush_radius,
            brush_shape,
            spray_density,
            stroke_line_start,
            stroke_segment_prev,
            brush_strength,
            brush_falloff,
            stroke_seed,
            brush_clip_bottom_half,
            extrude_profile,
            extrude_end_cap,
            extrude_taper,
            extrude_taper_start,
            extrude_taper_end,
            draw_normal_pos,
        )
    };
    if footprint.is_empty() {
        return Ok(Vec::new());
    }

    ensure_grid_fits_coords(file, footprint.iter().copied());
    let grid_size = file.grid_size.max(1);

    let spine = brush::stroke_anchor_centers_sculpt(
        mode,
        file,
        voxel_map,
        camera,
        width,
        height,
        sx,
        sy,
        stroke_line_start,
        stroke_segment_prev,
    );

    match mode {
        SculptStrokeMode::Draw | SculptStrokeMode::Wall | SculptStrokeMode::Extrude => {
            let mut out: Vec<VoxelEditDelta> = Vec::new();
            for (x, y, z) in footprint {
                if voxel_map.contains_key(&(x, y, z)) {
                    continue;
                }
                let nv = Voxel {
                    x,
                    y,
                    z,
                    color,
                    material,
                    object_id: file.active_object_id,
                };
                let idx = file.voxels.len();
                file.voxels.push(nv);
                voxel_map.insert((x, y, z), idx);
                out.push(VoxelEditDelta::Added(nv));
            }
            Ok(out)
        }
        SculptStrokeMode::Gouge => {
            let mut out: Vec<VoxelEditDelta> = Vec::new();
            for (x, y, z) in footprint {
                let Some(&remove_idx) = voxel_map.get(&(x, y, z)) else {
                    continue;
                };
                let removed_voxel = file.voxels[remove_idx];
                let last = file.voxels.len() - 1;
                if remove_idx != last {
                    file.voxels.swap(remove_idx, last);
                    let moved = file.voxels[remove_idx];
                    voxel_map.insert((moved.x, moved.y, moved.z), remove_idx);
                }
                file.voxels.pop();
                voxel_map.remove(&(x, y, z));
                out.push(VoxelEditDelta::Removed {
                    voxel: removed_voxel,
                });
            }
            Ok(out)
        }
        SculptStrokeMode::Smooth => {
            let mut seen_fp: AHashSet<VoxelCoord> = AHashSet::new();
            let mut deduped: Vec<VoxelCoord> = Vec::new();
            for &(x, y, z) in &footprint {
                if in_grid(x, y, z, grid_size) && seen_fp.insert((x, y, z)) {
                    deduped.push((x, y, z));
                }
            }
            if deduped.is_empty() {
                return Ok(Vec::new());
            }
            match sculpt_smooth_variant {
                SculptSmoothVariant::MeshLaplacian => {
                    let margin = (smooth_neighbor_radius as i32).min(6) + 2;
                    Ok(apply_sculpt_smooth_mesh_laplacian(
                        file,
                        voxel_map,
                        &deduped,
                        grid_size,
                        margin,
                        smooth_laplacian_iterations,
                        smooth_laplacian_relax_pct,
                        smooth_neighbor_radius,
                        smooth_aggressiveness,
                        color,
                        material,
                    ))
                }
                SculptSmoothVariant::Majority => {
                    let passes = smooth_neighbor_passes.max(1);
                    let mut out: Vec<VoxelEditDelta> = Vec::new();
                    for _ in 0..passes {
                        let pass_deltas = apply_sculpt_smooth_majority_pass(
                            file,
                            voxel_map,
                            &deduped,
                            grid_size,
                            smooth_neighbor_radius,
                            smooth_aggressiveness,
                            color,
                            material,
                        );
                        out.extend(pass_deltas);
                    }
                    Ok(out)
                }
            }
        }
        SculptStrokeMode::Terrain => {
            let op = terrain_op.unwrap_or(TerrainSculptOp::Raise);
            let base_y = terrain_base_y;
            // terrain_strength comes from sculptBrushStrength (0–100 percent).
            // Map to a voxel delta range of 1–10 so that 100% ≈ 10 voxels/sample.
            let strength = ((terrain_strength * 10 + 99) / 100).clamp(1, 10);
            let smooth_r = terrain_smooth_radius.clamp(0, 8);

            let mut xz_map: AHashMap<(i32, i32), (i32, i32)> = AHashMap::new();
            for (x, y, z) in &footprint {
                if in_grid(*x, *y, *z, grid_size) {
                    xz_map.entry((*x, *z)).or_insert((*x, *z));
                }
            }
            if xz_map.is_empty() {
                return Ok(Vec::new());
            }
            let cols: Vec<(i32, i32)> = xz_map.values().copied().collect();
            let brush_r_vox = (brush_radius + 1) as f32 / 2.0;

            let mut col_meta: Vec<(i32, i32, i32, i32, Voxel)> = Vec::new();
            for &(x, z) in &cols {
                let ext = column_top_bottom(voxel_map, grid_size, x, z);
                let y_fill = if let Some((min_y, _max_y)) = ext {
                    base_y.min(min_y)
                } else {
                    base_y
                };
                let old_max = ext.map(|(_, m)| m).unwrap_or(y_fill - 1);
                let template = if let Some(my) = column_max_y(voxel_map, grid_size, x, z) {
                    let idx = *voxel_map.get(&(x, my, z)).unwrap();
                    file.voxels[idx]
                } else {
                    Voxel {
                        x,
                        y: y_fill,
                        z,
                        color,
                        material,
                        object_id: file.active_object_id,
                    }
                };

                col_meta.push((x, z, y_fill, old_max, template));
            }

            let mut new_heights: Vec<i32> = vec![0; cols.len()];

            match op {
                TerrainSculptOp::Raise | TerrainSculptOp::Lower => {
                    for (i, &(x, z)) in cols.iter().enumerate() {
                        let meta = &col_meta[i];
                        let old_h = meta.3;
                        let y_fill = meta.2;
                        let t = terrain_brush_falloff(x, z, &spine, brush_r_vox);
                        let h = if terrain_sub_voxel {
                            // Accumulate fractional height changes; only commit whole voxels.
                            let raw = strength as f32 * t;
                            let acc = terrain_accum.entry((x, z)).or_insert(0.0);
                            *acc += raw;
                            let delta = (*acc).floor() as i32;
                            *acc -= delta as f32;
                            if matches!(op, TerrainSculptOp::Raise) {
                                old_h + delta
                            } else {
                                (old_h - delta).max(y_fill - 1)
                            }
                        } else {
                            let delta = (strength as f32 * t).round() as i32;
                            if matches!(op, TerrainSculptOp::Raise) {
                                old_h + delta
                            } else {
                                (old_h - delta).max(y_fill - 1)
                            }
                        };
                        new_heights[i] = h;
                    }
                }
                TerrainSculptOp::Smooth => {
                    let mut surface_cache: AHashMap<(i32, i32), i32> = AHashMap::new();
                    let mut surface_h = |sx: i32, sz: i32| -> i32 {
                        if let Some(&h) = surface_cache.get(&(sx, sz)) {
                            return h;
                        }
                        let h = column_max_y(voxel_map, grid_size, sx, sz).unwrap_or(base_y - 1);
                        surface_cache.insert((sx, sz), h);
                        h
                    };
                    for (i, &(x, z)) in cols.iter().enumerate() {
                        let mut sum: i32 = 0;
                        let mut cnt: i32 = 0;
                        for dz in -smooth_r..=smooth_r {
                            for dx in -smooth_r..=smooth_r {
                                let nx = x + dx;
                                let nz = z + dz;
                                if !in_grid(nx, base_y, nz, grid_size) {
                                    continue;
                                }
                                sum += surface_h(nx, nz);
                                cnt += 1;
                            }
                        }
                        let avg = if cnt > 0 { sum / cnt } else { surface_h(x, z) };
                        let meta = &col_meta[i];
                        let y_fill = meta.2;
                        new_heights[i] = avg.max(y_fill - 1);
                    }
                }
                TerrainSculptOp::Flatten => {
                    // Target Y: mean surface of non-empty columns, or explicit base_y.
                    let target_y: i32 = if terrain_flatten_use_base_y {
                        base_y
                    } else {
                        let mut sum = 0i64;
                        let mut cnt = 0i64;
                        for meta in &col_meta {
                            if meta.3 >= meta.2 {
                                sum += meta.3 as i64;
                                cnt += 1;
                            }
                        }
                        if cnt > 0 {
                            (sum / cnt) as i32
                        } else {
                            base_y
                        }
                    };
                    for (i, &(x, z)) in cols.iter().enumerate() {
                        let meta = &col_meta[i];
                        let old_h = meta.3;
                        let y_fill = meta.2;
                        let t = terrain_brush_falloff(x, z, &spine, brush_r_vox);
                        let new_h_f = old_h as f32 + (target_y - old_h) as f32 * t;
                        new_heights[i] = (new_h_f.round() as i32).max(y_fill - 1);
                    }
                }
                TerrainSculptOp::Erode => {
                    apply_terrain_erode(
                        &col_meta,
                        &cols,
                        &spine,
                        brush_r_vox,
                        strength,
                        stroke_seed,
                        grid_size,
                        voxel_map,
                        base_y,
                        &mut new_heights,
                    );
                }
            }

            Ok(apply_terrain_sculpt(
                file,
                voxel_map,
                grid_size,
                &cols,
                &col_meta,
                &new_heights,
            ))
        }
    }
}

// ── Selection transform ───────────────────────────────────────────────────────

/// Quarter-turn rotation of `(rx, ry, rz)` around the given axis.
/// `quarters` must be in `[1, 3]` (caller normalises with `rem_euclid(4)`).
fn rotate_rel_quarter(rx: f32, ry: f32, rz: f32, axis: u8, quarters: i32) -> (f32, f32, f32) {
    match (axis, quarters) {
        (0, 1) => (rx, -rz, ry),
        (0, 2) => (rx, -ry, -rz),
        (0, 3) => (rx, rz, -ry),
        (1, 1) => (rz, ry, -rx),
        (1, 2) => (-rx, ry, -rz),
        (1, 3) => (-rz, ry, rx),
        (2, 1) => (-ry, rx, rz),
        (2, 2) => (-rx, -ry, rz),
        (2, 3) => (ry, -rx, rz),
        _ => (rx, ry, rz),
    }
}

/// Translate all solid voxels in `selection` by `(dx, dy, dz)`.
/// Returns voxel-edit deltas (remove old + add new). Caller must update
/// `selection_cells` to the shifted positions.
pub fn translate_selected_voxels(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    selection: &AHashSet<VoxelCoord>,
    dx: i32,
    dy: i32,
    dz: i32,
) -> Vec<VoxelEditDelta> {
    // Snapshot data before any mutation.
    let to_move: Vec<(VoxelCoord, Voxel)> = selection
        .iter()
        .filter_map(|&c| voxel_map.get(&c).map(|&i| (c, file.voxels[i])))
        .collect();
    if to_move.is_empty() {
        return Vec::new();
    }

    ensure_grid_fits_coords(
        file,
        to_move.iter().map(|&(c, _)| (c.0 + dx, c.1 + dy, c.2 + dz)),
    );
    let grid_size = file.grid_size.max(1);
    let (lo, hi) = grid_valid_range(grid_size);

    let mut deltas = Vec::new();

    // Remove sources first.
    for &(coord, _) in &to_move {
        if let Some(d) = fill::remove_voxel_at_coord(file, voxel_map, coord) {
            deltas.push(d);
        }
    }

    // Add at destinations.
    let mut seen: HashSet<VoxelCoord> = HashSet::new();
    for (src, mut voxel) in to_move {
        let x = src.0 + dx;
        let y = src.1 + dy;
        let z = src.2 + dz;
        if x < lo || x > hi || y < lo || y > hi || z < lo || z > hi {
            continue;
        }
        if !seen.insert((x, y, z)) {
            continue;
        }
        // Evict any non-selection voxel already at destination.
        if voxel_map.contains_key(&(x, y, z)) {
            if let Some(d) = fill::remove_voxel_at_coord(file, voxel_map, (x, y, z)) {
                deltas.push(d);
            }
        }
        voxel.x = x;
        voxel.y = y;
        voxel.z = z;
        push_voxel_known(file, voxel_map, voxel);
        deltas.push(VoxelEditDelta::Added(voxel));
    }

    deltas
}

/// Rotate all solid voxels in `selection` by `quarters` quarter-turns around
/// `axis` (0=X, 1=Y, 2=Z). Returns voxel-edit deltas. The caller is
/// responsible for computing the new selection-cell set (use
/// [`rotate_selection_coords`]) and updating `selection_cells`.
pub fn rotate_selected_voxels(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    selection: &AHashSet<VoxelCoord>,
    axis: u8,
    quarters: i32,
) -> Vec<VoxelEditDelta> {
    let q = quarters.rem_euclid(4);
    if q == 0 {
        return Vec::new();
    }

    let pivot = selection_pivot(selection);

    let to_move: Vec<(VoxelCoord, Voxel)> = selection
        .iter()
        .filter_map(|&c| voxel_map.get(&c).map(|&i| (c, file.voxels[i])))
        .collect();
    if to_move.is_empty() {
        return Vec::new();
    }

    // Compute rotated destinations.
    let rotated: Vec<(VoxelCoord, Voxel)> = to_move
        .iter()
        .map(|&(src, mut voxel)| {
            let dest = rotate_coord(src, pivot, axis, q);
            voxel.x = dest.0;
            voxel.y = dest.1;
            voxel.z = dest.2;
            (dest, voxel)
        })
        .collect();

    ensure_grid_fits_coords(file, rotated.iter().map(|&(c, _)| c));
    let grid_size = file.grid_size.max(1);
    let (lo, hi) = grid_valid_range(grid_size);

    let mut deltas = Vec::new();

    for &(coord, _) in &to_move {
        if let Some(d) = fill::remove_voxel_at_coord(file, voxel_map, coord) {
            deltas.push(d);
        }
    }

    let mut seen: HashSet<VoxelCoord> = HashSet::new();
    for (dest, voxel) in rotated {
        if dest.0 < lo || dest.0 > hi || dest.1 < lo || dest.1 > hi || dest.2 < lo || dest.2 > hi {
            continue;
        }
        if !seen.insert(dest) {
            continue;
        }
        if voxel_map.contains_key(&dest) {
            if let Some(d) = fill::remove_voxel_at_coord(file, voxel_map, dest) {
                deltas.push(d);
            }
        }
        push_voxel_known(file, voxel_map, voxel);
        deltas.push(VoxelEditDelta::Added(voxel));
    }

    deltas
}

/// Compute the new selection-cell set after a quarter-turn rotation (no voxel
/// data needed — purely coordinate arithmetic).
pub fn rotate_selection_coords(
    selection: &AHashSet<VoxelCoord>,
    axis: u8,
    quarters: i32,
) -> AHashSet<VoxelCoord> {
    let q = quarters.rem_euclid(4);
    if q == 0 {
        return selection.clone();
    }
    let pivot = selection_pivot(selection);
    selection
        .iter()
        .map(|&c| rotate_coord(c, pivot, axis, q))
        .collect()
}

fn selection_pivot(selection: &AHashSet<VoxelCoord>) -> (f32, f32, f32) {
    let mut min_x = i32::MAX;
    let mut max_x = i32::MIN;
    let mut min_y = i32::MAX;
    let mut max_y = i32::MIN;
    let mut min_z = i32::MAX;
    let mut max_z = i32::MIN;
    for &(x, y, z) in selection {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
        min_z = min_z.min(z);
        max_z = max_z.max(z);
    }
    (
        (min_x + max_x) as f32 * 0.5,
        (min_y + max_y) as f32 * 0.5,
        (min_z + max_z) as f32 * 0.5,
    )
}

fn rotate_coord(coord: VoxelCoord, pivot: (f32, f32, f32), axis: u8, quarters: i32) -> VoxelCoord {
    let rx = coord.0 as f32 - pivot.0;
    let ry = coord.1 as f32 - pivot.1;
    let rz = coord.2 as f32 - pivot.2;
    let (nx, ny, nz) = rotate_rel_quarter(rx, ry, rz, axis, quarters);
    (
        (nx + pivot.0).round() as i32,
        (ny + pivot.1).round() as i32,
        (nz + pivot.2).round() as i32,
    )
}

/// Mirror solid selected voxels through the plane perpendicular to `axis`
/// that passes through the selection AABB center.
/// `axis`: 0=X, 1=Y, 2=Z.
pub fn mirror_selected_voxels(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    selection: &AHashSet<VoxelCoord>,
    axis: u8,
) -> Vec<VoxelEditDelta> {
    let to_move: Vec<(VoxelCoord, Voxel)> = selection
        .iter()
        .filter_map(|&c| voxel_map.get(&c).map(|&i| (c, file.voxels[i])))
        .collect();
    if to_move.is_empty() {
        return Vec::new();
    }
    let pivot = selection_pivot(selection);

    let mirrored: Vec<(VoxelCoord, Voxel)> = to_move
        .iter()
        .map(|&(src, mut voxel)| {
            let dest = mirror_coord(src, pivot, axis);
            voxel.x = dest.0;
            voxel.y = dest.1;
            voxel.z = dest.2;
            (dest, voxel)
        })
        .collect();

    ensure_grid_fits_coords(file, mirrored.iter().map(|&(c, _)| c));
    let grid_size = file.grid_size.max(1);
    let (lo, hi) = grid_valid_range(grid_size);

    let mut deltas = Vec::new();
    for &(coord, _) in &to_move {
        if let Some(d) = fill::remove_voxel_at_coord(file, voxel_map, coord) {
            deltas.push(d);
        }
    }
    let mut seen: HashSet<VoxelCoord> = HashSet::new();
    for (dest, voxel) in mirrored {
        if dest.0 < lo || dest.0 > hi || dest.1 < lo || dest.1 > hi || dest.2 < lo || dest.2 > hi {
            continue;
        }
        if !seen.insert(dest) {
            continue;
        }
        if voxel_map.contains_key(&dest) {
            if let Some(d) = fill::remove_voxel_at_coord(file, voxel_map, dest) {
                deltas.push(d);
            }
        }
        push_voxel_known(file, voxel_map, voxel);
        deltas.push(VoxelEditDelta::Added(voxel));
    }
    deltas
}

/// Compute the new selection-cell set after a mirror on `axis`.
pub fn mirror_selection_coords(selection: &AHashSet<VoxelCoord>, axis: u8) -> AHashSet<VoxelCoord> {
    let pivot = selection_pivot(selection);
    selection
        .iter()
        .map(|&c| mirror_coord(c, pivot, axis))
        .collect()
}

fn mirror_coord(coord: VoxelCoord, pivot: (f32, f32, f32), axis: u8) -> VoxelCoord {
    match axis {
        0 => (
            (2.0 * pivot.0 - coord.0 as f32).round() as i32,
            coord.1,
            coord.2,
        ),
        1 => (
            coord.0,
            (2.0 * pivot.1 - coord.1 as f32).round() as i32,
            coord.2,
        ),
        _ => (
            coord.0,
            coord.1,
            (2.0 * pivot.2 - coord.2 as f32).round() as i32,
        ),
    }
}

/// Scale solid selected voxels by `factor` around the selection AABB center
/// using inverse nearest-neighbor resampling.  Works for both upscale (>1)
/// and downscale (<1).
pub fn scale_selected_voxels(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    selection: &AHashSet<VoxelCoord>,
    factor: f64,
) -> Vec<VoxelEditDelta> {
    if factor <= 0.0 || (factor - 1.0).abs() < 1e-9 {
        return Vec::new();
    }

    let pivot = selection_pivot(selection);
    let px = pivot.0 as f64;
    let py = pivot.1 as f64;
    let pz = pivot.2 as f64;

    let source_voxels: AHashMap<VoxelCoord, Voxel> = selection
        .iter()
        .filter_map(|&c| voxel_map.get(&c).map(|&i| (c, file.voxels[i])))
        .collect();
    if source_voxels.is_empty() {
        return Vec::new();
    }

    let new_voxels = scale_resample(&source_voxels, selection, factor, px, py, pz);

    ensure_grid_fits_coords(file, new_voxels.iter().map(|&(c, _)| c));
    let grid_size = file.grid_size.max(1);
    let (lo, hi) = grid_valid_range(grid_size);

    let mut deltas = Vec::new();
    for &c in selection {
        if source_voxels.contains_key(&c) {
            if let Some(d) = fill::remove_voxel_at_coord(file, voxel_map, c) {
                deltas.push(d);
            }
        }
    }
    let mut seen: HashSet<VoxelCoord> = HashSet::new();
    for (dest, voxel) in new_voxels {
        if dest.0 < lo || dest.0 > hi || dest.1 < lo || dest.1 > hi || dest.2 < lo || dest.2 > hi {
            continue;
        }
        if !seen.insert(dest) {
            continue;
        }
        if voxel_map.contains_key(&dest) {
            if let Some(d) = fill::remove_voxel_at_coord(file, voxel_map, dest) {
                deltas.push(d);
            }
        }
        push_voxel_known(file, voxel_map, voxel);
        deltas.push(VoxelEditDelta::Added(voxel));
    }
    deltas
}

/// Compute the new selection-cell set after a uniform scale by `factor`.
pub fn scale_selection_coords(
    selection: &AHashSet<VoxelCoord>,
    factor: f64,
) -> AHashSet<VoxelCoord> {
    if factor <= 0.0 || (factor - 1.0).abs() < 1e-9 {
        return selection.clone();
    }
    let pivot = selection_pivot(selection);
    let px = pivot.0 as f64;
    let py = pivot.1 as f64;
    let pz = pivot.2 as f64;
    let inv = 1.0 / factor;
    let (nmin, nmax) = scale_dest_aabb(selection, factor, px, py, pz);
    let mut result = AHashSet::new();
    for nz in nmin.2..=nmax.2 {
        for ny in nmin.1..=nmax.1 {
            for nx in nmin.0..=nmax.0 {
                let sx = (px + (nx as f64 - px) * inv).round() as i32;
                let sy = (py + (ny as f64 - py) * inv).round() as i32;
                let sz = (pz + (nz as f64 - pz) * inv).round() as i32;
                if selection.contains(&(sx, sy, sz)) {
                    result.insert((nx, ny, nz));
                }
            }
        }
    }
    result
}

/// Inverse nearest-neighbor resample: iterate the destination AABB and pull
/// colour from the closest source voxel.
fn scale_resample(
    source_voxels: &AHashMap<VoxelCoord, Voxel>,
    selection: &AHashSet<VoxelCoord>,
    factor: f64,
    px: f64,
    py: f64,
    pz: f64,
) -> Vec<(VoxelCoord, Voxel)> {
    let inv = 1.0 / factor;
    let (nmin, nmax) = scale_dest_aabb(selection, factor, px, py, pz);
    let mut out = Vec::new();
    for nz in nmin.2..=nmax.2 {
        for ny in nmin.1..=nmax.1 {
            for nx in nmin.0..=nmax.0 {
                let sx = (px + (nx as f64 - px) * inv).round() as i32;
                let sy = (py + (ny as f64 - py) * inv).round() as i32;
                let sz = (pz + (nz as f64 - pz) * inv).round() as i32;
                if let Some(&voxel) = source_voxels.get(&(sx, sy, sz)) {
                    let mut v = voxel;
                    v.x = nx;
                    v.y = ny;
                    v.z = nz;
                    out.push(((nx, ny, nz), v));
                }
            }
        }
    }
    out
}

/// Compute the integer destination AABB after scaling the selection's AABB by
/// `factor` around `(px, py, pz)`.
fn scale_dest_aabb(
    selection: &AHashSet<VoxelCoord>,
    factor: f64,
    px: f64,
    py: f64,
    pz: f64,
) -> (VoxelCoord, VoxelCoord) {
    let mut smin = (i32::MAX, i32::MAX, i32::MAX);
    let mut smax = (i32::MIN, i32::MIN, i32::MIN);
    for &(x, y, z) in selection {
        smin.0 = smin.0.min(x);
        smin.1 = smin.1.min(y);
        smin.2 = smin.2.min(z);
        smax.0 = smax.0.max(x);
        smax.1 = smax.1.max(y);
        smax.2 = smax.2.max(z);
    }
    let map = |v: i32, p: f64| -> (f64, f64) {
        let a = p + (v as f64 - p) * factor;
        (a, a)
    };
    let mut nmin = (f64::MAX, f64::MAX, f64::MAX);
    let mut nmax = (f64::MIN, f64::MIN, f64::MIN);
    for &cx in &[smin.0, smax.0] {
        for &cy in &[smin.1, smax.1] {
            for &cz in &[smin.2, smax.2] {
                let (ax, _) = map(cx, px);
                let (ay, _) = map(cy, py);
                let (az, _) = map(cz, pz);
                nmin.0 = nmin.0.min(ax);
                nmin.1 = nmin.1.min(ay);
                nmin.2 = nmin.2.min(az);
                nmax.0 = nmax.0.max(ax);
                nmax.1 = nmax.1.max(ay);
                nmax.2 = nmax.2.max(az);
            }
        }
    }
    (
        (
            nmin.0.floor() as i32,
            nmin.1.floor() as i32,
            nmin.2.floor() as i32,
        ),
        (
            nmax.0.ceil() as i32,
            nmax.1.ceil() as i32,
            nmax.2.ceil() as i32,
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brush_extent_toward_solid_matches_radius_for_axis_aligned_normals() {
        // radius is a 0-based display index: size = radius + 1.
        // Sphere radius=4 → size 5 (odd): offsets span [-2, 2], extent = 2.
        assert_eq!(
            brush::brush_footprint_extent_toward_solid(BrushShape::Sphere, 4, (0, 1, 0), None),
            2
        );
        // Cube radius=3 → size 4 (even): offsets span [-1, 2], extent = 1.
        assert_eq!(
            brush::brush_footprint_extent_toward_solid(BrushShape::Cube, 3, (1, 0, 0), None),
            1
        );
        // Sphere radius=7 → size 8 (even): offsets span [-3, 4], extent = 3.
        assert_eq!(
            brush::brush_footprint_extent_toward_solid(BrushShape::Sphere, 7, (0, 0, 1), None),
            3
        );
        // Cube radius=0 → size 1: single voxel at origin, extent = 0.
        assert_eq!(
            brush::brush_footprint_extent_toward_solid(BrushShape::Cube, 0, (1, 0, 0), None),
            0
        );
    }

    #[test]
    fn world_to_voxel_negative() {
        let p = Vec3::new(-2.3, 0.4, 1.2);
        assert_eq!(world_to_voxel(p), (-2, 0, 1));
    }

    #[test]
    fn screen_ray_round_trips_through_vp() {
        let mut cam = OrbitCamera::new();
        cam.smooth_target = glam::Vec3::ZERO;
        cam.smooth_spherical = cam.spherical;
        let w = 1920.0_f32;
        let h = 1080.0_f32;
        let proj = cam.proj_matrix(w, h);
        let view = cam.view_matrix();
        let vp = proj * view;
        let world = glam::Vec3::new(1.5, 2.0, -3.0);
        let clip = vp * world.extend(1.0);
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        let sx = (ndc_x + 1.0) * 0.5 * w - 0.5;
        let sy = (1.0 - ndc_y) * 0.5 * h - 0.5;
        let (o, d) = raycasting::screen_to_world_ray(&cam, w, h, sx, sy);
        let t = (world - o).dot(d);
        let closest = o + d * t;
        assert!(
            (closest - world).length() < 5e-3,
            "ray miss: dist {}",
            (closest - world).length()
        );
    }

    #[test]
    fn ray_hits_center_voxel() {
        let mut m: AHashMap<VoxelCoord, usize> = AHashMap::new();
        m.insert((0, 0, 0), 0);
        let origin = Vec3::new(0.0, 0.0, 5.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);
        let r = raycasting::ray_first_solid(origin, dir, &m, 32);
        assert!(r.is_some());
        let ((x, y, z), prev) = r.unwrap();
        assert_eq!((x, y, z), (0, 0, 0));
        assert_eq!(prev, Some((0, 0, 1)));
    }

    #[test]
    fn collect_stroke_preview_and_edit_agree_when_empty_scene() {
        let file = VoxelleFile {
            version: 4,
            grid_size: 16,
            scene: crate::voxelle::Scene::default(),
            scene_extra: None,
            mood: None,
            lighting: None,
            voxels: vec![],
            objects: crate::voxelle::default_scene_objects(),
            active_object_id: 0,
        };
        let vm: AHashMap<VoxelCoord, usize> = AHashMap::new();
        let mut cam = OrbitCamera::new();
        cam.smooth_target = glam::Vec3::ZERO;
        cam.smooth_spherical = cam.spherical;
        let aux = StrokeAux::default();
        let w = 256.0_f32;
        let h = 256.0_f32;
        let sx = 128.0_f32;
        let sy = 128.0_f32;
        let preview = collect_stroke_preview_targets(
            &file,
            &vm,
            &cam,
            w,
            h,
            sx,
            sy,
            EditTool::Paint,
            0xff0000,
            MaterialId::Plastic,
            0,
            BrushShape::Sphere,
            0.0,
            None,
            None,
            DrawStrokeMode::Precise,
            PlaneAxis::Auto,
            &aux,
            None,
        );
        let edit = collect_stroke_edit_targets(
            &file,
            &vm,
            &cam,
            w,
            h,
            sx,
            sy,
            EditTool::Paint,
            0xff0000,
            MaterialId::Plastic,
            0,
            BrushShape::Sphere,
            0.0,
            None,
            None,
            DrawStrokeMode::Precise,
            PlaneAxis::Auto,
            &aux,
            None,
        );
        assert_eq!(preview, edit);
        assert!(preview.is_empty());
    }

    #[test]
    fn collect_stroke_remove_preview_includes_empty_footprint_cells() {
        let file = VoxelleFile {
            version: 4,
            grid_size: 16,
            scene: crate::voxelle::Scene::default(),
            scene_extra: None,
            mood: None,
            lighting: None,
            voxels: vec![Voxel {
                x: 0,
                y: 0,
                z: 0,
                color: 0xff0000,
                material: MaterialId::Plastic,
                object_id: 0,
            }],
            objects: crate::voxelle::default_scene_objects(),
            active_object_id: 0,
        };
        let mut vm: AHashMap<VoxelCoord, usize> = AHashMap::new();
        vm.insert((0, 0, 0), 0);
        let mut cam = OrbitCamera::new();
        cam.smooth_target = glam::Vec3::ZERO;
        cam.smooth_spherical = cam.spherical;
        let w = 256.0_f32;
        let h = 256.0_f32;
        let sx = 128.0_f32;
        let sy = 128.0_f32;
        let aux = StrokeAux::default();
        let preview = collect_stroke_preview_targets(
            &file,
            &vm,
            &cam,
            w,
            h,
            sx,
            sy,
            EditTool::Remove,
            0xff0000,
            MaterialId::Plastic,
            1,
            BrushShape::Sphere,
            0.0,
            None,
            None,
            DrawStrokeMode::Precise,
            PlaneAxis::Auto,
            &aux,
            None,
        );
        let edit = collect_stroke_edit_targets(
            &file,
            &vm,
            &cam,
            w,
            h,
            sx,
            sy,
            EditTool::Remove,
            0xff0000,
            MaterialId::Plastic,
            1,
            BrushShape::Sphere,
            0.0,
            None,
            None,
            DrawStrokeMode::Precise,
            PlaneAxis::Auto,
            &aux,
            None,
        );
        assert!(
            !preview.is_empty(),
            "preview should include brush footprint"
        );
        let ghosts = preview.iter().filter(|c| !vm.contains_key(c)).count();
        assert!(
            ghosts > 0,
            "preview should show empty cells along brush footprint"
        );
        assert!(preview.len() > edit.len());
        for c in &edit {
            assert!(vm.contains_key(c));
            assert!(preview.contains(c));
        }
    }

    #[test]
    fn sculpt_stroke_draw_adds_empty_cells() {
        let mut file = VoxelleFile {
            version: 4,
            grid_size: 16,
            scene: crate::voxelle::Scene::default(),
            scene_extra: None,
            mood: None,
            lighting: None,
            voxels: vec![Voxel {
                x: 0,
                y: 0,
                z: 0,
                color: 0xff0000,
                material: crate::voxelle::MaterialId::Plastic,
                object_id: 0,
            }],
            objects: crate::voxelle::default_scene_objects(),
            active_object_id: 0,
        };
        let mut vm: AHashMap<VoxelCoord, usize> = AHashMap::new();
        vm.insert((0, 0, 0), 0);
        let mut terrain_accum: AHashMap<(i32, i32), f32> = AHashMap::new();
        let mut cam = OrbitCamera::new();
        cam.smooth_target = glam::Vec3::ZERO;
        cam.smooth_spherical = cam.spherical;
        let w = 256.0_f32;
        let h = 256.0_f32;
        let deltas = apply_sculpt_stroke(
            &mut file,
            &mut vm,
            &cam,
            w,
            h,
            128.0,
            128.0,
            SculptStrokeMode::Draw,
            0x00ff00,
            crate::voxelle::MaterialId::Plastic,
            0,
            BrushShape::Sphere,
            0.0,
            false,
            None,
            None,
            None,
            0,
            4,
            2,
            false,
            false,
            &mut terrain_accum,
            1,
            100,
            0,
            0,
            WallAreaShape::Brush,
            SprayDirection::Auto,
            0,
            2,
            false,
            false,
            SculptSmoothVariant::Majority,
            0,
            100,
            4,
            50,
            None,
            ExtrudeProfile::Cube,
            ExtrudeEndCap::Flat,
            false,
            0.0,
            1.0,
            None,
        )
        .unwrap();
        assert!(!deltas.is_empty());
    }

    #[test]
    fn remove_voxel_at_coord_swap_remove_updates_map() {
        let mut file = VoxelleFile {
            version: 4,
            grid_size: 8,
            scene: crate::voxelle::Scene::default(),
            scene_extra: None,
            mood: None,
            lighting: None,
            voxels: vec![
                Voxel {
                    x: 0,
                    y: 0,
                    z: 0,
                    color: 0xff0000,
                    material: MaterialId::Plastic,
                    object_id: 0,
                },
                Voxel {
                    x: 1,
                    y: 0,
                    z: 0,
                    color: 0x00ff00,
                    material: MaterialId::Plastic,
                    object_id: 0,
                },
            ],
            objects: crate::voxelle::default_scene_objects(),
            active_object_id: 0,
        };
        let mut vm: AHashMap<VoxelCoord, usize> = AHashMap::new();
        vm.insert((0, 0, 0), 0);
        vm.insert((1, 0, 0), 1);
        assert!(fill::remove_voxel_at_coord(&mut file, &mut vm, (0, 0, 0)).is_some());
        assert_eq!(file.voxels.len(), 1);
        assert_eq!(vm.len(), 1);
        assert!(vm.contains_key(&(1, 0, 0)));
        assert!(!vm.contains_key(&(0, 0, 0)));
    }

    #[test]
    fn flood_fill_remove_deletes_full_same_color_region() {
        let mut file = VoxelleFile {
            version: 4,
            grid_size: 16,
            scene: crate::voxelle::Scene::default(),
            scene_extra: None,
            mood: None,
            lighting: None,
            voxels: vec![
                Voxel {
                    x: 0,
                    y: 0,
                    z: 0,
                    color: 0xff0000,
                    material: MaterialId::Plastic,
                    object_id: 0,
                },
                Voxel {
                    x: 1,
                    y: 0,
                    z: 0,
                    color: 0xff0000,
                    material: MaterialId::Plastic,
                    object_id: 0,
                },
            ],
            objects: crate::voxelle::default_scene_objects(),
            active_object_id: 0,
        };
        let mut vm: AHashMap<VoxelCoord, usize> = AHashMap::new();
        vm.insert((0, 0, 0), 0);
        vm.insert((1, 0, 0), 1);
        let mut cam = OrbitCamera::new();
        cam.smooth_target = glam::Vec3::ZERO;
        cam.smooth_spherical = cam.spherical;
        let w = 256.0_f32;
        let h = 256.0_f32;
        let sx = 128.0_f32;
        let sy = 128.0_f32;
        let n = flood_fill_remove_at_screen(
            &mut file,
            &mut vm,
            &cam,
            w,
            h,
            sx,
            sy,
            false,
            true,
            false,
            false,
            PlaneAxis::Auto,
            None,
            |_| {},
        )
        .unwrap()
        .deltas
        .len();
        assert_eq!(n, 2, "expected both red voxels removed");
        assert!(vm.is_empty());
    }

    #[test]
    fn flood_fill_respects_color_false_spans_colors() {
        let file = VoxelleFile {
            version: 4,
            grid_size: 16,
            scene: crate::voxelle::Scene::default(),
            scene_extra: None,
            mood: None,
            lighting: None,
            voxels: vec![
                Voxel {
                    x: 0,
                    y: 0,
                    z: 0,
                    color: 0xff0000,
                    material: MaterialId::Plastic,
                    object_id: 0,
                },
                Voxel {
                    x: 1,
                    y: 0,
                    z: 0,
                    color: 0x00ff00,
                    material: MaterialId::Plastic,
                    object_id: 0,
                },
            ],
            objects: crate::voxelle::default_scene_objects(),
            active_object_id: 0,
        };
        let mut vm: AHashMap<VoxelCoord, usize> = AHashMap::new();
        vm.insert((0, 0, 0), 0);
        vm.insert((1, 0, 0), 1);
        let mut cam = OrbitCamera::new();
        cam.smooth_target = glam::Vec3::ZERO;
        cam.smooth_spherical = cam.spherical;
        let w = 256.0_f32;
        let h = 256.0_f32;
        let sx = 128.0_f32;
        let sy = 128.0_f32;
        let coords = flood_fill_selection_coords(
            &file,
            &vm,
            &cam,
            w,
            h,
            sx,
            sy,
            false,
            false,
            false,
            false,
            PlaneAxis::Auto,
        );
        assert_eq!(
            coords.len(),
            2,
            "with respects_color false, both solids should connect"
        );
    }

    #[test]
    fn flood_fill_empty_places_adjacent_air() {
        let mut file = VoxelleFile {
            version: 4,
            grid_size: 16,
            scene: crate::voxelle::Scene::default(),
            scene_extra: None,
            mood: None,
            lighting: None,
            voxels: vec![Voxel {
                x: 0,
                y: 0,
                z: 0,
                color: 0xff0000,
                material: MaterialId::Plastic,
                object_id: 0,
            }],
            objects: crate::voxelle::default_scene_objects(),
            active_object_id: 0,
        };
        let mut vm: AHashMap<VoxelCoord, usize> = AHashMap::new();
        vm.insert((0, 0, 0), 0);
        let mut cam = OrbitCamera::new();
        cam.smooth_target = glam::Vec3::ZERO;
        cam.smooth_spherical = cam.spherical;
        let w = 256.0_f32;
        let h = 256.0_f32;
        let sx = 128.0_f32;
        let sy = 128.0_f32;
        let n = flood_fill_empty_at_screen(
            &mut file,
            &mut vm,
            &cam,
            w,
            h,
            sx,
            sy,
            false,
            |_, _, _| 0xabcdef,
            MaterialId::Plastic,
            false,
            PlaneAxis::Auto,
            None,
            |_| {},
        )
        .unwrap()
        .deltas
        .len();
        assert!(
            n >= 1,
            "expected at least one empty cell filled in front of solid"
        );
    }
}
