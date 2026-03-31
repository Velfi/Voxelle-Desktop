//! Screen-space ray → grid traversal for add/remove voxel editing.

use crate::camera::OrbitCamera;
use crate::greedy_mesh::VoxelCoord;
use crate::stroke_modes::{
    stroke_anchor_centers_with_mode, DrawStrokeMode, PlaneAxis, StrokeAux,
};
use crate::voxelle::{MaterialId, Voxel, VoxelleFile};
use ahash::{AHashMap, AHashSet};
use glam::{Vec3, Vec4};
use std::collections::{HashSet, VecDeque};

pub fn screen_to_world_ray(
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> (Vec3, Vec3) {
    let w = width.max(1.0);
    let h = height.max(1.0);
    // Pixel centers: map (sx,sy) through viewport so the ray matches fragment centers (GPU).
    let ndc_x = ((sx + 0.5) / w) * 2.0 - 1.0;
    let ndc_y = 1.0 - ((sy + 0.5) / h) * 2.0;
    let proj = camera.proj_matrix(width, height);
    let view = camera.view_matrix();
    let inv_vp = (proj * view).inverse();
    // Glam `perspective_rh` uses NDC z in [0,1] (WebGPU); unproject with inverse(view*proj).
    let mut near_h = inv_vp * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
    let mut far_h = inv_vp * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
    near_h /= near_h.w;
    far_h /= far_h.w;
    let o = near_h.truncate();
    let d = (far_h.truncate() - o).normalize();
    (o, d)
}

/// World-space point → physical viewport pixels `(sx, sy)` with top-left origin (+Y down), matching [`screen_to_world_ray`].
pub fn world_to_viewport_pixels(
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    wx: f32,
    wy: f32,
    wz: f32,
) -> Option<(f32, f32)> {
    let w = width.max(1.0);
    let h = height.max(1.0);
    let view = camera.view_matrix();
    let proj = camera.proj_matrix(w, h);
    let vp = proj * view;
    let p = Vec4::new(wx, wy, wz, 1.0);
    let clip = vp * p;
    if clip.w.abs() < 1e-5 {
        return None;
    }
    let ndc_x = clip.x / clip.w;
    let ndc_y = clip.y / clip.w;
    if ndc_x.abs() > 1.55 || ndc_y.abs() > 1.55 {
        return None;
    }
    let sx = (ndc_x + 1.0) * 0.5 * w - 0.5;
    let sy = (1.0 - ndc_y) * 0.5 * h - 0.5;
    Some((sx, sy))
}

#[inline]
fn world_to_voxel(p: Vec3) -> (i32, i32, i32) {
    (
        (p.x + 0.5).floor() as i32,
        (p.y + 0.5).floor() as i32,
        (p.z + 0.5).floor() as i32,
    )
}

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

fn ray_aabb_intersect(origin: Vec3, dir: Vec3, bmin: Vec3, bmax: Vec3) -> Option<(f32, f32)> {
    let mut tmin = f32::NEG_INFINITY;
    let mut tmax = f32::INFINITY;
    for i in 0..3 {
        let o = origin[i];
        let d = dir[i];
        let bn = bmin[i];
        let bx = bmax[i];
        if d.abs() < 1e-8 {
            if o < bn || o > bx {
                return None;
            }
            continue;
        }
        let inv_d = 1.0 / d;
        let mut t0 = (bn - o) * inv_d;
        let mut t1 = (bx - o) * inv_d;
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        tmin = tmin.max(t0);
        tmax = tmax.min(t1);
        if tmin > tmax {
            return None;
        }
    }
    Some((tmin, tmax))
}

fn exit_t_axis(ox: f32, dx: f32, c: i32, t_min: f32) -> f32 {
    const EPS: f32 = 1e-5;
    if dx.abs() < EPS {
        return f32::INFINITY;
    }
    let t = if dx > 0.0 {
        ((c as f32 + 0.5) - ox) / dx
    } else {
        ((c as f32 - 0.5) - ox) / dx
    };
    if t > t_min + EPS {
        t
    } else {
        f32::INFINITY
    }
}

/// First solid voxel along the ray within the grid bounding box, and the previous empty cell visited (adjacent along the ray).
pub(crate) fn ray_first_solid(
    origin: Vec3,
    dir: Vec3,
    occupied: &AHashMap<VoxelCoord, usize>,
    grid_size: i32,
) -> Option<((i32, i32, i32), Option<(i32, i32, i32)>)> {
    let (lo, hi) = grid_valid_range(grid_size);
    let bmin = Vec3::new(lo as f32 - 0.5, lo as f32 - 0.5, lo as f32 - 0.5);
    let bmax = Vec3::new(hi as f32 + 0.5, hi as f32 + 0.5, hi as f32 + 0.5);
    let (t_enter, t_exit) = ray_aabb_intersect(origin, dir, bmin, bmax)?;
    let mut t = t_enter.max(0.0) + 1e-4;
    let mut prev: Option<(i32, i32, i32)> = None;
    let ox = origin.x;
    let oy = origin.y;
    let oz = origin.z;
    let dx = dir.x;
    let dy = dir.y;
    let dz = dir.z;

    for _ in 0..200_000 {
        if t > t_exit {
            break;
        }
        let p = origin + dir * t;
        let c = world_to_voxel(p);
        if !in_grid(c.0, c.1, c.2, grid_size) {
            break;
        }
        if occupied.contains_key(&c) {
            return Some((c, prev));
        }
        prev = Some(c);
        let t_next = exit_t_axis(ox, dx, c.0, t)
            .min(exit_t_axis(oy, dy, c.1, t))
            .min(exit_t_axis(oz, dz, c.2, t));
        if !t_next.is_finite() {
            break;
        }
        t = t_next + 1e-4;
    }
    None
}

/// `true` if the ray from the screen point hits any solid voxel before exiting the grid (same test as edit/remove).
pub fn probe_solid_hit(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> bool {
    if file.voxels.is_empty() {
        return false;
    }
    let grid_size = file.grid_size.max(1);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    ray_first_solid(origin, dir, voxel_map, grid_size).is_some()
}

/// First solid voxel along the ray (for selection toggle).
pub fn pick_solid_coord_at_screen(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> Option<VoxelCoord> {
    if file.voxels.is_empty() {
        return None;
    }
    let grid_size = file.grid_size.max(1);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    ray_first_solid(origin, dir, voxel_map, grid_size).map(|(c, _)| c)
}

/// Cell where an add would place (empty cell in front of first solid along the ray), if valid.
pub fn preview_add_cell(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> Option<(i32, i32, i32)> {
    if file.voxels.is_empty() {
        return None;
    }
    let grid_size = file.grid_size.max(1);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let (_hit, prev) = ray_first_solid(origin, dir, voxel_map, grid_size)?;
    let (px, py, pz) = prev?;
    if in_grid(px, py, pz, grid_size) && !voxel_map.contains_key(&(px, py, pz)) {
        Some((px, py, pz))
    } else {
        None
    }
}

/// Solid voxel the ray would remove, if any.
pub fn preview_remove_cell(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> Option<(i32, i32, i32)> {
    if file.voxels.is_empty() {
        return None;
    }
    let grid_size = file.grid_size.max(1);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    ray_first_solid(origin, dir, voxel_map, grid_size).map(|(h, _)| h)
}

#[inline]
pub(crate) fn anchor_for_edit(
    tool: EditTool,
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> Option<(i32, i32, i32)> {
    match tool {
        EditTool::Add => preview_add_cell(file, voxel_map, camera, width, height, sx, sy),
        EditTool::Remove | EditTool::Paint => {
            preview_remove_cell(file, voxel_map, camera, width, height, sx, sy)
        }
    }
}

/// Result of a successful edit for GPU incremental brick updates.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum VoxelEditDelta {
    Added(Voxel),
    Removed { voxel: Voxel },
    /// Recolor / material change; `before` and `after` share the same cell.
    Painted { before: Voxel, after: Voxel },
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum EditTool {
    Add,
    Remove,
    Paint,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BrushShape {
    #[default]
    Sphere,
    Cube,
    Pyramid,
}

#[inline]
fn spray_passes(cell: (i32, i32, i32), spray: f32) -> bool {
    if spray <= 0.0 {
        return true;
    }
    let h = cell
        .0
        .wrapping_mul(73856093)
        ^ cell.1.wrapping_mul(19349663)
        ^ cell.2.wrapping_mul(83492791);
    let u = (h as u32 as f64 / u32::MAX as f64) as f32;
    u < spray.clamp(0.0, 1.0)
}

/// Voxel centers along a 3D line (inclusive endpoints).
pub(crate) fn voxel_line_dda(a: (i32, i32, i32), b: (i32, i32, i32)) -> Vec<(i32, i32, i32)> {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    let dz = b.2 - a.2;
    let steps = dx.abs().max(dy.abs()).max(dz.abs()).max(1);
    let mut pts = Vec::with_capacity((steps + 1) as usize);
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let x = (a.0 as f32 + dx as f32 * t).round() as i32;
        let y = (a.1 as f32 + dy as f32 * t).round() as i32;
        let z = (a.2 as f32 + dz as f32 * t).round() as i32;
        pts.push((x, y, z));
    }
    pts.sort_unstable();
    pts.dedup();
    pts
}

pub fn neighbors_6(c: VoxelCoord) -> [VoxelCoord; 6] {
    let (x, y, z) = c;
    [
        (x + 1, y, z),
        (x - 1, y, z),
        (x, y + 1, z),
        (x, y - 1, z),
        (x, y, z + 1),
        (x, y, z - 1),
    ]
}

/// Flood-fill paint: 6-connected voxels matching seed color (and optionally material).
pub fn flood_fill_paint_at_screen(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    new_color: u32,
    new_material: MaterialId,
    match_material: bool,
) -> Result<Vec<VoxelEditDelta>, String> {
    let grid_size = file.grid_size.max(1);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((hit, _)) = ray_first_solid(origin, dir, voxel_map, grid_size) else {
        return Ok(Vec::new());
    };
    let Some(&seed_idx) = voxel_map.get(&hit) else {
        return Ok(Vec::new());
    };
    let seed = file.voxels[seed_idx];
    let tc = seed.color;
    let tm = seed.material;

    let mut out: Vec<VoxelEditDelta> = Vec::new();
    let mut visited: AHashSet<VoxelCoord> = AHashSet::new();
    let mut queue: VecDeque<VoxelCoord> = VecDeque::new();
    visited.insert(hit);
    queue.push_back(hit);

    while let Some(c) = queue.pop_front() {
        let Some(&idx) = voxel_map.get(&c) else {
            continue;
        };
        let v = file.voxels[idx];
        if v.color != tc || (match_material && v.material != tm) {
            continue;
        }
        if v.color == new_color && v.material == new_material {
            continue;
        }
        let before = v;
        let after = Voxel {
            color: new_color,
            material: new_material,
            ..before
        };
        file.voxels[idx] = after;
        out.push(VoxelEditDelta::Painted { before, after });

        for n in neighbors_6(c) {
            if !in_grid(n.0, n.1, n.2, grid_size) {
                continue;
            }
            if visited.insert(n) {
                queue.push_back(n);
            }
        }
    }
    Ok(out)
}

/// 6-connected solid voxels matching the seed hit's color (and optionally material) — selection / flood without edits.
pub fn connected_solid_same_color_from_screen(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    match_material: bool,
) -> Option<Vec<VoxelCoord>> {
    let grid_size = file.grid_size.max(1);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((hit, _)) = ray_first_solid(origin, dir, voxel_map, grid_size) else {
        return None;
    };
    let Some(&seed_idx) = voxel_map.get(&hit) else {
        return None;
    };
    let seed = file.voxels[seed_idx];
    let tc = seed.color;
    let tm = seed.material;

    let mut out: Vec<VoxelCoord> = Vec::new();
    let mut visited: AHashSet<VoxelCoord> = AHashSet::new();
    let mut queue: VecDeque<VoxelCoord> = VecDeque::new();
    visited.insert(hit);
    queue.push_back(hit);

    while let Some(c) = queue.pop_front() {
        let Some(&idx) = voxel_map.get(&c) else {
            continue;
        };
        let v = file.voxels[idx];
        if v.color != tc || (match_material && v.material != tm) {
            continue;
        }
        out.push(c);
        for n in neighbors_6(c) {
            if !in_grid(n.0, n.1, n.2, grid_size) {
                continue;
            }
            if visited.insert(n) {
                queue.push_back(n);
            }
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Face-connected empty cells on the same plane as the add-cell in front of the ray hit.
pub fn coplanar_empty_connected_from_screen(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> Option<Vec<VoxelCoord>> {
    let grid_size = file.grid_size.max(1);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let (hit, prev) = ray_first_solid(origin, dir, voxel_map, grid_size)?;
    let prev = prev?;
    let (axis, fixed) = plane_axis_fixed(prev, hit)?;
    if voxel_map.contains_key(&prev) {
        return None;
    }
    if !voxel_on_plane(prev, axis, fixed) {
        return None;
    }

    let mut visited: AHashSet<VoxelCoord> = AHashSet::new();
    let mut queue: VecDeque<VoxelCoord> = VecDeque::new();
    let mut out: Vec<VoxelCoord> = Vec::new();
    visited.insert(prev);
    queue.push_back(prev);

    while let Some(c) = queue.pop_front() {
        if voxel_map.contains_key(&c) {
            continue;
        }
        if !voxel_on_plane(c, axis, fixed) {
            continue;
        }
        out.push(c);
        for n in neighbors_on_face_plane(axis, c) {
            if !in_grid(n.0, n.1, n.2, grid_size) {
                continue;
            }
            if !voxel_on_plane(n, axis, fixed) {
                continue;
            }
            if voxel_map.contains_key(&n) {
                continue;
            }
            if visited.insert(n) {
                queue.push_back(n);
            }
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Inclusive brush radius in voxels from the stroke center: `0` = single cell only.
pub fn brush_offset_cells(shape: BrushShape, radius: u32) -> Vec<(i32, i32, i32)> {
    let r = radius as i32;
    if r <= 0 {
        return vec![(0, 0, 0)];
    }
    let mut out = Vec::new();
    match shape {
        BrushShape::Cube => {
            for dx in -r..=r {
                for dy in -r..=r {
                    for dz in -r..=r {
                        out.push((dx, dy, dz));
                    }
                }
            }
        }
        BrushShape::Sphere => {
            let r2 = r * r;
            for dx in -r..=r {
                for dy in -r..=r {
                    for dz in -r..=r {
                        if dx * dx + dy * dy + dz * dz <= r2 {
                            out.push((dx, dy, dz));
                        }
                    }
                }
            }
        }
        BrushShape::Pyramid => {
            for dx in -r..=r {
                for dy in -r..=r {
                    for dz in -r..=r {
                        if dx.abs() + dy.abs() + dz.abs() <= r {
                            out.push((dx, dy, dz));
                        }
                    }
                }
            }
        }
    }
    out.sort_by_key(|(a, b, c)| (a.abs() + b.abs() + c.abs(), *a, *b, *c));
    out
}

pub fn pick_voxel_at_screen(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> Option<Voxel> {
    if file.voxels.is_empty() {
        return None;
    }
    let grid_size = file.grid_size.max(1);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let (hit, _) = ray_first_solid(origin, dir, voxel_map, grid_size)?;
    let idx = *voxel_map.get(&hit)?;
    Some(file.voxels[idx])
}

/// Apply add / remove / paint with optional brush; returns all atomic deltas (may be empty).
///
/// `spray_density`: `0` = full brush; `(0, 1]` thins voxels deterministically per cell.
/// `stroke_line_start`: when `Some`, brush samples along the 3D line between anchors at
/// pointer-down and `(sx, sy)` (Stroke / line mode).
/// `stroke_segment_prev`: when `stroke_line_start` is `None` and this is `Some`, samples along
/// the segment from the previous screen position to `(sx, sy)` (Brush path / web spray-style drag).
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
) -> Result<Vec<VoxelEditDelta>, String> {
    let grid_size = file.grid_size.max(1);
    let offsets = brush_offset_cells(brush_shape, brush_radius);
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
    );
    if centers.is_empty() {
        return Ok(Vec::new());
    }

    let mut seen: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut out: Vec<VoxelEditDelta> = Vec::new();

    match tool {
        EditTool::Add => {
            for (cx, cy, cz) in centers {
                for (dx, dy, dz) in &offsets {
                    let x = cx + dx;
                    let y = cy + dy;
                    let z = cz + dz;
                    if !in_grid(x, y, z, grid_size) {
                        continue;
                    }
                    if !spray_passes((x, y, z), spray) {
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
                        color,
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
            for (hx, hy, hz) in centers {
                for (dx, dy, dz) in &offsets {
                    let x = hx + dx;
                    let y = hy + dy;
                    let z = hz + dz;
                    if !spray_passes((x, y, z), spray) {
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
            for (hx, hy, hz) in centers {
                for (dx, dy, dz) in &offsets {
                    let x = hx + dx;
                    let y = hy + dy;
                    let z = hz + dz;
                    if !spray_passes((x, y, z), spray) {
                        continue;
                    }
                    if !seen.insert((x, y, z)) {
                        continue;
                    }
                    let Some(&idx) = voxel_map.get(&(x, y, z)) else {
                        continue;
                    };
                    let before = file.voxels[idx];
                    if before.color == color && before.material == material {
                        continue;
                    }
                    let after = Voxel {
                        color,
                        material,
                        ..before
                    };
                    file.voxels[idx] = after;
                    out.push(VoxelEditDelta::Painted { before, after });
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
        DrawStrokeMode::Plane | DrawStrokeMode::Spray => true,
        DrawStrokeMode::Precise => false,
        DrawStrokeMode::Line => stroke_line_start.is_none(),
        _ => false,
    }
}

/// Voxel coordinates one `apply_edit` call would affect (same rules; no mutation).
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
) -> Vec<VoxelCoord> {
    let grid_size = file.grid_size.max(1);
    let offsets = brush_offset_cells(brush_shape, brush_radius);
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
    );
    if centers.is_empty() {
        return Vec::new();
    }
    let mut seen: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut out: Vec<VoxelCoord> = Vec::new();

    match tool {
        EditTool::Add => {
            for (cx, cy, cz) in centers {
                for (dx, dy, dz) in &offsets {
                    let x = cx + dx;
                    let y = cy + dy;
                    let z = cz + dz;
                    if !in_grid(x, y, z, grid_size) {
                        continue;
                    }
                    if !spray_passes((x, y, z), spray) {
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
        EditTool::Remove => {
            for (hx, hy, hz) in centers {
                for (dx, dy, dz) in &offsets {
                    let x = hx + dx;
                    let y = hy + dy;
                    let z = hz + dz;
                    if !spray_passes((x, y, z), spray) {
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
        }
        EditTool::Paint => {
            for (hx, hy, hz) in centers {
                for (dx, dy, dz) in &offsets {
                    let x = hx + dx;
                    let y = hy + dy;
                    let z = hz + dz;
                    if !spray_passes((x, y, z), spray) {
                        continue;
                    }
                    if !seen.insert((x, y, z)) {
                        continue;
                    }
                    let Some(&idx) = voxel_map.get(&(x, y, z)) else {
                        continue;
                    };
                    let before = file.voxels[idx];
                    if before.color == color && before.material == material {
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
    color: u32,
    material: MaterialId,
    coords: &AHashSet<VoxelCoord>,
) -> Vec<VoxelEditDelta> {
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
                    color,
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
                if before.color == color && before.material == material {
                    continue;
                }
                let after = Voxel {
                    color,
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
) -> Vec<VoxelCoord> {
    let grid_size = file.grid_size.max(1);
    let offsets = brush_offset_cells(brush_shape, brush_radius);
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
    );
    if centers.is_empty() {
        return Vec::new();
    }

    let mut seen: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut out: Vec<VoxelCoord> = Vec::new();

    for (hx, hy, hz) in centers {
        for (dx, dy, dz) in &offsets {
            let x = hx + dx;
            let y = hy + dy;
            let z = hz + dz;
            if !in_grid(x, y, z, grid_size) {
                continue;
            }
            if !spray_passes((x, y, z), spray) {
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
) -> Vec<VoxelCoord> {
    let grid_size = file.grid_size.max(1);
    let offsets = brush_offset_cells(brush_shape, brush_radius);
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
    );
    if centers.is_empty() {
        return Vec::new();
    }

    let mut seen: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut out: Vec<VoxelCoord> = Vec::new();

    for (cx, cy, cz) in centers {
        for (dx, dy, dz) in &offsets {
            let x = cx + dx;
            let y = cy + dy;
            let z = cz + dz;
            if !in_grid(x, y, z, grid_size) {
                continue;
            }
            if !spray_passes((x, y, z), spray) {
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

/// Keep empty coords that lie on the coplanar-void plane from screen (same plane as `coplanar_empty_connected_from_screen`).
pub fn filter_coords_coplanar_empty_from_screen(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    coords: &[VoxelCoord],
) -> Vec<VoxelCoord> {
    let grid_size = file.grid_size.max(1);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((hit, prev)) = ray_first_solid(origin, dir, voxel_map, grid_size) else {
        return Vec::new();
    };
    let Some(prev) = prev else {
        return Vec::new();
    };
    let Some((axis, fixed)) = plane_axis_fixed(prev, hit) else {
        return Vec::new();
    };
    coords
        .iter()
        .copied()
        .filter(|c| {
            !voxel_map.contains_key(c) && voxel_on_plane(*c, axis, fixed)
        })
        .collect()
}

fn neighbors_26(c: VoxelCoord) -> Vec<VoxelCoord> {
    let (x, y, z) = c;
    let mut v = Vec::with_capacity(26);
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                v.push((x + dx, y + dy, z + dz));
            }
        }
    }
    v
}

/// Flood BFS over solid voxels from screen pick (selection fill). `respect_color`: only like-colored
/// to seed; if false, include any solid connected in the chosen adjacency.
pub fn flood_fill_selection_coords(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    fill_diagonals: bool,
    respect_color: bool,
    match_material: bool,
) -> Vec<VoxelCoord> {
    let grid_size = file.grid_size.max(1);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((hit, _)) = ray_first_solid(origin, dir, voxel_map, grid_size) else {
        return Vec::new();
    };
    let Some(&seed_idx) = voxel_map.get(&hit) else {
        return Vec::new();
    };
    let seed_v = file.voxels[seed_idx];
    let tc = seed_v.color;
    let tm = seed_v.material;

    let mut out: Vec<VoxelCoord> = Vec::new();
    let mut visited: AHashSet<VoxelCoord> = AHashSet::new();
    let mut queue: VecDeque<VoxelCoord> = VecDeque::new();
    queue.push_back(hit);

    while let Some(c) = queue.pop_front() {
        if visited.contains(&c) {
            continue;
        }
        visited.insert(c);
        let Some(&idx) = voxel_map.get(&c) else {
            continue;
        };
        let v = file.voxels[idx];
        if respect_color {
            if v.color != tc || (match_material && v.material != tm) {
                continue;
            }
        }
        out.push(c);

        let neigh: Vec<VoxelCoord> = if fill_diagonals {
            neighbors_26(c)
        } else {
            neighbors_6(c).to_vec()
        };
        for n in neigh {
            if !in_grid(n.0, n.1, n.2, grid_size) {
                continue;
            }
            if !visited.contains(&n) {
                queue.push_back(n);
            }
        }
    }
    out
}

/// Keep only coords whose voxels match `seed` color (and optionally material).
pub fn filter_coords_by_seed_color(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    coords: &[VoxelCoord],
    seed: Voxel,
    match_material: bool,
) -> Vec<VoxelCoord> {
    coords
        .iter()
        .copied()
        .filter(|c| {
            voxel_map.get(c).is_some_and(|&i| {
                let v = file.voxels[i];
                v.color == seed.color && (!match_material || v.material == seed.material)
            })
        })
        .collect()
}

/// Keep coords that lie on the same face plane as the coplanar pick from screen (solid hit).
pub fn filter_coords_coplanar_solid_from_screen(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    coords: &[VoxelCoord],
) -> Vec<VoxelCoord> {
    let grid_size = file.grid_size.max(1);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((hit, prev)) = ray_first_solid(origin, dir, voxel_map, grid_size) else {
        return Vec::new();
    };
    let Some(prev) = prev else {
        return Vec::new();
    };
    let Some((axis, fixed)) = plane_axis_fixed(prev, hit) else {
        return Vec::new();
    };
    coords
        .iter()
        .copied()
        .filter(|c| voxel_map.contains_key(c) && voxel_on_plane(*c, axis, fixed))
        .collect()
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

/// Web `SprayDirection` for wall extrusion.
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SprayDirection {
    #[default]
    Auto,
    None,
    Right,
    Left,
    Up,
    Down,
    Back,
    Forward,
}

#[inline]
fn sculpt_edit_tool(mode: SculptStrokeMode) -> EditTool {
    match mode {
        SculptStrokeMode::Draw
        | SculptStrokeMode::Wall
        | SculptStrokeMode::Extrude
        | SculptStrokeMode::Terrain => EditTool::Add,
        SculptStrokeMode::Smooth | SculptStrokeMode::Gouge => EditTool::Remove,
    }
}

fn stroke_anchor_centers_sculpt(
    mode: SculptStrokeMode,
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    stroke_line_start: Option<(f32, f32)>,
    stroke_segment_prev: Option<(f32, f32)>,
) -> Vec<(i32, i32, i32)> {
    let tool = sculpt_edit_tool(mode);
    stroke_anchor_centers_with_mode(
        DrawStrokeMode::Line,
        PlaneAxis::Auto,
        &StrokeAux::default(),
        tool,
        file,
        voxel_map,
        camera,
        width,
        height,
        sx,
        sy,
        0,
        stroke_line_start,
        stroke_segment_prev,
    )
}

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
    for y in (y_lo..=y_hi).rev() {
        if voxel_map.contains_key(&(x, y, z)) {
            return Some(y);
        }
    }
    None
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
) -> Vec<VoxelCoord> {
    let grid_size = file.grid_size.max(1);
    let offsets = brush_offset_cells(brush_shape, brush_radius);
    let spray = spray_density.clamp(0.0, 1.0);

    let spine = stroke_anchor_centers_sculpt(
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
            if !spray_passes((x, y, z), spray) {
                continue;
            }
            if seen.insert((x, y, z)) {
                footprint.push((x, y, z));
            }
        }
    }
    footprint
}

fn dist_sq_point_segment(
    px: f32,
    py: f32,
    pz: f32,
    ax: f32,
    ay: f32,
    az: f32,
    bx: f32,
    by: f32,
    bz: f32,
) -> f32 {
    let abx = bx - ax;
    let aby = by - ay;
    let abz = bz - az;
    let apx = px - ax;
    let apy = py - ay;
    let apz = pz - az;
    let ab_len_sq = abx * abx + aby * aby + abz * abz;
    if ab_len_sq < 1e-12 {
        let dx = px - ax;
        let dy = py - ay;
        let dz = pz - az;
        return dx * dx + dy * dy + dz * dz;
    }
    let mut t = (apx * abx + apy * aby + apz * abz) / ab_len_sq;
    t = t.clamp(0.0, 1.0);
    let qx = ax + t * abx;
    let qy = ay + t * aby;
    let qz = az + t * abz;
    let dx = px - qx;
    let dy = py - qy;
    let dz = pz - qz;
    dx * dx + dy * dy + dz * dz
}

fn min_dist_point_to_polyline(px: f32, py: f32, pz: f32, spine: &[(i32, i32, i32)]) -> f32 {
    if spine.is_empty() {
        return 0.0;
    }
    if spine.len() == 1 {
        let sx = spine[0].0 as f32 + 0.5;
        let sy = spine[0].1 as f32 + 0.5;
        let sz = spine[0].2 as f32 + 0.5;
        let dx = px - sx;
        let dy = py - sy;
        let dz = pz - sz;
        return (dx * dx + dy * dy + dz * dz).sqrt();
    }
    let mut min_d = f32::INFINITY;
    for i in 0..spine.len() - 1 {
        let ax = spine[i].0 as f32 + 0.5;
        let ay = spine[i].1 as f32 + 0.5;
        let az = spine[i].2 as f32 + 0.5;
        let bx = spine[i + 1].0 as f32 + 0.5;
        let by = spine[i + 1].1 as f32 + 0.5;
        let bz = spine[i + 1].2 as f32 + 0.5;
        let d = dist_sq_point_segment(px, py, pz, ax, ay, az, bx, by, bz).sqrt();
        if d < min_d {
            min_d = d;
        }
    }
    min_d
}

/// Mulberry32 — matches web `createSeededRng` (`strokeGeometry.ts`).
fn mulberry32_next(state: &mut u32) -> f32 {
    *state = state.wrapping_add(0x6d2b79f5);
    let mut t = *state;
    t = (t ^ (t >> 15)).wrapping_mul(t | 1);
    t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
    ((t ^ (t >> 14)) as f32) / 4294967296.0
}

/// Web `computeSculptVoxelWeights` + `filterPositionsBySculptBrush` (sculptBrushWeights.ts).
fn filter_sculpt_footprint_stochastic(
    footprint: Vec<VoxelCoord>,
    spine: &[(i32, i32, i32)],
    brush_radius: u32,
    falloff_100: u32,
    strength_100: u32,
    stroke_seed: u32,
) -> Vec<VoxelCoord> {
    let fall = (falloff_100.min(100) as f32) / 100.0;
    let str = (strength_100.max(1).min(100) as f32) / 100.0;
    if fall <= 1e-9 && str >= 1.0 - 1e-9 {
        return footprint;
    }

    let r_vox = (brush_radius as f32 * 0.5).max(1e-6);

    let mut spine_eff: Vec<(i32, i32, i32)> = spine.to_vec();
    if spine_eff.is_empty() && !footprint.is_empty() {
        spine_eff.push(footprint[0]);
    }

    let mut rng_state = stroke_seed;
    let mut out: Vec<VoxelCoord> = Vec::with_capacity(footprint.len());
    let mut seen: HashSet<(i32, i32, i32)> = HashSet::new();

    for (x, y, z) in footprint {
        if !seen.insert((x, y, z)) {
            continue;
        }
        let cx = x as f32 + 0.5;
        let cy = y as f32 + 0.5;
        let cz = z as f32 + 0.5;

        let mut w = 1.0f32;
        if fall > 1e-9 && !spine_eff.is_empty() {
            let d = min_dist_point_to_polyline(cx, cy, cz, &spine_eff);
            let t = (d / r_vox).min(1.0);
            let soft = (1.0 - t) * (1.0 - t);
            w = (1.0 - fall) + fall * soft;
            w = w.clamp(0.0, 1.0);
        }

        let p = w * str;
        if p >= 1.0 - 1e-9 {
            out.push((x, y, z));
            continue;
        }
        if p <= 1e-9 {
            continue;
        }
        let u = mulberry32_next(&mut rng_state);
        if u < p {
            out.push((x, y, z));
        }
    }
    out
}

fn snap_normal_to_axis(n: (i32, i32, i32)) -> (i32, i32, i32) {
    let ax = n.0.abs();
    let ay = n.1.abs();
    let az = n.2.abs();
    if ax >= ay && ax >= az {
        return (if n.0 >= 0 { 1 } else { -1 }, 0, 0);
    }
    if ay >= az {
        return (0, if n.1 >= 0 { 1 } else { -1 }, 0);
    }
    (0, 0, if n.2 >= 0 { 1 } else { -1 })
}

pub fn spray_direction_vector(
    dir: SprayDirection,
    face_normal: Option<(i32, i32, i32)>,
) -> Option<(i32, i32, i32)> {
    match dir {
        SprayDirection::Auto => face_normal.map(snap_normal_to_axis),
        SprayDirection::None => None,
        SprayDirection::Down => Some((0, -1, 0)),
        SprayDirection::Up => Some((0, 1, 0)),
        SprayDirection::Forward => Some((0, 0, -1)),
        SprayDirection::Back => Some((0, 0, 1)),
        SprayDirection::Left => Some((-1, 0, 0)),
        SprayDirection::Right => Some((1, 0, 0)),
    }
}

fn wall_lock_axis(dir: SprayDirection, face_n: Option<(i32, i32, i32)>) -> Option<usize> {
    match dir {
        SprayDirection::Auto => {
            let d = spray_direction_vector(SprayDirection::Auto, face_n)?;
            if d.0 != 0 {
                Some(0)
            } else if d.1 != 0 {
                Some(1)
            } else {
                Some(2)
            }
        }
        SprayDirection::Left | SprayDirection::Right => Some(0),
        SprayDirection::Down | SprayDirection::Up => Some(1),
        SprayDirection::Forward | SprayDirection::Back => Some(2),
        SprayDirection::None => None,
    }
}

fn perpendicular_step_thick(dir: (i32, i32, i32)) -> (i32, i32, i32) {
    if dir.0 != 0 {
        (0, 1, 0)
    } else if dir.1 != 0 {
        (1, 0, 0)
    } else {
        (0, 1, 0)
    }
}

fn thicken_path_in_plane_wall(
    positions: &[(i32, i32, i32)],
    radius: f32,
    plane_normal_axis: usize,
) -> Vec<(i32, i32, i32)> {
    if radius <= 0.0 {
        return positions.to_vec();
    }
    let lo = -radius.ceil() as i32;
    let hi = radius.floor() as i32;
    let mut seen: HashSet<(i32, i32, i32)> = positions.iter().copied().collect();
    let mut result: Vec<(i32, i32, i32)> = positions.to_vec();
    for &(px, py, pz) in positions {
        match plane_normal_axis {
            0 => {
                for dy in lo..=hi {
                    for dz in lo..=hi {
                        let p = (px, py + dy, pz + dz);
                        if seen.insert(p) {
                            result.push(p);
                        }
                    }
                }
            }
            1 => {
                for dx in lo..=hi {
                    for dz in lo..=hi {
                        let p = (px + dx, py, pz + dz);
                        if seen.insert(p) {
                            result.push(p);
                        }
                    }
                }
            }
            _ => {
                for dx in lo..=hi {
                    for dy in lo..=hi {
                        let p = (px + dx, py + dy, pz);
                        if seen.insert(p) {
                            result.push(p);
                        }
                    }
                }
            }
        }
    }
    result
}

fn directional_streak_wall(
    base: &[(i32, i32, i32)],
    direction: (i32, i32, i32),
    streak_len: i32,
) -> Vec<(i32, i32, i32)> {
    let len = streak_len.max(0);
    if len == 0 {
        return base.to_vec();
    }
    let (dx, dy, dz) = direction;
    let mut seen: HashSet<(i32, i32, i32)> = base.iter().copied().collect();
    let mut result = base.to_vec();
    for &(px, py, pz) in base {
        for k in 1..=len {
            let p = (px + k * dx, py + k * dy, pz + k * dz);
            if seen.insert(p) {
                result.push(p);
            }
        }
    }
    result
}

/// Outward face normal (empty cell before solid along the ray).
pub fn outward_face_normal_from_screen_ray(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> Option<(i32, i32, i32)> {
    let grid_size = file.grid_size.max(1);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let (hit, prev) = ray_first_solid(origin, dir, voxel_map, grid_size)?;
    let prev = prev?;
    Some((prev.0 - hit.0, prev.1 - hit.1, prev.2 - hit.2))
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
    spray_direction: SprayDirection,
    wall_width_index: u32,
    wall_height_vox: u32,
    wall_lock_start_height: bool,
    wall_axis_align: bool,
    brush_radius: u32,
    brush_falloff: u32,
    brush_strength: u32,
    stroke_seed: u32,
) -> Vec<VoxelCoord> {
    let _ = wall_area_shape;

    let mut spine = stroke_anchor_centers_sculpt(
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
    );
    if spine.is_empty() {
        return Vec::new();
    }

    if wall_axis_align && spine.len() >= 2 {
        let a = spine[0];
        let b = spine[spine.len() - 1];
        spine = voxel_line_dda(a, b);
    }

    let face_out = outward_face_normal_from_screen_ray(file, voxel_map, camera, width, height, sx, sy);
    let face_snapped = face_out.map(snap_normal_to_axis);

    if wall_lock_start_height {
        if let Some(axis) = wall_lock_axis(spray_direction, face_snapped) {
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

    let dir_vec = spray_direction_vector(spray_direction, face_snapped);
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
        let perp = perpendicular_step_thick(dir_for_plane);
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
        base_positions = thicken_path_in_plane_wall(&base_positions, r, plane_normal_axis);
    }

    let h = wall_height_vox.max(2) as i32;
    let mut out = if let Some(dv) = dir_vec {
        directional_streak_wall(&base_positions, dv, h)
    } else {
        base_positions
    };

    let grid_size = file.grid_size.max(1);
    out.retain(|&(x, y, z)| in_grid(x, y, z, grid_size));

    filter_sculpt_footprint_stochastic(
        out,
        &spine_for_weights,
        brush_radius,
        brush_falloff,
        brush_strength,
        stroke_seed,
    )
}

/// Brush footprint after web-style strength / falloff thinning. Terrain skips thinning (column ops).
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
) -> Vec<VoxelCoord> {
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
    );
    if footprint.is_empty() {
        return footprint;
    }
    if matches!(mode, SculptStrokeMode::Terrain) {
        return footprint;
    }
    let spine = stroke_anchor_centers_sculpt(
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
    filter_sculpt_footprint_stochastic(
        footprint,
        &spine,
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
    stroke_line_start: Option<(f32, f32)>,
    stroke_segment_prev: Option<(f32, f32)>,
    terrain_op: Option<TerrainSculptOp>,
    terrain_base_y: i32,
    terrain_strength: i32,
    terrain_smooth_radius: i32,
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
) -> Result<Vec<VoxelEditDelta>, String> {
    let grid_size = file.grid_size.max(1);
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
            spray_direction,
            wall_width_index,
            wall_height_vox,
            wall_lock_start_height,
            wall_axis_align,
            brush_radius,
            brush_falloff,
            brush_strength,
            stroke_seed,
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
        )
    };
    if footprint.is_empty() {
        return Ok(Vec::new());
    }

    let spine = stroke_anchor_centers_sculpt(
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
            let mut candidates: Vec<VoxelCoord> = Vec::new();
            for (x, y, z) in &footprint {
                if voxel_map.contains_key(&(*x, *y, *z)) {
                    candidates.push((*x, *y, *z));
                }
            }
            if candidates.is_empty() {
                return Ok(Vec::new());
            }
            let passes = smooth_neighbor_passes.max(1);
            let mut out: Vec<VoxelEditDelta> = Vec::new();
            for _ in 0..passes {
                let mut paints: Vec<(VoxelCoord, u32, MaterialId)> = Vec::new();
                for &(x, y, z) in &candidates {
                    let mut counts: AHashMap<(u32, MaterialId), u32> = AHashMap::new();
                    for (nx, ny, nz) in neighbors_6((x, y, z)) {
                        if let Some(&idx) = voxel_map.get(&(nx, ny, nz)) {
                            let v = file.voxels[idx];
                            *counts.entry((v.color, v.material)).or_insert(0) += 1;
                        }
                    }
                    let Some(&idx) = voxel_map.get(&(x, y, z)) else {
                        continue;
                    };
                    let before = file.voxels[idx];
                    *counts.entry((before.color, before.material)).or_insert(0) += 1;
                    let mut best: Option<((u32, MaterialId), u32)> = None;
                    for (k, c) in counts {
                        match best {
                            None => best = Some((k, c)),
                            Some((_bk, bc)) if c > bc => best = Some((k, c)),
                            Some((bk, bc)) if c == bc && k.0 < bk.0 => best = Some((k, c)),
                            _ => {}
                        }
                    }
                    let Some(((nc, nm), _)) = best else { continue };
                    if nc != before.color || nm != before.material {
                        paints.push(((x, y, z), nc, nm));
                    }
                }
                for ((x, y, z), nc, nm) in paints {
                    let Some(&idx) = voxel_map.get(&(x, y, z)) else {
                        continue;
                    };
                    let before = file.voxels[idx];
                    if before.color == nc && before.material == nm {
                        continue;
                    }
                    let after = Voxel {
                        color: nc,
                        material: nm,
                        ..before
                    };
                    file.voxels[idx] = after;
                    out.push(VoxelEditDelta::Painted { before, after });
                }
            }
            Ok(out)
        }
        SculptStrokeMode::Terrain => {
            let op = terrain_op.unwrap_or(TerrainSculptOp::Raise);
            let base_y = terrain_base_y;
            let strength = terrain_strength.max(0).min(64);
            let smooth_r = terrain_smooth_radius.max(0).min(8);

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
            let brush_r_vox = brush_radius as f32 * 0.5 + 0.5;

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
                        let old_max = meta.3;
                        let y_fill = meta.2;
                        let t = terrain_brush_falloff(x, z, &spine, brush_r_vox);
                        let delta = (strength as f32 * t).round() as i32;
                        let old_h = old_max;
                        let h = if matches!(op, TerrainSculptOp::Raise) {
                            old_h + delta
                        } else {
                            (old_h - delta).max(y_fill - 1)
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
    let Some(&remove_idx) = voxel_map.get(&coord) else {
        return None;
    };
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

fn plane_axis_fixed(prev: VoxelCoord, hit: VoxelCoord) -> Option<(usize, i32)> {
    let dx = hit.0 - prev.0;
    let dy = hit.1 - prev.1;
    let dz = hit.2 - prev.2;
    if dx != 0 && dy == 0 && dz == 0 {
        Some((0, hit.0))
    } else if dy != 0 && dx == 0 && dz == 0 {
        Some((1, hit.1))
    } else if dz != 0 && dx == 0 && dy == 0 {
        Some((2, hit.2))
    } else {
        None
    }
}

#[inline]
fn voxel_on_plane(c: VoxelCoord, axis: usize, fixed: i32) -> bool {
    match axis {
        0 => c.0 == fixed,
        1 => c.1 == fixed,
        2 => c.2 == fixed,
        _ => false,
    }
}

fn neighbors_on_face_plane(axis: usize, c: VoxelCoord) -> [(i32, i32, i32); 4] {
    let (x, y, z) = c;
    match axis {
        0 => [(x, y + 1, z), (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)],
        1 => [(x + 1, y, z), (x - 1, y, z), (x, y, z + 1), (x, y, z - 1)],
        2 => [(x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z)],
        _ => [(0, 0, 0); 4],
    }
}

/// Face-connected voxels on the axis-aligned plane through the hit face (from ray `prev` → `hit`).
pub fn coplanar_connected_from_screen(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> Option<Vec<VoxelCoord>> {
    if file.voxels.is_empty() {
        return None;
    }
    let grid_size = file.grid_size.max(1);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let (hit, prev) = ray_first_solid(origin, dir, voxel_map, grid_size)?;
    let prev = prev?;
    let (axis, fixed) = plane_axis_fixed(prev, hit)?;
    if !voxel_map.contains_key(&hit) {
        return None;
    }

    let mut visited: AHashSet<VoxelCoord> = AHashSet::new();
    let mut queue: VecDeque<VoxelCoord> = VecDeque::new();
    let mut out: Vec<VoxelCoord> = Vec::new();
    visited.insert(hit);
    queue.push_back(hit);

    while let Some(c) = queue.pop_front() {
        if !voxel_map.contains_key(&c) {
            continue;
        }
        if !voxel_on_plane(c, axis, fixed) {
            continue;
        }
        out.push(c);
        for n in neighbors_on_face_plane(axis, c) {
            if !in_grid(n.0, n.1, n.2, grid_size) {
                continue;
            }
            if !voxel_on_plane(n, axis, fixed) {
                continue;
            }
            if !voxel_map.contains_key(&n) {
                continue;
            }
            if visited.insert(n) {
                queue.push_back(n);
            }
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
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
    color: u32,
    material: MaterialId,
) -> Result<Vec<VoxelEditDelta>, String> {
    let grid_size = file.grid_size.max(1);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((_, prev)) = ray_first_solid(origin, dir, voxel_map, grid_size) else {
        return Ok(Vec::new());
    };
    let Some((ax, ay, az)) = prev else {
        return Ok(Vec::new());
    };
    let mut seen: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut out: Vec<VoxelEditDelta> = Vec::new();
    for &(dx, dy, dz, _src_color, _src_mat) in &clip.entries {
        let x = ax + dx;
        let y = ay + dy;
        let z = az + dz;
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
) -> Result<Vec<VoxelEditDelta>, String> {
    let grid_size = file.grid_size.max(1);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((hit, _)) = ray_first_solid(origin, dir, voxel_map, grid_size) else {
        return Ok(Vec::new());
    };
    let (hx, hy, hz) = hit;
    let mut seen: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut out: Vec<VoxelEditDelta> = Vec::new();
    for &(dx, dy, dz, _, _) in &clip.entries {
        let x = hx + dx;
        let y = hy + dy;
        let z = hz + dz;
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

/// One-voxel “raise” sculpt: add above the top face of the ray-hit solid (if empty).
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
    let grid_size = file.grid_size.max(1);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((hit, _)) = ray_first_solid(origin, dir, voxel_map, grid_size) else {
        return Ok(Vec::new());
    };
    let (hx, hy, hz) = hit;
    let x = hx;
    let y = hy + 1;
    let z = hz;
    if !in_grid(x, y, z, grid_size) || voxel_map.contains_key(&(x, y, z)) {
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let (o, d) = screen_to_world_ray(&cam, w, h, sx, sy);
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
        let r = ray_first_solid(origin, dir, &m, 32);
        assert!(r.is_some());
        let ((x, y, z), prev) = r.unwrap();
        assert_eq!((x, y, z), (0, 0, 0));
        assert_eq!(prev, Some((0, 0, 1)));
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
            None,
            None,
            None,
            0,
            4,
            2,
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
        )
        .unwrap();
        assert!(!deltas.is_empty());
    }
}
