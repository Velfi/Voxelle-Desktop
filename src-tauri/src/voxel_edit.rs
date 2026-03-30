//! Screen-space ray → grid traversal for add/remove voxel editing.

use crate::camera::OrbitCamera;
use crate::greedy_mesh::VoxelCoord;
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
fn ray_first_solid(
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
fn anchor_for_edit(
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

/// Stroke anchor cells: line between press and cursor, segment between previous and current sample,
/// or a single-ray sample when neither line nor segment is set.
fn stroke_anchor_centers(
    tool: EditTool,
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
    if let Some((lsx, lsy)) = stroke_line_start {
        match (
            anchor_for_edit(tool, file, voxel_map, camera, width, height, lsx, lsy),
            anchor_for_edit(tool, file, voxel_map, camera, width, height, sx, sy),
        ) {
            (Some(a), Some(b)) => voxel_line_dda(a, b),
            _ => anchor_for_edit(tool, file, voxel_map, camera, width, height, sx, sy)
                .into_iter()
                .collect(),
        }
    } else if let Some((px, py)) = stroke_segment_prev {
        match (
            anchor_for_edit(tool, file, voxel_map, camera, width, height, px, py),
            anchor_for_edit(tool, file, voxel_map, camera, width, height, sx, sy),
        ) {
            (Some(a), Some(b)) => voxel_line_dda(a, b),
            _ => anchor_for_edit(tool, file, voxel_map, camera, width, height, sx, sy)
                .into_iter()
                .collect(),
        }
    } else {
        anchor_for_edit(tool, file, voxel_map, camera, width, height, sx, sy)
            .into_iter()
            .collect()
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

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BrushShape {
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
fn voxel_line_dda(a: (i32, i32, i32), b: (i32, i32, i32)) -> Vec<(i32, i32, i32)> {
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

fn neighbors_6(c: VoxelCoord) -> [VoxelCoord; 6] {
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
) -> Result<Vec<VoxelEditDelta>, String> {
    let grid_size = file.grid_size.max(1);
    let offsets = brush_offset_cells(brush_shape, brush_radius);
    let spray = spray_density.clamp(0.0, 1.0);
    let centers = stroke_anchor_centers(
        tool,
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

/// Demo generator: filled sphere of voxels at the add-tool anchor.
pub fn generator_sphere_at_screen(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    radius: i32,
    color: u32,
    material: MaterialId,
) -> Result<Vec<VoxelEditDelta>, String> {
    let grid_size = file.grid_size.max(1);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((_, prev)) = ray_first_solid(origin, dir, voxel_map, grid_size) else {
        return Ok(Vec::new());
    };
    let Some((cx, cy, cz)) = prev else {
        return Ok(Vec::new());
    };
    let r = radius.max(1).min(12);
    let r2 = r * r;
    let mut seen: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut out: Vec<VoxelEditDelta> = Vec::new();
    for dx in -r..=r {
        for dy in -r..=r {
            for dz in -r..=r {
                if dx * dx + dy * dy + dz * dz > r2 {
                    continue;
                }
                let x = cx + dx;
                let y = cy + dy;
                let z = cz + dz;
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
        }
    }
    Ok(out)
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
}
