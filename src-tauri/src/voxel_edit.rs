//! Screen-space ray → grid traversal for add/remove voxel editing.

use crate::camera::OrbitCamera;
use crate::greedy_mesh::VoxelCoord;
use crate::voxelle::{MaterialId, Voxel, VoxelleFile};
use glam::{Vec3, Vec4};
use std::collections::HashMap;

pub fn screen_to_world_ray(camera: &OrbitCamera, width: f32, height: f32, sx: f32, sy: f32) -> (Vec3, Vec3) {
    let w = width.max(1.0);
    let h = height.max(1.0);
    let ndc_x = (sx / w) * 2.0 - 1.0;
    let ndc_y = 1.0 - (sy / h) * 2.0;
    let proj = camera.proj_matrix(width, height);
    let view = camera.view_matrix();
    let inv_proj = proj.inverse();
    let inv_view = view.inverse();
    let clip_near = Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
    let clip_far = Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
    let mut vn = inv_proj * clip_near;
    let mut vf = inv_proj * clip_far;
    vn /= vn.w;
    vf /= vf.w;
    let world_near = inv_view * Vec4::new(vn.x, vn.y, vn.z, 1.0);
    let world_far = inv_view * Vec4::new(vf.x, vf.y, vf.z, 1.0);
    let o = world_near.truncate();
    let d = (world_far.truncate() - o).normalize();
    (o, d)
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
    occupied: &HashMap<VoxelCoord, Voxel>,
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
    voxel_map: &HashMap<VoxelCoord, Voxel>,
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

/// Cell where an add would place (empty cell in front of first solid along the ray), if valid.
pub fn preview_add_cell(
    file: &VoxelleFile,
    voxel_map: &HashMap<VoxelCoord, Voxel>,
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
    voxel_map: &HashMap<VoxelCoord, Voxel>,
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

pub fn apply_edit(
    file: &mut VoxelleFile,
    voxel_map: &mut HashMap<VoxelCoord, Voxel>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    add: bool,
) -> Result<bool, String> {
    let grid_size = file.grid_size.max(1);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);

    if add {
        if let Some((_hit, prev)) = ray_first_solid(origin, dir, &*voxel_map, grid_size) {
            if let Some((px, py, pz)) = prev {
                if in_grid(px, py, pz, grid_size) && !voxel_map.contains_key(&(px, py, pz)) {
                    let nv = Voxel {
                        x: px,
                        y: py,
                        z: pz,
                        color: 0x8899aa,
                        material: MaterialId::Plastic,
                    };
                    file.voxels.push(nv);
                    voxel_map.insert((px, py, pz), nv);
                    return Ok(true);
                }
            }
        }
        Ok(false)
    } else {
        if let Some((hit, _)) = ray_first_solid(origin, dir, &*voxel_map, grid_size) {
            file.voxels.retain(|v| !(v.x == hit.0 && v.y == hit.1 && v.z == hit.2));
            voxel_map.remove(&hit);
            return Ok(true);
        }
        Ok(false)
    }
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
    fn ray_hits_center_voxel() {
        let mut m: HashMap<VoxelCoord, Voxel> = HashMap::new();
        m.insert(
            (0, 0, 0),
            Voxel {
                x: 0,
                y: 0,
                z: 0,
                color: 1,
                material: crate::voxelle::MaterialId::Plastic,
            },
        );
        let origin = Vec3::new(0.0, 0.0, 5.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);
        let r = ray_first_solid(origin, dir, &m, 32);
        assert!(r.is_some());
        let ((x, y, z), prev) = r.unwrap();
        assert_eq!((x, y, z), (0, 0, 0));
        assert_eq!(prev, Some((0, 0, 1)));
    }
}
