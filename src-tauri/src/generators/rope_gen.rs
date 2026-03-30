use crate::camera::OrbitCamera;
use crate::greedy_mesh::VoxelCoord;
use crate::voxel_edit::{in_grid, ray_first_solid, screen_to_world_ray, VoxelEditDelta};
use crate::voxelle::{MaterialId, Voxel, VoxelleFile};
use ahash::AHashMap;
use std::collections::HashSet;

/// Discrete voxel samples along a catenary between `a` and `b` in world space (grid coords).
pub fn catenary_voxel_arc(a: VoxelCoord, b: VoxelCoord, sag: f32, segments: i32) -> Vec<VoxelCoord> {
    let n = segments.max(4).min(128);
    let ax = a.0 as f32;
    let ay = a.1 as f32;
    let az = a.2 as f32;
    let bx = b.0 as f32;
    let by = b.1 as f32;
    let bz = b.2 as f32;
    let mut out = Vec::new();
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let x = ax + (bx - ax) * t;
        let y = ay + (by - ay) * t - sag * (1.0 - (2.0 * t - 1.0).powi(2)).max(0.0);
        let z = az + (bz - az) * t;
        let vx = x.round() as i32;
        let vy = y.round() as i32;
        let vz = z.round() as i32;
        if out.last().copied() != Some((vx, vy, vz)) {
            out.push((vx, vy, vz));
        }
    }
    out
}

pub fn generator_rope_between_screens(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx1: f32,
    sy1: f32,
    sx2: f32,
    sy2: f32,
    sag: f32,
    color: u32,
    material: MaterialId,
) -> Result<Vec<VoxelEditDelta>, String> {
    let grid_size = file.grid_size.max(1);
    let (o1, d1) = screen_to_world_ray(camera, width, height, sx1, sy1);
    let (o2, d2) = screen_to_world_ray(camera, width, height, sx2, sy2);
    let Some((h1, _)) = ray_first_solid(o1, d1, voxel_map, grid_size) else {
        return Ok(Vec::new());
    };
    let Some((h2, _)) = ray_first_solid(o2, d2, voxel_map, grid_size) else {
        return Ok(Vec::new());
    };
    let path = catenary_voxel_arc(h1, h2, sag, 48);
    let mut out = Vec::new();
    let mut seen: HashSet<VoxelCoord> = HashSet::new();
    for (x, y, z) in path {
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
