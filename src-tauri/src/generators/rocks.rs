use crate::camera::OrbitCamera;
use crate::greedy_mesh::VoxelCoord;
use crate::voxel_edit::{
    effective_ray_grid_size, ensure_grid_fits_coord, ray_first_solid, screen_to_world_ray,
    VoxelEditDelta,
};
use crate::voxelle::{MaterialId, Voxel, VoxelleFile};
use ahash::AHashMap;
use std::collections::HashSet;

fn hash3(x: i32, y: i32, z: i32, seed: i32) -> f32 {
    let h = (x.wrapping_mul(73856093) as i64)
        ^ (y.wrapping_mul(19349663) as i64)
        ^ (z.wrapping_mul(83492791) as i64)
        ^ ((seed as i64) << 20);
    let u = (h as u64).wrapping_mul(6364136223846793005);
    (u as f32) / (u64::MAX as f32)
}

/// Place a noisy ellipsoid rock in empty space, with its bottom near the clicked face.
/// `face_empty` is an empty voxel; `solid` is the voxel behind it along the ray (interior).
pub fn generate_rock_cluster_deltas(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    face_empty: VoxelCoord,
    solid: VoxelCoord,
    seed: i32,
    size: i32,
    roughness: f32,
    color: u32,
    material: MaterialId,
) -> Vec<VoxelEditDelta> {
    let nx = (face_empty.0 - solid.0).signum();
    let ny = (face_empty.1 - solid.1).signum();
    let nz = (face_empty.2 - solid.2).signum();
    let r = size.max(1).min(12);
    let lump = roughness.clamp(0.0, 1.0) * 0.45;
    let sx = 0.75 + hash3(seed, 1, 0, 0) * 0.5;
    let sy = 0.75 + hash3(seed, 2, 0, 0) * 0.5;
    let sz = 0.75 + hash3(seed, 3, 0, 0) * 0.5;
    // Center rock volume mostly in empty space in front of the face
    let cx = face_empty.0 + nx * (r / 2 + 1);
    let cy = face_empty.1 + ny * (r / 2 + 1);
    let cz = face_empty.2 + nz * (r / 2 + 1);
    let mut out = Vec::new();
    let mut seen: HashSet<VoxelCoord> = HashSet::new();
    let rmax = r + 2;
    for dx in -rmax..=rmax {
        for dy in -rmax..=rmax {
            for dz in -rmax..=rmax {
                let px = dx as f32 / sx;
                let py = dy as f32 / sy;
                let pz = dz as f32 / sz;
                let d = (px * px + py * py + pz * pz).sqrt();
                let n = hash3(dx + seed, dy, dz, seed);
                let perturb = (n - 0.5) * 2.0 * lump;
                let re = r as f32 * (1.0 + perturb);
                if d > re {
                    continue;
                }
                let x = cx + dx;
                let y = cy + dy;
                let z = cz + dz;
                ensure_grid_fits_coord(file, x, y, z);
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
    out
}

/// Preview-only: compute the set of voxel coords a rock cluster would occupy,
/// without mutating the file. Used for hover preview.
pub fn preview_rock_cluster_coords(
    face_empty: VoxelCoord,
    solid: VoxelCoord,
    seed: i32,
    size: i32,
    roughness: f32,
) -> Vec<VoxelCoord> {
    let nx = (face_empty.0 - solid.0).signum();
    let ny = (face_empty.1 - solid.1).signum();
    let nz = (face_empty.2 - solid.2).signum();
    let r = size.max(1).min(12);
    let lump = roughness.clamp(0.0, 1.0) * 0.45;
    let sx = 0.75 + hash3(seed, 1, 0, 0) * 0.5;
    let sy = 0.75 + hash3(seed, 2, 0, 0) * 0.5;
    let sz = 0.75 + hash3(seed, 3, 0, 0) * 0.5;
    let cx = face_empty.0 + nx * (r / 2 + 1);
    let cy = face_empty.1 + ny * (r / 2 + 1);
    let cz = face_empty.2 + nz * (r / 2 + 1);
    let mut out = Vec::new();
    let mut seen: HashSet<VoxelCoord> = HashSet::new();
    let rmax = r + 2;
    for dx in -rmax..=rmax {
        for dy in -rmax..=rmax {
            for dz in -rmax..=rmax {
                let px = dx as f32 / sx;
                let py = dy as f32 / sy;
                let pz = dz as f32 / sz;
                let d = (px * px + py * py + pz * pz).sqrt();
                let n = hash3(dx + seed, dy, dz, seed);
                let perturb = (n - 0.5) * 2.0 * lump;
                let re = r as f32 * (1.0 + perturb);
                if d > re {
                    continue;
                }
                let coord = (cx + dx, cy + dy, cz + dz);
                if seen.insert(coord) {
                    out.push(coord);
                }
            }
        }
    }
    out
}

/// Preview rock cluster from screen coordinates (for hover preview).
pub fn preview_rock_at_screen(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    seed: i32,
    size: i32,
    roughness: f32,
) -> Vec<VoxelCoord> {
    let grid_size = effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((solid, prev)) = ray_first_solid(origin, dir, voxel_map, grid_size) else {
        return Vec::new();
    };
    let Some(face_empty) = prev else {
        return Vec::new();
    };
    // Filter out coords that already have voxels
    preview_rock_cluster_coords(face_empty, solid, seed, size, roughness)
        .into_iter()
        .filter(|c| !voxel_map.contains_key(c))
        .collect()
}

/// Face-click rocks generator (web parity).
pub fn generator_rocks_at_screen(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    seed: i32,
    size: i32,
    roughness: f32,
    color: u32,
    material: MaterialId,
) -> Result<Vec<VoxelEditDelta>, String> {
    let grid_size = effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((solid, prev)) = ray_first_solid(origin, dir, voxel_map, grid_size) else {
        return Ok(Vec::new());
    };
    let Some(face_empty) = prev else {
        return Ok(Vec::new());
    };
    Ok(generate_rock_cluster_deltas(
        file,
        voxel_map,
        face_empty,
        solid,
        seed,
        size,
        roughness,
        color,
        material,
    ))
}
