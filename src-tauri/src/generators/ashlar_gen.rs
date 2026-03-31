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

fn seed_to_int(seed: i32, lo: i32, hi: i32) -> i32 {
    let mut h = (seed as u32).wrapping_mul(0x9e3779b9);
    h = (h ^ (h >> 16)).wrapping_mul(0x85ebca6b);
    h = (h ^ (h >> 13)).wrapping_mul(0xc2b2ae35);
    lo + ((h % ((hi - lo + 1) as u32)) as i32)
}

/// Generate an ashlar (dressed stone) block: axis-aligned box with rough edges and rounded corners.
/// Placed in empty space in front of the clicked face, centered on the result.
pub fn generate_ashlar_deltas(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    face_empty: VoxelCoord,
    solid: VoxelCoord,
    seed: i32,
    size: i32,
    roughness: f32,
    color: u32,
    material: MaterialId,
    thickness: Option<i32>,
    thickness_axis: Option<i32>,
) -> Vec<VoxelEditDelta> {
    let s = size.max(1);
    let lo = (s / 2).max(3);
    let hi = (s + s / 2).min(20).max(lo);
    let mut wx = seed_to_int(seed, lo, hi).max(3);
    let mut wy = seed_to_int(seed + 1, lo, hi).max(3);
    let mut wz = seed_to_int(seed + 2, lo, hi).max(3);

    if let (Some(t), Some(axis)) = (thickness, thickness_axis) {
        let t = t.max(1).min(20);
        match axis {
            0 => wx = t,
            1 => wy = t,
            _ => wz = t,
        }
    }

    let rough = roughness.clamp(0.0, 1.0);
    let round_radius_sq = 1.4_f32 * 1.4;

    // Collect local-space voxels
    let mut local_cells: Vec<(i32, i32, i32)> = Vec::new();

    let corners: [(i32, i32, i32); 8] = [
        (0, 0, 0),
        (wx - 1, 0, 0),
        (0, wy - 1, 0),
        (wx - 1, wy - 1, 0),
        (0, 0, wz - 1),
        (wx - 1, 0, wz - 1),
        (0, wy - 1, wz - 1),
        (wx - 1, wy - 1, wz - 1),
    ];

    for x in 0..wx {
        for y in 0..wy {
            for z in 0..wz {
                let on_boundary = x == 0
                    || x == wx - 1
                    || y == 0
                    || y == wy - 1
                    || z == 0
                    || z == wz - 1;
                if on_boundary && rough > 0.0 {
                    let h = hash3(seed + 0xabcd_i32, x, y, z);
                    if h < rough * 0.4 {
                        continue;
                    }
                }
                // Rounded corners
                let near_corner = corners.iter().any(|&(cx, cy, cz)| {
                    let dx = (x - cx) as f32;
                    let dy = (y - cy) as f32;
                    let dz = (z - cz) as f32;
                    dx * dx + dy * dy + dz * dz < round_radius_sq
                });
                if near_corner {
                    continue;
                }
                local_cells.push((x, y, z));
            }
        }
    }

    if local_cells.is_empty() {
        return Vec::new();
    }

    // Center the block
    let cx = (wx - 1) as f32 / 2.0;
    let cy = (wy - 1) as f32 / 2.0;
    let cz = (wz - 1) as f32 / 2.0;

    // Place centered at face_empty, offset along normal
    let nx = (face_empty.0 - solid.0).signum();
    let ny = (face_empty.1 - solid.1).signum();
    let nz = (face_empty.2 - solid.2).signum();
    let half = ((wx.max(wy).max(wz)) as f32 / 2.0) as i32 + 1;
    let ox = face_empty.0 + nx * half;
    let oy = face_empty.1 + ny * half;
    let oz = face_empty.2 + nz * half;

    let mut out = Vec::new();
    let mut seen: HashSet<VoxelCoord> = HashSet::new();

    for &(lx, ly, lz) in &local_cells {
        let x = ox + (lx as f32 - cx).round() as i32;
        let y = oy + (ly as f32 - cy).round() as i32;
        let z = oz + (lz as f32 - cz).round() as i32;
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
    out
}

/// Face-click ashlar generator (web parity).
pub fn generator_ashlar_at_screen(
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
    thickness: Option<i32>,
    thickness_axis: Option<i32>,
) -> Result<Vec<VoxelEditDelta>, String> {
    let grid_size = effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((solid, prev)) = ray_first_solid(origin, dir, voxel_map, grid_size) else {
        return Ok(Vec::new());
    };
    let Some(face_empty) = prev else {
        return Ok(Vec::new());
    };
    Ok(generate_ashlar_deltas(
        file,
        voxel_map,
        face_empty,
        solid,
        seed,
        size,
        roughness,
        color,
        material,
        thickness,
        thickness_axis,
    ))
}
