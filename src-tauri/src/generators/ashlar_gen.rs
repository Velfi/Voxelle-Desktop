use crate::camera::OrbitCamera;
use crate::greedy_mesh::VoxelCoord;
use crate::voxel_edit::{
    effective_ray_grid_size, ensure_grid_fits_coord, ray_first_solid, screen_to_world_ray,
    VoxelEditDelta,
};
use crate::voxelle::{MaterialId, Voxel, VoxelleFile};
use ahash::AHashMap;
use std::collections::HashSet;

fn hash3(seed: i32, x: i32, y: i32, z: i32) -> f32 {
    let mut h = (seed as u32)
        ^ (x.wrapping_mul(73856093) as u32)
        ^ (y.wrapping_mul(19349663) as u32)
        ^ (z.wrapping_mul(83492791) as u32);
    h = (h ^ (h >> 16)).wrapping_mul(0x85ebca6b);
    h = (h ^ (h >> 13)).wrapping_mul(0xc2b2ae35);
    h = h ^ (h >> 16);
    (h as f32) / (u32::MAX as f32)
}

fn seed_to_int(seed: i32, lo: i32, hi: i32) -> i32 {
    let mut h = (seed as u32).wrapping_mul(0x9e3779b9);
    h = (h ^ (h >> 16)).wrapping_mul(0x85ebca6b);
    h = (h ^ (h >> 13)).wrapping_mul(0xc2b2ae35);
    lo + ((h % ((hi - lo + 1) as u32)) as i32)
}

/// Pure coordinate generator — returns world-space voxel positions for an ashlar block.
/// Does not mutate any state.
pub fn ashlar_world_coords(
    face_empty: VoxelCoord,
    solid: VoxelCoord,
    seed: i32,
    size: i32,
    roughness: f32,
    thickness: Option<i32>,
    thickness_axis: Option<i32>,
) -> Vec<VoxelCoord> {
    let s = size.max(1);
    let lo = (s / 2).max(3);
    let hi = (s + s / 2).min(20).max(lo);
    let mut wx = seed_to_int(seed, lo, hi).max(3);
    let mut wy = seed_to_int(seed + 1, lo, hi).max(3);
    let mut wz = seed_to_int(seed + 2, lo, hi).max(3);

    let fn_x = (face_empty.0 - solid.0).signum();
    let fn_y = (face_empty.1 - solid.1).signum();
    let fn_z = (face_empty.2 - solid.2).signum();

    // Resolve thickness axis: use explicit value, or auto-detect from face normal.
    let resolved_axis = thickness_axis.or({
        if fn_x != 0 {
            Some(0)
        } else if fn_y != 0 {
            Some(1)
        } else if fn_z != 0 {
            Some(2)
        } else {
            None
        }
    });
    if let (Some(t), Some(axis)) = (thickness, resolved_axis) {
        let t = t.clamp(1, 20);
        match axis {
            0 => wx = t,
            1 => wy = t,
            _ => wz = t,
        }
    }

    let rough = roughness.clamp(0.0, 1.0);
    let round_radius_sq = 1.4_f32 * 1.4;

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

    let mut local_cells: Vec<(i32, i32, i32)> = Vec::new();
    for x in 0..wx {
        for y in 0..wy {
            for z in 0..wz {
                let on_boundary =
                    x == 0 || x == wx - 1 || y == 0 || y == wy - 1 || z == 0 || z == wz - 1;
                if on_boundary && rough > 0.0 {
                    let h = hash3(seed + 0xabcd_i32, x, y, z);
                    if h < rough * 0.4 {
                        continue;
                    }
                }
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

    // Center of the block in local space
    let cx = (wx - 1) as f32 / 2.0;
    let cy = (wy - 1) as f32 / 2.0;
    let cz = (wz - 1) as f32 / 2.0;

    // Offset along the face normal by half the normal-axis extent so the block
    // sits adjacent to the clicked face (not floating with a large gap).
    let normal_dim = if fn_x != 0 {
        wx
    } else if fn_y != 0 {
        wy
    } else {
        wz
    };
    let half = ((normal_dim as f32) / 2.0).ceil() as i32;
    let ox = face_empty.0 + fn_x * half;
    let oy = face_empty.1 + fn_y * half;
    let oz = face_empty.2 + fn_z * half;

    // Use floor(offset + 0.5) to match JavaScript's Math.round behaviour:
    // round(-0.5) = 0 (not -1), so even-dimension blocks have no gap at centre.
    let mut out = Vec::new();
    let mut seen: HashSet<VoxelCoord> = HashSet::new();
    for &(lx, ly, lz) in &local_cells {
        let x = ox + (lx as f32 - cx + 0.5).floor() as i32;
        let y = oy + (ly as f32 - cy + 0.5).floor() as i32;
        let z = oz + (lz as f32 - cz + 0.5).floor() as i32;
        if seen.insert((x, y, z)) {
            out.push((x, y, z));
        }
    }
    out
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
    let coords = ashlar_world_coords(
        face_empty,
        solid,
        seed,
        size,
        roughness,
        thickness,
        thickness_axis,
    );
    let mut out = Vec::new();
    for (x, y, z) in coords {
        ensure_grid_fits_coord(file, x, y, z);
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

/// Preview coords for the ashlar generator (no world mutation).
pub fn preview_ashlar_at_screen(
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
    thickness: i32,
) -> Vec<VoxelCoord> {
    let grid_size = effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((solid, prev)) = ray_first_solid(origin, dir, voxel_map, grid_size) else {
        return Vec::new();
    };
    let Some(face_empty) = prev else {
        return Vec::new();
    };
    ashlar_world_coords(
        face_empty,
        solid,
        seed,
        size,
        roughness,
        Some(thickness),
        None,
    )
    .into_iter()
    .filter(|c| !voxel_map.contains_key(c))
    .collect()
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
