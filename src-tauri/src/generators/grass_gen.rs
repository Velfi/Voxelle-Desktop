use crate::camera::OrbitCamera;
use crate::greedy_mesh::VoxelCoord;
use crate::voxel_edit::{
    effective_ray_grid_size, ensure_grid_fits_coord, ray_first_solid, screen_to_world_ray,
    VoxelEditDelta,
};
use crate::voxelle::{MaterialId, Voxel, VoxelleFile};
use ahash::AHashMap;
use std::collections::HashSet;

fn rnd(x: i32, z: i32, s: i32) -> f32 {
    let h = (x.wrapping_mul(92837111) ^ z.wrapping_mul(689287499) ^ s) as u32;
    (h as f32) / (u32::MAX as f32)
}

/// Short blades along outward normal from the face plane.
pub fn generate_grass_on_face_deltas(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    face_empty: VoxelCoord,
    solid: VoxelCoord,
    seed: i32,
    density: i32,
    max_height: i32,
    color: u32,
    material: MaterialId,
) -> Vec<VoxelEditDelta> {
    let nx = face_empty.0 - solid.0;
    let ny = face_empty.1 - solid.1;
    let nz = face_empty.2 - solid.2;
    if nx.abs() + ny.abs() + nz.abs() != 1 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut seen: HashSet<VoxelCoord> = HashSet::new();
    let spread = density.max(1).min(8);
    let mh = max_height.max(1).min(8);
    for u in -spread..=spread {
        for v in -spread..=spread {
            if rnd(u + seed, v, seed) > 0.4 {
                continue;
            }
            let base = if nx != 0 {
                (
                    face_empty.0,
                    face_empty.1 + u,
                    face_empty.2 + v,
                )
            } else if ny != 0 {
                (
                    face_empty.0 + u,
                    face_empty.1,
                    face_empty.2 + v,
                )
            } else {
                (
                    face_empty.0 + u,
                    face_empty.1 + v,
                    face_empty.2,
                )
            };
            let h = 1 + (rnd(base.1, base.2, seed + 9) * mh as f32) as i32;
            for t in 0..h {
                let x = base.0 + nx * t;
                let y = base.1 + ny * t;
                let z = base.2 + nz * t;
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

pub fn generator_grass_at_screen(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    seed: i32,
    density: i32,
    max_height: i32,
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
    Ok(generate_grass_on_face_deltas(
        file,
        voxel_map,
        face_empty,
        solid,
        seed,
        density,
        max_height,
        color,
        material,
    ))
}
