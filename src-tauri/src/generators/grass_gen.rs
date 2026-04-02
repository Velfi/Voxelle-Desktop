use crate::camera::OrbitCamera;
use crate::greedy_mesh::VoxelCoord;
use crate::voxel_edit::{
    effective_ray_grid_size, ensure_grid_fits_coord, ray_first_solid, screen_to_world_ray,
    VoxelEditDelta,
};
use crate::voxelle::{MaterialId, Voxel, VoxelleFile};
use ahash::AHashMap;
use std::collections::HashSet;

/// Mulberry32 seeded PRNG – matches the web version exactly.
struct Rng {
    state: u32,
}

impl Rng {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    /// Returns a value in [0, 1).
    fn next(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x6d2b79f5);
        let mut t = self.state;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        ((t ^ (t >> 14)) as f32) / 4_294_967_296.0
    }
}

/// Blades shorter than this grow perfectly straight.
const BEND_THRESHOLD: i32 = 5;
/// Base lateral drift per voxel of height above the threshold.
const BEND_RATE_BASE: f32 = 0.15;
/// +/- variation applied to the bend rate per blade.
const BEND_RATE_VARIANCE: f32 = 0.10;

/// Two integer tangent vectors perpendicular to the face normal.
fn tangent_vectors(nx: i32, ny: i32) -> ([i32; 3], [i32; 3]) {
    if nx != 0 {
        ([0, 1, 0], [0, 0, 1])
    } else if ny != 0 {
        ([1, 0, 0], [0, 0, 1])
    } else {
        ([1, 0, 0], [0, 1, 0])
    }
}

/// Yields voxel positions for one grass blade, with optional bending for tall blades.
fn blade_coords(
    bx: i32,
    by: i32,
    bz: i32,
    nx: i32,
    ny: i32,
    nz: i32,
    t1: [i32; 3],
    t2: [i32; 3],
    blade_h: i32,
    bend_angle: f32,
    bend_rate: f32,
) -> impl Iterator<Item = (i32, i32, i32)> {
    (0..blade_h).map(move |k| {
        let (off1, off2) = if blade_h >= BEND_THRESHOLD && k >= BEND_THRESHOLD {
            let progress = (k - BEND_THRESHOLD) as f32;
            (
                (bend_angle.cos() * bend_rate * progress).round() as i32,
                (bend_angle.sin() * bend_rate * progress).round() as i32,
            )
        } else {
            (0, 0)
        };
        (
            bx + k * nx + off1 * t1[0] + off2 * t2[0],
            by + k * ny + off1 * t1[1] + off2 * t2[1],
            bz + k * nz + off1 * t1[2] + off2 * t2[2],
        )
    })
}

/// Generate grass voxels on a surface face. Blades grow along +normal from a
/// circular disk in the tangent plane — matching the web version's algorithm.
pub fn generate_grass_on_face_deltas(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    face_empty: VoxelCoord,
    solid: VoxelCoord,
    seed: i32,
    radius: i32,
    density: f32,
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

    let r = radius.max(0);
    let max_h = max_height.clamp(1, 40);
    let dens = density.clamp(0.0, 1.0);
    let (t1, t2) = tangent_vectors(nx, ny);
    let (cx, cy, cz) = face_empty;
    let seed_u = seed as u32;

    let mut out = Vec::new();
    let mut seen: HashSet<VoxelCoord> = HashSet::new();

    for i in -r..=r {
        for j in -r..=r {
            // Circular disk – skip corners outside the radius
            if i * i + j * j > r * r {
                continue;
            }
            let blade_seed =
                (seed_u ^ (i as u32).wrapping_mul(73856093) ^ (j as u32).wrapping_mul(19349663))
                    & 0xFFFF_FFFF;
            let mut rng = Rng::new(blade_seed);
            if rng.next() > dens {
                continue;
            }
            let bx = cx + i * t1[0] + j * t2[0];
            let by = cy + i * t1[1] + j * t2[1];
            let bz = cz + i * t1[2] + j * t2[2];
            let blade_h = 1 + (rng.next() * max_h as f32) as i32;
            let bend_angle = rng.next() * std::f32::consts::TAU;
            let bend_rate =
                (BEND_RATE_BASE + (rng.next() - 0.5) * 2.0 * BEND_RATE_VARIANCE).max(0.02);
            for (x, y, z) in blade_coords(
                bx, by, bz, nx, ny, nz, t1, t2, blade_h, bend_angle, bend_rate,
            ) {
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

/// Preview-only: compute the set of voxel coords grass would occupy,
/// without mutating the file. Used for hover preview.
pub fn preview_grass_coords(
    face_empty: VoxelCoord,
    solid: VoxelCoord,
    seed: i32,
    radius: i32,
    density: f32,
    max_height: i32,
) -> Vec<VoxelCoord> {
    let nx = face_empty.0 - solid.0;
    let ny = face_empty.1 - solid.1;
    let nz = face_empty.2 - solid.2;
    if nx.abs() + ny.abs() + nz.abs() != 1 {
        return Vec::new();
    }

    let r = radius.max(0);
    let max_h = max_height.clamp(1, 40);
    let dens = density.clamp(0.0, 1.0);
    let (t1, t2) = tangent_vectors(nx, ny);
    let (cx, cy, cz) = face_empty;
    let seed_u = seed as u32;

    let mut out = Vec::new();
    let mut seen: HashSet<VoxelCoord> = HashSet::new();

    for i in -r..=r {
        for j in -r..=r {
            if i * i + j * j > r * r {
                continue;
            }
            let blade_seed =
                (seed_u ^ (i as u32).wrapping_mul(73856093) ^ (j as u32).wrapping_mul(19349663))
                    & 0xFFFF_FFFF;
            let mut rng = Rng::new(blade_seed);
            if rng.next() > dens {
                continue;
            }
            let bx = cx + i * t1[0] + j * t2[0];
            let by = cy + i * t1[1] + j * t2[1];
            let bz = cz + i * t1[2] + j * t2[2];
            let blade_h = 1 + (rng.next() * max_h as f32) as i32;
            let bend_angle = rng.next() * std::f32::consts::TAU;
            let bend_rate =
                (BEND_RATE_BASE + (rng.next() - 0.5) * 2.0 * BEND_RATE_VARIANCE).max(0.02);
            for coord in blade_coords(
                bx, by, bz, nx, ny, nz, t1, t2, blade_h, bend_angle, bend_rate,
            ) {
                if seen.insert(coord) {
                    out.push(coord);
                }
            }
        }
    }
    out
}

/// Preview grass from screen coordinates (for hover preview).
pub fn preview_grass_at_screen(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    seed: i32,
    radius: i32,
    density: f32,
    max_height: i32,
) -> Vec<VoxelCoord> {
    let grid_size = effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((solid, prev)) = ray_first_solid(origin, dir, voxel_map, grid_size) else {
        return Vec::new();
    };
    let Some(face_empty) = prev else {
        return Vec::new();
    };
    preview_grass_coords(face_empty, solid, seed, radius, density, max_height)
        .into_iter()
        .filter(|c| !voxel_map.contains_key(c))
        .collect()
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
    radius: i32,
    density: f32,
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
        file, voxel_map, face_empty, solid, seed, radius, density, max_height, color, material,
    ))
}
