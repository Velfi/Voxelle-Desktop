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

/// Smooth 3D value noise at fractional coords (seeded). Returns 0–1.
fn value_noise3(seed: i32, x: f32, y: f32, z: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let z0 = z.floor() as i32;
    let fx = x - x.floor();
    let fy = y - y.floor();
    let fz = z - z.floor();
    let u = fx * fx * (3.0 - 2.0 * fx);
    let v = fy * fy * (3.0 - 2.0 * fy);
    let w = fz * fz * (3.0 - 2.0 * fz);
    let n000 = hash3(seed, x0, y0, z0);
    let n100 = hash3(seed, x0 + 1, y0, z0);
    let n010 = hash3(seed, x0, y0 + 1, z0);
    let n110 = hash3(seed, x0 + 1, y0 + 1, z0);
    let n001 = hash3(seed, x0, y0, z0 + 1);
    let n101 = hash3(seed, x0 + 1, y0, z0 + 1);
    let n011 = hash3(seed, x0, y0 + 1, z0 + 1);
    let n111 = hash3(seed, x0 + 1, y0 + 1, z0 + 1);
    let nx00 = n000 * (1.0 - u) + n100 * u;
    let nx10 = n010 * (1.0 - u) + n110 * u;
    let nx01 = n001 * (1.0 - u) + n101 * u;
    let nx11 = n011 * (1.0 - u) + n111 * u;
    let nxy0 = nx00 * (1.0 - v) + nx10 * v;
    let nxy1 = nx01 * (1.0 - v) + nx11 * v;
    nxy0 * (1.0 - w) + nxy1 * w
}

/// Derive a float in [lo, hi] from seed.
fn seed_to_range(seed: i32, lo: f32, hi: f32) -> f32 {
    let mut h = (seed as u32).wrapping_mul(0x9e3779b9);
    h = (h ^ (h >> 16)).wrapping_mul(0x85ebca6b);
    h = (h ^ (h >> 13)).wrapping_mul(0xc2b2ae35);
    lo + ((h as f32) / (u32::MAX as f32)) * (hi - lo)
}

/// Generate rock voxel coordinates in local space (origin at center), matching the web version.
/// Returns Vec of (local_x, local_y, local_z) offsets.
fn generate_rock_local(seed: i32, size: i32, roughness: f32) -> Vec<(i32, i32, i32)> {
    let r = size.max(1).min(20);
    let lumpiness = roughness.clamp(0.0, 1.0) * 0.6;
    let scale = 2.5 / (r as f32).max(1.0);

    // Asymmetric stretch per axis (0.6–1.4) so rocks aren't round
    let sx = seed_to_range(seed + 1, 0.6, 1.4);
    let sy = seed_to_range(seed + 2, 0.6, 1.4);
    let sz = seed_to_range(seed + 3, 0.6, 1.4);

    // Optional planar fracture facet (55% chance)
    let do_facet = seed_to_range(seed + 4, 0.0, 1.0) < 0.55;
    let fnx = seed_to_range(seed + 5, -1.0, 1.0);
    let fny = seed_to_range(seed + 6, -1.0, 1.0);
    let fnz = seed_to_range(seed + 7, -1.0, 1.0);
    let flen = (fnx * fnx + fny * fny + fnz * fnz).sqrt().max(0.001);
    let facet_depth = r as f32 * (0.15 + seed_to_range(seed + 8, 0.0, 0.35));

    let rf = r as f32;
    let mut out = Vec::new();

    for x in -r..=r {
        for y in -r..=r {
            for z in -r..=r {
                let px = x as f32 / sx;
                let py = y as f32 / sy;
                let pz = z as f32 / sz;
                let d = (px * px + py * py + pz * pz).sqrt();
                if d > rf + 1.0 {
                    continue;
                }
                let xf = x as f32;
                let yf = y as f32;
                let zf = z as f32;
                let n = value_noise3(seed + 0x1234, xf * scale, yf * scale, zf * scale);
                let n2 = value_noise3(
                    seed + 0x5678,
                    xf * scale * 1.7 + 3.0,
                    yf * scale * 1.7,
                    zf * scale * 1.7,
                );
                let perturb = (n - 0.5) * 2.0 * lumpiness + (n2 - 0.5) * lumpiness * 0.5;
                let r_effective = rf * (1.0 + perturb);
                if d > r_effective {
                    continue;
                }
                // Fracture facet: exclude voxels on one side of a plane
                if do_facet {
                    let dist = xf * (fnx / flen) + yf * (fny / flen) + zf * (fnz / flen);
                    if dist < -facet_depth {
                        continue;
                    }
                }
                out.push((x, y, z));
            }
        }
    }

    // Flat base – clip bottom so rock has a resting face
    let floor_y = -(rf * 0.4);
    out.retain(|&(_, y, _)| y as f32 >= floor_y);

    out
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
    count: i32,
    cluster_radius: i32,
    sink_direction: i32, // 0=none, -1=under, 1=over
    sink_amount: i32,
) -> Vec<VoxelEditDelta> {
    let nx = (face_empty.0 - solid.0).signum();
    let ny = (face_empty.1 - solid.1).signum();
    let nz = (face_empty.2 - solid.2).signum();

    let count = count.max(1).min(5);
    let cluster_r = cluster_radius.max(0).min(3);
    let sink_n = if sink_direction != 0 {
        sink_amount.max(0).min(5)
    } else {
        0
    };

    // Compute the surface target based on sink direction
    let (stx, sty, stz) = if sink_direction < 0 {
        // under: push into surface
        (
            face_empty.0 - (1 + sink_n) * nx,
            face_empty.1 - (1 + sink_n) * ny,
            face_empty.2 - (1 + sink_n) * nz,
        )
    } else if sink_direction > 0 {
        // over: float above surface
        (
            face_empty.0 + (sink_n - 1) * nx,
            face_empty.1 + (sink_n - 1) * ny,
            face_empty.2 + (sink_n - 1) * nz,
        )
    } else {
        (face_empty.0 - nx, face_empty.1 - ny, face_empty.2 - nz)
    };

    let mut out = Vec::new();
    let mut seen: HashSet<VoxelCoord> = HashSet::new();

    for i in 0..count {
        let rock_seed = seed.wrapping_add(i);

        // Cluster offset
        let (dx, dy, dz) = if cluster_r > 0 && count > 1 {
            let cr = cluster_r as f32;
            let diam = 2.0 * cr + 1.0;
            let ox = (seed_to_range(rock_seed + 100, 0.0, diam) - cr).floor() as i32;
            let oy = (seed_to_range(rock_seed + 200, 0.0, diam) - cr).floor() as i32;
            let oz = (seed_to_range(rock_seed + 300, 0.0, diam) - cr).floor() as i32;
            (ox, oy, oz)
        } else {
            (0, 0, 0)
        };

        let local_voxels = generate_rock_local(rock_seed, size, roughness);
        if local_voxels.is_empty() {
            continue;
        }

        // Compute local bounds
        let mut min_x = i32::MAX;
        let mut max_x = i32::MIN;
        let mut min_y = i32::MAX;
        let mut max_y = i32::MIN;
        let mut min_z = i32::MAX;
        let mut max_z = i32::MIN;
        for &(lx, ly, lz) in &local_voxels {
            min_x = min_x.min(lx);
            max_x = max_x.max(lx);
            min_y = min_y.min(ly);
            max_y = max_y.max(ly);
            min_z = min_z.min(lz);
            max_z = max_z.max(lz);
        }
        let half_x = (max_x - min_x) / 2;
        let half_y = (max_y - min_y) / 2;
        let half_z = (max_z - min_z) / 2;

        // Place rock: align to surface along the normal axis, center on tangent axes
        // The "stamp offset" centers the rock on the surface target
        let ox = if nx != 0 {
            // Normal is along X: align to surface
            if nx > 0 {
                stx - min_x
            } else {
                stx - max_x
            }
        } else {
            face_empty.0 + dx - half_x - min_x
        };
        let oy = if ny != 0 {
            if ny > 0 {
                sty - min_y
            } else {
                sty - max_y
            }
        } else {
            face_empty.1 + dy - half_y - min_y
        };
        let oz = if nz != 0 {
            if nz > 0 {
                stz - min_z
            } else {
                stz - max_z
            }
        } else {
            face_empty.2 + dz - half_z - min_z
        };

        for &(lx, ly, lz) in &local_voxels {
            let x = lx + ox;
            let y = ly + oy;
            let z = lz + oz;
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
    count: i32,
    cluster_radius: i32,
    sink_direction: i32,
    sink_amount: i32,
) -> Vec<VoxelCoord> {
    let nx = (face_empty.0 - solid.0).signum();
    let ny = (face_empty.1 - solid.1).signum();
    let nz = (face_empty.2 - solid.2).signum();

    let count = count.max(1).min(5);
    let cluster_r = cluster_radius.max(0).min(3);
    let sink_n = if sink_direction != 0 {
        sink_amount.max(0).min(5)
    } else {
        0
    };

    let (stx, sty, stz) = if sink_direction < 0 {
        (
            face_empty.0 - (1 + sink_n) * nx,
            face_empty.1 - (1 + sink_n) * ny,
            face_empty.2 - (1 + sink_n) * nz,
        )
    } else if sink_direction > 0 {
        (
            face_empty.0 + (sink_n - 1) * nx,
            face_empty.1 + (sink_n - 1) * ny,
            face_empty.2 + (sink_n - 1) * nz,
        )
    } else {
        (face_empty.0 - nx, face_empty.1 - ny, face_empty.2 - nz)
    };

    let mut out = Vec::new();
    let mut seen: HashSet<VoxelCoord> = HashSet::new();

    for i in 0..count {
        let rock_seed = seed.wrapping_add(i);

        let (dx, dy, dz) = if cluster_r > 0 && count > 1 {
            let cr = cluster_r as f32;
            let diam = 2.0 * cr + 1.0;
            let ox = (seed_to_range(rock_seed + 100, 0.0, diam) - cr).floor() as i32;
            let oy = (seed_to_range(rock_seed + 200, 0.0, diam) - cr).floor() as i32;
            let oz = (seed_to_range(rock_seed + 300, 0.0, diam) - cr).floor() as i32;
            (ox, oy, oz)
        } else {
            (0, 0, 0)
        };

        let local_voxels = generate_rock_local(rock_seed, size, roughness);
        if local_voxels.is_empty() {
            continue;
        }

        let mut min_x = i32::MAX;
        let mut max_x = i32::MIN;
        let mut min_y = i32::MAX;
        let mut max_y = i32::MIN;
        let mut min_z = i32::MAX;
        let mut max_z = i32::MIN;
        for &(lx, ly, lz) in &local_voxels {
            min_x = min_x.min(lx);
            max_x = max_x.max(lx);
            min_y = min_y.min(ly);
            max_y = max_y.max(ly);
            min_z = min_z.min(lz);
            max_z = max_z.max(lz);
        }
        let half_x = (max_x - min_x) / 2;
        let half_y = (max_y - min_y) / 2;
        let half_z = (max_z - min_z) / 2;

        let ox = if nx != 0 {
            if nx > 0 {
                stx - min_x
            } else {
                stx - max_x
            }
        } else {
            face_empty.0 + dx - half_x - min_x
        };
        let oy = if ny != 0 {
            if ny > 0 {
                sty - min_y
            } else {
                sty - max_y
            }
        } else {
            face_empty.1 + dy - half_y - min_y
        };
        let oz = if nz != 0 {
            if nz > 0 {
                stz - min_z
            } else {
                stz - max_z
            }
        } else {
            face_empty.2 + dz - half_z - min_z
        };

        for &(lx, ly, lz) in &local_voxels {
            let coord = (lx + ox, ly + oy, lz + oz);
            if seen.insert(coord) {
                out.push(coord);
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
    count: i32,
    cluster_radius: i32,
    sink_direction: i32,
    sink_amount: i32,
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
    preview_rock_cluster_coords(
        face_empty,
        solid,
        seed,
        size,
        roughness,
        count,
        cluster_radius,
        sink_direction,
        sink_amount,
    )
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
    count: i32,
    cluster_radius: i32,
    sink_direction: i32,
    sink_amount: i32,
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
        count,
        cluster_radius,
        sink_direction,
        sink_amount,
    ))
}
