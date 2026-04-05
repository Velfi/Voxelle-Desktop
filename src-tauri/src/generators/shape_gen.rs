//! Shape generator — places geometric primitives (cube, orb, cylinder, etc.)
//! at a clicked surface position with optional Euler rotation.
//!
//! Mirrors the web's `store/shapes.ts` `getShapePositionsAt` logic.

use crate::camera::OrbitCamera;
use crate::greedy_mesh::VoxelCoord;
use crate::voxel_edit::{
    effective_ray_grid_size, ensure_grid_fits_coords, ray_first_solid, screen_to_world_ray,
    VoxelEditDelta,
};
use crate::voxelle::start_shape::StartShape;
use crate::voxelle::{MaterialId, Voxel, VoxelleFile};
use ahash::AHashMap;

// ---------------------------------------------------------------------------
// Euler rotation (matches web `rotatePositionAroundOrigin`)
// ---------------------------------------------------------------------------

fn rotate_x_rad(x: f64, y: f64, z: f64, rad: f64) -> (f64, f64, f64) {
    if rad.abs() < 1e-9 {
        return (x, y, z);
    }
    let c = rad.cos();
    let s = rad.sin();
    (x, y * c - z * s, y * s + z * c)
}

fn rotate_y_rad(x: f64, y: f64, z: f64, rad: f64) -> (f64, f64, f64) {
    if rad.abs() < 1e-9 {
        return (x, y, z);
    }
    let c = rad.cos();
    let s = rad.sin();
    (x * c + z * s, y, -x * s + z * c)
}

fn rotate_z_rad(x: f64, y: f64, z: f64, rad: f64) -> (f64, f64, f64) {
    if rad.abs() < 1e-9 {
        return (x, y, z);
    }
    let c = rad.cos();
    let s = rad.sin();
    (x * c - y * s, x * s + y * c, z)
}

/// Rotate position by Euler degrees around origin. Order: X, Y, Z.
fn rotate_position_around_origin(
    pos: (f64, f64, f64),
    rot_deg: (f64, f64, f64),
) -> (i32, i32, i32) {
    let (mut x, mut y, mut z) = pos;
    let rx = rot_deg.0 * std::f64::consts::PI / 180.0;
    let ry = rot_deg.1 * std::f64::consts::PI / 180.0;
    let rz = rot_deg.2 * std::f64::consts::PI / 180.0;
    (x, y, z) = rotate_x_rad(x, y, z, rx);
    (x, y, z) = rotate_y_rad(x, y, z, ry);
    (x, y, z) = rotate_z_rad(x, y, z, rz);
    (x.round() as i32, y.round() as i32, z.round() as i32)
}

// ---------------------------------------------------------------------------
// Shape position computation
// ---------------------------------------------------------------------------

/// Compute world-space voxel positions for a shape at a given origin with rotation.
/// Reuses the same inclusion logic as `voxels_for_start_shape` in `start_shape.rs`.
pub fn compute_shape_positions(
    shape: StartShape,
    size: i32,
    origin: (i32, i32, i32),
    rot_deg: (f32, f32, f32),
) -> Vec<VoxelCoord> {
    if size < 1 || shape == StartShape::Empty {
        return Vec::new();
    }
    let lo = -size / 2;
    let hi = (size - 1) / 2;
    let r = (size - 1) as f64 * 0.5;
    let r_sq = r * r;
    let rot = (rot_deg.0 as f64, rot_deg.1 as f64, rot_deg.2 as f64);
    let has_rotation = rot_deg.0.abs() > 1e-6 || rot_deg.1.abs() > 1e-6 || rot_deg.2.abs() > 1e-6;

    let mut out = Vec::new();
    for x in lo..=hi {
        for y in lo..=hi {
            for z in lo..=hi {
                let include = match shape {
                    StartShape::Cube => true,
                    StartShape::Orb => {
                        let (xf, yf, zf) = (x as f64, y as f64, z as f64);
                        xf * xf + yf * yf + zf * zf <= r_sq
                    }
                    StartShape::Cylinder => {
                        let (xf, zf) = (x as f64, z as f64);
                        xf * xf + zf * zf <= r_sq
                    }
                    StartShape::HollowCube => {
                        x == lo || x == hi || y == lo || y == hi || z == lo || z == hi
                    }
                    StartShape::Plane => y == 0,
                    StartShape::Circle => {
                        y == 0 && {
                            let (xf, zf) = (x as f64, z as f64);
                            xf * xf + zf * zf <= r_sq
                        }
                    }
                    StartShape::Empty => unreachable!(),
                };
                if !include {
                    continue;
                }
                let (wx, wy, wz) = if has_rotation {
                    let (rx, ry, rz) =
                        rotate_position_around_origin((x as f64, y as f64, z as f64), rot);
                    (rx + origin.0, ry + origin.1, rz + origin.2)
                } else {
                    (x + origin.0, y + origin.1, z + origin.2)
                };
                out.push((wx, wy, wz));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Preview
// ---------------------------------------------------------------------------

/// Return preview cells for a shape at screen coords (filtered: only empty cells).
pub fn preview_shape_at_screen(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    shape: StartShape,
    size: i32,
    rot_x: f32,
    rot_y: f32,
    rot_z: f32,
    overwrite: bool,
) -> Vec<VoxelCoord> {
    let grid_size = effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((_solid, prev)) = ray_first_solid(origin, dir, voxel_map, grid_size) else {
        return Vec::new();
    };
    let Some(face_empty) = prev else {
        return Vec::new();
    };
    let cells = compute_shape_positions(shape, size, face_empty, (rot_x, rot_y, rot_z));
    if overwrite {
        cells
    } else {
        cells
            .into_iter()
            .filter(|c| !voxel_map.contains_key(c))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Commit
// ---------------------------------------------------------------------------

/// Generate shape voxels at screen coords and return edit deltas.
pub fn generator_shape_at_screen(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    shape: StartShape,
    size: i32,
    rot_x: f32,
    rot_y: f32,
    rot_z: f32,
    color: u32,
    material: MaterialId,
    overwrite: bool,
) -> Result<Vec<VoxelEditDelta>, String> {
    let grid_size = effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((_solid, prev)) = ray_first_solid(origin, dir, voxel_map, grid_size) else {
        return Ok(Vec::new());
    };
    let Some(face_empty) = prev else {
        return Ok(Vec::new());
    };

    let positions = compute_shape_positions(shape, size, face_empty, (rot_x, rot_y, rot_z));
    if positions.is_empty() {
        return Ok(Vec::new());
    }

    // Ensure grid is large enough.
    ensure_grid_fits_coords(file, positions.iter().copied());

    let mut deltas = Vec::with_capacity(positions.len());
    for (x, y, z) in positions {
        if !overwrite && voxel_map.contains_key(&(x, y, z)) {
            continue;
        }
        deltas.push(VoxelEditDelta::Added(Voxel {
            x,
            y,
            z,
            color,
            material,
            object_id: 0,
        }));
    }
    Ok(deltas)
}
