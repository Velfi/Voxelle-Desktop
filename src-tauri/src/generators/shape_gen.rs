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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxelle::start_shape::StartShape;
    use std::f64::consts::PI;

    const EPS: f64 = 1e-9;

    fn approx_f64(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    // ── Rotation helpers ───────────────────────────────────────────────

    #[test]
    fn rotate_x_identity_near_zero() {
        let (x, y, z) = rotate_x_rad(1.0, 2.0, 3.0, 0.0);
        assert_eq!((x, y, z), (1.0, 2.0, 3.0));
    }

    #[test]
    fn rotate_x_quarter_turn() {
        // (0, 1, 0) rotated 90° around X → (0, 0, 1)
        let (x, y, z) = rotate_x_rad(0.0, 1.0, 0.0, PI / 2.0);
        assert!(approx_f64(x, 0.0));
        assert!(approx_f64(y, 0.0));
        assert!(approx_f64(z, 1.0));
    }

    #[test]
    fn rotate_y_identity_near_zero() {
        let (x, y, z) = rotate_y_rad(5.0, 3.0, 1.0, 0.0);
        assert_eq!((x, y, z), (5.0, 3.0, 1.0));
    }

    #[test]
    fn rotate_y_quarter_turn() {
        // (1, 0, 0) rotated 90° around Y → (0, 0, -1)
        let (x, y, z) = rotate_y_rad(1.0, 0.0, 0.0, PI / 2.0);
        assert!(approx_f64(x, 0.0));
        assert!(approx_f64(y, 0.0));
        assert!(approx_f64(z, -1.0));
    }

    #[test]
    fn rotate_z_identity_near_zero() {
        let (x, y, z) = rotate_z_rad(1.0, 2.0, 3.0, 0.0);
        assert_eq!((x, y, z), (1.0, 2.0, 3.0));
    }

    #[test]
    fn rotate_z_quarter_turn() {
        // (1, 0, 0) rotated 90° around Z → (0, 1, 0)
        let (x, y, z) = rotate_z_rad(1.0, 0.0, 0.0, PI / 2.0);
        assert!(approx_f64(x, 0.0));
        assert!(approx_f64(y, 1.0));
        assert!(approx_f64(z, 0.0));
    }

    // ── rotate_position_around_origin ─────────────────────────────────

    #[test]
    fn rotate_position_identity() {
        assert_eq!(
            rotate_position_around_origin((3.0, -1.0, 5.0), (0.0, 0.0, 0.0)),
            (3, -1, 5)
        );
    }

    #[test]
    fn rotate_position_180_around_y() {
        // (1, 0, 0) rotated 180° around Y → (-1, 0, 0)
        let (x, y, z) = rotate_position_around_origin((1.0, 0.0, 0.0), (0.0, 180.0, 0.0));
        assert_eq!(x, -1);
        assert_eq!(y, 0);
        assert_eq!(z, 0);
    }

    // ── compute_shape_positions ────────────────────────────────────────

    #[test]
    fn compute_shape_positions_empty_shape_returns_nothing() {
        let positions = compute_shape_positions(StartShape::Empty, 5, (0, 0, 0), (0.0, 0.0, 0.0));
        assert!(positions.is_empty());
    }

    #[test]
    fn compute_shape_positions_size_zero_returns_nothing() {
        let positions = compute_shape_positions(StartShape::Cube, 0, (0, 0, 0), (0.0, 0.0, 0.0));
        assert!(positions.is_empty());
    }

    #[test]
    fn compute_shape_positions_cube_size_1() {
        let positions = compute_shape_positions(StartShape::Cube, 1, (0, 0, 0), (0.0, 0.0, 0.0));
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0], (0, 0, 0));
    }

    #[test]
    fn compute_shape_positions_cube_size_3() {
        let positions = compute_shape_positions(StartShape::Cube, 3, (0, 0, 0), (0.0, 0.0, 0.0));
        assert_eq!(positions.len(), 27);
    }

    #[test]
    fn compute_shape_positions_origin_offset_applied() {
        let positions = compute_shape_positions(StartShape::Cube, 1, (10, 20, 30), (0.0, 0.0, 0.0));
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0], (10, 20, 30));
    }

    #[test]
    fn compute_shape_positions_plane_is_flat() {
        let positions = compute_shape_positions(StartShape::Plane, 3, (0, 0, 0), (0.0, 0.0, 0.0));
        // All y == 0
        assert!(!positions.is_empty());
        for &(_, y, _) in &positions {
            assert_eq!(y, 0);
        }
    }

    #[test]
    fn compute_shape_positions_orb_within_sphere() {
        let size = 5;
        let positions = compute_shape_positions(StartShape::Orb, size, (0, 0, 0), (0.0, 0.0, 0.0));
        let r = (size - 1) as f64 * 0.5;
        for &(x, y, z) in &positions {
            let d2 = (x as f64).powi(2) + (y as f64).powi(2) + (z as f64).powi(2);
            assert!(
                d2 <= r * r + 1e-6,
                "voxel ({x},{y},{z}) outside sphere r={r}"
            );
        }
    }
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
