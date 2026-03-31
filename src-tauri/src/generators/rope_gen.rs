use crate::camera::OrbitCamera;
use crate::greedy_mesh::VoxelCoord;
use crate::voxel_edit::{
    effective_ray_grid_size, ensure_grid_fits_coord, ray_first_solid, screen_to_world_ray,
    BrushShape, VoxelEditDelta,
};
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

/// Web `ropeBrushRadius` index → approximate Chebyshev radius for thickening (voxels).
fn brush_expand_radius_vox(brush_radius_index: u32) -> i32 {
    let r = brush_radius_index as f32 * 0.5 + 0.5;
    r.ceil() as i32
}

fn brush_offset_list(r: i32, shape: BrushShape) -> Vec<(i32, i32, i32)> {
    let r = r.max(1);
    let mut v = Vec::new();
    match shape {
        BrushShape::Sphere => {
            let r2 = (r as f32 + 0.4).powi(2);
            for dz in -r..=r {
                for dy in -r..=r {
                    for dx in -r..=r {
                        if ((dx * dx + dy * dy + dz * dz) as f32) <= r2 {
                            v.push((dx, dy, dz));
                        }
                    }
                }
            }
        }
        BrushShape::Cube | BrushShape::Pyramid | BrushShape::Square | BrushShape::Circle => {
            for dz in -r..=r {
                for dy in -r..=r {
                    for dx in -r..=r {
                        v.push((dx, dy, dz));
                    }
                }
            }
        }
    }
    if v.is_empty() {
        v.push((0, 0, 0));
    }
    v
}

/// Expand centerline voxels with a brush (web `applyBrushAlongPath`).
/// Shared with [`crate::generators::cloth_gen`] for rope/cloth generators.
pub fn thicken_centerline_voxels(
    path: &[VoxelCoord],
    brush_radius_index: u32,
    shape: BrushShape,
) -> Vec<VoxelCoord> {
    let r = brush_expand_radius_vox(brush_radius_index);
    let offs = brush_offset_list(r, shape);
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for &(cx, cy, cz) in path {
        for &(dx, dy, dz) in &offs {
            let p = (cx + dx, cy + dy, cz + dz);
            if seen.insert(p) {
                out.push(p);
            }
        }
    }
    out
}

/// Rope voxel footprint for hover preview (no file mutation).
pub fn preview_rope_voxels_between_screens(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx1: f32,
    sy1: f32,
    sx2: f32,
    sy2: f32,
    sag: f32,
    tension: f32,
    brush_radius_index: u32,
    brush_shape: BrushShape,
) -> Vec<VoxelCoord> {
    let grid_size = effective_ray_grid_size(file);
    let (o1, d1) = screen_to_world_ray(camera, width, height, sx1, sy1);
    let (o2, d2) = screen_to_world_ray(camera, width, height, sx2, sy2);
    let Some((h1, _)) = ray_first_solid(o1, d1, voxel_map, grid_size) else {
        return Vec::new();
    };
    let Some((h2, _)) = ray_first_solid(o2, d2, voxel_map, grid_size) else {
        return Vec::new();
    };
    let t = tension.clamp(0.0, 1.0);
    let sag_eff = sag * (1.0 - t * 0.95).max(0.05);
    let path = catenary_voxel_arc(h1, h2, sag_eff, 48);
    thicken_centerline_voxels(&path, brush_radius_index, brush_shape)
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
    // 0 = loose (full sag), 1 = nearly straight (web ropeTension).
    tension: f32,
    brush_radius_index: u32,
    brush_shape: BrushShape,
    color: u32,
    material: MaterialId,
) -> Result<Vec<VoxelEditDelta>, String> {
    let grid_size = effective_ray_grid_size(file);
    let (o1, d1) = screen_to_world_ray(camera, width, height, sx1, sy1);
    let (o2, d2) = screen_to_world_ray(camera, width, height, sx2, sy2);
    let Some((h1, _)) = ray_first_solid(o1, d1, voxel_map, grid_size) else {
        return Ok(Vec::new());
    };
    let Some((h2, _)) = ray_first_solid(o2, d2, voxel_map, grid_size) else {
        return Ok(Vec::new());
    };
    let t = tension.clamp(0.0, 1.0);
    let sag_eff = sag * (1.0 - t * 0.95).max(0.05);
    let path = catenary_voxel_arc(h1, h2, sag_eff, 48);
    let cells = thicken_centerline_voxels(&path, brush_radius_index, brush_shape);
    let mut out = Vec::new();
    let mut seen: HashSet<VoxelCoord> = HashSet::new();
    for (x, y, z) in cells {
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
    Ok(out)
}
