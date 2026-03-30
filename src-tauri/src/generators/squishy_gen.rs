//! Single-metaball voxelization (web squishy subset): place a soft blob in empty space.
use crate::camera::OrbitCamera;
use crate::greedy_mesh::VoxelCoord;
use crate::voxel_edit::{in_grid, preview_add_cell, VoxelEditDelta};
use crate::voxelle::{MaterialId, Voxel, VoxelleFile};
use ahash::AHashMap;
use std::collections::HashSet;

pub(crate) fn field_strength(cx: f32, cy: f32, cz: f32, px: i32, py: i32, pz: i32, r: f32) -> f32 {
    let dx = px as f32 - cx;
    let dy = py as f32 - cy;
    let dz = pz as f32 - cz;
    let d2 = dx * dx + dy * dy + dz * dz;
    r * r / d2.max(0.25)
}

pub fn squishy_metaball_at_screen(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    radius: i32,
    color: u32,
    material: MaterialId,
) -> Result<Vec<VoxelEditDelta>, String> {
    let grid_size = file.grid_size.max(1);
    let Some(anchor) = preview_add_cell(file, voxel_map, camera, width, height, sx, sy) else {
        return Ok(Vec::new());
    };
    let r = radius.max(2).min(10) as f32;
    let cx = anchor.0 as f32 + 0.5;
    let cy = anchor.1 as f32 + 0.5;
    let cz = anchor.2 as f32 + 0.5;
    let ri = radius.max(2).min(10);
    let threshold = 1.15_f32;
    let mut out = Vec::new();
    let mut seen: HashSet<VoxelCoord> = HashSet::new();
    for dx in -ri..=ri {
        for dy in -ri..=ri {
            for dz in -ri..=ri {
                let f = field_strength(cx, cy, cz, anchor.0 + dx, anchor.1 + dy, anchor.2 + dz, r);
                if f < threshold {
                    continue;
                }
                let x = anchor.0 + dx;
                let y = anchor.1 + dy;
                let z = anchor.2 + dz;
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
        }
    }
    Ok(out)
}
