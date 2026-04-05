//! Multi-metaball editor session (web `squishy/state.ts` parity).

use crate::camera::OrbitCamera;
use crate::greedy_mesh::VoxelCoord;
use crate::voxel_edit::{in_grid, preview_add_cell, VoxelEditDelta};
use crate::voxelle::{MaterialId, Voxel, VoxelleFile};
use ahash::AHashMap;
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};

const FIELD_THRESHOLD: f32 = 1.15;

/// Minimum grid size needed to contain all metaballs' influence regions
/// without clipping.  Returns `max(current_gs, needed)`.
pub fn min_grid_size_for_balls(balls: &[Metaball], current_gs: i32) -> i32 {
    let mut max_abs = 0i32;
    for b in balls {
        let r_pad = (b.radius * 3.0).max(4.0).ceil() as i32 + 2;
        max_abs = max_abs
            .max((b.x - r_pad).abs())
            .max((b.x + r_pad).abs())
            .max((b.y - r_pad).abs())
            .max((b.y + r_pad).abs())
            .max((b.z - r_pad).abs())
            .max((b.z + r_pad).abs());
    }
    let need = (2 * (max_abs + 1)).max(1);
    current_gs
        .max(1)
        .max(need)
        .min(crate::voxel_edit::MAX_GRID_SIZE)
}

/// Compute the scan bounding box as the union of per-ball influence regions,
/// clamped to the grid.  Each ball's pad is proportional to its own radius,
/// so distant small balls don't inflate the box for large ones (and vice-versa).
fn scan_bbox(balls: &[Metaball], gs: i32) -> (i32, i32, i32, i32, i32, i32) {
    let mut min_x = i32::MAX;
    let mut max_x = i32::MIN;
    let mut min_y = i32::MAX;
    let mut max_y = i32::MIN;
    let mut min_z = i32::MAX;
    let mut max_z = i32::MIN;
    for b in balls {
        let r_pad = (b.radius * 3.0).max(4.0).ceil() as i32 + 2;
        min_x = min_x.min(b.x - r_pad);
        max_x = max_x.max(b.x + r_pad);
        min_y = min_y.min(b.y - r_pad);
        max_y = max_y.max(b.y + r_pad);
        min_z = min_z.min(b.z - r_pad);
        max_z = max_z.max(b.z + r_pad);
    }
    let (gx0, gx1) = crate::voxel_edit::grid_valid_range(gs);
    (
        min_x.max(gx0),
        max_x.min(gx1),
        min_y.max(gx0),
        max_y.min(gx1),
        min_z.max(gx0),
        max_z.min(gx1),
    )
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Metaball {
    pub id: u32,
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub radius: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
#[derive(Default)]
pub enum SquishyMode {
    #[default]
    Add,
    Edit,
    Delete,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SquishySession {
    pub mode: SquishyMode,
    pub balls: Vec<Metaball>,
    pub selected_id: Option<u32>,
    pub hollow: bool,
    pub wall_thickness: i32,
    pub add_snap_to_surface: bool,
    #[serde(skip)]
    next_id: u32,
}

impl Default for SquishySession {
    fn default() -> Self {
        Self::new()
    }
}

impl SquishySession {
    pub fn new() -> Self {
        Self {
            mode: SquishyMode::Add,
            balls: Vec::new(),
            selected_id: None,
            hollow: false,
            wall_thickness: 1,
            add_snap_to_surface: true,
            next_id: 1,
        }
    }

    fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        id
    }

    pub fn clear(&mut self) {
        self.balls.clear();
        self.selected_id = None;
    }

    pub fn remove_ball(&mut self, id: u32) -> bool {
        let n = self.balls.len();
        self.balls.retain(|b| b.id != id);
        if self.selected_id == Some(id) {
            self.selected_id = None;
        }
        self.balls.len() < n
    }

    pub fn set_ball_transform(&mut self, id: u32, x: i32, y: i32, z: i32, radius: f32) -> bool {
        if let Some(b) = self.balls.iter_mut().find(|b| b.id == id) {
            b.x = x;
            b.y = y;
            b.z = z;
            b.radius = radius.max(0.5);
            return true;
        }
        false
    }

    pub fn add_ball(&mut self, x: i32, y: i32, z: i32, radius: f32) -> u32 {
        let id = self.alloc_id();
        self.balls.push(Metaball {
            id,
            x,
            y,
            z,
            radius: radius.max(0.5_f32),
        });
        id
    }

    pub fn field_sum_at(&self, px: i32, py: i32, pz: i32) -> f32 {
        let mut s = 0.0_f32;
        for b in &self.balls {
            let cx = b.x as f32 + 0.5;
            let cy = b.y as f32 + 0.5;
            let cz = b.z as f32 + 0.5;
            let r = b.radius.max(0.5);
            let dx = px as f32 - cx;
            let dy = py as f32 - cy;
            let dz = pz as f32 - cz;
            let d2 = dx * dx + dy * dy + dz * dz;
            // Skip balls whose contribution is < 0.0001 (d² > r² × 10 000).
            // Max accumulated error with 50 balls ≈ 0.005, negligible vs threshold 1.15.
            if d2 > r * r * 10_000.0 {
                continue;
            }
            s += r * r / d2.max(0.25);
            // Early exit: field_strength is non-negative, so once we exceed the
            // threshold the result can only grow.  The caller only tests ≥ threshold.
            if s >= FIELD_THRESHOLD {
                return s;
            }
        }
        s
    }
}

fn neighbors_6(c: VoxelCoord) -> [VoxelCoord; 6] {
    let (x, y, z) = c;
    [
        (x + 1, y, z),
        (x - 1, y, z),
        (x, y + 1, z),
        (x, y - 1, z),
        (x, y, z + 1),
        (x, y, z - 1),
    ]
}

/// Voxels inside the isosurface; optionally only the outer shell (hollow).
pub fn voxel_coords_for_session(session: &SquishySession, grid_size: i32) -> Vec<VoxelCoord> {
    if session.balls.is_empty() {
        return Vec::new();
    }
    let gs = grid_size.max(1);
    let (min_x, max_x, min_y, max_y, min_z, max_z) = scan_bbox(&session.balls, gs);

    let mut solid: HashSet<VoxelCoord> = (min_x..=max_x)
        .into_par_iter()
        .flat_map(|x| {
            let mut local = Vec::new();
            for y in min_y..=max_y {
                for z in min_z..=max_z {
                    if !in_grid(x, y, z, gs) {
                        continue;
                    }
                    if session.field_sum_at(x, y, z) >= FIELD_THRESHOLD {
                        local.push((x, y, z));
                    }
                }
            }
            local
        })
        .collect::<Vec<_>>()
        .into_iter()
        .collect();

    if session.hollow {
        let wt = session.wall_thickness.max(1);
        solid = hollow_shell_layers(solid, wt);
    }

    solid.into_iter().collect()
}

/// Preview path: same field as commit, but stops after `max_voxels` (web ~24k preview cap).
pub fn voxel_coords_for_session_with_limit(
    session: &SquishySession,
    grid_size: i32,
    max_voxels: usize,
) -> Vec<VoxelCoord> {
    if session.balls.is_empty() || max_voxels == 0 {
        return Vec::new();
    }
    let gs = grid_size.max(1);
    let (min_x, max_x, min_y, max_y, min_z, max_z) = scan_bbox(&session.balls, gs);

    let count = AtomicUsize::new(0);
    let results: Vec<VoxelCoord> = (min_x..=max_x)
        .into_par_iter()
        .flat_map(|x| {
            let mut local = Vec::new();
            for y in min_y..=max_y {
                if count.load(Ordering::Relaxed) >= max_voxels {
                    break;
                }
                for z in min_z..=max_z {
                    if count.load(Ordering::Relaxed) >= max_voxels {
                        break;
                    }
                    if !in_grid(x, y, z, gs) {
                        continue;
                    }
                    if session.field_sum_at(x, y, z) >= FIELD_THRESHOLD {
                        local.push((x, y, z));
                        count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            local
        })
        .collect();

    results.into_iter().take(max_voxels).collect()
}

/// Outer `wall_thickness` layers: peel boundary repeatedly and union those layers (web hollow shell).
fn hollow_shell_layers(mut s: HashSet<VoxelCoord>, wt: i32) -> HashSet<VoxelCoord> {
    let wt = wt.max(1) as usize;
    let mut shell = HashSet::new();
    for _ in 0..wt {
        if s.is_empty() {
            break;
        }
        let boundary: HashSet<VoxelCoord> = s
            .iter()
            .copied()
            .filter(|&c| neighbors_6(c).iter().any(|n| !s.contains(n)))
            .collect();
        if boundary.is_empty() {
            break;
        }
        shell.extend(boundary.iter().copied());
        s = s.difference(&boundary).copied().collect();
    }
    shell
}

/// Add metaball from screen (add mode); uses empty cell in front of hit.
pub fn squishy_add_ball_at_screen(
    session: &mut SquishySession,
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    radius: i32,
) -> Option<u32> {
    let anchor = if session.add_snap_to_surface {
        preview_add_cell(file, voxel_map, camera, width, height, sx, sy).map(|(c, _)| c)
    } else {
        crate::voxel_edit::pick_solid_coord_at_screen(
            file, voxel_map, camera, width, height, sx, sy,
        )
    };
    let anchor = anchor?;
    let r = radius.clamp(2, 64) as f32;
    let id = session.add_ball(anchor.0, anchor.1, anchor.2, r);
    Some(id)
}

/// Ray vs small spheres around metaball centers; returns nearest ball id.
pub fn pick_metaball_at_screen(
    session: &SquishySession,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> Option<u32> {
    use glam::Vec3;
    let (ro, rd) = crate::voxel_edit::screen_to_world_ray(camera, width, height, sx, sy);
    let o = Vec3::new(ro.x, ro.y, ro.z);
    let d = Vec3::new(rd.x, rd.y, rd.z).normalize();
    let mut best: Option<(u32, f32)> = None;
    for b in &session.balls {
        let c = Vec3::new(b.x as f32 + 0.5, b.y as f32 + 0.5, b.z as f32 + 0.5);
        let pick_r = b.radius.clamp(0.2_f32, 64.0_f32);
        if let Some(t) = ray_sphere_intersect(o, d, c, pick_r) {
            if t >= 0.0 {
                let replace = best.map(|(_, bt)| t < bt).unwrap_or(true);
                if replace {
                    best = Some((b.id, t));
                }
            }
        }
    }
    best.map(|(id, _)| id)
}

fn ray_sphere_intersect(o: glam::Vec3, d: glam::Vec3, c: glam::Vec3, r: f32) -> Option<f32> {
    let oc = o - c;
    let b = oc.dot(d);
    let c0 = oc.dot(oc) - r * r;
    let disc = b * b - c0;
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    let t0 = -b - s;
    let t1 = -b + s;
    let t = if t0 >= 0.0 {
        t0
    } else if t1 >= 0.0 {
        t1
    } else {
        return None;
    };
    Some(t)
}

pub fn squishy_commit_session(
    session: &SquishySession,
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    color: u32,
    material: MaterialId,
) -> Result<Vec<VoxelEditDelta>, String> {
    // Grow the grid so metaballs near the edge aren't clipped.
    let grid_size = min_grid_size_for_balls(&session.balls, file.grid_size);
    file.grid_size = grid_size;
    let coords = voxel_coords_for_session(session, grid_size);
    let mut out = Vec::new();
    for (x, y, z) in coords {
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
