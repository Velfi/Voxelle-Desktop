//! Multi-metaball editor session (web `squishy/state.ts` parity).

use crate::camera::OrbitCamera;
use crate::generators::squishy_gen::field_strength;
use crate::greedy_mesh::VoxelCoord;
use crate::voxel_edit::{in_grid, preview_add_cell, VoxelEditDelta};
use crate::voxelle::{MaterialId, Voxel, VoxelleFile};
use ahash::AHashMap;
use std::collections::HashSet;

const FIELD_THRESHOLD: f32 = 1.15;

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
pub enum SquishyMode {
    Add,
    Edit,
    Delete,
}

impl Default for SquishyMode {
    fn default() -> Self {
        Self::Add
    }
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
            s += field_strength(cx, cy, cz, px, py, pz, r);
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
    let mut min_x = i32::MAX;
    let mut max_x = i32::MIN;
    let mut min_y = i32::MAX;
    let mut max_y = i32::MIN;
    let mut min_z = i32::MAX;
    let mut max_z = i32::MIN;
    let mut max_r = 4.0_f32;
    for b in &session.balls {
        max_r = max_r.max(b.radius * 3.0);
        min_x = min_x.min(b.x);
        max_x = max_x.max(b.x);
        min_y = min_y.min(b.y);
        max_y = max_y.max(b.y);
        min_z = min_z.min(b.z);
        max_z = max_z.max(b.z);
    }
    let pad = max_r.ceil() as i32 + 4;
    min_x -= pad;
    max_x += pad;
    min_y -= pad;
    max_y += pad;
    min_z -= pad;
    max_z += pad;

    let (gx0, gx1) = crate::voxel_edit::grid_valid_range(gs);
    min_x = min_x.max(gx0);
    max_x = max_x.min(gx1);
    min_y = min_y.max(gx0);
    max_y = max_y.min(gx1);
    min_z = min_z.max(gx0);
    max_z = max_z.min(gx1);

    let mut solid: HashSet<VoxelCoord> = HashSet::new();
    for x in min_x..=max_x {
        for y in min_y..=max_y {
            for z in min_z..=max_z {
                if !in_grid(x, y, z, gs) {
                    continue;
                }
                if session.field_sum_at(x, y, z) >= FIELD_THRESHOLD {
                    solid.insert((x, y, z));
                }
            }
        }
    }

    if session.hollow {
        let wt = session.wall_thickness.max(1);
        solid = hollow_shell_layers(solid, wt);
    }

    solid.into_iter().collect()
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
        preview_add_cell(file, voxel_map, camera, width, height, sx, sy)
    } else {
        crate::voxel_edit::pick_solid_coord_at_screen(file, voxel_map, camera, width, height, sx, sy)
    };
    let Some(anchor) = anchor else {
        return None;
    };
    let r = radius.max(2).min(64) as f32;
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
        if let Some(t) = ray_sphere_intersect(o, d, c, 0.55) {
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
    let grid_size = file.grid_size.max(1);
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
