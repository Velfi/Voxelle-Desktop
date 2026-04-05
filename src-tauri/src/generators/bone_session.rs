//! Bone armature editor session: joints, bones, picking, and voxelization.

use crate::camera::OrbitCamera;
use crate::greedy_mesh::VoxelCoord;
use crate::voxel_edit::{in_grid, VoxelEditDelta};
use crate::voxelle::{MaterialId, Voxel, VoxelleFile};
use ahash::AHashMap;
use glam::Vec3;
use std::collections::HashSet;

// ── Data types ───────────────────────────────────────────────────────

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Joint {
    pub id: u32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub radius: f32,
}

impl Joint {
    pub fn pos(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Bone {
    pub id: u32,
    pub joint_a: u32,
    pub joint_b: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BoneSelection {
    Joint(u32),
    Bone(u32),
}

// ── Session ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BoneSession {
    pub joints: Vec<Joint>,
    pub bones: Vec<Bone>,
    pub selected: Option<BoneSelection>,
    /// Build phase: first joint placed, awaiting a second click to form a bone.
    pub pending_joint: Option<u32>,
    #[serde(skip)]
    next_id: u32,
}

impl Default for BoneSession {
    fn default() -> Self {
        Self::new()
    }
}

impl BoneSession {
    pub fn new() -> Self {
        Self {
            joints: Vec::new(),
            bones: Vec::new(),
            selected: None,
            pending_joint: None,
            next_id: 1,
        }
    }

    fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        id
    }

    pub fn clear(&mut self) {
        self.joints.clear();
        self.bones.clear();
        self.selected = None;
        self.pending_joint = None;
    }

    // ── Joint operations ─────────────────────────────────────────────

    pub fn add_joint(&mut self, x: f32, y: f32, z: f32, radius: f32) -> u32 {
        let id = self.alloc_id();
        self.joints.push(Joint {
            id,
            x,
            y,
            z,
            radius: radius.max(0.5),
        });
        id
    }

    pub fn find_joint(&self, id: u32) -> Option<&Joint> {
        self.joints.iter().find(|j| j.id == id)
    }

    pub fn find_joint_mut(&mut self, id: u32) -> Option<&mut Joint> {
        self.joints.iter_mut().find(|j| j.id == id)
    }

    pub fn set_joint_position(&mut self, id: u32, x: f32, y: f32, z: f32) -> bool {
        if let Some(j) = self.find_joint_mut(id) {
            j.x = x;
            j.y = y;
            j.z = z;
            true
        } else {
            false
        }
    }

    pub fn set_joint_radius(&mut self, id: u32, radius: f32) -> bool {
        if let Some(j) = self.find_joint_mut(id) {
            j.radius = radius.max(0.5);
            true
        } else {
            false
        }
    }

    /// Remove a joint and all bones connected to it.
    pub fn remove_joint(&mut self, id: u32) -> bool {
        let before = self.joints.len();
        self.joints.retain(|j| j.id != id);
        if self.joints.len() == before {
            return false;
        }
        self.bones.retain(|b| b.joint_a != id && b.joint_b != id);
        if self.selected == Some(BoneSelection::Joint(id)) {
            self.selected = None;
        }
        if self.pending_joint == Some(id) {
            self.pending_joint = None;
        }
        // Clean up bone selections that reference removed bones
        if let Some(BoneSelection::Bone(bid)) = self.selected {
            if !self.bones.iter().any(|b| b.id == bid) {
                self.selected = None;
            }
        }
        true
    }

    // ── Bone operations ──────────────────────────────────────────────

    /// Connect two joints with a bone. Returns None if either joint doesn't
    /// exist or if a bone already connects them.
    pub fn add_bone(&mut self, joint_a: u32, joint_b: u32) -> Option<u32> {
        if joint_a == joint_b {
            return None;
        }
        if self.find_joint(joint_a).is_none() || self.find_joint(joint_b).is_none() {
            return None;
        }
        // Check for duplicates (either direction).
        let dup = self.bones.iter().any(|b| {
            (b.joint_a == joint_a && b.joint_b == joint_b)
                || (b.joint_a == joint_b && b.joint_b == joint_a)
        });
        if dup {
            return None;
        }
        let id = self.alloc_id();
        self.bones.push(Bone {
            id,
            joint_a,
            joint_b,
        });
        Some(id)
    }

    pub fn remove_bone(&mut self, id: u32) -> bool {
        let before = self.bones.len();
        self.bones.retain(|b| b.id != id);
        if self.selected == Some(BoneSelection::Bone(id)) {
            self.selected = None;
        }
        self.bones.len() < before
    }

    /// All bone IDs connected to a given joint.
    pub fn connected_bones(&self, joint_id: u32) -> Vec<u32> {
        self.bones
            .iter()
            .filter(|b| b.joint_a == joint_id || b.joint_b == joint_id)
            .map(|b| b.id)
            .collect()
    }

    /// The other joint at the far end of a bone from `from_joint`.
    pub fn other_joint(&self, bone_id: u32, from_joint: u32) -> Option<u32> {
        self.bones.iter().find(|b| b.id == bone_id).map(|b| {
            if b.joint_a == from_joint {
                b.joint_b
            } else {
                b.joint_a
            }
        })
    }
}

// ── Picking ──────────────────────────────────────────────────────────

/// Ray-sphere intersection; returns the nearest positive `t`.
fn ray_sphere_intersect(o: Vec3, d: Vec3, c: Vec3, r: f32) -> Option<f32> {
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

/// Ray vs. capsule (cylinder with hemispherical caps).
/// Returns the nearest positive `t` along the ray.
fn ray_capsule_intersect(o: Vec3, d: Vec3, a: Vec3, b: Vec3, r: f32) -> Option<f32> {
    let ab = b - a;
    let ab_len = ab.length();
    if ab_len < 1e-6 {
        return ray_sphere_intersect(o, d, a, r);
    }
    let ab_n = ab / ab_len;

    // Infinite cylinder intersection.
    let ao = o - a;
    let d_perp = d - ab_n * d.dot(ab_n);
    let ao_perp = ao - ab_n * ao.dot(ab_n);

    let qa = d_perp.dot(d_perp);
    let qb = 2.0 * d_perp.dot(ao_perp);
    let qc = ao_perp.dot(ao_perp) - r * r;

    let mut best: Option<f32> = None;
    let mut consider = |t: f32| {
        if t >= 0.0 {
            if best.map(|bt| t < bt).unwrap_or(true) {
                best = Some(t);
            }
        }
    };

    // Cylinder body
    let disc = qb * qb - 4.0 * qa * qc;
    if disc >= 0.0 && qa.abs() > 1e-9 {
        let sq = disc.sqrt();
        for t in [(-qb - sq) / (2.0 * qa), (-qb + sq) / (2.0 * qa)] {
            if t >= 0.0 {
                let p = o + d * t;
                let proj = (p - a).dot(ab_n);
                if proj >= 0.0 && proj <= ab_len {
                    consider(t);
                }
            }
        }
    }

    // Hemispherical caps
    if let Some(t) = ray_sphere_intersect(o, d, a, r) {
        let p = o + d * t;
        if (p - a).dot(ab_n) <= 0.0 {
            consider(t);
        }
    }
    if let Some(t) = ray_sphere_intersect(o, d, b, r) {
        let p = o + d * t;
        if (p - b).dot(ab_n) >= 0.0 {
            consider(t);
        }
    }

    best
}

/// Pick nearest joint or bone; joints take priority when overlapping.
pub fn pick_at_screen(
    session: &BoneSession,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> Option<BoneSelection> {
    let (ro, rd) = crate::voxel_edit::screen_to_world_ray(camera, width, height, sx, sy);
    let o = Vec3::new(ro.x, ro.y, ro.z);
    let d = Vec3::new(rd.x, rd.y, rd.z).normalize();

    // Find nearest joint hit.
    let mut joint_best_t = f32::MAX;
    let mut joint_result: Option<u32> = None;
    for j in &session.joints {
        let pick_r = j.radius.max(1.0);
        if let Some(t) = ray_sphere_intersect(o, d, j.pos(), pick_r) {
            if t >= 0.0 && t < joint_best_t {
                joint_best_t = t;
                joint_result = Some(j.id);
            }
        }
    }

    // If a joint was hit, use it — joints always take priority over bones.
    if let Some(id) = joint_result {
        return Some(BoneSelection::Joint(id));
    }

    // No joint hit; try bones.
    let mut bone_best_t = f32::MAX;
    let mut bone_result: Option<u32> = None;
    for bone in &session.bones {
        let Some(ja) = session.find_joint(bone.joint_a) else {
            continue;
        };
        let Some(jb) = session.find_joint(bone.joint_b) else {
            continue;
        };
        let r = (ja.radius + jb.radius) * 0.5;
        let pick_r = r.max(0.5);
        if let Some(t) = ray_capsule_intersect(o, d, ja.pos(), jb.pos(), pick_r) {
            if t >= 0.0 && t < bone_best_t {
                bone_best_t = t;
                bone_result = Some(bone.id);
            }
        }
    }

    bone_result.map(BoneSelection::Bone)
}

// ── Ray-plane helpers ────────────────────────────────────────────────

/// Intersect a ray with a plane. Returns the world-space hit point.
pub fn ray_plane_intersect(ro: Vec3, rd: Vec3, plane_n: Vec3, plane_p: Vec3) -> Option<Vec3> {
    let denom = rd.dot(plane_n);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (plane_p - ro).dot(plane_n) / denom;
    if t < 0.0 {
        return None;
    }
    Some(ro + rd * t)
}

/// Resolve a screen position to a world-space point: try voxel surface first,
/// then fall back to a camera-facing plane at the camera target depth.
pub fn screen_to_world_pos(
    file: &crate::voxelle::VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    w: f32,
    h: f32,
    sx: f32,
    sy: f32,
) -> Option<Vec3> {
    // Try voxel surface hit first.
    if let Some(((ax, ay, az), _)) =
        crate::voxel_edit::preview_add_cell(file, voxel_map, camera, w, h, sx, sy)
    {
        return Some(Vec3::new(ax as f32 + 0.5, ay as f32 + 0.5, az as f32 + 0.5));
    }
    // Fallback: camera-facing plane at target depth.
    let (ro, rd) = crate::voxel_edit::screen_to_world_ray(camera, w, h, sx, sy);
    let ro = Vec3::new(ro.x, ro.y, ro.z);
    let rd = Vec3::new(rd.x, rd.y, rd.z).normalize();
    let eye = camera.smooth_eye();
    let target = camera.smooth_target;
    let plane_n = (eye - target).normalize();
    ray_plane_intersect(ro, rd, plane_n, target)
}

// ── Voxelization ─────────────────────────────────────────────────────

/// Minimum grid size to contain all joints with their radii.
pub fn min_grid_size_for_joints(joints: &[Joint], current_gs: i32) -> i32 {
    let mut max_abs = 0i32;
    for j in joints {
        let r_pad = (j.radius + 2.0).ceil() as i32;
        max_abs = max_abs
            .max((j.x.round() as i32 - r_pad).abs())
            .max((j.x.round() as i32 + r_pad).abs())
            .max((j.y.round() as i32 - r_pad).abs())
            .max((j.y.round() as i32 + r_pad).abs())
            .max((j.z.round() as i32 - r_pad).abs())
            .max((j.z.round() as i32 + r_pad).abs());
    }
    let need = (2 * (max_abs + 1)).max(1);
    current_gs.max(1).max(need).min(crate::voxel_edit::MAX_GRID_SIZE)
}

/// Generate voxel coordinates for the entire armature (capsule fills along bones,
/// sphere fills at isolated joints). Used by both live preview and final commit.
pub fn voxel_coords_for_session(
    session: &BoneSession,
    grid_size: i32,
    max_voxels: usize,
) -> Vec<VoxelCoord> {
    if session.joints.is_empty() {
        return Vec::new();
    }
    let gs = grid_size.max(1);
    let mut coords: HashSet<VoxelCoord> = HashSet::new();

    // Track which joints are used by bones (to render isolated joints as spheres).
    let mut used_joints: HashSet<u32> = HashSet::new();

    for bone in &session.bones {
        let Some(ja) = session.find_joint(bone.joint_a) else {
            continue;
        };
        let Some(jb) = session.find_joint(bone.joint_b) else {
            continue;
        };
        used_joints.insert(bone.joint_a);
        used_joints.insert(bone.joint_b);

        fill_capsule(
            ja.pos(),
            jb.pos(),
            ja.radius,
            jb.radius,
            gs,
            max_voxels,
            &mut coords,
        );
        if coords.len() >= max_voxels {
            break;
        }
    }

    // Isolated joints (not connected to any bone) get a sphere fill.
    for j in &session.joints {
        if used_joints.contains(&j.id) {
            continue;
        }
        fill_sphere(j.pos(), j.radius, gs, max_voxels, &mut coords);
        if coords.len() >= max_voxels {
            break;
        }
    }

    coords.into_iter().collect()
}

/// Fill a tapered capsule between two joint positions.
fn fill_capsule(
    a: Vec3,
    b: Vec3,
    r_a: f32,
    r_b: f32,
    gs: i32,
    max_voxels: usize,
    out: &mut HashSet<VoxelCoord>,
) {
    let dir = b - a;
    let length = dir.length();
    if length < 1e-6 {
        fill_sphere(a, r_a.max(r_b), gs, max_voxels, out);
        return;
    }

    // Step along the bone axis in half-voxel increments.
    let steps = (length * 2.0).ceil() as i32 + 1;
    for i in 0..=steps {
        if out.len() >= max_voxels {
            return;
        }
        let t = i as f32 / steps as f32;
        let center = a.lerp(b, t);
        let radius = lerp_f32(r_a, r_b, t);
        fill_disc_sphere(center, radius, gs, max_voxels, out);
    }
}

/// Fill a sphere of voxels at the given center and radius.
fn fill_sphere(
    center: Vec3,
    radius: f32,
    gs: i32,
    max_voxels: usize,
    out: &mut HashSet<VoxelCoord>,
) {
    fill_disc_sphere(center, radius, gs, max_voxels, out);
}

/// Fill a sphere (all voxels within `radius` of `center`).
fn fill_disc_sphere(
    center: Vec3,
    radius: f32,
    gs: i32,
    max_voxels: usize,
    out: &mut HashSet<VoxelCoord>,
) {
    let r_ceil = radius.ceil() as i32 + 1;
    let cx = center.x.round() as i32;
    let cy = center.y.round() as i32;
    let cz = center.z.round() as i32;
    let r2 = radius * radius;

    for dx in -r_ceil..=r_ceil {
        for dy in -r_ceil..=r_ceil {
            for dz in -r_ceil..=r_ceil {
                if out.len() >= max_voxels {
                    return;
                }
                let x = cx + dx;
                let y = cy + dy;
                let z = cz + dz;
                if !in_grid(x, y, z, gs) {
                    continue;
                }
                let fx = x as f32 - center.x;
                let fy = y as f32 - center.y;
                let fz = z as f32 - center.z;
                if fx * fx + fy * fy + fz * fz <= r2 {
                    out.insert((x, y, z));
                }
            }
        }
    }
}

#[inline]
fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// ── Commit ───────────────────────────────────────────────────────────

/// Voxelize the armature and write to the file. Returns edit deltas for
/// undo and GPU upload.
pub fn bone_commit_session(
    session: &BoneSession,
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    color: u32,
    material: MaterialId,
) -> Result<Vec<VoxelEditDelta>, String> {
    let grid_size = min_grid_size_for_joints(&session.joints, file.grid_size);
    file.grid_size = grid_size;
    let coords = voxel_coords_for_session(session, grid_size, usize::MAX);
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
