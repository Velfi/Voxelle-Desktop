//! Bone edit gizmo: screen-space handle pick + plane drag for joint positioning
//! and radius adjustment.

use crate::camera::OrbitCamera;
use crate::generators::bone_session::{BoneSelection, BoneSession, Joint};
use crate::greedy_mesh::MeshBuffers;
use crate::voxel_edit::{screen_to_world_ray, world_to_viewport_pixels};
use glam::Vec3;

// ── Handle types ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoneGizmoHandle {
    MoveX,
    MoveY,
    MoveZ,
    Scale, // radius ring
}

#[derive(Clone)]
pub struct BoneGizmoDrag {
    pub handle: BoneGizmoHandle,
    pub joint_id: u32,
    pub start: Joint,
    pub plane_n: Vec3,
    pub plane_p: Vec3,
    pub start_hit: Vec3,
}

// ── Layout constants ─────────────────────────────────────────────────

const HANDLE_BASE_SCALE: f32 = 0.4;
const SHAFT_LEN: f32 = 1.75;
const CONE_H: f32 = 0.52;
const ARROW_LEN: f32 = SHAFT_LEN + CONE_H;
const AXIS_ARROW_SIZE_MULT: f32 = 5.0;
const PICK_PX: f32 = 24.0;

fn gizmo_layout(center: Vec3, selected_radius: f32, eye: Vec3) -> (f32, f32) {
    let dist = center.distance(eye);
    let handle_scale = (dist * 0.028).clamp(0.22, 0.58);
    let radius_offset = (selected_radius + 0.9).max(1.2);
    let s = HANDLE_BASE_SCALE * handle_scale;
    let arrow_s = s * AXIS_ARROW_SIZE_MULT;
    let arrow_world_len = ARROW_LEN * arrow_s;
    let arm_base = (radius_offset - arrow_world_len).max(0.12);
    (arm_base, arrow_world_len)
}

fn screen_dist_to_segment(
    camera: &OrbitCamera,
    w: f32,
    h: f32,
    sx: f32,
    sy: f32,
    a: Vec3,
    b: Vec3,
) -> f32 {
    const SAMPLES: usize = 20;
    let mut best = f32::MAX;
    for i in 0..=SAMPLES {
        let t = i as f32 / SAMPLES as f32;
        let p = a.lerp(b, t);
        if let Some((px, py)) = world_to_viewport_pixels(camera, w, h, p.x, p.y, p.z) {
            let d = (px - sx).hypot(py - sy);
            best = best.min(d);
        }
    }
    best
}

// ── Hit-test ─────────────────────────────────────────────────────────

/// Returns the gizmo handle under the cursor (if any) for the currently
/// selected joint.
pub fn pick_bone_gizmo_handle(
    session: &BoneSession,
    camera: &OrbitCamera,
    w: f32,
    h: f32,
    sx: f32,
    sy: f32,
) -> Option<(BoneGizmoHandle, u32)> {
    // Gizmos only apply to selected joints.
    let joint_id = match session.selected {
        Some(BoneSelection::Joint(id)) => id,
        Some(BoneSelection::Bone(bone_id)) => {
            // When a bone is selected, show gizmos at both endpoints; pick the
            // nearest handle across both joints.
            let bone = session.bones.iter().find(|b| b.id == bone_id)?;
            let a = pick_joint_gizmo(session, camera, w, h, sx, sy, bone.joint_a);
            let b = pick_joint_gizmo(session, camera, w, h, sx, sy, bone.joint_b);
            return match (a, b) {
                (Some((ha, da)), Some((hb, db))) => {
                    if da <= db {
                        Some((ha, bone.joint_a))
                    } else {
                        Some((hb, bone.joint_b))
                    }
                }
                (Some((h, _)), None) => Some((h, bone.joint_a)),
                (None, Some((h, _))) => Some((h, bone.joint_b)),
                _ => None,
            };
        }
        None => return None,
    };
    pick_joint_gizmo(session, camera, w, h, sx, sy, joint_id)
        .map(|(handle, _)| (handle, joint_id))
}

fn pick_joint_gizmo(
    session: &BoneSession,
    camera: &OrbitCamera,
    w: f32,
    h: f32,
    sx: f32,
    sy: f32,
    joint_id: u32,
) -> Option<(BoneGizmoHandle, f32)> {
    let joint = session.find_joint(joint_id)?;
    let center = joint.pos();
    let eye = camera.smooth_eye();
    let (arm_base, arrow_world_len) = gizmo_layout(center, joint.radius, eye);

    let mut best: Option<(BoneGizmoHandle, f32)> = None;
    let axes = [
        (BoneGizmoHandle::MoveX, Vec3::X),
        (BoneGizmoHandle::MoveY, Vec3::Y),
        (BoneGizmoHandle::MoveZ, Vec3::Z),
    ];
    for (kind, axis) in axes {
        let a = center + axis * arm_base;
        let b = center + axis * (arm_base + arrow_world_len);
        let d = screen_dist_to_segment(camera, w, h, sx, sy, a, b);
        let replace = best.map(|(_, bd)| d < bd).unwrap_or(true);
        if replace {
            best = Some((kind, d));
        }
    }

    // Scale handle (diagonal, same as squishy)
    let radius_offset = (joint.radius + 0.9).max(1.2);
    let sp = radius_offset * 0.82;
    let scale_center = center + Vec3::new(sp, sp, sp);
    let scale_len = arrow_world_len * 0.45;
    let diag = Vec3::new(1.0, 1.0, 1.0).normalize();
    let sa = scale_center - diag * (scale_len * 0.5);
    let sb = scale_center + diag * (scale_len * 0.5);
    let sd = screen_dist_to_segment(camera, w, h, sx, sy, sa, sb);
    let replace = best.map(|(_, bd)| sd < bd).unwrap_or(true);
    if replace {
        best = Some((BoneGizmoHandle::Scale, sd));
    }

    let (kind, d) = best?;
    if d <= PICK_PX {
        Some((kind, d))
    } else {
        None
    }
}

// ── Drag ─────────────────────────────────────────────────────────────

use super::bone_session::ray_plane_intersect;

pub fn bone_gizmo_begin_drag(
    session: &BoneSession,
    camera: &OrbitCamera,
    w: f32,
    h: f32,
    sx: f32,
    sy: f32,
    handle: BoneGizmoHandle,
    joint_id: u32,
) -> Option<BoneGizmoDrag> {
    let joint = session.find_joint(joint_id)?.clone();
    let center = joint.pos();
    let (ro, rd) = screen_to_world_ray(camera, w, h, sx, sy);
    let ro = Vec3::new(ro.x, ro.y, ro.z);
    let rd = Vec3::new(rd.x, rd.y, rd.z).normalize();
    let eye = camera.smooth_eye();
    let plane_n = (center - eye).normalize();
    let plane_p = center;
    let start_hit = ray_plane_intersect(ro, rd, plane_n, plane_p)?;

    Some(BoneGizmoDrag {
        handle,
        joint_id,
        start: joint,
        plane_n,
        plane_p,
        start_hit,
    })
}

pub fn bone_gizmo_apply_drag(
    session: &mut BoneSession,
    camera: &OrbitCamera,
    w: f32,
    h: f32,
    sx: f32,
    sy: f32,
    drag: &BoneGizmoDrag,
) {
    let (ro, rd) = screen_to_world_ray(camera, w, h, sx, sy);
    let ro = Vec3::new(ro.x, ro.y, ro.z);
    let rd = Vec3::new(rd.x, rd.y, rd.z).normalize();
    let Some(hit) = ray_plane_intersect(ro, rd, drag.plane_n, drag.plane_p) else {
        return;
    };
    let delta = hit - drag.start_hit;
    let start = &drag.start;
    let view = camera.view_matrix();
    let inv_view = view.inverse();
    let camera_right = inv_view.x_axis.truncate().normalize();

    match drag.handle {
        BoneGizmoHandle::MoveX => {
            let nx = start.x + delta.x;
            session.set_joint_position(drag.joint_id, nx, start.y, start.z);
        }
        BoneGizmoHandle::MoveY => {
            let ny = start.y + delta.y;
            session.set_joint_position(drag.joint_id, start.x, ny, start.z);
        }
        BoneGizmoHandle::MoveZ => {
            let nz = start.z + delta.z;
            session.set_joint_position(drag.joint_id, start.x, start.y, nz);
        }
        BoneGizmoHandle::Scale => {
            let signed = delta.dot(camera_right);
            let nr = (start.radius + signed).clamp(0.5, 64.0);
            session.set_joint_radius(drag.joint_id, nr);
        }
    }
}

// ── Skeleton wireframe (Blender-style) ──────────────────────────────

/// Minimum screen fraction for bone/joint wireframe visibility.
const MIN_SCREEN_FRAC: f32 = 0.012;

/// Compute the visual radius for a joint, clamped to a minimum screen size.
fn visual_radius(eye: Vec3, pos: Vec3, radius: f32) -> f32 {
    let dist = eye.distance(pos).max(0.1);
    radius.max(dist * MIN_SCREEN_FRAC)
}

/// Draw the entire skeleton: octahedral (diamond) bones and sphere joints.
pub fn append_bone_skeleton_wire(
    session: &BoneSession,
    camera: &OrbitCamera,
    out: &mut MeshBuffers,
) {
    let eye = camera.smooth_eye();
    let joint_color = [0.3, 0.75, 1.0_f32];
    let joint_sel_color = [1.0, 0.85, 0.2_f32];
    let bone_color = [0.7, 0.7, 0.72_f32];
    let bone_sel_color = [1.0, 0.85, 0.2_f32];
    let pending_color = [0.3, 0.9, 0.4_f32];

    // Draw bones as octahedral diamonds
    for bone in &session.bones {
        let Some(ja) = session.find_joint(bone.joint_a) else {
            continue;
        };
        let Some(jb) = session.find_joint(bone.joint_b) else {
            continue;
        };
        let is_sel = session.selected == Some(BoneSelection::Bone(bone.id));
        let color = if is_sel { bone_sel_color } else { bone_color };
        let avg_r = (ja.radius + jb.radius) * 0.5;
        let mid = ja.pos().lerp(jb.pos(), 0.5);
        let width = visual_radius(eye, mid, avg_r * 0.35).max(0.15);
        draw_octahedral_bone(out, ja.pos(), jb.pos(), width, color);
    }

    // Draw joints as icospheres
    for j in &session.joints {
        let is_sel = session.selected == Some(BoneSelection::Joint(j.id));
        let is_pending = session.pending_joint == Some(j.id);
        let color = if is_pending {
            pending_color
        } else if is_sel {
            joint_sel_color
        } else {
            joint_color
        };
        let r = visual_radius(eye, j.pos(), j.radius * 0.4).max(0.12);
        draw_sphere(out, j.pos(), r, color);
    }
}

/// Emit an octahedral (diamond) bone between two joint positions.
/// The octahedron has its tips at `a` and `b`, and 4 equatorial vertices
/// at the midpoint offset by `width` in the perpendicular plane.
fn draw_octahedral_bone(
    out: &mut MeshBuffers,
    a: Vec3,
    b: Vec3,
    width: f32,
    color: [f32; 3],
) {
    let dir = b - a;
    let length = dir.length();
    if length < 1e-6 {
        return;
    }
    let axis = dir / length;

    // Build perpendicular basis
    let perp = if axis.y.abs() < 0.9 {
        axis.cross(Vec3::Y).normalize()
    } else {
        axis.cross(Vec3::X).normalize()
    };
    let perp2 = axis.cross(perp).normalize();

    let mid = a.lerp(b, 0.5);
    let v0 = a;                          // tip A
    let v1 = b;                          // tip B
    let v2 = mid + perp * width;         // equator +perp
    let v3 = mid + perp2 * width;        // equator +perp2
    let v4 = mid - perp * width;         // equator -perp
    let v5 = mid - perp2 * width;        // equator -perp2

    // 8 triangular faces (4 on each half)
    let faces: [[Vec3; 3]; 8] = [
        // Top half (a → equator)
        [v0, v2, v3],
        [v0, v3, v4],
        [v0, v4, v5],
        [v0, v5, v2],
        // Bottom half (equator → b)
        [v1, v3, v2],
        [v1, v4, v3],
        [v1, v5, v4],
        [v1, v2, v5],
    ];

    let base_idx = (out.positions.len() / 3) as u32;
    let mut vi = 0u32;
    for face in &faces {
        let e1 = face[1] - face[0];
        let e2 = face[2] - face[0];
        let n = e1.cross(e2).normalize();
        for &vert in face {
            out.positions.extend_from_slice(&[vert.x, vert.y, vert.z]);
            out.normals.extend_from_slice(&[n.x, n.y, n.z]);
            out.colors.extend_from_slice(&color);
            out.mat_kind.push(0.0);
            out.ao.push(1.0);
            out.emission_tint.extend_from_slice(&[0.0, 0.0, 0.0]);
        }
        out.indices.push(base_idx + vi);
        out.indices.push(base_idx + vi + 1);
        out.indices.push(base_idx + vi + 2);
        vi += 3;
    }
}

/// Emit a low-poly sphere (icosphere, 1 subdivision = 80 triangles) at the
/// given position.
fn draw_sphere(out: &mut MeshBuffers, center: Vec3, radius: f32, color: [f32; 3]) {
    // Start with icosahedron vertices
    let t = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let raw_verts: [[f32; 3]; 12] = [
        [-1.0,  t, 0.0], [ 1.0,  t, 0.0], [-1.0, -t, 0.0], [ 1.0, -t, 0.0],
        [ 0.0, -1.0,  t], [ 0.0,  1.0,  t], [ 0.0, -1.0, -t], [ 0.0,  1.0, -t],
        [  t, 0.0, -1.0], [  t, 0.0,  1.0], [ -t, 0.0, -1.0], [ -t, 0.0,  1.0],
    ];
    let ico_faces: [[usize; 3]; 20] = [
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ];

    // Subdivide once for smoother sphere (20 → 80 tris)
    let mut verts: Vec<Vec3> = raw_verts
        .iter()
        .map(|v| Vec3::new(v[0], v[1], v[2]).normalize())
        .collect();
    let mut faces: Vec<[usize; 3]> = ico_faces.to_vec();

    let mut midpoint_cache = std::collections::HashMap::new();
    let get_mid = |a: usize, b: usize, vs: &mut Vec<Vec3>, cache: &mut std::collections::HashMap<(usize,usize), usize>| -> usize {
        let key = if a < b { (a, b) } else { (b, a) };
        if let Some(&idx) = cache.get(&key) {
            return idx;
        }
        let mid = (vs[a] + vs[b]).normalize();
        let idx = vs.len();
        vs.push(mid);
        cache.insert(key, idx);
        idx
    };
    let mut new_faces = Vec::with_capacity(faces.len() * 4);
    for f in &faces {
        let a = get_mid(f[0], f[1], &mut verts, &mut midpoint_cache);
        let b = get_mid(f[1], f[2], &mut verts, &mut midpoint_cache);
        let c = get_mid(f[2], f[0], &mut verts, &mut midpoint_cache);
        new_faces.push([f[0], a, c]);
        new_faces.push([f[1], b, a]);
        new_faces.push([f[2], c, b]);
        new_faces.push([a, b, c]);
    }
    faces = new_faces;

    // Emit triangles
    let base_idx = (out.positions.len() / 3) as u32;
    for v in &verts {
        let p = center + *v * radius;
        out.positions.extend_from_slice(&[p.x, p.y, p.z]);
        out.normals.extend_from_slice(&[v.x, v.y, v.z]);
        out.colors.extend_from_slice(&color);
        out.mat_kind.push(0.0);
        out.ao.push(1.0);
        out.emission_tint.extend_from_slice(&[0.0, 0.0, 0.0]);
    }
    for f in &faces {
        out.indices.push(base_idx + f[0] as u32);
        out.indices.push(base_idx + f[1] as u32);
        out.indices.push(base_idx + f[2] as u32);
    }
}
