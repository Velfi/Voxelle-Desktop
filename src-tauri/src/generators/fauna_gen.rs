//! Fauna (creature) generator (web parity): skeleton-based quadruped/biped creatures with
//! spine chain, body voxelization, limbs via 3-bone FABRIK IK, head, and tail.

use super::common::{
    v3_add, v3_cross, v3_len, v3_lerp, v3_normalize, v3_round, v3_scale, v3_sub, V3,
};
use crate::camera::OrbitCamera;
use crate::greedy_mesh::VoxelCoord;
use crate::voxel_edit::{
    effective_ray_grid_size, ensure_grid_fits_coord, ray_first_solid, screen_to_world_ray,
    VoxelEditDelta,
};
use crate::voxelle::{MaterialId, Scene, Voxel, VoxelleFile};
use ahash::AHashMap;
use std::collections::HashSet;

const FAUNA_MAX_VOXELS: usize = 12_000;

// ── Local alias ─────────────────────────────────────────────────────────────

// `v3_norm` was the name used throughout this file; alias to common's
// `v3_normalize` so call sites don't need to change.
#[inline(always)]
fn v3_norm(v: V3) -> V3 {
    v3_normalize(v)
}

// ── Body frame ──────────────────────────────────────────────────────────────

/// Build a right-handed creature frame from a face normal.
/// Returns (forward, side, up) in world space.
fn body_frame_from_normal(face_normal: [f32; 3], body_yaw: f32) -> ([f32; 3], [f32; 3], [f32; 3]) {
    // Up is the face normal (creature stands on the face)
    let up = v3_norm(face_normal);

    // Pick a provisional forward perpendicular to up
    let world_fwd = if up[1].abs() > 0.9 {
        [0.0, 0.0, -1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let side0 = v3_norm(v3_cross(up, world_fwd));
    let fwd0 = v3_norm(v3_cross(side0, up));

    // Apply yaw rotation around up axis
    let cos_y = body_yaw.cos();
    let sin_y = body_yaw.sin();
    let forward = v3_norm(v3_add(v3_scale(fwd0, cos_y), v3_scale(side0, sin_y)));
    let side = v3_norm(v3_cross(up, forward));

    (forward, side, up)
}

/// Transform a creature-local [forward, side, up] offset to world space.
fn local_to_world(
    center: [f32; 3],
    forward: [f32; 3],
    side: [f32; 3],
    up: [f32; 3],
    local: [f32; 3],
) -> [f32; 3] {
    v3_add(
        center,
        v3_add(
            v3_scale(forward, local[0]),
            v3_add(v3_scale(side, local[1]), v3_scale(up, local[2])),
        ),
    )
}

// ── Spine ───────────────────────────────────────────────────────────────────

/// Spine bone: position in world space.
struct SpineBone {
    pos: [f32; 3],
}

/// Build spine chain from pelvis to head.
/// Returns a Vec of positions along the spine, then neck, then head.
fn build_spine_chain(
    center: [f32; 3],
    _forward: [f32; 3],
    _up: [f32; 3],
    body_arch: f32,
    spine_segments: i32,
    body_length: i32,
    neck_length: i32,
    head_length: i32,
    spine_pose_chest: [f32; 3],
    spine_pose_neck: [f32; 3],
    spine_pose_head: [f32; 3],
    fwd_axis: [f32; 3],
    side_axis: [f32; 3],
    up_axis: [f32; 3],
) -> (Vec<SpineBone>, Vec<SpineBone>, Vec<SpineBone>) {
    let segs = spine_segments.max(2).min(20);
    let half_len = body_length as f32 * 0.5;

    // Body spine: from rear (-half_len) to front (+half_len)
    let mut body_bones = Vec::with_capacity(segs as usize + 1);
    for i in 0..=segs {
        let t = i as f32 / segs as f32;
        let fwd_offset = -half_len + body_length as f32 * t;
        // Arch: parabolic lift peaking at mid-torso
        let arch_lift = body_arch * 4.0 * t * (1.0 - t);
        let pos = v3_add(
            center,
            v3_add(v3_scale(fwd_axis, fwd_offset), v3_scale(up_axis, arch_lift)),
        );
        body_bones.push(SpineBone { pos });
    }

    // Apply chest pose offset to front-most body bone
    if let Some(last) = body_bones.last_mut() {
        last.pos = local_to_world(last.pos, fwd_axis, side_axis, up_axis, spine_pose_chest);
    }

    // Neck chain: from chest end forward
    let chest_pos = body_bones.last().map(|b| b.pos).unwrap_or(center);
    let neck_segs = (neck_length.max(1) as usize).min(10);
    let mut neck_bones = Vec::with_capacity(neck_segs + 1);
    for i in 0..=neck_segs {
        let t = i as f32 / neck_segs as f32;
        let offset_fwd = neck_length as f32 * t;
        // Neck curves slightly upward
        let offset_up = neck_length as f32 * 0.3 * t;
        let mut pos = v3_add(
            chest_pos,
            v3_add(v3_scale(fwd_axis, offset_fwd), v3_scale(up_axis, offset_up)),
        );
        if i == neck_segs {
            pos = local_to_world(pos, fwd_axis, side_axis, up_axis, spine_pose_neck);
        }
        neck_bones.push(SpineBone { pos });
    }

    // Head chain: from neck end forward
    let neck_end = neck_bones.last().map(|b| b.pos).unwrap_or(chest_pos);
    let head_segs = (head_length.max(1) as usize).min(8);
    let mut head_bones = Vec::with_capacity(head_segs + 1);
    for i in 0..=head_segs {
        let t = i as f32 / head_segs as f32;
        let offset_fwd = head_length as f32 * t;
        let mut pos = v3_add(neck_end, v3_scale(fwd_axis, offset_fwd));
        if i == head_segs {
            pos = local_to_world(pos, fwd_axis, side_axis, up_axis, spine_pose_head);
        }
        head_bones.push(SpineBone { pos });
    }

    (body_bones, neck_bones, head_bones)
}

// ── Voxel fill helpers ──────────────────────────────────────────────────────

/// Insert a voxel at (x,y,z) if not already occupied, respecting the cap.
fn place_voxel(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    seen: &mut HashSet<VoxelCoord>,
    out: &mut Vec<VoxelEditDelta>,
    x: i32,
    y: i32,
    z: i32,
    color: u32,
    material: MaterialId,
) -> bool {
    if out.len() >= FAUNA_MAX_VOXELS {
        return false;
    }
    let coord = (x, y, z);
    if !seen.insert(coord) {
        return true;
    }
    if voxel_map.contains_key(&coord) {
        return true;
    }
    ensure_grid_fits_coord(file, x, y, z);
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
    voxel_map.insert(coord, idx);
    out.push(VoxelEditDelta::Added(nv));
    true
}

/// Fill an elliptical cross-section at `center` on the plane defined by `side` and `up` axes.
fn fill_slice(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    seen: &mut HashSet<VoxelCoord>,
    out: &mut Vec<VoxelEditDelta>,
    center: [f32; 3],
    side: [f32; 3],
    up: [f32; 3],
    half_w: f32,
    half_h: f32,
    color: u32,
    material: MaterialId,
) {
    let rw = half_w.ceil() as i32;
    let rh = half_h.ceil() as i32;
    for du in -rh..=rh {
        for dv in -rw..=rw {
            let fu = du as f32;
            let fv = dv as f32;
            // Ellipse test
            let ex = if half_w > 0.01 { fv / half_w } else { 999.0 };
            let ey = if half_h > 0.01 { fu / half_h } else { 999.0 };
            if ex * ex + ey * ey > 1.0 {
                continue;
            }
            let p = v3_add(center, v3_add(v3_scale(side, fv), v3_scale(up, fu)));
            let (x, y, z) = v3_round(p);
            if !place_voxel(file, voxel_map, seen, out, x, y, z, color, material) {
                return;
            }
        }
    }
}

/// Interpolate elliptical slices between two points.
fn fill_slice_bridge(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    seen: &mut HashSet<VoxelCoord>,
    out: &mut Vec<VoxelEditDelta>,
    p_start: [f32; 3],
    p_end: [f32; 3],
    side: [f32; 3],
    up: [f32; 3],
    hw_start: f32,
    hh_start: f32,
    hw_end: f32,
    hh_end: f32,
    color: u32,
    material: MaterialId,
) {
    let dist = v3_len(v3_sub(p_end, p_start));
    let steps = (dist.ceil() as i32).max(1);
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let center = v3_lerp(p_start, p_end, t);
        let hw = hw_start + (hw_end - hw_start) * t;
        let hh = hh_start + (hh_end - hh_start) * t;
        fill_slice(
            file, voxel_map, seen, out, center, side, up, hw, hh, color, material,
        );
    }
}

/// Fill a sphere at `center` with given radius.
fn fill_sphere(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    seen: &mut HashSet<VoxelCoord>,
    out: &mut Vec<VoxelEditDelta>,
    center: [f32; 3],
    radius: f32,
    color: u32,
    material: MaterialId,
) {
    let r = radius.ceil() as i32;
    let r2 = (radius + 0.4) * (radius + 0.4);
    let (cx, cy, cz) = v3_round(center);
    for dx in -r..=r {
        for dy in -r..=r {
            for dz in -r..=r {
                if (dx * dx + dy * dy + dz * dz) as f32 <= r2 {
                    if !place_voxel(
                        file,
                        voxel_map,
                        seen,
                        out,
                        cx + dx,
                        cy + dy,
                        cz + dz,
                        color,
                        material,
                    ) {
                        return;
                    }
                }
            }
        }
    }
}

/// Fill an oriented ellipsoid at `center` using the creature frame axes.
fn fill_oriented_ellipsoid(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    seen: &mut HashSet<VoxelCoord>,
    out: &mut Vec<VoxelEditDelta>,
    center: [f32; 3],
    fwd: [f32; 3],
    side: [f32; 3],
    up: [f32; 3],
    half_fwd: f32,
    half_side: f32,
    half_up: f32,
    color: u32,
    material: MaterialId,
) {
    let max_r = half_fwd.max(half_side).max(half_up).ceil() as i32;
    for df in -max_r..=max_r {
        for ds in -max_r..=max_r {
            for du in -max_r..=max_r {
                let ef = if half_fwd > 0.01 {
                    df as f32 / half_fwd
                } else {
                    999.0
                };
                let es = if half_side > 0.01 {
                    ds as f32 / half_side
                } else {
                    999.0
                };
                let eu = if half_up > 0.01 {
                    du as f32 / half_up
                } else {
                    999.0
                };
                if ef * ef + es * es + eu * eu > 1.0 {
                    continue;
                }
                let p = v3_add(
                    center,
                    v3_add(
                        v3_scale(fwd, df as f32),
                        v3_add(v3_scale(side, ds as f32), v3_scale(up, du as f32)),
                    ),
                );
                let (x, y, z) = v3_round(p);
                if !place_voxel(file, voxel_map, seen, out, x, y, z, color, material) {
                    return;
                }
            }
        }
    }
}

/// Fill a tapered capsule between two points.
fn fill_segment_capsule(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    seen: &mut HashSet<VoxelCoord>,
    out: &mut Vec<VoxelEditDelta>,
    p_start: [f32; 3],
    p_end: [f32; 3],
    r_start: f32,
    r_end: f32,
    color: u32,
    material: MaterialId,
) {
    let dist = v3_len(v3_sub(p_end, p_start));
    let steps = (dist.ceil() as i32).max(1);
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let center = v3_lerp(p_start, p_end, t);
        let r = r_start + (r_end - r_start) * t;
        fill_sphere(file, voxel_map, seen, out, center, r, color, material);
    }
}

// ── 3-Bone FABRIK IK ───────────────────────────────────────────────────────

/// Solve 3-bone FABRIK: root(p0) → mid(p1) → distal(p2) → end(p3).
/// `pole` biases the mid-joint direction.
fn fabrik_3bone(
    root: [f32; 3],
    target: [f32; 3],
    pole: [f32; 3],
    l1: f32,
    l2: f32,
    l3: f32,
    iterations: u32,
) -> [[f32; 3]; 4] {
    // Initialize chain with hints
    let total = l1 + l2 + l3;
    let to_target = v3_sub(target, root);
    let reach = v3_len(to_target);

    // If fully stretched, distribute points along line
    if reach >= total - 0.01 {
        let dir = v3_norm(to_target);
        return [
            root,
            v3_add(root, v3_scale(dir, l1)),
            v3_add(root, v3_scale(dir, l1 + l2)),
            v3_add(root, v3_scale(dir, total.min(reach))),
        ];
    }

    // Initial hint positions using pole direction
    let fwd = v3_norm(to_target);
    let pole_dir = v3_norm(v3_sub(pole, root));
    // Bias mid-joint toward pole
    let mid_hint = v3_add(
        root,
        v3_add(v3_scale(fwd, l1 * 0.7), v3_scale(pole_dir, l1 * 0.5)),
    );

    let mut p0 = root;
    let mut p1 = mid_hint;
    let mut p2 = v3_lerp(p1, target, 0.5);
    let mut p3 = target;

    for _ in 0..iterations {
        // Backward pass: p3 = target, pull backward
        p3 = target;
        let d23 = v3_norm(v3_sub(p2, p3));
        p2 = v3_add(p3, v3_scale(d23, l3));
        let d12 = v3_norm(v3_sub(p1, p2));
        p1 = v3_add(p2, v3_scale(d12, l2));
        let d01 = v3_norm(v3_sub(p0, p1));
        let _ = v3_add(p1, v3_scale(d01, l1));

        // Forward pass: p0 = root, push forward
        p0 = root;
        let d01 = v3_norm(v3_sub(p1, p0));
        p1 = v3_add(p0, v3_scale(d01, l1));
        let d12 = v3_norm(v3_sub(p2, p1));
        p2 = v3_add(p1, v3_scale(d12, l2));
        let d23 = v3_norm(v3_sub(p3, p2));
        p3 = v3_add(p2, v3_scale(d23, l3));
    }

    [p0, p1, p2, p3]
}

// ── Profile multipliers ─────────────────────────────────────────────────────

/// Body profile multiplier for a position along the spine (0=rear, 1=front).
fn body_profile_multiplier(t: f32, stance: &str) -> (f32, f32) {
    // Returns (width_mult, height_mult)
    match stance {
        "biped" => {
            // Wider shoulders, narrows at waist, widens at hips
            let w = if t < 0.3 {
                // Hips
                0.9 + 0.1 * (1.0 - t / 0.3)
            } else if t < 0.6 {
                // Waist narrows
                let u = (t - 0.3) / 0.3;
                0.9 - 0.2 * (1.0 - (2.0 * u - 1.0).powi(2))
            } else {
                // Chest/shoulders widen
                let u = (t - 0.6) / 0.4;
                0.9 + 0.3 * u
            };
            let h = if t < 0.3 {
                0.85
            } else if t < 0.7 {
                0.7
            } else {
                0.9 + 0.15 * ((t - 0.7) / 0.3)
            };
            (w, h)
        }
        _ => {
            // Quadruped: narrows mid-torso, widens at ends
            let u = (t - 0.5).abs() * 2.0; // 1 at ends, 0 at middle
            let w = 0.7 + 0.3 * u;
            let h = 0.75 + 0.25 * u;
            (w, h)
        }
    }
}

// ── Foot archetypes ─────────────────────────────────────────────────────────

/// Archetype distal offset (raise for digitigrade/ungulate).
fn archetype_distal_raise(archetype: &str) -> f32 {
    match archetype {
        "digitigrade" => 1.5,
        "ungulate" => 2.5,
        _ => 0.0, // plantigrade: flat foot
    }
}

/// Place a foot/paw/hoof at the end effector.
fn fill_foot(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    seen: &mut HashSet<VoxelCoord>,
    out: &mut Vec<VoxelEditDelta>,
    pos: [f32; 3],
    fwd: [f32; 3],
    side: [f32; 3],
    _up: [f32; 3],
    archetype: &str,
    color: u32,
    material: MaterialId,
) {
    match archetype {
        "digitigrade" => {
            // Elongated paw: 3 forward, narrow
            for i in 0..3 {
                let p = v3_add(pos, v3_scale(fwd, i as f32));
                fill_sphere(file, voxel_map, seen, out, p, 0.8, color, material);
            }
        }
        "ungulate" => {
            // Small hard hoof: compact sphere
            fill_sphere(file, voxel_map, seen, out, pos, 1.0, color, material);
        }
        _ => {
            // Plantigrade: wide flat foot
            for i in 0..2 {
                let p = v3_add(pos, v3_scale(fwd, i as f32));
                let (x, y, z) = v3_round(p);
                for ds in -1..=1 {
                    let pp = v3_add(p, v3_scale(side, ds as f32));
                    let (fx, fy, fz) = v3_round(pp);
                    place_voxel(file, voxel_map, seen, out, fx, fy, fz, color, material);
                }
                let _ = place_voxel(file, voxel_map, seen, out, x, y, z, color, material);
            }
        }
    }
}

// ── Main generator ──────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn generator_fauna_at_screen(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    stance: &str,
    archetype: &str,
    anchor_offset_u: i32,
    anchor_offset_v: i32,
    body_yaw: f32,
    body_arch: f32,
    spine_segments: i32,
    body_length: i32,
    body_half_width: i32,
    body_half_height: i32,
    neck_length: i32,
    neck_half_width: i32,
    neck_half_height: i32,
    head_length: i32,
    head_half_width: i32,
    head_half_height: i32,
    tail_length: i32,
    shoulder_offset_forward: i32,
    hip_offset_forward: i32,
    front_upper_length: i32,
    front_lower_length: i32,
    hind_upper_length: i32,
    hind_lower_length: i32,
    limb_targets: &[[f32; 3]; 4],
    limb_poles: &[[f32; 3]; 4],
    spine_pose_chest: [f32; 3],
    spine_pose_neck: [f32; 3],
    spine_pose_head: [f32; 3],
    auto_foot_placement: bool,
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

    // Face normal (inward → outward)
    let face_normal = [
        (face_empty.0 - solid.0) as f32,
        (face_empty.1 - solid.1) as f32,
        (face_empty.2 - solid.2) as f32,
    ];

    // Anchor: face_empty + UV offsets along the face plane
    let (forward, side, up) = body_frame_from_normal(face_normal, body_yaw);

    // Tangent offsets for anchor
    let anchor_base = [
        face_empty.0 as f32
            + side[0] * anchor_offset_u as f32
            + forward[0] * anchor_offset_v as f32,
        face_empty.1 as f32
            + side[1] * anchor_offset_u as f32
            + forward[1] * anchor_offset_v as f32,
        face_empty.2 as f32
            + side[2] * anchor_offset_u as f32
            + forward[2] * anchor_offset_v as f32,
    ];

    // Compute standing lift: creature center is raised so feet touch the surface
    let max_leg = if stance == "biped" {
        (front_upper_length + front_lower_length) as f32
    } else {
        ((front_upper_length + front_lower_length) as f32)
            .max((hind_upper_length + hind_lower_length) as f32)
    };
    let lift = max_leg * 0.85;
    let center = v3_add(anchor_base, v3_scale(up, lift));

    let mut out = Vec::new();
    let mut seen: HashSet<VoxelCoord> = HashSet::new();

    // ── Build spine ─────────────────────────────────────────────────────
    let (body_bones, neck_bones, head_bones) = build_spine_chain(
        center,
        forward,
        up,
        body_arch,
        spine_segments,
        body_length,
        neck_length,
        head_length,
        spine_pose_chest,
        spine_pose_neck,
        spine_pose_head,
        forward,
        side,
        up,
    );

    // ── Voxelize body along spine ───────────────────────────────────────
    let segs = body_bones.len();
    for i in 0..segs {
        let t = if segs > 1 {
            i as f32 / (segs - 1) as f32
        } else {
            0.5
        };
        let (w_mult, h_mult) = body_profile_multiplier(t, stance);
        let hw = body_half_width as f32 * w_mult;
        let hh = body_half_height as f32 * h_mult;
        fill_slice(
            file,
            voxel_map,
            &mut seen,
            &mut out,
            body_bones[i].pos,
            side,
            up,
            hw,
            hh,
            color,
            material,
        );
        // Bridge to next segment
        if i + 1 < segs {
            let t_next = (i + 1) as f32 / (segs - 1).max(1) as f32;
            let (w_next, h_next) = body_profile_multiplier(t_next, stance);
            let hw_next = body_half_width as f32 * w_next;
            let hh_next = body_half_height as f32 * h_next;
            fill_slice_bridge(
                file,
                voxel_map,
                &mut seen,
                &mut out,
                body_bones[i].pos,
                body_bones[i + 1].pos,
                side,
                up,
                hw,
                hh,
                hw_next,
                hh_next,
                color,
                material,
            );
        }
    }

    // ── Localized masses ────────────────────────────────────────────────

    // Pelvis mass at rear of body
    let pelvis_pos = body_bones.first().map(|b| b.pos).unwrap_or(center);
    fill_oriented_ellipsoid(
        file,
        voxel_map,
        &mut seen,
        &mut out,
        pelvis_pos,
        forward,
        side,
        up,
        body_half_height as f32 * 0.8,
        body_half_width as f32 * 1.1,
        body_half_height as f32 * 0.9,
        color,
        material,
    );

    // Chest mass at front of body
    let chest_pos = body_bones.last().map(|b| b.pos).unwrap_or(center);
    fill_oriented_ellipsoid(
        file,
        voxel_map,
        &mut seen,
        &mut out,
        chest_pos,
        forward,
        side,
        up,
        body_half_height as f32 * 0.9,
        body_half_width as f32 * 1.15,
        body_half_height as f32 * 1.0,
        color,
        material,
    );

    // Rump (quadruped only)
    if stance == "quadruped" {
        let rump_pos = v3_add(pelvis_pos, v3_scale(up, -(body_half_height as f32 * 0.2)));
        fill_oriented_ellipsoid(
            file,
            voxel_map,
            &mut seen,
            &mut out,
            rump_pos,
            forward,
            side,
            up,
            body_half_height as f32 * 0.6,
            body_half_width as f32 * 1.0,
            body_half_height as f32 * 0.7,
            color,
            material,
        );
    }

    // Belly mass at mid-body
    let belly_idx = segs / 2;
    if belly_idx < body_bones.len() {
        let belly_pos = v3_add(
            body_bones[belly_idx].pos,
            v3_scale(up, -(body_half_height as f32 * 0.3)),
        );
        fill_oriented_ellipsoid(
            file,
            voxel_map,
            &mut seen,
            &mut out,
            belly_pos,
            forward,
            side,
            up,
            body_length as f32 * 0.2,
            body_half_width as f32 * 0.8,
            body_half_height as f32 * 0.5,
            color,
            material,
        );
    }

    // ── Neck ────────────────────────────────────────────────────────────
    if neck_bones.len() >= 2 {
        for i in 0..neck_bones.len() - 1 {
            let t0 = i as f32 / (neck_bones.len() - 1) as f32;
            let t1 = (i + 1) as f32 / (neck_bones.len() - 1) as f32;
            let hw0 = neck_half_width as f32 * (1.0 - t0 * 0.3);
            let hh0 = neck_half_height as f32 * (1.0 - t0 * 0.3);
            let hw1 = neck_half_width as f32 * (1.0 - t1 * 0.3);
            let hh1 = neck_half_height as f32 * (1.0 - t1 * 0.3);
            fill_slice_bridge(
                file,
                voxel_map,
                &mut seen,
                &mut out,
                neck_bones[i].pos,
                neck_bones[i + 1].pos,
                side,
                up,
                hw0,
                hh0,
                hw1,
                hh1,
                color,
                material,
            );
        }
    }

    // ── Head ────────────────────────────────────────────────────────────
    let head_start = head_bones.first().map(|b| b.pos).unwrap_or(center);
    let head_end = head_bones.last().map(|b| b.pos).unwrap_or(head_start);

    if stance == "biped" {
        // Spherical head
        let head_center = v3_lerp(head_start, head_end, 0.5);
        let head_r = head_half_width.max(head_half_height) as f32;
        fill_sphere(
            file,
            voxel_map,
            &mut seen,
            &mut out,
            head_center,
            head_r,
            color,
            material,
        );
    } else {
        // Quadruped: bridged head with snout
        // Main head mass
        fill_slice_bridge(
            file,
            voxel_map,
            &mut seen,
            &mut out,
            head_start,
            head_end,
            side,
            up,
            head_half_width as f32,
            head_half_height as f32,
            head_half_width as f32 * 0.8,
            head_half_height as f32 * 0.7,
            color,
            material,
        );
        // Snout extends forward
        let snout_len = (head_length as f32 * 0.6).max(1.0);
        let snout_start = head_end;
        let snout_end = v3_add(head_end, v3_scale(forward, snout_len));
        fill_slice_bridge(
            file,
            voxel_map,
            &mut seen,
            &mut out,
            snout_start,
            snout_end,
            side,
            up,
            head_half_width as f32 * 0.6,
            head_half_height as f32 * 0.5,
            head_half_width as f32 * 0.3,
            head_half_height as f32 * 0.3,
            color,
            material,
        );
    }

    // ── Tail ────────────────────────────────────────────────────────────
    if tail_length > 0 {
        let tail_start = pelvis_pos;
        let tail_dir = v3_norm(v3_add(v3_scale(forward, -1.0), v3_scale(up, -0.3)));
        for i in 0..tail_length {
            let p = v3_add(tail_start, v3_scale(tail_dir, i as f32 + 1.0));
            let (x, y, z) = v3_round(p);
            if !place_voxel(
                file, voxel_map, &mut seen, &mut out, x, y, z, color, material,
            ) {
                break;
            }
        }
    }

    // ── Limbs ───────────────────────────────────────────────────────────
    // Limb order: frontLeft(0), frontRight(1), hindLeft(2), hindRight(3)
    let shoulder_fwd = shoulder_offset_forward as f32;
    let hip_fwd = hip_offset_forward as f32;

    struct LimbDef {
        root_local_fwd: f32,
        root_local_side: f32,
        upper_len: f32,
        lower_len: f32,
        target_idx: usize,
        is_front: bool,
    }

    let body_hw = body_half_width as f32;

    let limbs = [
        LimbDef {
            root_local_fwd: shoulder_fwd,
            root_local_side: -body_hw,
            upper_len: front_upper_length as f32,
            lower_len: front_lower_length as f32,
            target_idx: 0,
            is_front: true,
        },
        LimbDef {
            root_local_fwd: shoulder_fwd,
            root_local_side: body_hw,
            upper_len: front_upper_length as f32,
            lower_len: front_lower_length as f32,
            target_idx: 1,
            is_front: true,
        },
        LimbDef {
            root_local_fwd: hip_fwd,
            root_local_side: -body_hw,
            upper_len: hind_upper_length as f32,
            lower_len: hind_lower_length as f32,
            target_idx: 2,
            is_front: false,
        },
        LimbDef {
            root_local_fwd: hip_fwd,
            root_local_side: body_hw,
            upper_len: hind_upper_length as f32,
            lower_len: hind_lower_length as f32,
            target_idx: 3,
            is_front: false,
        },
    ];

    let distal_raise = archetype_distal_raise(archetype);

    for limb in &limbs {
        // Skip hind limbs for biped (biped only uses front limbs as arms)
        // Actually biped has all four: two arms (front) and two legs (hind)

        // Root position on body surface
        let root_pos = local_to_world(
            center,
            forward,
            side,
            up,
            [limb.root_local_fwd, limb.root_local_side, 0.0],
        );

        // 3-bone split: upper, lower_fore (60%), lower_hand (40%)
        let l1 = limb.upper_len;
        let l2 = limb.lower_len * 0.6;
        let l3 = limb.lower_len * 0.4;

        // Target in world space
        let local_target = if auto_foot_placement && stance == "quadruped" {
            // Auto: place feet under shoulder/hip lines at ground level
            [
                limb.root_local_fwd,
                limb.root_local_side * 1.2,
                -lift, // ground level relative to center
            ]
        } else {
            limb_targets[limb.target_idx]
        };

        let mut world_target = local_to_world(center, forward, side, up, local_target);

        // Apply archetype distal offset (raise foot for digitigrade/ungulate)
        world_target = v3_add(world_target, v3_scale(up, distal_raise));

        // Pole hint in world space
        let pole_world = local_to_world(center, forward, side, up, limb_poles[limb.target_idx]);

        // Solve FABRIK
        let chain = fabrik_3bone(root_pos, world_target, pole_world, l1, l2, l3, 10);

        // Fill capsules along each bone segment (tapered)
        let r_root = if limb.is_front { 1.8_f32 } else { 2.0 };
        let r_mid = if limb.is_front { 1.4 } else { 1.5 };
        let r_distal = 1.0_f32;
        let r_end = 0.7_f32;

        fill_segment_capsule(
            file, voxel_map, &mut seen, &mut out, chain[0], chain[1], r_root, r_mid, color,
            material,
        );
        fill_segment_capsule(
            file, voxel_map, &mut seen, &mut out, chain[1], chain[2], r_mid, r_distal, color,
            material,
        );
        fill_segment_capsule(
            file, voxel_map, &mut seen, &mut out, chain[2], chain[3], r_distal, r_end, color,
            material,
        );

        // Place foot at end effector
        fill_foot(
            file, voxel_map, &mut seen, &mut out, chain[3], forward, side, up, archetype, color,
            material,
        );
    }

    Ok(out)
}

/// Preview-only: compute the set of voxel coords a fauna creature would occupy,
/// without mutating the real file. Used for hover preview.
#[allow(clippy::too_many_arguments)]
pub fn preview_fauna_at_screen(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    stance: &str,
    archetype: &str,
    anchor_offset_u: i32,
    anchor_offset_v: i32,
    body_yaw: f32,
    body_arch: f32,
    spine_segments: i32,
    body_length: i32,
    body_half_width: i32,
    body_half_height: i32,
    neck_length: i32,
    neck_half_width: i32,
    neck_half_height: i32,
    head_length: i32,
    head_half_width: i32,
    head_half_height: i32,
    tail_length: i32,
    shoulder_offset_forward: i32,
    hip_offset_forward: i32,
    front_upper_length: i32,
    front_lower_length: i32,
    hind_upper_length: i32,
    hind_lower_length: i32,
    auto_foot_placement: bool,
    color: u32,
    material: MaterialId,
) -> Vec<(VoxelCoord, u32)> {
    let limb_targets: [[f32; 3]; 4] = [
        [20.0, -2.1, -19.0],
        [20.0, 2.1, -19.0],
        [-3.5, -2.2, -20.0],
        [-3.5, 2.2, -20.0],
    ];
    let limb_poles: [[f32; 3]; 4] = [
        [20.0, -2.4, 0.6],
        [20.0, 2.4, 0.6],
        [1.8, -2.8, 1.2],
        [1.8, 2.8, 1.2],
    ];
    let spine_pose_chest = [0.0f32; 3];
    let spine_pose_neck = [0.0f32; 3];
    let spine_pose_head = [0.0f32; 3];

    let mut stub_file = VoxelleFile {
        version: 0,
        grid_size: file.grid_size,
        scene: Scene::default(),
        scene_extra: None,
        mood: None,
        lighting: None,
        voxels: Vec::new(),
        objects: Vec::new(),
        active_object_id: 0,
    };
    let mut stub_map: AHashMap<VoxelCoord, usize> = AHashMap::new();
    match generator_fauna_at_screen(
        &mut stub_file,
        &mut stub_map,
        camera,
        width,
        height,
        sx,
        sy,
        stance,
        archetype,
        anchor_offset_u,
        anchor_offset_v,
        body_yaw,
        body_arch,
        spine_segments,
        body_length,
        body_half_width,
        body_half_height,
        neck_length,
        neck_half_width,
        neck_half_height,
        head_length,
        head_half_width,
        head_half_height,
        tail_length,
        shoulder_offset_forward,
        hip_offset_forward,
        front_upper_length,
        front_lower_length,
        hind_upper_length,
        hind_lower_length,
        &limb_targets,
        &limb_poles,
        spine_pose_chest,
        spine_pose_neck,
        spine_pose_head,
        auto_foot_placement,
        color,
        material,
    ) {
        Ok(deltas) => deltas
            .into_iter()
            .filter_map(|d| {
                if let VoxelEditDelta::Added(v) = d {
                    if !voxel_map.contains_key(&(v.x, v.y, v.z)) {
                        return Some(((v.x, v.y, v.z), v.color));
                    }
                }
                None
            })
            .collect(),
        Err(_) => Vec::new(),
    }
}
