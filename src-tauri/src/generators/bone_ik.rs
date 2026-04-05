//! FABRIK inverse-kinematics solver for bone armatures.

use crate::camera::OrbitCamera;
use crate::generators::bone_session::BoneSession;
use crate::voxel_edit::screen_to_world_ray;
use glam::Vec3;

/// Maximum joints in an IK chain.
const MAX_CHAIN_LEN: usize = 16;

// ── IK drag state ────────────────────────────────────────────────────

#[derive(Clone)]
#[allow(dead_code)]
pub struct IkDrag {
    pub effector_joint_id: u32,
    /// Ordered joint IDs from root to effector.
    pub chain: Vec<u32>,
    /// Cached distances between consecutive joints.
    pub bone_lengths: Vec<f32>,
    /// Anchored root position (does not move).
    pub root_pos: Vec3,
    /// Drag plane normal (camera-facing).
    pub plane_n: Vec3,
    /// Drag plane point (effector start).
    pub plane_p: Vec3,
}

// ── Chain discovery ──────────────────────────────────────────────────

/// Walk backward from the effector through connected bones to build the
/// IK chain. At each step, prefer the neighbor with the most downstream
/// connections (longest-chain heuristic). Stops at a leaf joint or after
/// `MAX_CHAIN_LEN` joints.
pub fn find_ik_chain(session: &BoneSession, effector_id: u32) -> Vec<u32> {
    let mut chain = vec![effector_id];
    let mut visited = std::collections::HashSet::new();
    visited.insert(effector_id);

    let mut current = effector_id;
    while chain.len() < MAX_CHAIN_LEN {
        let connected = session.connected_bones(current);
        let mut best: Option<(u32, usize)> = None;
        for &bone_id in &connected {
            if let Some(other) = session.other_joint(bone_id, current) {
                if visited.contains(&other) {
                    continue;
                }
                // Prefer joints with more connections (deeper chain).
                let depth = session.connected_bones(other).len();
                let replace = best.map(|(_, bd)| depth > bd).unwrap_or(true);
                if replace {
                    best = Some((other, depth));
                }
            }
        }
        match best {
            Some((next, _)) => {
                chain.push(next);
                visited.insert(next);
                current = next;
            }
            None => break,
        }
    }

    // Reverse so chain goes root → effector.
    chain.reverse();
    chain
}

// ── FABRIK solver ────────────────────────────────────────────────────

/// Run the FABRIK algorithm on a set of positions with fixed bone lengths.
/// `positions[0]` is the root (pinned), `positions[last]` is the effector
/// (pulled toward `target`).
fn fabrik_solve(positions: &mut [Vec3], lengths: &[f32], target: Vec3, iterations: u32) {
    let n = positions.len();
    if n < 2 {
        return;
    }
    assert_eq!(lengths.len(), n - 1);

    let root = positions[0];

    for _ in 0..iterations {
        // Forward pass: move effector to target, walk back to root.
        positions[n - 1] = target;
        for i in (0..n - 1).rev() {
            let dir = positions[i] - positions[i + 1];
            let len = dir.length();
            if len < 1e-9 {
                // Degenerate: push along arbitrary direction.
                positions[i] = positions[i + 1] + Vec3::Y * lengths[i];
            } else {
                positions[i] = positions[i + 1] + dir / len * lengths[i];
            }
        }

        // Backward pass: pin root, walk forward to effector.
        positions[0] = root;
        for i in 0..n - 1 {
            let dir = positions[i + 1] - positions[i];
            let len = dir.length();
            if len < 1e-9 {
                positions[i + 1] = positions[i] + Vec3::Y * lengths[i];
            } else {
                positions[i + 1] = positions[i] + dir / len * lengths[i];
            }
        }
    }
}

// ── Public API ───────────────────────────────────────────────────────

/// Begin an IK drag on the given effector joint.
pub fn ik_drag_begin(
    session: &BoneSession,
    camera: &OrbitCamera,
    w: f32,
    h: f32,
    sx: f32,
    sy: f32,
    effector_id: u32,
) -> Option<IkDrag> {
    let chain = find_ik_chain(session, effector_id);
    if chain.len() < 2 {
        return None;
    }

    // Cache bone lengths.
    let mut bone_lengths = Vec::with_capacity(chain.len() - 1);
    for i in 0..chain.len() - 1 {
        let a = session.find_joint(chain[i])?.pos();
        let b = session.find_joint(chain[i + 1])?.pos();
        bone_lengths.push(a.distance(b));
    }

    let root_pos = session.find_joint(chain[0])?.pos();
    let effector_pos = session.find_joint(effector_id)?.pos();

    let eye = camera.smooth_eye();
    let plane_n = (effector_pos - eye).normalize();
    let plane_p = effector_pos;

    // Verify we can hit the drag plane from the initial cursor.
    let (ro, rd) = screen_to_world_ray(camera, w, h, sx, sy);
    let ro = Vec3::new(ro.x, ro.y, ro.z);
    let rd = Vec3::new(rd.x, rd.y, rd.z).normalize();
    ray_plane_intersect(ro, rd, plane_n, plane_p)?;

    Some(IkDrag {
        effector_joint_id: effector_id,
        chain,
        bone_lengths,
        root_pos,
        plane_n,
        plane_p,
    })
}

/// Update the IK chain during a drag.
pub fn ik_drag_update(
    session: &mut BoneSession,
    camera: &OrbitCamera,
    w: f32,
    h: f32,
    sx: f32,
    sy: f32,
    drag: &IkDrag,
) {
    let (ro, rd) = screen_to_world_ray(camera, w, h, sx, sy);
    let ro = Vec3::new(ro.x, ro.y, ro.z);
    let rd = Vec3::new(rd.x, rd.y, rd.z).normalize();
    let Some(target) = ray_plane_intersect(ro, rd, drag.plane_n, drag.plane_p) else {
        return;
    };

    // Collect current positions.
    let mut positions: Vec<Vec3> = drag
        .chain
        .iter()
        .filter_map(|&id| session.find_joint(id).map(|j| j.pos()))
        .collect();
    if positions.len() != drag.chain.len() {
        return;
    }

    // Solve.
    fabrik_solve(&mut positions, &drag.bone_lengths, target, 10);

    // Write back (skip root which is pinned).
    for (i, p) in positions.iter().enumerate().skip(1) {
        let p = *p;
        session.set_joint_position(drag.chain[i], p.x, p.y, p.z);
    }
}

use super::bone_session::ray_plane_intersect;
