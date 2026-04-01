//! Squishy edit gizmo: screen-space handle pick + plane drag (web `squishyGizmo.ts` parity).

use crate::camera::OrbitCamera;
use crate::generators::squishy_session::{Metaball, SquishyMode, SquishySession};
use crate::greedy_mesh::{self, MeshBuffers};
use crate::voxel_edit::{screen_to_world_ray, world_to_viewport_pixels};
use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SquishyGizmoHandle {
    MoveX,
    MoveY,
    MoveZ,
    Scale,
}

#[derive(Clone)]
pub struct SquishyGizmoDrag {
    pub handle: SquishyGizmoHandle,
    pub ball_id: u32,
    pub start: Metaball,
    pub plane_n: Vec3,
    pub plane_p: Vec3,
    pub start_hit: Vec3,
}

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

pub fn pick_squishy_gizmo_handle(
    session: &SquishySession,
    camera: &OrbitCamera,
    w: f32,
    h: f32,
    sx: f32,
    sy: f32,
) -> Option<SquishyGizmoHandle> {
    if session.mode != SquishyMode::Edit {
        return None;
    }
    let sel_id = session.selected_id?;
    let ball = session.balls.iter().find(|b| b.id == sel_id)?;
    let center = Vec3::new(
        ball.x as f32 + 0.5,
        ball.y as f32 + 0.5,
        ball.z as f32 + 0.5,
    );
    let eye = camera.smooth_eye();
    let (arm_base, arrow_world_len) = gizmo_layout(center, ball.radius, eye);

    let mut best: Option<(SquishyGizmoHandle, f32)> = None;
    let axes = [
        (SquishyGizmoHandle::MoveX, Vec3::X),
        (SquishyGizmoHandle::MoveY, Vec3::Y),
        (SquishyGizmoHandle::MoveZ, Vec3::Z),
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
    let radius_offset = (ball.radius + 0.9).max(1.2);
    let sp = radius_offset * 0.82;
    let scale_center = center + Vec3::new(sp, sp, sp);
    let scale_len = arrow_world_len * 0.45;
    let diag = Vec3::new(1.0, 1.0, 1.0).normalize();
    let sa = scale_center - diag * (scale_len * 0.5);
    let sb = scale_center + diag * (scale_len * 0.5);
    let sd = screen_dist_to_segment(camera, w, h, sx, sy, sa, sb);
    let replace = best.map(|(_, bd)| sd < bd).unwrap_or(true);
    if replace {
        best = Some((SquishyGizmoHandle::Scale, sd));
    }

    let (kind, d) = best?;
    if d <= PICK_PX {
        Some(kind)
    } else {
        None
    }
}

fn ray_plane_intersect(ro: Vec3, rd: Vec3, plane_n: Vec3, plane_p: Vec3) -> Option<Vec3> {
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

pub fn squishy_gizmo_begin_drag(
    session: &SquishySession,
    camera: &OrbitCamera,
    w: f32,
    h: f32,
    sx: f32,
    sy: f32,
    handle: SquishyGizmoHandle,
) -> Option<SquishyGizmoDrag> {
    let sel_id = session.selected_id?;
    let ball = session.balls.iter().find(|b| b.id == sel_id)?.clone();
    let center = Vec3::new(
        ball.x as f32 + 0.5,
        ball.y as f32 + 0.5,
        ball.z as f32 + 0.5,
    );
    let (ro, rd) = screen_to_world_ray(camera, w, h, sx, sy);
    let ro = Vec3::new(ro.x, ro.y, ro.z);
    let rd = Vec3::new(rd.x, rd.y, rd.z).normalize();
    let eye = camera.smooth_eye();
    let plane_n = (center - eye).normalize();
    let plane_p = center;
    let start_hit = ray_plane_intersect(ro, rd, plane_n, plane_p)?;

    Some(SquishyGizmoDrag {
        handle,
        ball_id: sel_id,
        start: ball,
        plane_n,
        plane_p,
        start_hit,
    })
}

pub fn squishy_gizmo_apply_drag(
    session: &mut SquishySession,
    camera: &OrbitCamera,
    w: f32,
    h: f32,
    sx: f32,
    sy: f32,
    drag: &SquishyGizmoDrag,
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
        SquishyGizmoHandle::MoveX => {
            let nx = (start.x as f32 + delta.x).round() as i32;
            session.set_ball_transform(drag.ball_id, nx, start.y, start.z, start.radius);
        }
        SquishyGizmoHandle::MoveY => {
            let ny = (start.y as f32 + delta.y).round() as i32;
            session.set_ball_transform(drag.ball_id, start.x, ny, start.z, start.radius);
        }
        SquishyGizmoHandle::MoveZ => {
            let nz = (start.z as f32 + delta.z).round() as i32;
            session.set_ball_transform(drag.ball_id, start.x, start.y, nz, start.radius);
        }
        SquishyGizmoHandle::Scale => {
            let signed = delta.dot(camera_right);
            let nr = (start.radius + signed).clamp(0.5, 64.0);
            session.set_ball_transform(drag.ball_id, start.x, start.y, start.z, nr);
        }
    }
}

/// Colored wire stubs along axes + diagonal scale (matches web gizmo readability).
pub fn append_squishy_gizmo_wire(
    session: &SquishySession,
    camera: &OrbitCamera,
    out: &mut MeshBuffers,
) {
    if session.mode != SquishyMode::Edit {
        return;
    }
    let Some(sel_id) = session.selected_id else {
        return;
    };
    let Some(ball) = session.balls.iter().find(|b| b.id == sel_id) else {
        return;
    };
    let center = Vec3::new(
        ball.x as f32 + 0.5,
        ball.y as f32 + 0.5,
        ball.z as f32 + 0.5,
    );
    let eye = camera.smooth_eye();
    let (arm_base, arrow_world_len) = gizmo_layout(center, ball.radius, eye);
    let colors = [[1.0, 0.36, 0.4], [0.34, 0.84, 0.43], [0.36, 0.63, 1.0]];
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];
    for i in 0..3 {
        let a = center + axes[i] * arm_base;
        let b = center + axes[i] * (arm_base + arrow_world_len);
        for k in 0..=12 {
            let t = k as f32 / 12.0;
            let p = a.lerp(b, t);
            let wfm = greedy_mesh::preview_cube_wireframe_mesh(p.x, p.y, p.z, 0.04, colors[i], 2.0);
            greedy_mesh::append_mesh_buffers(out, wfm);
        }
    }
    let radius_offset = (ball.radius + 0.9).max(1.2);
    let sp = radius_offset * 0.82;
    let scale_center = center + Vec3::new(sp, sp, sp);
    let scale_len = arrow_world_len * 0.45;
    let diag = Vec3::new(1.0, 1.0, 1.0).normalize();
    let sa = scale_center - diag * (scale_len * 0.5);
    let sb = scale_center + diag * (scale_len * 0.5);
    for k in 0..=10 {
        let t = k as f32 / 10.0;
        let p = sa.lerp(sb, t);
        let wfm =
            greedy_mesh::preview_cube_wireframe_mesh(p.x, p.y, p.z, 0.035, [1.0, 0.82, 0.4], 2.0);
        greedy_mesh::append_mesh_buffers(out, wfm);
    }
}
