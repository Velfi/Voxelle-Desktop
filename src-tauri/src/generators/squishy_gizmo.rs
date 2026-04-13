//! Squishy edit gizmo: screen-space handle pick + plane drag.
//! Hit-testing matches the shared sync_gizmo_gpu layout (move arrows + scale ring).

use crate::camera::OrbitCamera;
use crate::generators::common::axis_drag_scalar_from_ray;
use crate::generators::squishy_session::{Metaball, SquishyMode, SquishySession};
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
    pub axis_origin: Vec3,
    pub axis_dir: Vec3,
    pub start_axis_scalar: f32,
    pub plane_n: Vec3,
    pub plane_p: Vec3,
    pub start_hit: Vec3,
}

const PICK_PX: f32 = 24.0;

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
    let dist = (eye - center).length().max(1.0);
    // Arm length matches sync_gizmo_gpu exactly.
    let arm = (dist * 0.13_f32).clamp(1.5, 20.0);

    // Test move arrows — both directions of each axis, same handle per axis.
    let mut best: Option<(SquishyGizmoHandle, f32)> = None;
    let axes = [
        (SquishyGizmoHandle::MoveX, Vec3::X),
        (SquishyGizmoHandle::MoveY, Vec3::Y),
        (SquishyGizmoHandle::MoveZ, Vec3::Z),
    ];
    for (kind, axis) in axes {
        for &dir in &[axis, -axis] {
            let d = screen_dist_to_segment(camera, w, h, sx, sy, center, center + dir * arm);
            if best.map(|(_, bd)| d < bd).unwrap_or(true) {
                best = Some((kind, d));
            }
        }
    }

    // Test scale ring — camera-facing circle at ball.radius (matches sync_gizmo_gpu ring).
    let inv_view = camera.view_matrix().inverse();
    let cam_right = inv_view.x_axis.truncate().normalize();
    let cam_up = inv_view.y_axis.truncate().normalize();
    let r = ball.radius.max(dist * 0.015);
    const RING_N: usize = 32;
    let mut ring_d = f32::MAX;
    for i in 0..RING_N {
        let a0 = i as f32 * 2.0 * std::f32::consts::PI / RING_N as f32;
        let a1 = (i + 1) as f32 * 2.0 * std::f32::consts::PI / RING_N as f32;
        let p0 = center + (cam_right * a0.cos() + cam_up * a0.sin()) * r;
        let p1 = center + (cam_right * a1.cos() + cam_up * a1.sin()) * r;
        ring_d = ring_d.min(screen_dist_to_segment(camera, w, h, sx, sy, p0, p1));
    }
    if best.map(|(_, bd)| ring_d < bd).unwrap_or(true) {
        best = Some((SquishyGizmoHandle::Scale, ring_d));
    }

    let (kind, d) = best?;
    if d <= PICK_PX {
        Some(kind)
    } else {
        None
    }
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
    let start_hit = crate::generators::common::ray_plane_intersect(ro, rd, plane_n, plane_p)?;
    let axis_dir = match handle {
        SquishyGizmoHandle::MoveX => Vec3::X,
        SquishyGizmoHandle::MoveY => Vec3::Y,
        SquishyGizmoHandle::MoveZ => Vec3::Z,
        SquishyGizmoHandle::Scale => Vec3::ZERO,
    };
    let start_axis_scalar = if axis_dir.length_squared() > 0.0 {
        axis_drag_scalar_from_ray(center, axis_dir, ro, rd, plane_n, plane_p).unwrap_or(0.0)
    } else {
        0.0
    };

    Some(SquishyGizmoDrag {
        handle,
        ball_id: sel_id,
        start: ball,
        axis_origin: center,
        axis_dir,
        start_axis_scalar,
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
    let start = &drag.start;
    let inv_view = camera.view_matrix().inverse();
    let camera_right = inv_view.x_axis.truncate().normalize();

    match drag.handle {
        SquishyGizmoHandle::MoveX => {
            let Some(axis_scalar) = axis_drag_scalar_from_ray(
                drag.axis_origin,
                drag.axis_dir,
                ro,
                rd,
                drag.plane_n,
                drag.plane_p,
            ) else {
                return;
            };
            let nx = (start.x as f32 + (axis_scalar - drag.start_axis_scalar)).round() as i32;
            session.set_ball_transform(drag.ball_id, nx, start.y, start.z, start.radius);
        }
        SquishyGizmoHandle::MoveY => {
            let Some(axis_scalar) = axis_drag_scalar_from_ray(
                drag.axis_origin,
                drag.axis_dir,
                ro,
                rd,
                drag.plane_n,
                drag.plane_p,
            ) else {
                return;
            };
            let ny = (start.y as f32 + (axis_scalar - drag.start_axis_scalar)).round() as i32;
            session.set_ball_transform(drag.ball_id, start.x, ny, start.z, start.radius);
        }
        SquishyGizmoHandle::MoveZ => {
            let Some(axis_scalar) = axis_drag_scalar_from_ray(
                drag.axis_origin,
                drag.axis_dir,
                ro,
                rd,
                drag.plane_n,
                drag.plane_p,
            ) else {
                return;
            };
            let nz = (start.z as f32 + (axis_scalar - drag.start_axis_scalar)).round() as i32;
            session.set_ball_transform(drag.ball_id, start.x, start.y, nz, start.radius);
        }
        SquishyGizmoHandle::Scale => {
            let Some(hit) =
                crate::generators::common::ray_plane_intersect(ro, rd, drag.plane_n, drag.plane_p)
            else {
                return;
            };
            let delta = hit - drag.start_hit;
            // Drag right = grow, drag left = shrink.
            let signed = delta.dot(camera_right);
            let nr = (start.radius + signed).clamp(0.5, 64.0);
            session.set_ball_transform(drag.ball_id, start.x, start.y, start.z, nr);
        }
    }
}
