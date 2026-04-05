//! Presence data: camera sync, avatar positions, peer pings.

use crate::camera::Spherical;
use crate::ViewerState;
use glam::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct CameraPresence {
    pub target: [f32; 3],
    pub radius: f32,
    pub theta: f32,
    pub phi: f32,
    pub perspective: bool,
    pub fov_y: f32,
    pub ortho_half_height: f32,
}

/// World-space eye position (orbit target + spherical offset), matching [`crate::camera::OrbitCamera::smooth_eye`].
pub fn presence_eye(p: &CameraPresence) -> Vec3 {
    let target = Vec3::new(p.target[0], p.target[1], p.target[2]);
    let s = Spherical {
        radius: p.radius,
        theta: p.theta,
        phi: p.phi,
    };
    target + s.to_offset()
}

/// Ephemeral world highlight when a peer pings a voxel cell (`collab-ping`).
#[derive(Clone)]
pub struct PingFlash {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub color_rgb: u32,
    pub until: std::time::Instant,
    pub started: std::time::Instant,
    pub display_name: String,
    /// Optional emoji reaction shown above the ping (e.g. "👍").
    pub emoji: String,
}

pub fn record_ping_flash_colored(
    state: &ViewerState,
    x: i32,
    y: i32,
    z: i32,
    color_rgb: u32,
    display_name: String,
    emoji: String,
) {
    let now = std::time::Instant::now();
    {
        let mut g = state.ping_flash.lock();
        *g = Some(PingFlash {
            x,
            y,
            z,
            color_rgb,
            until: now + std::time::Duration::from_secs_f32(7.0),
            started: now,
            display_name,
            emoji,
        });
    }
}

/// Resolves accent color and display name from the roster. Do **not** call while holding [`ViewerState::collab`].
pub fn record_ping_flash(state: &ViewerState, peer_id: u32, x: i32, y: i32, z: i32, emoji: String) {
    let (color_rgb, display_name) = {
        let c = state.collab.lock();
        c.roster.iter().find(|r| r.peer_id == peer_id).map(|r| {
            (
                r.color_rgb,
                if r.display_name.is_empty() {
                    "Guest".to_string()
                } else {
                    r.display_name.clone()
                },
            )
        })
    }
    .unwrap_or((0xffff44, "Guest".to_string()));
    record_ping_flash_colored(state, x, y, z, color_rgb, display_name, emoji);
}
