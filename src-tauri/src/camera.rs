//! Orbit camera with damping (Three.js OrbitControls–like).

use glam::{Mat4, Vec3};

const DAMPING: f32 = 0.05;
/// Same defaults as Three.js `OrbitControls` (`rotateSpeed`, `panSpeed`, `zoomSpeed`).
const ROTATE_SPEED: f32 = 1.0;
const PAN_SPEED: f32 = 1.0;
const ZOOM_SPEED: f32 = 1.0;

#[derive(Clone, Copy, Debug)]
pub struct Spherical {
    pub radius: f32,
    /// Azimuth around Y, from +Z toward +X.
    pub theta: f32,
    /// Polar angle from +Y axis.
    pub phi: f32,
}

impl Spherical {
    /// Matches Three.js `Spherical` (Y-up): `phi` from +Y, `theta` in XZ from +Z toward +X.
    pub fn from_offset(offset: Vec3) -> Self {
        let radius = offset.length().max(1e-4);
        let phi = (offset.y / radius).clamp(-1.0, 1.0).acos();
        let theta = offset.x.atan2(offset.z);
        Self { radius, theta, phi }
    }

    pub fn to_offset(self) -> Vec3 {
        let sin_p = self.phi.sin() * self.radius;
        Vec3::new(
            sin_p * self.theta.sin(),
            self.phi.cos() * self.radius,
            sin_p * self.theta.cos(),
        )
    }
}

pub struct OrbitCamera {
    pub target: Vec3,
    pub spherical: Spherical,
    /// Smoothed spherical + target (damping).
    pub smooth_target: Vec3,
    pub smooth_spherical: Spherical,
    pub min_radius: f32,
    pub max_radius: f32,
    pub min_polar: f32,
    pub max_polar: f32,
    pub perspective: bool,
    /// Vertical FOV radians (perspective).
    pub fov_y: f32,
    pub ortho_half_height: f32,
    pub near: f32,
    pub far: f32,
}

impl OrbitCamera {
    pub fn new() -> Self {
        let s = Spherical {
            radius: 10.0,
            theta: 0.0,
            phi: std::f32::consts::FRAC_PI_4,
        };
        Self {
            target: Vec3::ZERO,
            spherical: s,
            smooth_target: Vec3::ZERO,
            smooth_spherical: s,
            min_radius: 0.5,
            max_radius: 1e6,
            min_polar: 0.01,
            max_polar: std::f32::consts::PI - 0.01,
            perspective: true,
            fov_y: std::f32::consts::FRAC_PI_4,
            ortho_half_height: 10.0,
            near: 0.05,
            far: 5000.0,
        }
    }

    pub fn eye(&self) -> Vec3 {
        self.target + self.smooth_spherical.to_offset()
    }

    pub fn smooth_eye(&self) -> Vec3 {
        self.smooth_target + self.smooth_spherical.to_offset()
    }

    /// Call each frame.
    pub fn update_damping(&mut self) {
        self.smooth_target = self.smooth_target.lerp(self.target, DAMPING);
        let sr = self.smooth_spherical.radius
            + (self.spherical.radius - self.smooth_spherical.radius) * DAMPING;
        let st = self.smooth_spherical.theta
            + (self.spherical.theta - self.smooth_spherical.theta) * DAMPING;
        let sp =
            self.smooth_spherical.phi + (self.spherical.phi - self.smooth_spherical.phi) * DAMPING;
        self.smooth_spherical = Spherical {
            radius: sr.clamp(self.min_radius, self.max_radius),
            theta: st,
            phi: sp.clamp(self.min_polar, self.max_polar),
        };
    }

    pub fn needs_redraw(&self) -> bool {
        let d_target = (self.target - self.smooth_target).length();
        let d_r = (self.spherical.radius - self.smooth_spherical.radius).abs();
        let d_t = (self.spherical.theta - self.smooth_spherical.theta).abs();
        let d_p = (self.spherical.phi - self.smooth_spherical.phi).abs();
        d_target > 1e-4 || d_r > 1e-4 || d_t > 1e-5 || d_p > 1e-5
    }

    /// Mouse orbit: `_handleMouseMoveRotate` uses `2π * delta / clientHeight` per axis (× `rotateSpeed`).
    pub fn rotate_screen(&mut self, dx: f32, dy: f32, viewport_height_px: f32) {
        let h = viewport_height_px.max(1.0);
        let k = std::f32::consts::TAU * ROTATE_SPEED / h;
        self.spherical.theta -= dx * k;
        self.spherical.phi -= dy * k;
        self.spherical.phi = self.spherical.phi.clamp(self.min_polar, self.max_polar);
    }

    /// Pan in pixels, matching Three.js `OrbitControls` (`screenSpacePanning = true`): uses
    /// `2 * delta * R * tan(fov/2) / clientHeight` for perspective and frustum extents / client size for ortho.
    pub fn pan_screen(
        &mut self,
        dx: f32,
        dy: f32,
        viewport_width_px: f32,
        viewport_height_px: f32,
    ) {
        let dx = dx * PAN_SPEED;
        let dy = dy * PAN_SPEED;
        let h_px = viewport_height_px.max(1.0);
        let w_px = viewport_width_px.max(1.0);
        let aspect = w_px / h_px;

        let eye = self.target + self.spherical.to_offset();
        let view = Mat4::look_at_rh(eye, self.target, Vec3::Y);
        let world_from_view = view.inverse();
        let right = world_from_view.x_axis.truncate().normalize();
        let up = world_from_view.y_axis.truncate().normalize();

        if self.perspective {
            // OrbitControls._pan (perspective): scale uses clientHeight only for both axes.
            let r = self.spherical.radius.max(1e-4);
            let td = r * (self.fov_y * 0.5).tan();
            let s = 2.0 * td / h_px;
            // _panLeft: -s*dx*right; _panUp: +s*dy*up (screen space)
            self.target += -right * (s * dx) + up * (s * dy);
        } else {
            // OrbitControls._pan (orthographic): (right-left)/zoom/width, (top-bottom)/zoom/height
            let hh = self.ortho_half_height;
            let ww = hh * aspect;
            let zoom = 1.0_f32;
            let pan_x = dx * (2.0 * ww) / zoom / w_px;
            let pan_y = dy * (2.0 * hh) / zoom / h_px;
            self.target += -right * pan_x + up * pan_y;
        }
    }

    /// `_getZoomScale` in Three.js `OrbitControls`.
    #[inline]
    fn zoom_scale(delta: f32) -> f32 {
        let normalized = (delta * 0.01).abs();
        0.95_f32.powf(ZOOM_SPEED * normalized)
    }

    /// Wheel and middle-button dolly: `_handleMouseWheel` / `_handleMouseMoveDolly` (same `delta * 0.01` scaling).
    pub fn dolly_delta(&mut self, delta_y: f32) {
        if delta_y == 0.0 {
            return;
        }
        let zs = Self::zoom_scale(delta_y);
        let r = if delta_y < 0.0 {
            // _dollyIn → radius *= scale
            self.spherical.radius * zs
        } else {
            // _dollyOut → radius /= scale
            self.spherical.radius / zs
        };
        self.spherical.radius = r.clamp(self.min_radius, self.max_radius);
    }

    /// First-person style move on XZ + world Y (fly mode); `forward`/`right` are -1..1.
    pub fn fly_move(&mut self, forward: f32, right: f32, up: f32, dt: f32, speed: f32) {
        let dt = dt.max(0.0);
        if dt == 0.0 {
            return;
        }
        let eye = self.target + self.spherical.to_offset();
        let mut flat = self.target - eye;
        flat.y = 0.0;
        if flat.length_squared() < 1e-8 {
            flat = Vec3::new(0.0, 0.0, -1.0);
        } else {
            flat = flat.normalize();
        }
        let world_right = flat.cross(Vec3::Y).normalize();
        let delta = (flat * forward + world_right * right + Vec3::Y * up) * (speed * dt);
        self.target += delta;
    }

    pub fn fit_sphere(&mut self, center: Vec3, radius: f32, width: f32, height: f32) {
        self.target = center;
        let aspect = (width / height.max(1.0)).max(1e-4);
        let fov = self.fov_y;
        let dist_persp = radius / (0.5 * fov).sin().max(0.01);
        let dist_ortho = radius * 2.0;
        let dist = if self.perspective {
            dist_persp * 1.2
        } else {
            dist_ortho * 1.2
        };
        self.spherical = Spherical {
            radius: dist.max(self.min_radius),
            theta: std::f32::consts::FRAC_PI_4,
            phi: std::f32::consts::FRAC_PI_3,
        };
        self.smooth_target = self.target;
        self.smooth_spherical = self.spherical;
        let _ = aspect; // could tweak theta from aspect
    }

    pub fn view_matrix(&self) -> Mat4 {
        let eye = self.smooth_target + self.smooth_spherical.to_offset();
        Mat4::look_at_rh(eye, self.smooth_target, Vec3::Y)
    }

    pub fn proj_matrix(&self, width: f32, height: f32) -> Mat4 {
        let aspect = (width / height.max(1.0)).max(1e-4);
        if self.perspective {
            Mat4::perspective_rh(self.fov_y, aspect, self.near, self.far)
        } else {
            let hh = self.ortho_half_height;
            let ww = hh * aspect;
            Mat4::orthographic_rh(-ww, ww, -hh, hh, self.near, self.far)
        }
    }
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self::new()
    }
}
