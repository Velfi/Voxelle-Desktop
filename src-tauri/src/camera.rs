//! Orbit camera with damping (Three.js OrbitControls–like).

use glam::{Mat4, Quat, Vec3};

const DAMPING: f32 = 0.05;
/// Logo splash: max deviation from rest for drag + hover (±75°).
fn logo_splash_orbit_half_span_rad() -> f32 {
    75.0_f32.to_radians()
}
/// Subtle cursor parallax on cold-start logo (radians at viewport edges).
const LOGO_SPLASH_HOVER_MAX_RAD: f32 = 0.038;
/// Same defaults as Three.js `OrbitControls` (`rotateSpeed`, `panSpeed`, `zoomSpeed`).
const ROTATE_SPEED: f32 = 1.0;
/// Scales [`Self::fly_look_rotate_screen`] only (orbit drag uses unscaled `TAU/h`).
const FLY_LOOK_SENSITIVITY: f32 = 1.0;
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

#[derive(Clone)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub spherical: Spherical,
    /// Smoothed spherical + target (damping).
    pub smooth_target: Vec3,
    pub smooth_spherical: Spherical,
    /// Cold-start logo splash: rest pose after [`Self::configure_logo_splash_after_fit`]; orbit is clamped and snaps back on release.
    pub logo_splash_rest: Option<Spherical>,
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
            logo_splash_rest: None,
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

    /// After [`Self::fit_sphere`], apply splash offsets (theta −1 rad, phi +0 vs previous phi −1) and record rest for clamped orbit + release reset.
    pub fn configure_logo_splash_after_fit(&mut self) {
        const TH_OFF: f32 = -1.0;
        const PH_OFF: f32 = 0.0;
        self.spherical.theta += TH_OFF;
        self.spherical.phi += PH_OFF;
        self.spherical.phi = self.spherical.phi.clamp(self.min_polar, self.max_polar);
        self.smooth_target = self.target;
        self.smooth_spherical = self.spherical;
        self.logo_splash_rest = Some(self.spherical);
    }

    /// Logo splash drag: same as [`Self::rotate_screen`], then clamp ±75° from rest on theta and phi.
    pub fn rotate_screen_logo_splash(&mut self, dx: f32, dy: f32, viewport_height_px: f32) {
        let Some(rest) = self.logo_splash_rest else {
            self.rotate_screen(dx, dy, viewport_height_px);
            return;
        };
        let h = viewport_height_px.max(1.0);
        let k = std::f32::consts::TAU * ROTATE_SPEED / h;
        self.spherical.theta -= dx * k;
        self.spherical.phi -= dy * k;
        let half_span = logo_splash_orbit_half_span_rad();
        self.spherical.theta = Self::clamp_theta_near_rest(self.spherical.theta, rest.theta, half_span);
        self.spherical.phi = (self.spherical.phi)
            .clamp(rest.phi - half_span, rest.phi + half_span)
            .clamp(self.min_polar, self.max_polar);
    }

    /// No-button move: nudge orbit slightly from [`Self::logo_splash_rest`] from normalized viewport position.
    pub fn set_logo_splash_hover_from_viewport_px(
        &mut self,
        x_px: f32,
        y_px: f32,
        viewport_w: f32,
        viewport_h: f32,
    ) {
        let Some(rest) = self.logo_splash_rest else {
            return;
        };
        let vw = viewport_w.max(1.0);
        let vh = viewport_h.max(1.0);
        let nx = ((x_px / vw) - 0.5) * 2.0;
        let ny = -(((y_px / vh) - 0.5) * 2.0);
        let nx = nx.clamp(-1.0, 1.0);
        let ny = ny.clamp(-1.0, 1.0);

        let half_span = logo_splash_orbit_half_span_rad();
        let theta_t = rest.theta + nx * LOGO_SPLASH_HOVER_MAX_RAD;
        let phi_t = rest.phi + ny * LOGO_SPLASH_HOVER_MAX_RAD;
        self.spherical.theta = Self::clamp_theta_near_rest(theta_t, rest.theta, half_span);
        self.spherical.phi = phi_t
            .clamp(rest.phi - half_span, rest.phi + half_span)
            .clamp(self.min_polar, self.max_polar);
        self.smooth_spherical.theta = self.spherical.theta;
        self.smooth_spherical.phi = self.spherical.phi;
    }

    fn clamp_theta_near_rest(theta: f32, rest: f32, half_span: f32) -> f32 {
        const PI: f32 = std::f32::consts::PI;
        const TAU: f32 = std::f32::consts::TAU;
        let mut d = theta - rest;
        while d > PI {
            d -= TAU;
        }
        while d < -PI {
            d += TAU;
        }
        rest + d.clamp(-half_span, half_span)
    }

    /// Pointer released: damp back to [`Self::logo_splash_rest`].
    pub fn reset_logo_splash_orbit(&mut self) {
        if let Some(rest) = self.logo_splash_rest {
            self.spherical = rest;
        }
    }

    /// FPS-style mouse look: yaw around world +Y, then pitch around camera right — pivot at **eye**,
    /// so you turn where you look instead of orbiting a fixed point in space (`rotate_screen`).
    pub fn fly_look_rotate_screen(&mut self, dx: f32, dy: f32, viewport_height_px: f32) {
        let h = viewport_height_px.max(1.0);
        let k = std::f32::consts::TAU * ROTATE_SPEED * FLY_LOOK_SENSITIVITY / h;
        // Horizontal matches orbit (`theta -= dx*k` → same sign as delta yaw here).
        let yaw = -dx * k;
        // Vertical: spherical `phi -= dy*k` vs pitch about camera-right; opposite sign so mouse-down looks down like orbit.
        let pitch = dy * k;

        let eye = self.target + self.spherical.to_offset();
        let r = self.spherical.radius.max(1e-4);

        let mut forward = self.target - eye;
        if forward.length_squared() < 1e-12 {
            forward = Vec3::new(0.0, 0.0, -1.0);
        } else {
            forward = forward.normalize();
        }

        let yaw_q = Quat::from_axis_angle(Vec3::Y, yaw);
        forward = (yaw_q * forward).normalize();

        let mut pitch_axis = Vec3::Y.cross(forward);
        if pitch_axis.length_squared() < 1e-12 {
            pitch_axis = Vec3::X;
        } else {
            pitch_axis = pitch_axis.normalize();
        }
        let pitch_q = Quat::from_axis_angle(pitch_axis, pitch);
        forward = (pitch_q * forward).normalize();

        self.target = eye + forward * r;
        let mut s = Spherical::from_offset(eye - self.target);
        s.phi = s.phi.clamp(self.min_polar, self.max_polar);
        self.spherical = s;
        self.target = eye - self.spherical.to_offset();
        self.smooth_target = self.target;
        self.smooth_spherical = self.spherical;
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

    /// Fly mode: move along the camera view axis (W/S) and strafe (A/D) relative to look; E/Q use world +Y.
    /// Matches web `applyFlyMovement`: forward is full look direction from eye toward `target`, strafe is
    /// `look × world_up` (no roll), vertical keys move in world Y.
    pub fn fly_move(&mut self, forward: f32, right: f32, up: f32, dt: f32, speed: f32) {
        let dt = dt.max(0.0);
        if dt == 0.0 {
            return;
        }
        let eye = self.target + self.spherical.to_offset();
        let mut look = self.target - eye;
        if look.length_squared() < 1e-12 {
            look = Vec3::new(0.0, 0.0, -1.0);
        } else {
            look = look.normalize();
        }
        // Strafe right (camera local +X with world up, roll = 0): forward × up → right-handed
        let mut strafe_right = look.cross(Vec3::Y);
        if strafe_right.length_squared() < 1e-12 {
            strafe_right = Vec3::X;
        } else {
            strafe_right = strafe_right.normalize();
        }
        let delta =
            (look * forward + strafe_right * right + Vec3::Y * up) * (speed * dt);
        self.target += delta;
    }

    pub fn fit_sphere(&mut self, center: Vec3, radius: f32, width: f32, height: f32) {
        self.target = center;
        let aspect = (width / height.max(1.0)).max(1e-4);
        let fov = self.fov_y;
        let dist_persp = radius / (0.5 * fov).sin().max(0.01);
        let dist_ortho = radius * 2.0;
        // Closer than tight fit (was 1.2×, then 1.05×); small margin vs clipping at viewport edges.
        let dist = if self.perspective {
            dist_persp * 0.92
        } else {
            dist_ortho * 0.92
        };
        self.spherical = Spherical {
            radius: dist.max(self.min_radius),
            // Default orbit: π/4 from +Z toward +X, then 90° right around +Y (world vertical).
            theta: std::f32::consts::FRAC_PI_4 + std::f32::consts::FRAC_PI_2,
            // π/3 + a bit: lower eye toward the horizon so framing reads more head-on than top-down.
            phi: std::f32::consts::FRAC_PI_3 + 0.22_f32,
        };
        self.smooth_target = self.target;
        self.smooth_spherical = self.spherical;
        let _ = aspect; // could tweak theta from aspect
    }

    /// Web `VoxelCanvas` fit-to-view: keep current view orientation, frame AABB in the viewport.
    pub fn fit_to_aabb_preserving_view(&mut self, min: Vec3, max: Vec3, width: f32, height: f32) {
        let h = height.max(1.0);
        let w = width.max(1.0);
        let aspect = (w / h).max(1e-4);
        let center = (min + max) * 0.5;
        let corners = [
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(min.x, max.y, min.z),
            Vec3::new(min.x, max.y, max.z),
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(max.x, min.y, max.z),
            Vec3::new(max.x, max.y, min.z),
            Vec3::new(max.x, max.y, max.z),
        ];

        let eye = self.target + self.spherical.to_offset();
        let view = Mat4::look_at_rh(eye, self.target, Vec3::Y);
        let inv = view.inverse();
        let right = inv.x_axis.truncate().normalize();
        let up = inv.y_axis.truncate().normalize();
        let forward_scene = (self.target - eye).normalize();

        const FIT_PADDING: f32 = 1.08;

        if self.perspective {
            let v_fov_tan = (self.fov_y * 0.5).tan();
            let h_fov_tan = v_fov_tan * aspect;
            let mut fit_dist = self.min_radius;
            for c in corners {
                let rel = c - center;
                let x = rel.dot(right).abs();
                let y = rel.dot(up).abs();
                let depth = rel.dot(forward_scene);
                fit_dist = fit_dist.max((x * FIT_PADDING) / h_fov_tan.max(1e-6) - depth);
                fit_dist = fit_dist.max((y * FIT_PADDING) / v_fov_tan.max(1e-6) - depth);
            }
            fit_dist = fit_dist.clamp(self.min_radius, self.max_radius);
            self.target = center;
            let mut dir = eye - center;
            if dir.length_squared() < 1e-8 {
                dir = Vec3::new(0.6, 0.8, 1.0);
            }
            let dir = dir.normalize();
            self.spherical = Spherical::from_offset(dir * fit_dist);
            self.smooth_target = self.target;
            self.smooth_spherical = self.spherical;
        } else {
            let mut max_x = 0.0_f32;
            let mut max_y = 0.0_f32;
            for c in corners {
                let rel = c - center;
                max_x = max_x.max(rel.dot(right).abs());
                max_y = max_y.max(rel.dot(up).abs());
            }
            let hh = (max_y * FIT_PADDING)
                .max((max_x * FIT_PADDING) / aspect)
                .max(0.5);
            self.target = center;
            self.ortho_half_height = hh;
            let old_off = self.spherical.to_offset();
            let dir = old_off.normalize();
            let dist = old_off.length().max(self.min_radius);
            self.spherical = Spherical::from_offset(dir * dist);
            self.smooth_target = self.target;
            self.smooth_spherical = self.spherical;
        }
    }

    /// Web-style reset: orbit target at content center, camera offset along (0.6, 0.8, 1) scaled.
    pub fn reset_view_to_bounds(&mut self, min: Vec3, max: Vec3, empty_extent: f32) {
        let extent = if (max - min).length() > 1e-3 {
            (max.x - min.x)
                .max(max.y - min.y)
                .max(max.z - min.z)
                + 2.0
        } else {
            empty_extent
        };
        let dist = extent * 2.5;
        let center = if (max - min).length() > 1e-3 {
            (min + max) * 0.5
        } else {
            Vec3::ZERO
        };
        self.target = center;
        let offset = Vec3::new(dist * 0.6, dist * 0.8, dist);
        self.spherical = Spherical::from_offset(offset);
        self.smooth_target = self.target;
        self.smooth_spherical = self.spherical;
    }

    /// Orbit gizmo: drag on the widget (matches web `0.008` rad/pixel).
    pub fn orbit_gizmo_drag(&mut self, dx: f32, dy: f32, theta_only: bool) {
        const S: f32 = 0.008;
        let mut s = self.spherical;
        s.theta -= dx * S;
        if !theta_only {
            s.phi -= dy * S;
        }
        s.phi = s.phi.clamp(self.min_polar, self.max_polar);
        self.spherical = s;
    }

    /// Snap to an axis-aligned view (+X,+Y,+Z,-X,-Y,-Z); preserves orbit radius.
    pub fn snap_to_axis(&mut self, axis_idx: u8) {
        let dirs = [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            -Vec3::X,
            -Vec3::Y,
            -Vec3::Z,
        ];
        let mut dir = dirs[(axis_idx as usize).min(5)];
        if dir.y.abs() > 0.9 {
            dir.x += 0.0001;
            dir = dir.normalize();
        }
        let r = self.spherical.radius;
        self.spherical = Spherical::from_offset(dir * r);
        self.smooth_target = self.target;
        self.smooth_spherical = self.spherical;
    }

    /// Zoom toolbar: perspective dolly, orthographic half-height (web uses 1.2× per step).
    pub fn zoom_step(&mut self, inward: bool) {
        const ZOOM_STEP: f32 = 1.2;
        if self.perspective {
            let f = if inward {
                1.0 / ZOOM_STEP
            } else {
                ZOOM_STEP
            };
            self.spherical.radius = (self.spherical.radius * f).clamp(self.min_radius, self.max_radius);
        } else {
            let f = if inward {
                1.0 / ZOOM_STEP
            } else {
                ZOOM_STEP
            };
            self.ortho_half_height = (self.ortho_half_height * f).max(0.01);
        }
    }

    /// Six axis directions in camera space `(sx, sy, depth)` for the orbit gizmo overlay (matches web).
    pub fn gizmo_axis_projections(&self) -> [[f32; 3]; 6] {
        let eye = self.smooth_target + self.smooth_spherical.to_offset();
        let view = Mat4::look_at_rh(eye, self.smooth_target, Vec3::Y);
        let axes = [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            -Vec3::X,
            -Vec3::Y,
            -Vec3::Z,
        ];
        let mut out = [[0f32; 3]; 6];
        for (i, ax) in axes.iter().enumerate() {
            let p = view.transform_vector3(*ax);
            out[i] = [p.x, p.y, p.z];
        }
        out
    }

    /// Percent label: perspective uses `base_dist / radius`, ortho uses `ref_half_h / ortho_half_height`.
    pub fn zoom_percent_for_display(&self, base_perspective_dist: f32, ortho_ref_half_h: f32) -> i32 {
        if self.perspective {
            let r = self.smooth_spherical.radius.max(1e-4);
            ((base_perspective_dist / r) * 100.0).round().clamp(1.0, 9999.0) as i32
        } else {
            let h = self.ortho_half_height.max(1e-4);
            ((ortho_ref_half_h / h) * 100.0).round().clamp(1.0, 9999.0) as i32
        }
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
