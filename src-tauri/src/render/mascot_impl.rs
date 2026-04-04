//! Mascot and logo overlay types and rendering methods for [`super::WgpuViewer`].

use super::*;

/// GPU overlay for the start-screen logo (rendered like a mascot but at full viewport size
/// with interactive orbit instead of bobbing animation).
pub struct LogoOverlay {
    pub vertex_buffer: Option<wgpu::Buffer>,
    pub index_buffer: Option<wgpu::Buffer>,
    pub index_count: u32,
    pub(crate) uniforms_buffer: wgpu::Buffer,
    pub(crate) bind_group: wgpu::BindGroup,
    pub bounds: Option<MeshBounds>,
    pub visible: bool,
    /// Light direction (toward light source); computed from azimuth/elevation.
    pub light_dir: [f32; 3],
    /// Azimuth in degrees (0-360, CCW from +X in XZ plane).
    pub light_azimuth_deg: f32,
    /// Elevation in degrees (5-90, above XZ plane).
    pub light_elevation_deg: f32,
    /// Sun/direct light intensity (default 0.7).
    pub light_intensity: f32,
    /// Camera distance from origin.
    pub cam_dist: f32,
    /// Current orbit angles.
    pub theta: f32,
    pub phi: f32,
    /// Rest pose recorded after load (snap-back target).
    pub rest_theta: f32,
    pub rest_phi: f32,
    /// When true, theta/phi lerp back toward rest each frame.
    pub returning_to_rest: bool,
    /// Mouse position in NDC [-1,1] for the explode effect.
    pub mouse_ndc_x: f32,
    pub mouse_ndc_y: f32,
    /// Whether the mouse is currently over the logo.
    pub mouse_active: bool,
    /// Accumulated animation time (seconds) for wobble.
    pub anim_t: f32,
}

impl LogoOverlay {
    /// Logo splash: max deviation from rest for drag + hover (+-75 deg).
    const ORBIT_HALF_SPAN: f32 = 75.0 * (std::f32::consts::PI / 180.0);
    /// Subtle cursor parallax at viewport edges (radians).
    const HOVER_MAX_RAD: f32 = 0.038;

    /// Logo splash drag: rotate then clamp +-75 deg from rest on both axes.
    pub fn rotate_drag(&mut self, dx: f32, dy: f32, viewport_height_px: f32) {
        self.returning_to_rest = false;
        let h = viewport_height_px.max(1.0);
        let k = std::f32::consts::TAU / h;
        self.theta -= dx * k;
        self.phi -= dy * k;
        self.theta = Self::clamp_near(self.theta, self.rest_theta, Self::ORBIT_HALF_SPAN);
        self.phi = Self::clamp_near(self.phi, self.rest_phi, Self::ORBIT_HALF_SPAN)
            .clamp(0.01, std::f32::consts::PI - 0.01);
    }

    /// Hover parallax: nudge orbit slightly from rest based on normalised viewport position.
    /// Skipped while the return-to-rest lerp is still running so it doesn't fight the animation.
    pub fn hover_parallax(&mut self, x_px: f32, y_px: f32, vw: f32, vh: f32) {
        if self.returning_to_rest {
            return;
        }
        let vw = vw.max(1.0);
        let vh = vh.max(1.0);
        let nx = ((x_px / vw) - 0.5) * 2.0;
        let ny = -(((y_px / vh) - 0.5) * 2.0);
        let nx = nx.clamp(-1.0, 1.0);
        let ny = ny.clamp(-1.0, 1.0);

        let theta_t = self.rest_theta + nx * Self::HOVER_MAX_RAD;
        let phi_t = self.rest_phi + ny * Self::HOVER_MAX_RAD;
        self.theta = Self::clamp_near(theta_t, self.rest_theta, Self::ORBIT_HALF_SPAN);
        self.phi = Self::clamp_near(phi_t, self.rest_phi, Self::ORBIT_HALF_SPAN)
            .clamp(0.01, std::f32::consts::PI - 0.01);
    }

    /// Begin lerping back to rest pose on pointer release.
    pub fn reset_orbit(&mut self) {
        self.returning_to_rest = true;
    }

    /// Exponential decay toward rest pose; call once per frame.
    /// Returns `true` while the animation is still in progress.
    pub fn tick_return_to_rest(&mut self, dt: f32) -> bool {
        if !self.returning_to_rest {
            return false;
        }
        // Exponential decay: half-life ≈ 0.15 s → λ = ln(2)/0.15 ≈ 4.6
        let speed = 2.0;
        let t = 1.0 - (-speed * dt).exp();
        self.theta += (self.rest_theta - self.theta) * t;
        self.phi += (self.rest_phi - self.phi) * t;
        // Snap when close enough to avoid endless micro-updates.
        let eps = 1e-4;
        if (self.theta - self.rest_theta).abs() < eps && (self.phi - self.rest_phi).abs() < eps {
            self.theta = self.rest_theta;
            self.phi = self.rest_phi;
            self.returning_to_rest = false;
        }
        true
    }

    /// Update mouse position in NDC for the explode effect.
    pub fn update_mouse_ndc(&mut self, x_px: f32, y_px: f32, vw: f32, vh: f32) {
        let vw = vw.max(1.0);
        let vh = vh.max(1.0);
        self.mouse_ndc_x = (x_px / vw) * 2.0 - 1.0;
        self.mouse_ndc_y = -((y_px / vh) * 2.0 - 1.0); // flip Y for NDC
        self.mouse_active = true;
    }

    /// Clear mouse position (mouse left the viewport).
    pub fn clear_mouse_ndc(&mut self) {
        self.mouse_ndc_x = 99.0;
        self.mouse_ndc_y = 99.0;
        self.mouse_active = false;
    }

    fn clamp_near(val: f32, rest: f32, half_span: f32) -> f32 {
        let mut d = val - rest;
        while d > std::f32::consts::PI {
            d -= std::f32::consts::TAU;
        }
        while d < -std::f32::consts::PI {
            d += std::f32::consts::TAU;
        }
        rest + d.clamp(-half_span, half_span)
    }
}

/// Per-mascot GPU state for start-screen floating model views.
pub struct MascotEntry {
    pub id: u32,
    pub vertex_buffer: Option<wgpu::Buffer>,
    pub index_buffer: Option<wgpu::Buffer>,
    pub index_count: u32,
    pub(crate) uniforms_buffer: wgpu::Buffer,
    pub(crate) bind_group: wgpu::BindGroup,
    /// Animation phase in seconds; incremented each frame.
    pub anim_t: f32,
    /// Viewport-relative screen rect [x, y, w, h] in physical pixels.
    pub screen_rect: [f32; 4],
    pub visible: bool,
    /// World-space AABB for auto-framing the mascot camera.
    pub bounds: Option<MeshBounds>,
}

impl WgpuViewer {
    /// Interleave mesh buffers for the mascot/logo pipeline (17 floats per vertex).
    /// Appends `voxel_center` (vec3) after the standard 14-float opaque layout.
    /// The voxel center is derived from each quad's geometry: the face center
    /// offset inward by half the normal, so all faces of the same voxel share it.
    fn interleaved_for_mascot(mesh: &MeshBuffers) -> Vec<f32> {
        let n = mesh.positions.len() / 3;
        // Pre-compute per-vertex voxel center from quad geometry (groups of 4 vertices).
        let mut voxel_centers = vec![[0.0f32; 3]; n];
        let quads = n / 4;
        for q in 0..quads {
            let base = q * 4;
            // Face center = average of the 4 corner positions.
            let mut cx = 0.0f32;
            let mut cy = 0.0f32;
            let mut cz = 0.0f32;
            for k in 0..4 {
                let i = (base + k) * 3;
                cx += mesh.positions[i];
                cy += mesh.positions[i + 1];
                cz += mesh.positions[i + 2];
            }
            cx *= 0.25;
            cy *= 0.25;
            cz *= 0.25;
            // Offset by -0.5 * normal to get the voxel center.
            let ni = base * 3;
            let nx = mesh.normals[ni];
            let ny = mesh.normals[ni + 1];
            let nz = mesh.normals[ni + 2];
            let vc = [cx - 0.5 * nx, cy - 0.5 * ny, cz - 0.5 * nz];
            for k in 0..4 {
                voxel_centers[base + k] = vc;
            }
        }

        let mut interleaved: Vec<f32> = Vec::with_capacity(n * 17);
        for i in 0..n {
            interleaved.push(mesh.positions[i * 3]);
            interleaved.push(mesh.positions[i * 3 + 1]);
            interleaved.push(mesh.positions[i * 3 + 2]);
            interleaved.push(mesh.normals[i * 3]);
            interleaved.push(mesh.normals[i * 3 + 1]);
            interleaved.push(mesh.normals[i * 3 + 2]);
            interleaved.push(mesh.colors[i * 3]);
            interleaved.push(mesh.colors[i * 3 + 1]);
            interleaved.push(mesh.colors[i * 3 + 2]);
            interleaved.push(mesh.mat_kind[i]);
            interleaved.push(mesh.ao.get(i).copied().unwrap_or(1.0));
            let ei = i * 3;
            interleaved.push(mesh.emission_tint.get(ei).copied().unwrap_or(0.0));
            interleaved.push(mesh.emission_tint.get(ei + 1).copied().unwrap_or(0.0));
            interleaved.push(mesh.emission_tint.get(ei + 2).copied().unwrap_or(0.0));
            interleaved.push(voxel_centers[i][0]);
            interleaved.push(voxel_centers[i][1]);
            interleaved.push(voxel_centers[i][2]);
        }
        interleaved
    }

    /// Load (or replace) the voxel mesh for mascot `id`.
    /// Creates the slot if it does not yet exist.
    pub fn load_mascot_mesh(&mut self, id: u32, mesh: &MeshBuffers, bounds: MeshBounds) {
        let interleaved = Self::interleaved_for_mascot(mesh);
        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mascot_vtx"),
                contents: bytemuck::cast_slice(&interleaved),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mascot_idx"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        let index_count = mesh.indices.len() as u32;

        if let Some(entry) = self.mascots.iter_mut().find(|m| m.id == id) {
            entry.vertex_buffer = Some(vertex_buffer);
            entry.index_buffer = Some(index_buffer);
            entry.index_count = index_count;
            entry.bounds = Some(bounds);
        } else {
            let uniforms_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mascot_uniforms"),
                size: std::mem::size_of::<MascotUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mascot_bg"),
                layout: &self.mascot_bind_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buffer.as_entire_binding(),
                }],
            });
            self.mascots.push(MascotEntry {
                id,
                vertex_buffer: Some(vertex_buffer),
                index_buffer: Some(index_buffer),
                index_count,
                uniforms_buffer,
                bind_group,
                anim_t: 0.0,
                screen_rect: [0.0, 0.0, 200.0, 200.0],
                visible: false,
                bounds: Some(bounds),
            });
        }
    }

    /// Update the viewport-relative screen rect for mascot `id`.
    /// Recreates the depth buffer if the dimensions change.
    pub fn set_mascot_screen_rect(&mut self, id: u32, x: f32, y: f32, w: f32, h: f32) {
        if let Some(entry) = self.mascots.iter_mut().find(|m| m.id == id) {
            entry.screen_rect = [x, y, w, h];
        }
    }

    /// Show or hide a mascot.
    pub fn set_mascot_visible(&mut self, id: u32, visible: bool) {
        if let Some(entry) = self.mascots.iter_mut().find(|m| m.id == id) {
            entry.visible = visible;
        }
    }

    /// Returns true if any mascot is currently visible (used to keep the render loop alive).
    pub fn any_mascot_visible(&self) -> bool {
        self.mascots
            .iter()
            .any(|m| m.visible && m.vertex_buffer.is_some())
    }

    fn mascot_uniforms(&self, i: usize) -> MascotUniforms {
        let m = &self.mascots[i];
        let rw = m.screen_rect[2].max(1.0);
        let rh = m.screen_rect[3].max(1.0);
        let aspect = rw / rh;
        if let Some(bounds) = &m.bounds {
            let center = bounds.center();
            let extent = (bounds.max - bounds.min).max_element().max(0.001);
            let bob_y = m.anim_t.sin() * 0.08;
            let model = Mat4::from_scale(Vec3::splat(1.2 / extent))
                * Mat4::from_rotation_y(-std::f32::consts::FRAC_PI_4)
                * Mat4::from_translation(Vec3::new(0.0, bob_y, 0.0))
                * Mat4::from_translation(-center);
            let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 3.5), Vec3::ZERO, Vec3::Y);
            let proj = Mat4::perspective_rh(0.5, aspect, 0.1, 50.0);
            MascotUniforms {
                mvp: (proj * view * model).to_cols_array_2d(),
                light_dir: [0.6, 0.8, 0.5, 0.0],
                ambient: 0.70,
                sun: 2.0,
                explode_radius: 0.0,
                explode_strength: 0.0,
                mouse_ndc: [99.0, 99.0],
                mouse_active: 0.0,
                time_seconds: 0.0,
            }
        } else {
            MascotUniforms {
                mvp: Mat4::IDENTITY.to_cols_array_2d(),
                light_dir: [0.6, 0.8, 0.5, 0.0],
                ambient: 0.70,
                sun: 2.0,
                explode_radius: 0.0,
                explode_strength: 0.0,
                mouse_ndc: [99.0, 99.0],
                mouse_active: 0.0,
                time_seconds: 0.0,
            }
        }
    }

    pub(crate) fn render_mascots(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        swap_view: &wgpu::TextureView,
    ) {
        // Skip when the swapchain is in HDR mode -- pipeline target is sdr_format only.
        // TODO: recreate mascot_pipeline when config.format changes.
        if self.config.format != self.sdr_format {
            return;
        }

        let now = std::time::Instant::now();
        let dt = now
            .duration_since(self.mascot_last_tick)
            .as_secs_f32()
            .clamp(0.0, 0.1);
        self.mascot_last_tick = now;

        const BOB_SPEED: f32 = 1.8; // radians / second

        // Advance animation phase.
        for m in &mut self.mascots {
            m.anim_t += dt * BOB_SPEED;
        }

        // Upload uniforms, then issue render passes.
        for i in 0..self.mascots.len() {
            if !self.mascots[i].visible || self.mascots[i].vertex_buffer.is_none() {
                continue;
            }

            let uniforms = self.mascot_uniforms(i);
            self.queue.write_buffer(
                &self.mascots[i].uniforms_buffer,
                0,
                bytemuck::bytes_of(&uniforms),
            );

            let rx = self.viewport_x as f32 + self.mascots[i].screen_rect[0];
            let ry = self.viewport_y as f32 + self.mascots[i].screen_rect[1];
            let rw = self.mascots[i].screen_rect[2].max(1.0);
            let rh = self.mascots[i].screen_rect[3].max(1.0);
            let index_count = self.mascots[i].index_count;

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("mascot"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: swap_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.mascot_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });

                pass.set_viewport(rx, ry, rw, rh, 0.0, 1.0);
                pass.set_scissor_rect(rx as u32, ry as u32, rw as u32, rh as u32);
                pass.set_pipeline(&self.mascot_pipeline);
                pass.set_bind_group(0, &self.mascots[i].bind_group, &[]);
                pass.set_vertex_buffer(
                    0,
                    self.mascots[i].vertex_buffer.as_ref().unwrap().slice(..),
                );
                pass.set_index_buffer(
                    self.mascots[i].index_buffer.as_ref().unwrap().slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                pass.draw_indexed(0..index_count, 0, 0..1);
            }
        }
    }

    // ── Logo overlay helpers ─────────────────────────────────────────────────

    /// Load (or replace) the start-screen logo mesh as an overlay.
    pub fn load_logo_mesh(&mut self, mesh: &MeshBuffers, bounds: MeshBounds) {
        let interleaved = Self::interleaved_for_mascot(mesh);
        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("logo_vtx"),
                contents: bytemuck::cast_slice(&interleaved),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("logo_idx"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        let index_count = mesh.indices.len() as u32;

        // Rest orbit: azimuth 60 deg, elevation 4 deg.
        let rest_theta = 60.0_f32.to_radians();
        let rest_phi = (90.0 - 4.0_f32).to_radians();

        if let Some(logo) = &mut self.logo_overlay {
            logo.vertex_buffer = Some(vertex_buffer);
            logo.index_buffer = Some(index_buffer);
            logo.index_count = index_count;
            logo.bounds = Some(bounds);
            logo.theta = rest_theta;
            logo.phi = rest_phi;
            logo.rest_theta = rest_theta;
            logo.rest_phi = rest_phi;
            logo.returning_to_rest = false;
        } else {
            let uniforms_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("logo_uniforms"),
                size: std::mem::size_of::<MascotUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("logo_bg"),
                layout: &self.mascot_bind_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buffer.as_entire_binding(),
                }],
            });
            self.logo_overlay = Some(LogoOverlay {
                vertex_buffer: Some(vertex_buffer),
                index_buffer: Some(index_buffer),
                index_count,
                uniforms_buffer,
                bind_group,
                bounds: Some(bounds),
                visible: false,
                light_dir: light_dir_from_azimuth_elevation_deg(0.0, 30.0).to_array(),
                light_azimuth_deg: 0.0,
                light_elevation_deg: 30.0,
                light_intensity: 3.0,
                cam_dist: 2.4,
                theta: rest_theta,
                phi: rest_phi,
                rest_theta,
                rest_phi,
                returning_to_rest: false,
                mouse_ndc_x: 99.0,
                mouse_ndc_y: 99.0,
                mouse_active: false,
                anim_t: 0.0,
            });
        }
    }

    fn logo_uniforms(&self) -> MascotUniforms {
        let logo = self.logo_overlay.as_ref().unwrap();
        let vw = self.viewport_width.max(1) as f32;
        let vh = self.viewport_height.max(1) as f32;
        let aspect = vw / vh;
        if let Some(bounds) = &logo.bounds {
            let center = bounds.center();
            let extent = (bounds.max - bounds.min).max_element().max(0.001);

            // Spherical to cartesian (physics convention: theta=azimuth, phi=polar from +Y).
            let sin_phi = logo.phi.sin();
            let cos_phi = logo.phi.cos();
            let sin_theta = logo.theta.sin();
            let cos_theta = logo.theta.cos();
            let eye_dir = Vec3::new(sin_phi * sin_theta, cos_phi, sin_phi * cos_theta);

            let eye = eye_dir * logo.cam_dist;
            let model =
                Mat4::from_scale(Vec3::splat(1.2 / extent)) * Mat4::from_translation(-center);
            let view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
            let proj = Mat4::perspective_rh(0.5, aspect, 0.1, 50.0);
            let ld = logo.light_dir;
            MascotUniforms {
                mvp: (proj * view * model).to_cols_array_2d(),
                light_dir: [ld[0], ld[1], ld[2], 0.0],
                ambient: 0.70,
                sun: logo.light_intensity,
                explode_radius: 0.25,
                explode_strength: 5.0,
                mouse_ndc: [logo.mouse_ndc_x, logo.mouse_ndc_y],
                mouse_active: if logo.mouse_active { 1.0 } else { 0.0 },
                time_seconds: logo.anim_t,
            }
        } else {
            let ld = logo.light_dir;
            MascotUniforms {
                mvp: Mat4::IDENTITY.to_cols_array_2d(),
                light_dir: [ld[0], ld[1], ld[2], 0.0],
                ambient: 0.70,
                sun: logo.light_intensity,
                explode_radius: 0.0,
                explode_strength: 0.0,
                mouse_ndc: [99.0, 99.0],
                mouse_active: 0.0,
                time_seconds: 0.0,
            }
        }
    }

    pub(crate) fn render_logo_overlay(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        swap_view: &wgpu::TextureView,
    ) {
        // Skip when the swapchain is in HDR mode (same limitation as mascots).
        if self.config.format != self.sdr_format {
            return;
        }
        let visible = self
            .logo_overlay
            .as_ref()
            .map_or(false, |l| l.visible && l.vertex_buffer.is_some());
        if !visible {
            return;
        }

        // Tick the return-to-rest lerp before computing uniforms.
        let now = std::time::Instant::now();
        let dt = now
            .duration_since(self.mascot_last_tick)
            .as_secs_f32()
            .clamp(0.0, 0.1);
        if let Some(logo) = self.logo_overlay.as_mut() {
            logo.tick_return_to_rest(dt);
            logo.anim_t += dt;
        }

        let uniforms = self.logo_uniforms();
        let logo = self.logo_overlay.as_ref().unwrap();
        self.queue
            .write_buffer(&logo.uniforms_buffer, 0, bytemuck::bytes_of(&uniforms));

        let vx = self.viewport_x as f32;
        let vy = self.viewport_y as f32;
        let vw = self.viewport_width.max(1) as f32;
        let vh = self.viewport_height.max(1) as f32;
        let index_count = logo.index_count;

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("logo_overlay"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: swap_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.mascot_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            pass.set_viewport(vx, vy, vw, vh, 0.0, 1.0);
            pass.set_scissor_rect(vx as u32, vy as u32, vw as u32, vh as u32);
            pass.set_pipeline(&self.mascot_pipeline);
            pass.set_bind_group(0, &logo.bind_group, &[]);
            pass.set_vertex_buffer(0, logo.vertex_buffer.as_ref().unwrap().slice(..));
            pass.set_index_buffer(
                logo.index_buffer.as_ref().unwrap().slice(..),
                wgpu::IndexFormat::Uint32,
            );
            pass.draw_indexed(0..index_count, 0, 0..1);
        }
    }

    /// Returns true if the logo overlay is currently visible (used to keep the render loop alive).
    pub fn logo_overlay_visible(&self) -> bool {
        self.logo_overlay
            .as_ref()
            .map_or(false, |l| l.visible && l.vertex_buffer.is_some())
    }
}
