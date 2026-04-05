use super::*;

// ── Progressive raytracer types ───────────────────────────────────────────────

#[derive(serde::Serialize, Clone)]
pub struct RaytraceBenchmarkResult {
    pub frame_count: u32,
    pub viewport_width: u32,
    pub viewport_height: u32,
    /// Total wall-clock time for all frames (ms), GPU synced.
    pub total_ms: f64,
    pub avg_ms: f64,
    pub stddev_ms: f64,
    pub min_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
    /// Megapixels per second (frames * pixels / total_ms * 1000).
    pub mpix_per_sec: f64,
    /// Individual frame times in ms.
    pub frame_times_ms: Vec<f64>,
}

impl WgpuViewer {
    /// Run `frame_count` ray-trace frames off-screen (no surface present), GPU-synced.
    /// Returns timing statistics. Raytrace mode does not need to be active; this temporarily
    /// forces it on regardless of the current mode flag.
    pub fn run_raytrace_benchmark(&mut self, frame_count: u32) -> RaytraceBenchmarkResult {
        let n = frame_count.max(1);
        let mut frame_ms = Vec::with_capacity(n as usize);

        // Reset accumulation so every frame is a fresh sample-0 pass.
        self.rt_sample_n = 0;
        self.rt_fast_preview = false;

        for _ in 0..n {
            let flip = self.rt_accum_flip as usize;
            let rt_u = RtUniform {
                frame_seed: self.rt_sample_n.wrapping_mul(2654435761).wrapping_add(1),
                sample_n: self.rt_sample_n,
                fast_preview: 0,
                surface_mode: self.rt_surface_mode,
            };
            self.queue
                .write_buffer(&self.rt_uniform_buf, 0, bytemuck::bytes_of(&rt_u));

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rt_bench"),
                });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("rt_bench_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.rt_accum_views[flip],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_raytrace);
                pass.set_bind_group(0, &self.rt_scene_bg, &[]);
                pass.set_bind_group(1, &self.rt_accum_bgs[flip], &[]);
                pass.draw(0..3, 0..1);
            }

            let t0 = Instant::now();
            self.queue.submit(std::iter::once(encoder.finish()));
            self.device.poll(wgpu::Maintain::Wait);
            frame_ms.push(t0.elapsed().as_secs_f64() * 1000.0);

            self.rt_accum_flip = !self.rt_accum_flip;
            self.rt_sample_n = self.rt_sample_n.saturating_add(1);
        }

        // Reset so normal rendering starts fresh.
        self.rt_sample_n = 0;
        self.rt_accum_flip = false;

        let total_ms: f64 = frame_ms.iter().sum();
        let avg_ms = total_ms / n as f64;
        let variance = frame_ms.iter().map(|&t| (t - avg_ms).powi(2)).sum::<f64>() / n as f64;
        let stddev_ms = variance.sqrt();
        let min_ms = frame_ms.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_ms = frame_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let pixels = self.viewport_width as f64 * self.viewport_height as f64;
        let mpix_per_sec = (n as f64 * pixels) / (total_ms / 1000.0) / 1_000_000.0;

        let mut sorted = frame_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let percentile = |p: f64| -> f64 {
            let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
            sorted[idx.min(sorted.len() - 1)]
        };

        RaytraceBenchmarkResult {
            frame_count: n,
            viewport_width: self.viewport_width,
            viewport_height: self.viewport_height,
            total_ms,
            avg_ms,
            stddev_ms,
            min_ms,
            p50_ms: percentile(50.0),
            p95_ms: percentile(95.0),
            p99_ms: percentile(99.0),
            max_ms,
            mpix_per_sec,
            frame_times_ms: frame_ms,
        }
    }

    pub fn update_uniforms(&mut self, camera: &OrbitCamera) {
        let w = self.viewport_width.max(1) as f32;
        let h = self.viewport_height.max(1) as f32;
        let proj = camera.proj_matrix(w, h);
        let view = camera.view_matrix();

        // Detect camera change for RT accumulation reset.
        if self.raytrace_enabled {
            let eye = camera.smooth_eye();
            let inv_v = view.inverse();
            let eye_arr = [eye.x, eye.y, eye.z];
            let inv_v_arr = inv_v.to_cols_array_2d();
            if eye_arr != self.rt_prev_eye || inv_v_arr != self.rt_prev_inv_view {
                self.rt_sample_n = 0;
                self.rt_prev_eye = eye_arr;
                self.rt_prev_inv_view = inv_v_arr;
                self.rt_fast_preview = true;
            } else {
                self.rt_fast_preview = false;
            }
        }
        let vp = proj * view;
        let inv_v = view.inverse();
        let inv_p = proj.inverse();
        let lvp = light_view_proj(&self.scene_bounds, self.light_dir);
        let eye = camera.smooth_eye();
        let gs = GlobalState {
            view_proj: vp.to_cols_array_2d(),
            inv_view: inv_v.to_cols_array_2d(),
            inv_proj: inv_p.to_cols_array_2d(),
            light_view_proj: lvp.to_cols_array_2d(),
            light_dir: [self.light_dir.x, self.light_dir.y, self.light_dir.z, 0.0],
            cam_pos: [eye.x, eye.y, eye.z, 0.0],
            brick_origin: [
                self.brick_origin_iv.x as f32,
                self.brick_origin_iv.y as f32,
                self.brick_origin_iv.z as f32,
                0.0,
            ],
            brick_dims: [
                self.brick_dims_u.0 as f32,
                self.brick_dims_u.1 as f32,
                self.brick_dims_u.2 as f32,
                0.0,
            ],
            screen: [w, h, 1.0 / w.max(1.0), 1.0 / h.max(1.0)],
            params: [
                self.start_screen_appearance,
                if self.soft_shadows { 1.0 } else { 0.0 },
                BLOOM_STRENGTH,
                camera.near,
            ],
            light_params: [
                self.light_ambient,
                self.light_sun,
                if self.shadows_enabled { 1.0 } else { 0.0 },
                if self.sky_enabled { 1.0 } else { 0.0 },
            ],
            sun_color: [
                self.sun_color_linear.x,
                self.sun_color_linear.y,
                self.sun_color_linear.z,
                0.0,
            ],
            bg_color: [
                self.bg_color_linear.x,
                self.bg_color_linear.y,
                self.bg_color_linear.z,
                0.0,
            ],
        };
        self.queue
            .write_buffer(&self.global_buffer, 0, bytemuck::bytes_of(&gs));

        // Sun shafts: project light direction to screen UV for ray marching origin.
        let far_pt = glam::Vec4::new(
            eye.x + self.light_dir.x * 1000.0,
            eye.y + self.light_dir.y * 1000.0,
            eye.z + self.light_dir.z * 1000.0,
            1.0,
        );
        let clip = vp * far_pt;
        if clip.w.abs() > 1e-6 {
            let ndc = clip / clip.w;
            self.post_composite_opts.ss_sun_uv_x = ndc.x * 0.5 + 0.5;
            self.post_composite_opts.ss_sun_uv_y = 1.0 - (ndc.y * 0.5 + 0.5);
        }
    }

    pub fn render(&mut self) -> Result<(), String> {
        let frame = self
            .surface
            .get_current_texture()
            .map_err(|e| e.to_string())?;
        let tex_size = frame.texture.size();
        // Keep CPU-side surface size in sync with the actual swapchain (configure can differ slightly).
        self.surface_size = (tex_size.width.max(1), tex_size.height.max(1));
        // If the swapchain resized without a `resize()` call (Windows DPI / restore), the mascot
        // depth texture would mismatch the color attachment and cause a wgpu validation panic.
        if self.surface_size != self.mascot_depth_size {
            self.mascot_depth_view =
                Self::make_mascot_depth(&self.device, self.surface_size.0, self.surface_size.1).1;
            self.mascot_depth_size = self.surface_size;
        }
        // #region agent log
        static RENDER_LOG_COUNTER: AtomicU32 = AtomicU32::new(0);
        let render_log_idx = RENDER_LOG_COUNTER.fetch_add(1, Ordering::Relaxed);
        if render_log_idx < 8 {
            debug_log(
                "H5",
                "src-tauri/src/render/mod.rs:render",
                "frame-begin",
                json!({
                    "index": render_log_idx,
                    "viewport": [self.viewport_x, self.viewport_y, self.viewport_width, self.viewport_height],
                    "surface": [self.surface_size.0, self.surface_size.1]
                }),
            );
        }
        // #endregion
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        if self.raytrace_enabled {
            // ── Progressive raytracer ─────────────────────────────────────────
            // Upload uniform: frame seed + current sample count.
            let flip = self.rt_accum_flip as usize;
            let rt_u = RtUniform {
                frame_seed: self.rt_sample_n.wrapping_mul(2654435761).wrapping_add(1),
                sample_n: self.rt_sample_n,
                fast_preview: self.rt_fast_preview as u32,
                surface_mode: self.rt_surface_mode,
            };
            self.queue
                .write_buffer(&self.rt_uniform_buf, 0, bytemuck::bytes_of(&rt_u));

            if self.rt_fast_preview {
                // ── Fast-preview path: render at half resolution, then upscale ──
                // Rendering to w/2 × h/2 cuts pixel count to 1/4, giving ~4× speedup
                // before the reduced-step DDA savings on top.
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("raytrace_preview"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.rt_preview_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.pipeline_raytrace);
                    pass.set_bind_group(0, &self.rt_scene_bg, &[]);
                    pass.set_bind_group(1, &self.rt_accum_bgs[flip], &[]);
                    pass.draw(0..3, 0..1);
                }
                // Bilinear upscale to full-size accum texture.
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("raytrace_upscale"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.rt_accum_views[flip],
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.pipeline_blit);
                    pass.set_bind_group(0, &self.rt_preview_bg, &[]);
                    pass.draw(0..3, 0..1);
                }
            } else {
                // ── Full-quality path ─────────────────────────────────────────────
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("raytrace"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.rt_accum_views[flip],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_raytrace);
                pass.set_bind_group(0, &self.rt_scene_bg, &[]);
                pass.set_bind_group(1, &self.rt_accum_bgs[flip], &[]);
                pass.draw(0..3, 0..1);
            }

            // Copy raytrace result → hdr_texture for bloom + composite.
            let ext = wgpu::Extent3d {
                width: self.viewport_width.max(1),
                height: self.viewport_height.max(1),
                depth_or_array_layers: 1,
            };
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.rt_accum_textures[flip],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyTexture {
                    texture: &self.hdr_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                ext,
            );

            // Overlay pass: draw editing overlays (preview, selection, gizmo, grid) on top.
            // Depth is cleared fresh so overlays z-test against each other but not the scene.
            // All overlay pipelines write to two color targets (HDR + normal) but the normal
            // target has ColorWrites::empty(), so it is safe to Load/Discard it.
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("rt_overlay"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: &self.hdr_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &self.normal_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Discard,
                            },
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                self.draw_indexed_selection_overlay_solid(&mut pass);
                self.draw_indexed_preview(&mut pass);
                self.draw_selection_overlay_lines(&mut pass);
                self.draw_grid_border_lines(&mut pass);
                self.render_avatars(&mut pass);
                self.draw_ping_wave_lines(&mut pass);
                self.draw_indexed_ping(&mut pass);
                self.draw_gizmo(&mut pass);
            }

            self.rt_accum_flip = !self.rt_accum_flip;
            self.rt_sample_n = self.rt_sample_n.saturating_add(1);
        } else {
            // ── Rasterized path (default) ─────────────────────────────────────────

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shadow"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.shadow_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_shadow);
                pass.set_bind_group(0, &self.bind_shadow_pass, &[]);
                self.draw_indexed_mesh_all(&mut pass);
            }

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("opaque"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: &self.hdr_opaque_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &self.normal_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.5,
                                    g: 0.5,
                                    b: 1.0,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                if self.start_screen_transparent {
                    pass.set_pipeline(&self.pipeline_start_screen_bg);
                    pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
                    pass.draw(0..3, 0..1);
                } else {
                    pass.set_pipeline(&self.pipeline_sky);
                    pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
                    pass.draw(0..3, 0..1);
                }
                pass.set_pipeline(&self.pipeline_opaque);
                pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
                self.draw_indexed_mesh(&mut pass);
                self.draw_indexed_selection_overlay_solid(&mut pass);
                self.draw_indexed_preview(&mut pass);
                self.draw_selection_overlay_lines(&mut pass);
                self.draw_grid_border_lines(&mut pass);
                self.render_avatars(&mut pass);
                self.draw_ping_wave_lines(&mut pass);
                self.draw_indexed_ping(&mut pass);
                self.draw_gizmo(&mut pass);
            }

            let ext = wgpu::Extent3d {
                width: self.viewport_width.max(1),
                height: self.viewport_height.max(1),
                depth_or_array_layers: 1,
            };
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.hdr_opaque_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyTexture {
                    texture: &self.hdr_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                ext,
            );
            // Copy depth to a read-only snapshot: next passes sample depth while the main depth
            // attachment may still be bound in overlapping use; WebGPU disallows attach+sample.
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.depth_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::DepthOnly,
                },
                wgpu::ImageCopyTexture {
                    texture: &self.depth_snapshot_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::DepthOnly,
                },
                ext,
            );

            // ── SSR fullscreen pass (opaque metals) ─────────────────────────────
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ssr_fullscreen"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.ssr_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_ssr_fullscreen);
                pass.set_bind_group(0, &self.bind_ssr, &[]);
                pass.draw(0..3, 0..1);
            }

            // ── OIT accumulation pass ────────────────────────────────────────────
            if let Some(ref trans_bg) = self.bind_trans {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("oit_accum"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: &self.oit_accum_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &self.oit_revealage_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 1.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 0.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_oit_accum);
                pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
                pass.set_bind_group(1, trans_bg, &[]);
                self.draw_indexed_oit_mesh(&mut pass);
            }

            // ── OIT composite pass ───────────────────────────────────────────────
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("oit_composite"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.hdr_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_oit_composite);
                pass.set_bind_group(0, &self.bind_oit_composite, &[]);
                pass.draw(0..3, 0..1);
            }
        } // end rasterized path

        // Bloom extract reads `bloom_extract_buf` uploaded at end of the *previous* frame (after
        // meter readback). This frame's upload happens below, after the meter pass.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_ex"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_a_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_bloom_extract);
            pass.set_bind_group(0, &self.bind_bloom_extract, &[]);
            pass.draw(0..3, 0..1);
        }

        // Bloom mip pyramid — matches the web version's BloomNode / UnrealBloomPass approach:
        //   1. Downsample bloom_a through 5 half-res levels (hardware bilinear blit).
        //   2. H+V Gaussian blur at each level (step=1 at native mip res = natural wide radius).
        //   3. Upsample additively from coarsest → finest to accumulate all scales.
        //   4. Replace-blit pyr_a[0] back into bloom_a for the composite pass.

        // --- Downsample chain ---
        for i in 0..BLOOM_LEVELS {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit_down"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_pyramid_a_views[i],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_blit);
            pass.set_bind_group(0, &self.bind_blit_down[i], &[]);
            pass.draw(0..3, 0..1);
        }

        // --- Blur each pyramid level (H then V, step=1 at that mip's native res) ---
        let blur_h_u = PostBlurUniform {
            blur_dir: [1.0, 0.0, 1.0, 0.0],
        };
        let blur_v_u = PostBlurUniform {
            blur_dir: [0.0, 1.0, 1.0, 0.0],
        };
        for i in 0..BLOOM_LEVELS {
            self.queue
                .write_buffer(&self.post_blur_buf, 0, bytemuck::bytes_of(&blur_h_u));
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("blur_pyr_h"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.bloom_pyramid_b_views[i],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_blur);
                pass.set_bind_group(0, &self.bind_blur_pyr_h[i], &[]);
                pass.draw(0..3, 0..1);
            }
            self.queue
                .write_buffer(&self.post_blur_buf, 0, bytemuck::bytes_of(&blur_v_u));
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("blur_pyr_v"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.bloom_pyramid_a_views[i],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_blur);
                pass.set_bind_group(0, &self.bind_blur_pyr_v[i], &[]);
                pass.draw(0..3, 0..1);
            }
        }

        // --- Upsample chain: weighted additive blend from coarser into finer levels ---
        // Each coarser level is multiplied by 0.75 before addition, giving a geometric falloff
        // (fine detail dominates, large-scale haze contributes progressively less) which better
        // matches the 1/r² energy falloff of real optical bloom.
        for i in (0..BLOOM_LEVELS - 1).rev() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit_up"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_pyramid_a_views[i],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_blit_weighted_add);
            pass.set_bind_group(0, &self.bind_blit_up_weighted[i], &[]);
            pass.draw(0..3, 0..1);
        }

        // --- Final blit: replace bloom_a with the combined pyramid result ---
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit_final"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_a_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_blit);
            pass.set_bind_group(0, &self.bind_blit_final, &[]);
            pass.draw(0..3, 0..1);
        }

        if self.auto_exposure_enabled {
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("meter_lum"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.meter_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.18,
                                g: 0.0,
                                b: 0.0,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_meter);
                pass.set_bind_group(0, &self.bind_meter, &[]);
                pass.draw(0..3, 0..1);
            }
            encoder.copy_texture_to_buffer(
                wgpu::ImageCopyTexture {
                    texture: &self.meter_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyBuffer {
                    buffer: &self.meter_staging,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
        }

        self.post_composite_opts.transparent_bg = 0.0;
        self.sync_composite_exposure_ev();
        // Animated effects: monotonic time
        self.post_composite_opts.time_seconds = self.creation_instant.elapsed().as_secs_f32();
        self.queue.write_buffer(
            &self.post_composite_opts_buf,
            0,
            bytemuck::bytes_of(&self.post_composite_opts),
        );
        // EV for the *next* frame's bloom_extract pass (this frame's bloom already used prior value).
        self.queue.write_buffer(
            &self.bloom_extract_buf,
            0,
            bytemuck::bytes_of(&BloomExtractUniform {
                exposure_ev: self.post_composite_opts.exposure_ev,
                _pad: [0.0; 3],
            }),
        );

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.present_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.06,
                            g: 0.07,
                            b: 0.09,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_composite);
            pass.set_bind_group(0, &self.bind_composite, &[]);
            pass.draw(0..3, 0..1);
        }

        // ── Gizmo move-drag delta label (GPU text via glyphon) ──
        if let Some(label) = &self.gizmo_delta_label {
            let vw = self.viewport_width.max(1);
            let vh = self.viewport_height.max(1);
            self.glyphon_viewport.update(
                &self.queue,
                Resolution {
                    width: vw,
                    height: vh,
                },
            );
            let font_size = 36.0_f32;
            let line_height = 46.0_f32;
            let mut buf = GlyphonBuffer::new(
                &mut self.glyphon_font_system,
                Metrics::new(font_size, line_height),
            );
            buf.set_size(
                &mut self.glyphon_font_system,
                Some(300.0),
                Some(line_height),
            );
            buf.set_text(
                &mut self.glyphon_font_system,
                &label.name,
                Attrs::new()
                    .family(Family::Monospace)
                    .color(GlyphonColor::rgb(159, 216, 255)),
                Shaping::Basic,
            );
            buf.shape_until_scroll(&mut self.glyphon_font_system, false);
            let text_width = buf.layout_runs().next().map(|r| r.line_w).unwrap_or(0.0);
            let areas = [TextArea {
                buffer: &buf,
                left: label.x - text_width * 0.5,
                top: label.y - line_height - 6.0,
                scale: 1.0,
                bounds: TextBounds {
                    left: 0,
                    top: 0,
                    right: vw as i32,
                    bottom: vh as i32,
                },
                default_color: GlyphonColor::rgb(159, 216, 255),
                custom_glyphs: &[],
            }];
            let _ = self.glyphon_text_renderer.prepare(
                &self.device,
                &self.queue,
                &mut self.glyphon_font_system,
                &mut self.glyphon_atlas,
                &self.glyphon_viewport,
                areas,
                &mut self.glyphon_swash_cache,
            );
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("gizmo_delta_label"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.present_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                let _ = self.glyphon_text_renderer.render(
                    &self.glyphon_atlas,
                    &self.glyphon_viewport,
                    &mut pass,
                );
            }
            self.glyphon_atlas.trim();
        }

        // ── Peer + ping name labels (GPU text via glyphon) ──
        let has_text = !self.peer_label_data.is_empty() || self.ping_label_data.is_some();
        if has_text {
            let vw = self.viewport_width.max(1);
            let vh = self.viewport_height.max(1);

            self.glyphon_viewport.update(
                &self.queue,
                Resolution {
                    width: vw,
                    height: vh,
                },
            );

            let font_size = 52.0;
            let line_height = 68.0;

            // Collect all labels: peers + optional ping
            let all_labels: Vec<&GpuPeerLabel> = self
                .peer_label_data
                .iter()
                .chain(self.ping_label_data.iter())
                .collect();

            let mut text_areas: Vec<TextArea<'_>> = Vec::new();
            let mut buffers: Vec<GlyphonBuffer> = Vec::new();

            for label in &all_labels {
                let mut buffer = GlyphonBuffer::new(
                    &mut self.glyphon_font_system,
                    Metrics::new(font_size, line_height),
                );
                buffer.set_size(
                    &mut self.glyphon_font_system,
                    Some(400.0),
                    Some(line_height),
                );

                let r = ((label.color_rgb >> 16) & 0xff) as u8;
                let g = ((label.color_rgb >> 8) & 0xff) as u8;
                let b = (label.color_rgb & 0xff) as u8;

                buffer.set_text(
                    &mut self.glyphon_font_system,
                    &label.name,
                    Attrs::new()
                        .family(Family::SansSerif)
                        .color(GlyphonColor::rgb(r, g, b)),
                    Shaping::Advanced,
                );
                buffer.shape_until_scroll(&mut self.glyphon_font_system, false);
                buffers.push(buffer);
            }

            for (i, label) in all_labels.iter().enumerate() {
                let buf = &buffers[i];
                let text_width = buf
                    .layout_runs()
                    .next()
                    .map(|run| run.line_w)
                    .unwrap_or(0.0);

                text_areas.push(TextArea {
                    buffer: buf,
                    left: label.x - text_width * 0.5,
                    top: label.y - line_height - 4.0,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: 0,
                        top: 0,
                        right: vw as i32,
                        bottom: vh as i32,
                    },
                    default_color: GlyphonColor::rgb(255, 255, 255),
                    custom_glyphs: &[],
                });
            }

            let _ = self.glyphon_text_renderer.prepare(
                &self.device,
                &self.queue,
                &mut self.glyphon_font_system,
                &mut self.glyphon_atlas,
                &self.glyphon_viewport,
                text_areas,
                &mut self.glyphon_swash_cache,
            );

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("peer_labels_text"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.present_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                let _ = self.glyphon_text_renderer.render(
                    &self.glyphon_atlas,
                    &self.glyphon_viewport,
                    &mut pass,
                );
            }

            self.glyphon_atlas.trim();
        }

        let vw = self.viewport_width.max(1);
        let vh = self.viewport_height.max(1);
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: &self.present_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &frame.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: self.viewport_x,
                    y: self.viewport_y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: vw,
                height: vh,
                depth_or_array_layers: 1,
            },
        );

        // Logo overlay, mascots, and speech bubbles render directly on the swapchain surface.
        let needs_swap_pass = self.start_screen_transparent || self.has_visible_speech_bubbles();
        if needs_swap_pass {
            let swap_view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            if self.start_screen_transparent {
                // Logo first, then mascots on top.
                self.render_logo_overlay(&mut encoder, &swap_view);
                self.render_mascots(&mut encoder, &swap_view);
            }
            self.render_speech_bubbles(&mut encoder, &swap_view);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        if self.auto_exposure_enabled {
            self.read_meter_luminance_and_update_auto_exposure();
        }
        frame.present();
        // #region agent log
        if render_log_idx < 8 {
            debug_log(
                "H6",
                "src-tauri/src/render/mod.rs:render",
                "frame-end",
                json!({
                    "index": render_log_idx,
                    "auto_exposure": self.auto_exposure_enabled,
                    "raytrace": self.raytrace_enabled
                }),
            );
        }
        // #endregion
        Ok(())
    }
}
