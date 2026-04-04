//! Speech bubble types and rendering methods for [`super::WgpuViewer`].

use super::*;

/// Animation state for a speech bubble.
#[derive(Debug, Clone, PartialEq)]
pub enum BubbleState {
    /// Showing normally; click advances page or begins shake on last page.
    Active,
    /// Shaking side-to-side; auto-dismisses when shake_t >= SHAKE_DURATION.
    Shaking { shake_t: f32 },
    /// Hidden; pending removal from the Vec.
    Dismissed,
}

/// One GPU-rendered speech bubble / floating note.
pub struct SpeechBubble {
    pub id: u32,
    /// Text pages shown in sequence; empty = no text.
    pub pages: Vec<String>,
    pub current_page: usize,
    /// Viewport-relative rect [x, y, w, h] in physical pixels (before viewport_x/y offset).
    pub screen_rect: [f32; 4],
    /// Viewport-relative tail tip [x, y] in physical pixels.
    pub tail_tip: [f32; 2],
    pub state: BubbleState,
    pub(crate) uniforms_buffer: wgpu::Buffer,
    pub(crate) bind_group: wgpu::BindGroup,
}

impl WgpuViewer {
    // ── Speech bubble helpers ─────────────────────────────────────────────────

    /// Returns true when at least one speech bubble is visible.
    pub fn has_visible_speech_bubbles(&self) -> bool {
        self.speech_bubbles
            .iter()
            .any(|b| !matches!(b.state, BubbleState::Dismissed))
    }

    /// Create or replace a speech bubble.
    /// `screen_rect` and `tail_tip` are in viewport-relative physical pixels.
    /// Returns the actual bubble height (physical pixels) after fitting to content.
    pub fn show_speech_bubble(
        &mut self,
        id: u32,
        pages: Vec<String>,
        mut screen_rect: [f32; 4],
        tail_tip: [f32; 2],
    ) -> f32 {
        // Measure each page's wrapped text height and use the tallest.
        const FONT_SIZE: f32 = 44.0;
        const LINE_HEIGHT: f32 = 56.0;
        const PADDING: f32 = 18.0;
        let wrap_w = (screen_rect[2] - PADDING * 2.0).max(1.0);
        let mut max_content_h: f32 = LINE_HEIGHT;
        for text in &pages {
            let mut buf = GlyphonBuffer::new(
                &mut self.glyphon_font_system,
                Metrics::new(FONT_SIZE, LINE_HEIGHT),
            );
            buf.set_size(&mut self.glyphon_font_system, Some(wrap_w), None);
            buf.set_text(
                &mut self.glyphon_font_system,
                text,
                Attrs::new()
                    .family(Family::Name("Zelda Sans"))
                    .color(GlyphonColor::rgb(30, 30, 35)),
                Shaping::Advanced,
            );
            buf.shape_until_scroll(&mut self.glyphon_font_system, false);
            if let Some(last) = buf.layout_runs().last() {
                max_content_h = max_content_h.max(last.line_top + last.line_height);
            }
        }
        screen_rect[3] = (max_content_h + PADDING * 2.0).ceil();

        // Remove any existing bubble with the same id.
        self.speech_bubbles.retain(|b| b.id != id);

        let uniforms_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("speech_bubble_uniforms"),
            size: std::mem::size_of::<SpeechBubbleUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("speech_bubble_bg"),
            layout: &self.speech_bubble_bind_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms_buffer.as_entire_binding(),
            }],
        });
        self.speech_bubbles.push(SpeechBubble {
            id,
            pages,
            current_page: 0,
            screen_rect,
            tail_tip,
            state: BubbleState::Active,
            uniforms_buffer,
            bind_group,
        });
        screen_rect[3]
    }

    /// Handle a click on bubble `id`.
    /// Returns `true` if the click caused a page advance (caller can re-render).
    /// Transitions to `Shaking` on the last page instead of immediately dismissing.
    pub fn click_speech_bubble(&mut self, id: u32) -> bool {
        if let Some(b) = self.speech_bubbles.iter_mut().find(|b| b.id == id) {
            match b.state {
                BubbleState::Active => {
                    if b.current_page + 1 < b.pages.len() {
                        b.current_page += 1;
                    } else {
                        b.state = BubbleState::Shaking { shake_t: 0.0 };
                    }
                    true
                }
                BubbleState::Shaking { .. } => {
                    // Click during shake -> dismiss immediately.
                    b.state = BubbleState::Dismissed;
                    true
                }
                BubbleState::Dismissed => false,
            }
        } else {
            false
        }
    }

    /// Update the screen rect and tail tip of an existing bubble without resetting its page or state.
    pub fn reposition_speech_bubble(&mut self, id: u32, screen_rect: [f32; 4], tail_tip: [f32; 2]) {
        if let Some(b) = self.speech_bubbles.iter_mut().find(|b| b.id == id) {
            b.screen_rect = screen_rect;
            b.tail_tip = tail_tip;
        }
    }

    /// Forcibly dismiss a bubble without shaking.
    pub fn dismiss_speech_bubble(&mut self, id: u32) {
        if let Some(b) = self.speech_bubbles.iter_mut().find(|b| b.id == id) {
            b.state = BubbleState::Dismissed;
        }
    }

    /// Advance shake animation; returns ids of bubbles that just finished dismissing.
    pub fn tick_speech_bubbles(&mut self, dt: f32) -> Vec<u32> {
        const SHAKE_DURATION: f32 = 0.65;
        let mut dismissed = Vec::new();
        for b in &mut self.speech_bubbles {
            match &mut b.state {
                BubbleState::Shaking { shake_t } => {
                    *shake_t += dt;
                    if *shake_t >= SHAKE_DURATION {
                        b.state = BubbleState::Dismissed;
                        dismissed.push(b.id);
                    }
                }
                // Bubble was click-dismissed during shake before the timer expired.
                // Must be collected here so the frontend receives the dismissed event.
                BubbleState::Dismissed => dismissed.push(b.id),
                BubbleState::Active => {}
            }
        }
        // Prune dismissed bubbles from the vec.
        self.speech_bubbles
            .retain(|b| !matches!(b.state, BubbleState::Dismissed));
        dismissed
    }

    pub(crate) fn speech_bubble_uniforms(
        b: &SpeechBubble,
        vx: f32,
        vy: f32,
        time_secs: f32,
    ) -> SpeechBubbleUniforms {
        const SHAKE_FREQ: f32 = 28.0;
        const SHAKE_AMP: f32 = 50.0;
        const SWAY_FREQ: f32 = 1.0;
        const SWAY_AMP: f32 = 8.0;
        let shake_x = match &b.state {
            BubbleState::Shaking { shake_t } => {
                let t = *shake_t;
                let decay = (-t * 6.0_f32).exp();
                (t * SHAKE_FREQ).sin() * SHAKE_AMP * decay
            }
            _ => 0.0,
        };
        let sway_x = (time_secs * SWAY_FREQ).sin() * SWAY_AMP;
        // Arc: bob upward (negative Y) when crossing center, down at the extremes.
        let sway_y = -(time_secs * SWAY_FREQ * 2.0).cos() * SWAY_AMP * 0.5;
        let shake_x = shake_x + sway_x;
        // Convert viewport-relative -> swapchain-absolute.
        let rect = [
            b.screen_rect[0] + vx,
            b.screen_rect[1] + vy + sway_y,
            b.screen_rect[2],
            b.screen_rect[3],
        ];
        // Clamp the tail so it protrudes at most MAX_TAIL_LEN beyond the
        // nearest body edge (must match the shader constant).
        const MAX_TAIL_LEN: f32 = 30.0;
        let raw_tip = [b.tail_tip[0] + vx, b.tail_tip[1] + vy];
        let cx = rect[0] + rect[2] * 0.5;
        let cy = rect[1] + rect[3] * 0.5;
        let dx = raw_tip[0] - cx;
        let dy = raw_tip[1] - cy;
        // Distance from center to the nearest edge along the direction to the tip.
        let half_w = rect[2] * 0.5;
        let half_h = rect[3] * 0.5;
        let dist = (dx * dx + dy * dy).sqrt();
        let tail_tip = if dist > 0.001 {
            let nx = dx / dist;
            let ny = dy / dist;
            // How far along (nx,ny) until we hit the box edge.
            let tx = if nx.abs() > 0.001 {
                half_w / nx.abs()
            } else {
                f32::MAX
            };
            let ty = if ny.abs() > 0.001 {
                half_h / ny.abs()
            } else {
                f32::MAX
            };
            let edge_dist = tx.min(ty);
            let max_dist = edge_dist + MAX_TAIL_LEN;
            if dist > max_dist {
                [cx + nx * max_dist, cy + ny * max_dist]
            } else {
                raw_tip
            }
        } else {
            raw_tip
        };
        SpeechBubbleUniforms {
            rect,
            tail_tip,
            shake_x,
            corner_r: 12.0,
            bg_color: [1.0, 1.0, 1.0, 0.96],
            border_color: [0.18, 0.18, 0.22, 1.0],
            border_w: 2.5,
            _pad: [0.0; 3],
        }
    }

    pub(crate) fn render_speech_bubbles(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        swap_view: &wgpu::TextureView,
    ) {
        if self.config.format != self.sdr_format {
            return;
        }

        // Advance animations.
        let now = std::time::Instant::now();
        let dt = now
            .duration_since(self.speech_bubble_last_tick)
            .as_secs_f32()
            .clamp(0.0, 0.1);
        self.speech_bubble_last_tick = now;

        // Advance shake; store dismissed ids for the event loop to emit.
        let dismissed = self.tick_speech_bubbles(dt);
        self.pending_dismissed_bubble_ids.extend(dismissed);

        let sw = self.config.width as f32;
        let sh = self.config.height as f32;
        let vx = self.viewport_x as f32;
        let vy = self.viewport_y as f32;

        // Update speech bubble glyphon viewport to swapchain dimensions.
        self.speech_bubble_glyphon_viewport.update(
            &self.queue,
            Resolution {
                width: self.config.width.max(1),
                height: self.config.height.max(1),
            },
        );

        // Upload uniforms and draw the bubble shapes.
        // We collect text areas separately; Glyphon needs all areas in one prepare call.
        let bubble_count = self.speech_bubbles.len();
        if bubble_count == 0 {
            return;
        }

        let time_secs = self.creation_instant.elapsed().as_secs_f32();

        for i in 0..bubble_count {
            let uniforms = Self::speech_bubble_uniforms(&self.speech_bubbles[i], vx, vy, time_secs);
            self.queue.write_buffer(
                &self.speech_bubbles[i].uniforms_buffer,
                0,
                bytemuck::bytes_of(&uniforms),
            );

            // Scissor to the AABB enclosing both the body and the tail tip,
            // with a small margin for anti-aliasing and shake/sway.
            let margin = 4.0;
            let body_left = uniforms.rect[0] + uniforms.shake_x.min(0.0);
            let body_right = uniforms.rect[0] + uniforms.rect[2] + uniforms.shake_x.max(0.0);
            let body_top = uniforms.rect[1];
            let body_bottom = uniforms.rect[1] + uniforms.rect[3];
            let rx = (body_left.min(uniforms.tail_tip[0]) - margin).max(0.0);
            let ry = (body_top.min(uniforms.tail_tip[1]) - margin).max(0.0);
            let rr = (body_right.max(uniforms.tail_tip[0]) + margin).min(sw);
            let rb = (body_bottom.max(uniforms.tail_tip[1]) + margin).min(sh);
            let rw = rr - rx;
            let rh = rb - ry;

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("speech_bubble"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: swap_view,
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
                pass.set_scissor_rect(rx as u32, ry as u32, rw.max(1.0) as u32, rh.max(1.0) as u32);
                pass.set_pipeline(&self.speech_bubble_pipeline);
                pass.set_bind_group(0, &self.speech_bubbles[i].bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        // Render text for each bubble via Glyphon.
        let font_size = 44.0_f32;
        let line_height = 56.0_f32;
        let padding = 18.0_f32;

        let mut buffers: Vec<GlyphonBuffer> = Vec::new();
        let mut text_areas: Vec<TextArea<'_>> = Vec::new();

        // Collect per-bubble offsets so text moves with the bubble during shake/sway.
        let offsets: Vec<(f32, f32)> = self
            .speech_bubbles
            .iter()
            .map(|b| {
                let u = Self::speech_bubble_uniforms(b, vx, vy, time_secs);
                // shake_x includes sway_x; sway_y is baked into rect.y, recover it
                // as the difference from the base position.
                let sway_y = u.rect[1] - (b.screen_rect[1] + vy);
                (u.shake_x, sway_y)
            })
            .collect();

        for b in &self.speech_bubbles {
            let text = b
                .pages
                .get(b.current_page)
                .map(|s| s.as_str())
                .unwrap_or("");
            let wrap_w = (b.screen_rect[2] - padding * 2.0).max(1.0);

            let mut buf = GlyphonBuffer::new(
                &mut self.glyphon_font_system,
                Metrics::new(font_size, line_height),
            );
            buf.set_size(
                &mut self.glyphon_font_system,
                Some(wrap_w),
                Some(b.screen_rect[3] - padding * 2.0),
            );
            buf.set_text(
                &mut self.glyphon_font_system,
                text,
                Attrs::new()
                    .family(Family::Name("Zelda Sans"))
                    .color(GlyphonColor::rgb(30, 30, 35)),
                Shaping::Advanced,
            );
            buf.shape_until_scroll(&mut self.glyphon_font_system, false);
            buffers.push(buf);
        }

        // Build TextArea slice (must be done in a second pass because buffers is borrowed).
        for (i, b) in self.speech_bubbles.iter().enumerate() {
            let (sx, sy) = offsets[i];
            let abs_x = b.screen_rect[0] + vx + sx;
            let abs_y = b.screen_rect[1] + vy + sy;
            text_areas.push(TextArea {
                buffer: &buffers[i],
                left: abs_x + padding,
                top: abs_y + padding,
                scale: 1.0,
                bounds: TextBounds {
                    left: (abs_x + padding) as i32,
                    top: (abs_y + padding) as i32,
                    right: (abs_x + b.screen_rect[2] - padding) as i32,
                    bottom: (abs_y + b.screen_rect[3] - padding) as i32,
                },
                default_color: GlyphonColor::rgb(30, 30, 35),
                custom_glyphs: &[],
            });
        }

        let _ = self.speech_bubble_text_renderer.prepare(
            &self.device,
            &self.queue,
            &mut self.glyphon_font_system,
            &mut self.speech_bubble_glyphon_atlas,
            &self.speech_bubble_glyphon_viewport,
            text_areas,
            &mut self.glyphon_swash_cache,
        );

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("speech_bubble_text"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: swap_view,
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
            let _ = self.speech_bubble_text_renderer.render(
                &self.speech_bubble_glyphon_atlas,
                &self.speech_bubble_glyphon_viewport,
                &mut pass,
            );
        }
        self.speech_bubble_glyphon_atlas.trim();
    }
}
