//! Draw helper methods: indexed mesh draws, preview, selection overlay, gizmo, ping, avatar rendering.

use super::*;

impl WgpuViewer {
    /// Draw only the opaque index range (or full range when un-partitioned — shader discards glass).
    pub(crate) fn draw_indexed_mesh(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.opaque_chunked {
            for ch in self.opaque_chunks.values() {
                if ch.opaque_index_count == 0 {
                    continue;
                }
                pass.set_vertex_buffer(0, ch.vertex_buffer.slice(..));
                pass.set_index_buffer(ch.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..ch.opaque_index_count, 0, 0..1);
            }
            return;
        }
        if let (Some(vb), Some(ib)) = (&self.vertex_buffer, &self.index_buffer) {
            // When un-partitioned (opaque_index_split == 0), draw all; shader discards glass.
            let count = if self.opaque_index_split > 0 {
                self.opaque_index_split
            } else {
                self.index_count
            };
            if count > 0 {
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..count, 0, 0..1);
            }
        }
    }

    /// Draw only the transparent index range for the OIT accumulation pass.
    /// When un-partitioned (GPU path), draws full range — shader discards opaque.
    pub(crate) fn draw_indexed_oit_mesh(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.opaque_chunked {
            for ch in self.opaque_chunks.values() {
                if ch.transparent_index_count == 0 {
                    continue;
                }
                pass.set_vertex_buffer(0, ch.vertex_buffer.slice(..));
                pass.set_index_buffer(ch.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                let start = if ch.partitioned {
                    ch.opaque_index_count
                } else {
                    0
                };
                pass.draw_indexed(start..start + ch.transparent_index_count, 0, 0..1);
            }
            return;
        }
        if let (Some(vb), Some(ib)) = (&self.vertex_buffer, &self.index_buffer) {
            let trans_count = self.index_count.saturating_sub(self.opaque_index_split);
            if trans_count > 0 {
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(self.opaque_index_split..self.index_count, 0, 0..1);
            }
        }
    }

    /// Draw the full index range (opaque + transparent). Used by the shadow pass
    /// where glass needs `glass_shadow_push` applied.
    pub(crate) fn draw_indexed_mesh_all(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.opaque_chunked {
            for ch in self.opaque_chunks.values() {
                let total = ch.opaque_index_count + ch.transparent_index_count;
                if total == 0 {
                    continue;
                }
                pass.set_vertex_buffer(0, ch.vertex_buffer.slice(..));
                pass.set_index_buffer(ch.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..total, 0, 0..1);
            }
            return;
        }
        if let (Some(vb), Some(ib)) = (&self.vertex_buffer, &self.index_buffer) {
            if self.index_count > 0 {
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.index_count, 0, 0..1);
            }
        }
    }

    pub(crate) fn draw_indexed_preview(&self, pass: &mut wgpu::RenderPass<'_>) {
        // GPU-instanced solid cubes
        if let (Some(pvb), Some(pib), Some(ibuf)) = (
            &self.preview_solid_proto_vb,
            &self.preview_solid_proto_ib,
            &self.preview_solid_instance_buf,
        ) {
            if self.preview_solid_instance_count > 0 && self.preview_solid_proto_idx_count > 0 {
                pass.set_vertex_buffer(0, pvb.slice(..));
                pass.set_vertex_buffer(1, ibuf.slice(..));
                pass.set_index_buffer(pib.slice(..), wgpu::IndexFormat::Uint32);
                pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
                pass.set_pipeline(&self.pipeline_preview_inst_occluded);
                pass.draw_indexed(
                    0..self.preview_solid_proto_idx_count,
                    0,
                    0..self.preview_solid_instance_count,
                );
                pass.set_pipeline(&self.pipeline_preview_inst_front);
                pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
                pass.draw_indexed(
                    0..self.preview_solid_proto_idx_count,
                    0,
                    0..self.preview_solid_instance_count,
                );
            }
        }
        // GPU-instanced wireframe
        if let (Some(pvb), Some(pib), Some(ibuf)) = (
            &self.preview_wire_proto_vb,
            &self.preview_wire_proto_ib,
            &self.preview_wire_instance_buf,
        ) {
            if self.preview_wire_instance_count > 0 && self.preview_wire_proto_idx_count > 0 {
                pass.set_vertex_buffer(0, pvb.slice(..));
                pass.set_vertex_buffer(1, ibuf.slice(..));
                pass.set_index_buffer(pib.slice(..), wgpu::IndexFormat::Uint32);
                pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
                pass.set_pipeline(&self.pipeline_preview_inst_front_wire);
                pass.draw_indexed(
                    0..self.preview_wire_proto_idx_count,
                    0,
                    0..self.preview_wire_instance_count,
                );
            }
        }
        // Non-instanced extras (gizmos, polygon markers)
        if let (Some(vb), Some(ib)) = (&self.preview_vertex_buffer, &self.preview_index_buffer) {
            if self.preview_index_count > 0 {
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
                pass.set_pipeline(&self.pipeline_preview_occluded);
                pass.draw_indexed(0..self.preview_index_count, 0, 0..1);
                pass.set_pipeline(&self.pipeline_preview_front);
                pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
                pass.draw_indexed(0..self.preview_index_count, 0, 0..1);
            }
        }
        if let (Some(wvb), Some(wib)) = (
            &self.preview_wire_vertex_buffer,
            &self.preview_wire_index_buffer,
        ) {
            if self.preview_wire_index_count > 0 {
                pass.set_vertex_buffer(0, wvb.slice(..));
                pass.set_index_buffer(wib.slice(..), wgpu::IndexFormat::Uint32);
                pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
                pass.set_pipeline(&self.pipeline_preview_front_wire);
                pass.draw_indexed(0..self.preview_wire_index_count, 0, 0..1);
            }
        }
        // Lit generator preview solid cubes (opaque, self-shadowing)
        if let (Some(pvb), Some(pib), Some(ibuf)) = (
            &self.gen_preview_solid_proto_vb,
            &self.gen_preview_solid_proto_ib,
            &self.gen_preview_solid_instance_buf,
        ) {
            if self.gen_preview_solid_instance_count > 0
                && self.gen_preview_solid_proto_idx_count > 0
            {
                pass.set_vertex_buffer(0, pvb.slice(..));
                pass.set_vertex_buffer(1, ibuf.slice(..));
                pass.set_index_buffer(pib.slice(..), wgpu::IndexFormat::Uint32);
                pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
                pass.set_pipeline(&self.pipeline_gen_preview_inst_occluded);
                pass.draw_indexed(
                    0..self.gen_preview_solid_proto_idx_count,
                    0,
                    0..self.gen_preview_solid_instance_count,
                );
                pass.set_pipeline(&self.pipeline_gen_preview_inst_front);
                pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
                pass.draw_indexed(
                    0..self.gen_preview_solid_proto_idx_count,
                    0,
                    0..self.gen_preview_solid_instance_count,
                );
            }
        }
        // Lit generator preview wireframe
        if let (Some(pvb), Some(pib), Some(ibuf)) = (
            &self.gen_preview_wire_proto_vb,
            &self.gen_preview_wire_proto_ib,
            &self.gen_preview_wire_instance_buf,
        ) {
            if self.gen_preview_wire_instance_count > 0 && self.gen_preview_wire_proto_idx_count > 0
            {
                pass.set_vertex_buffer(0, pvb.slice(..));
                pass.set_vertex_buffer(1, ibuf.slice(..));
                pass.set_index_buffer(pib.slice(..), wgpu::IndexFormat::Uint32);
                pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
                pass.set_pipeline(&self.pipeline_gen_preview_inst_front_wire);
                pass.draw_indexed(
                    0..self.gen_preview_wire_proto_idx_count,
                    0,
                    0..self.gen_preview_wire_instance_count,
                );
            }
        }
    }

    pub(crate) fn draw_indexed_selection_overlay_solid(&self, pass: &mut wgpu::RenderPass<'_>) {
        if let (Some(vb), Some(ib)) = (
            &self.selection_overlay_vertex_buffer,
            &self.selection_overlay_index_buffer,
        ) {
            if self.selection_overlay_index_count > 0 {
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
                pass.set_pipeline(&self.pipeline_preview_occluded);
                pass.draw_indexed(0..self.selection_overlay_index_count, 0, 0..1);
                pass.set_pipeline(&self.pipeline_preview_front);
                pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
                pass.draw_indexed(0..self.selection_overlay_index_count, 0, 0..1);
            }
        }
    }

    pub(crate) fn draw_selection_overlay_lines(&self, pass: &mut wgpu::RenderPass<'_>) {
        let Some(ref vb) = self.selection_overlay_line_vertex_buffer else {
            return;
        };
        if self.selection_overlay_line_vertex_count < 2 {
            return;
        }
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
        pass.set_pipeline(&self.pipeline_collab_lines_occluded);
        pass.draw(0..self.selection_overlay_line_vertex_count, 0..1);
        pass.set_pipeline(&self.pipeline_collab_lines_front);
        pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
        pass.draw(0..self.selection_overlay_line_vertex_count, 0..1);
    }

    pub(crate) fn draw_grid_border_lines(&self, pass: &mut wgpu::RenderPass<'_>) {
        let Some(ref vb) = self.grid_border_line_vertex_buffer else {
            return;
        };
        let Some(ref ib) = self.grid_border_line_index_buffer else {
            return;
        };
        if self.grid_border_line_index_count < 2 {
            return;
        }
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
        pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
        pass.set_pipeline(&self.pipeline_grid_border_lines);
        pass.draw_indexed(0..self.grid_border_line_index_count, 0, 0..1);
    }

    pub(crate) fn draw_gizmo(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
        if let Some(ref vb) = self.gizmo_line_vertex_buffer {
            if self.gizmo_line_vertex_count >= 3 {
                pass.set_vertex_buffer(0, vb.slice(..));
                if self.gizmo_on_top {
                    pass.set_pipeline(&self.pipeline_gizmo_lines_always);
                    pass.draw(0..self.gizmo_line_vertex_count, 0..1);
                } else {
                    pass.set_pipeline(&self.pipeline_gizmo_lines_occluded);
                    pass.draw(0..self.gizmo_line_vertex_count, 0..1);
                    pass.set_pipeline(&self.pipeline_gizmo_lines_front);
                    pass.draw(0..self.gizmo_line_vertex_count, 0..1);
                }
            }
        }
        if let Some(ref tb) = self.gizmo_tri_vertex_buffer {
            if self.gizmo_tri_vertex_count >= 3 {
                pass.set_vertex_buffer(0, tb.slice(..));
                if self.gizmo_on_top {
                    pass.set_pipeline(&self.pipeline_gizmo_tris_always);
                    pass.draw(0..self.gizmo_tri_vertex_count, 0..1);
                } else {
                    pass.set_pipeline(&self.pipeline_gizmo_tris_occluded);
                    pass.draw(0..self.gizmo_tri_vertex_count, 0..1);
                    pass.set_pipeline(&self.pipeline_gizmo_tris_front);
                    pass.draw(0..self.gizmo_tri_vertex_count, 0..1);
                }
            }
        }
    }

    pub(crate) fn draw_indexed_ping(&self, pass: &mut wgpu::RenderPass<'_>) {
        if let (Some(vb), Some(ib)) = (&self.ping_vertex_buffer, &self.ping_index_buffer) {
            if self.ping_index_count > 0 {
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
                pass.set_pipeline(&self.pipeline_preview_occluded);
                pass.draw_indexed(0..self.ping_index_count, 0, 0..1);
                pass.set_pipeline(&self.pipeline_preview_front);
                pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
                pass.draw_indexed(0..self.ping_index_count, 0, 0..1);
            }
        }
        if let (Some(wvb), Some(wib)) =
            (&self.ping_wire_vertex_buffer, &self.ping_wire_index_buffer)
        {
            if self.ping_wire_index_count > 0 {
                pass.set_vertex_buffer(0, wvb.slice(..));
                pass.set_index_buffer(wib.slice(..), wgpu::IndexFormat::Uint32);
                pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
                pass.set_pipeline(&self.pipeline_preview_front_wire);
                pass.draw_indexed(0..self.ping_wire_index_count, 0, 0..1);
            }
        }
    }

    pub(crate) fn draw_ping_wave_lines(&self, pass: &mut wgpu::RenderPass<'_>) {
        let Some(ref vb) = self.ping_wave_line_vertex_buffer else {
            return;
        };
        if self.ping_wave_line_vertex_count < 2 {
            return;
        }
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
        pass.set_pipeline(&self.pipeline_collab_lines_occluded);
        pass.draw(0..self.ping_wave_line_vertex_count, 0..1);
        pass.set_pipeline(&self.pipeline_collab_lines_front);
        pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
        pass.draw(0..self.ping_wave_line_vertex_count, 0..1);
    }
}
