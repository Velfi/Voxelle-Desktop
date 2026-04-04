//! Avatar system: per-peer collab voxel avatars (mesh cache, peer entries, rendering).

use super::*;

/// Shared GPU mesh for one named avatar, stored in local space (centered, facing +Z).
pub struct AvatarMeshData {
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) index_buffer: wgpu::Buffer,
    pub(crate) index_count: u32,
    /// Translation to apply before scale: `-bounds.center()`.
    pub center_offset: Vec3,
    /// Uniform scale so the avatar fits in a ~1.5-unit cube.
    pub scale: f32,
}

/// Per-peer GPU draw state for collab avatar rendering.
pub struct AvatarPeerEntry {
    pub peer_id: u32,
    /// Key into `WgpuViewer::avatar_mesh_cache`; `""` = default glow dot.
    pub mesh_name: String,
    pub(crate) uniforms_buf: wgpu::Buffer,
    pub(crate) bind_group: wgpu::BindGroup,
}

impl WgpuViewer {
    /// Store a decoded voxel mesh as a named avatar in the shared cache.
    /// No-op if the name is already cached.  Pass `name = ""` for the default glow dot.
    /// Store a decoded voxel mesh as a named avatar.  `centroid` is the world-space point that
    /// will be placed at the peer's eye position; `scale` maps mesh units to world units.
    /// No-op if the name is already cached.
    pub fn cache_avatar_mesh(
        &mut self,
        name: String,
        mesh: &MeshBuffers,
        centroid: Vec3,
        scale: f32,
    ) {
        if self.avatar_mesh_cache.contains_key(&name) {
            return;
        }
        let interleaved = Self::interleaved_from_mesh(mesh);
        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("avatar_vtx"),
                contents: bytemuck::cast_slice(&interleaved),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("avatar_idx"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        self.avatar_mesh_cache.insert(
            name,
            AvatarMeshData {
                vertex_buffer,
                index_buffer,
                index_count: mesh.indices.len() as u32,
                center_offset: -centroid,
                scale,
            },
        );
    }

    /// Update or insert a peer's avatar draw state.  Called every frame from
    /// `sync_collab_peer_avatars` with the pre-computed MVP, tint, and rotation.
    /// `rot_cols` is the upper-left 3×3 of the model rotation matrix, column-major,
    /// each column stored as `[x, y, z, 0.0]` to match WGSL `mat3x3` std140 layout.
    pub fn update_avatar_peer(
        &mut self,
        peer_id: u32,
        mesh_name: String,
        mvp: [[f32; 4]; 4],
        tint: [f32; 3],
        rot_cols: [[f32; 4]; 3],
    ) {
        let uniforms = AvatarUniforms {
            mvp,
            light_dir: [0.6, 0.8, 0.5, 0.0],
            color_tint: [tint[0], tint[1], tint[2], 0.0],
            ambient: 0.55,
            sun: 0.7,
            _pad: [0.0; 2],
            normal_mat: rot_cols,
        };
        if let Some(idx) = self.avatar_peers.iter().position(|p| p.peer_id == peer_id) {
            // Reuse existing bind group; just update mesh_name and uniforms.
            self.avatar_peers[idx].mesh_name = mesh_name;
            self.queue.write_buffer(
                &self.avatar_peers[idx].uniforms_buf,
                0,
                bytemuck::bytes_of(&uniforms),
            );
        } else {
            let uniforms_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("avatar_peer_uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("avatar_peer_bg"),
                layout: &self.avatar_bind_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buf.as_entire_binding(),
                }],
            });
            self.avatar_peers.push(AvatarPeerEntry {
                peer_id,
                mesh_name,
                uniforms_buf,
                bind_group,
            });
        }
    }

    /// Remove a peer's avatar entry (called when they leave the session).
    pub fn remove_avatar_peer(&mut self, peer_id: u32) {
        self.avatar_peers.retain(|p| p.peer_id != peer_id);
    }

    /// Remove all peer avatar entries (called on collab disconnect).
    pub fn clear_avatar_peers(&mut self) {
        self.avatar_peers.clear();
    }

    pub(crate) fn render_avatars(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.avatar_peers.is_empty() {
            return;
        }
        pass.set_pipeline(&self.pipeline_avatar);
        for peer in &self.avatar_peers {
            let mesh = self
                .avatar_mesh_cache
                .get(&peer.mesh_name)
                .or_else(|| self.avatar_mesh_cache.get(""));
            let Some(mesh) = mesh else { continue };
            if mesh.index_count == 0 {
                continue;
            }
            pass.set_bind_group(0, &peer.bind_group, &[]);
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }
    }
}
