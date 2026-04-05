//! Mesh upload methods: CPU mesh building, GPU buffer creation, preview/overlay/gizmo uploads.

use super::*;

/// Timings from [`WgpuViewer::remesh_opaque_chunks`] (incremental CPU chunked path).
#[derive(Clone, Debug, Default)]
pub struct RemeshOpaquePerf {
    /// Cold [`greedy_mesh::SpatialMeshCache::from_voxels`] when cache was missing.
    pub buckets_ms: f64,
    /// Wall time in greedy phase (`greedy_gpu_ms` + `greedy_cpu_ms` when GPU chunk path is enabled).
    pub greedy_ms: f64,
    /// [`WgpuViewer::run_mesh_greedy_compute_with_brick`] for dirty chunks (subset of `greedy_ms`).
    pub greedy_gpu_ms: f64,
    /// CPU [`greedy_mesh::mesh_buffers_for_chunk_key`] (subset of `greedy_ms`).
    pub greedy_cpu_ms: f64,
    pub chunk_buffers_ms: f64,
    /// Full [`WgpuViewer::upload_cpu_mesh_chunked_full`] when chunk origin drifted or equivalent.
    pub full_chunked_rebuild_ms: f64,
}

/// CPU-built opaque mesh for [`WgpuViewer::upload_prepared_opaque`] (main thread uploads buffers only).
pub(crate) enum PreparedOpaqueUpload {
    Empty,
    Single(MeshBuffers),
    Chunked {
        chunk_origin: IVec3,
        meshes: BTreeMap<ChunkKey, MeshBuffers>,
        spatial_cache: greedy_mesh::SpatialMeshCache,
    },
    /// Scene set up for chunked rendering but meshes arrive later via background streaming.
    ChunkedDeferred {
        chunk_origin: IVec3,
    },
}

/// CPU-only phase of [`WgpuViewer::rebuild_mesh_gpu_greedy`] (safe to run off the main thread).
pub(crate) enum PreparedGreedyRebuild {
    /// Same as an empty voxel file: clear draw buffers.
    NoVoxels,
    /// Nothing visible after filtering.
    AllHidden,
    /// Upload via [`WgpuViewer::upload_prepared_opaque`].
    Opaque {
        opaque: PreparedOpaqueUpload,
        bounds: MeshBounds,
        last_route: String,
    },
    /// GPU slice pack done; main thread runs compute + buffer copy.
    GpuGreedyPack {
        bounds: MeshBounds,
        headers: Vec<greedy_mesh::GpuSliceHeader>,
        bits: Vec<u32>,
        fallback_voxels: Vec<Voxel>,
        fallback_objects: Vec<SceneObject>,
    },
}

/// GPU buffers for one spatial chunk of opaque greedy mesh.
pub(crate) struct OpaqueChunkDraw {
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) index_buffer: wgpu::Buffer,
    pub(crate) opaque_index_count: u32,
    pub(crate) transparent_index_count: u32,
    /// When false (GPU mesh path), indices are not partitioned: both opaque and OIT
    /// passes draw `0..total` and rely on shader-side `mat_kind` discard.
    pub(crate) partitioned: bool,
}

pub(crate) fn cpu_mesh_fallback_prepare(
    voxels: &[Voxel],
    objects: &[SceneObject],
    grid_size: i32,
    chunk_progress: Option<&(dyn Fn(f32, u32, u32) + Sync)>,
) -> Result<(PreparedOpaqueUpload, MeshBounds, String), String> {
    let default_objs = crate::voxelle::default_scene_objects();
    let objs: &[SceneObject] = if objects.is_empty() {
        default_objs.as_slice()
    } else {
        objects
    };
    let work = crate::voxelle::scene::visible_voxels_for_meshing(voxels, objs);
    if work.is_empty() {
        return Ok((
            PreparedOpaqueUpload::Empty,
            greedy_mesh::mesh_bounds_for_cube_side(grid_size),
            "cpu_empty".to_string(),
        ));
    }
    let bounds = greedy_mesh::mesh_bounds_from_voxels_world(&work, objs)
        .or_else(|| greedy_mesh::mesh_bounds_from_voxels(&work))
        .unwrap_or_else(|| greedy_mesh::mesh_bounds_for_cube_side(grid_size));
    let multi = work
        .iter()
        .map(|v| v.object_id)
        .collect::<HashSet<_>>()
        .len()
        > 1;
    if work.len() >= greedy_mesh::CHUNKED_CPU_MESH_MIN_VOXELS && !multi {
        let Some((origin, meshes, spatial_cache)) =
            greedy_mesh::build_chunk_meshes_and_spatial_cache(
                &work,
                greedy_mesh::SPATIAL_CHUNK_SIZE,
                |frac, done, total| {
                    if let Some(cb) = chunk_progress {
                        cb(frac, done, total);
                    }
                },
            )
        else {
            return Ok((
                PreparedOpaqueUpload::Empty,
                bounds,
                "cpu_chunked_none".to_string(),
            ));
        };
        if meshes.is_empty() {
            return Ok((
                PreparedOpaqueUpload::Empty,
                bounds,
                "cpu_chunked_empty".to_string(),
            ));
        }
        let chunk_origin = IVec3::new(origin.0, origin.1, origin.2);
        Ok((
            PreparedOpaqueUpload::Chunked {
                chunk_origin,
                meshes,
                spatial_cache,
            },
            bounds,
            "cpu_chunked".to_string(),
        ))
    } else {
        let (mesh, _) = greedy_mesh::build_greedy_mesh(voxels, objs);
        Ok((
            PreparedOpaqueUpload::Single(mesh),
            bounds,
            "cpu".to_string(),
        ))
    }
}

/// CPU work for a full greedy mesh rebuild (background thread + [`WgpuViewer::rebuild_mesh_gpu_greedy`]).
/// `chunk_progress` is invoked while building large chunked meshes (fraction, completed buckets, total buckets).
pub(crate) fn compute_greedy_rebuild_cpu(
    voxels: &[Voxel],
    objects: &[SceneObject],
    grid_size: i32,
    chunk_progress: Option<&(dyn Fn(f32, u32, u32) + Sync)>,
) -> Result<PreparedGreedyRebuild, String> {
    if voxels.is_empty() {
        return Ok(PreparedGreedyRebuild::NoVoxels);
    }
    let default_objs = crate::voxelle::default_scene_objects();
    let objs: &[SceneObject] = if objects.is_empty() {
        default_objs.as_slice()
    } else {
        objects
    };
    let work = crate::voxelle::scene::visible_voxels_for_meshing(voxels, objs);
    if work.is_empty() {
        return Ok(PreparedGreedyRebuild::AllHidden);
    }
    let multi = work
        .iter()
        .map(|v| v.object_id)
        .collect::<HashSet<_>>()
        .len()
        > 1;
    let transformed = objs.iter().any(|o| {
        o.visible
            && (o.translation != [0.0, 0.0, 0.0]
                || o.rotation[0].abs() + o.rotation[1].abs() + o.rotation[2].abs() > 1e-5
                || o.rotation[3] < 0.9999
                || o.scale != [1.0, 1.0, 1.0])
    });
    if multi || transformed {
        let (mesh, _) = greedy_mesh::build_greedy_mesh(voxels, objs);
        let bounds = greedy_mesh::mesh_bounds_from_voxels_world(voxels, objs)
            .or_else(|| greedy_mesh::mesh_bounds_from_voxels(voxels))
            .ok_or_else(|| "mesh bounds".to_string())?;
        let last_route = if multi {
            "cpu_multi_object"
        } else {
            "cpu_object_transform"
        }
        .to_string();
        return Ok(PreparedGreedyRebuild::Opaque {
            opaque: PreparedOpaqueUpload::Single(mesh),
            bounds,
            last_route,
        });
    }
    let bounds = greedy_mesh::mesh_bounds_from_voxels_world(&work, objs)
        .or_else(|| greedy_mesh::mesh_bounds_from_voxels(&work))
        .ok_or("mesh bounds")?;
    if std::env::var("VOXELLE_CPU_MESH").is_ok() {
        let (opaque, b, route) =
            cpu_mesh_fallback_prepare(voxels, objs, grid_size, chunk_progress)?;
        return Ok(PreparedGreedyRebuild::Opaque {
            opaque,
            bounds: b,
            last_route: route,
        });
    }
    let map = greedy_mesh::voxel_map(voxels);
    let (headers, bits) = match greedy_mesh::pack_gpu_greedy_slices(&map, &work) {
        Ok(x) => x,
        Err(()) => {
            let (opaque, b, route) =
                cpu_mesh_fallback_prepare(voxels, objs, grid_size, chunk_progress)?;
            return Ok(PreparedGreedyRebuild::Opaque {
                opaque,
                bounds: b,
                last_route: route,
            });
        }
    };
    if headers.is_empty() {
        return Ok(PreparedGreedyRebuild::Opaque {
            opaque: PreparedOpaqueUpload::Empty,
            bounds,
            last_route: "gpu_no_headers".to_string(),
        });
    }
    Ok(PreparedGreedyRebuild::GpuGreedyPack {
        bounds,
        headers,
        bits,
        fallback_voxels: voxels.to_vec(),
        fallback_objects: objects.to_vec(),
    })
}

impl WgpuViewer {
    pub(crate) fn interleaved_from_mesh(mesh: &MeshBuffers) -> Vec<f32> {
        let n = mesh.positions.len() / 3;
        let mut interleaved: Vec<f32> = Vec::with_capacity(n * 14);
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
        }
        interleaved
    }

    pub(crate) fn opaque_draw_from_mesh(&self, mesh: &MeshBuffers, opaque_split: u32) -> OpaqueChunkDraw {
        let interleaved = Self::interleaved_from_mesh(mesh);
        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vtx_chunk"),
                contents: bytemuck::cast_slice(&interleaved),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("idx_chunk"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            });
        OpaqueChunkDraw {
            vertex_buffer,
            index_buffer,
            opaque_index_count: opaque_split,
            transparent_index_count: (mesh.indices.len() as u32).saturating_sub(opaque_split),
            partitioned: true,
        }
    }

    /// If existing chunk buffers are large enough, overwrite with [`queue::write_buffer`]; else allocate new.
    pub(crate) fn upload_or_replace_chunk_mesh(
        &mut self,
        key: ChunkKey,
        mesh: &MeshBuffers,
        opaque_split: u32,
    ) {
        let n = mesh.positions.len() / 3;
        let vtx_need = (n as u64).saturating_mul(OPAQUE_VERTEX_STRIDE);
        let idx_need = (mesh.indices.len() * 4) as u64;
        let can_reuse = self.opaque_chunks.get(&key).map(|d| {
            d.vertex_buffer
                .usage()
                .contains(wgpu::BufferUsages::COPY_DST)
                && d.index_buffer
                    .usage()
                    .contains(wgpu::BufferUsages::COPY_DST)
                && d.vertex_buffer.size() >= vtx_need
                && d.index_buffer.size() >= idx_need
        }) == Some(true);
        if can_reuse {
            let interleaved = Self::interleaved_from_mesh(mesh);
            let draw = self.opaque_chunks.get_mut(&key).expect("reuse");
            self.queue
                .write_buffer(&draw.vertex_buffer, 0, bytemuck::cast_slice(&interleaved));
            self.queue
                .write_buffer(&draw.index_buffer, 0, bytemuck::cast_slice(&mesh.indices));
            draw.opaque_index_count = opaque_split;
            draw.transparent_index_count = (mesh.indices.len() as u32).saturating_sub(opaque_split);
            draw.partitioned = true;
        } else {
            self.opaque_chunks
                .insert(key, self.opaque_draw_from_mesh(mesh, opaque_split));
        }
    }

    pub fn upload_mesh(&mut self, mesh: &mut MeshBuffers) {
        self.opaque_chunked = false;
        self.opaque_chunks.clear();
        self.spatial_mesh_cache = None;
        let split = greedy_mesh::partition_indices_by_transparency(mesh);
        let interleaved = Self::interleaved_from_mesh(mesh);
        self.vertex_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("vtx"),
                contents: bytemuck::cast_slice(&interleaved),
                usage: wgpu::BufferUsages::VERTEX,
            },
        ));
        self.index_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("idx"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            },
        ));
        self.index_count = mesh.indices.len() as u32;
        self.opaque_index_split = split;
    }

    pub(crate) fn has_pending_chunk_uploads(&self) -> bool {
        !self.pending_chunk_uploads.is_empty()
    }

    /// Move chunks from an external inbox (background thread) into the pending upload queue.
    pub(crate) fn enqueue_chunk_uploads(&mut self, inbox: &mut VecDeque<(ChunkKey, MeshBuffers)>) {
        self.pending_chunk_uploads.extend(inbox.drain(..));
    }

    pub(crate) fn has_spatial_mesh_cache(&self) -> bool {
        self.spatial_mesh_cache.is_some()
    }

    pub(crate) fn set_spatial_mesh_cache(&mut self, cache: greedy_mesh::SpatialMeshCache) {
        self.spatial_mesh_cache = Some(cache);
    }

    /// Upload queued chunks until `budget` elapses. Returns `true` if more remain.
    pub(crate) fn drain_pending_chunk_uploads(&mut self, budget: Duration) -> bool {
        if self.pending_chunk_uploads.is_empty() {
            return false;
        }
        let start = Instant::now();
        let mut uploaded = 0u32;
        while let Some((key, mut mesh)) = self.pending_chunk_uploads.pop_front() {
            if !self.opaque_chunks.contains_key(&key) {
                let split = greedy_mesh::partition_indices_by_transparency(&mut mesh);
                self.opaque_chunks
                    .insert(key, self.opaque_draw_from_mesh(&mesh, split));
            }
            uploaded += 1;
            if start.elapsed() >= budget {
                break;
            }
        }
        let remaining = self.pending_chunk_uploads.len();
        if remaining == 0 {
            log::info!(
                target: "voxelle_load",
                "progressive upload: done ({uploaded} chunks this frame)"
            );
        }
        remaining > 0
    }

    /// Full CPU chunked mesh upload (all spatial chunks). Used on load and when chunk origin shifts.
    pub fn upload_cpu_mesh_chunked_full<R: Runtime>(
        &mut self,
        voxels: &[Voxel],
        work_progress: Option<&AppHandle<R>>,
    ) {
        self.vertex_buffer = None;
        self.index_buffer = None;
        self.index_count = 0;
        self.opaque_index_split = 0;
        self.opaque_chunks.clear();
        if let Some(app) = work_progress {
            crate::emit_work_progress(app, 0.38, "Indexing voxels for mesh…");
        }
        let last_permille = AtomicU32::new(0);
        let Some((origin, meshes, spatial_cache)) =
            greedy_mesh::build_chunk_meshes_and_spatial_cache(
                voxels,
                greedy_mesh::SPATIAL_CHUNK_SIZE,
                |frac: f32, done: u32, total: u32| {
                    if let Some(app) = work_progress {
                        let permille = (frac * 1000.0).min(1000.0) as u32;
                        let prev = last_permille.load(Ordering::Relaxed);
                        if permille.saturating_sub(prev) >= 40 || done == total {
                            last_permille.store(permille, Ordering::Relaxed);
                            crate::emit_work_progress(
                                app,
                                0.38 + 0.35 * frac,
                                format!("Mesh chunks {done}/{total}…"),
                            );
                        }
                    }
                },
            )
        else {
            self.opaque_chunked = false;
            self.spatial_mesh_cache = None;
            return;
        };
        self.chunk_grid_origin = IVec3::new(origin.0, origin.1, origin.2);
        if meshes.is_empty() {
            self.opaque_chunked = false;
            self.spatial_mesh_cache = None;
            return;
        }
        self.opaque_chunked = true;
        for (i, (key, mut mesh)) in meshes.into_iter().enumerate() {
            if i > 0 && i % 4 == 0 {
                std::thread::yield_now();
            }
            let split = greedy_mesh::partition_indices_by_transparency(&mut mesh);
            self.opaque_chunks
                .insert(key, self.opaque_draw_from_mesh(&mesh, split));
        }
        self.spatial_mesh_cache = Some(spatial_cache);
        self.last_mesh_route = "cpu_chunked".to_string();
    }

    /// CPU greedy mesh, using chunked construction for very large voxel counts.
    pub fn cpu_mesh_fallback(&mut self, voxels: &[Voxel], objects: &[SceneObject]) {
        let default_objs = crate::voxelle::default_scene_objects();
        let objs: &[SceneObject] = if objects.is_empty() {
            default_objs.as_slice()
        } else {
            objects
        };
        let work = crate::voxelle::scene::visible_voxels_for_meshing(voxels, objs);
        if work.is_empty() {
            self.upload_mesh(&mut greedy_mesh::MeshBuffers::default());
            self.last_mesh_route = "cpu_empty".to_string();
            return;
        }
        let multi = work
            .iter()
            .map(|v| v.object_id)
            .collect::<std::collections::HashSet<_>>()
            .len()
            > 1;
        if work.len() >= greedy_mesh::CHUNKED_CPU_MESH_MIN_VOXELS && !multi {
            self.upload_cpu_mesh_chunked_full(&work, None::<&AppHandle<tauri::Wry>>);
        } else {
            let (mut mesh, _) = greedy_mesh::build_greedy_mesh(voxels, objs);
            self.upload_mesh(&mut mesh);
            self.last_mesh_route = "cpu".to_string();
        }
    }

    pub fn upload_preview_mesh(&mut self, solid: &MeshBuffers, wire: &MeshBuffers) {
        let solid_v = Self::interleaved_from_mesh(solid);
        let wire_v = Self::interleaved_from_mesh(wire);
        self.preview_vertex_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("preview_vtx"),
                contents: bytemuck::cast_slice(&solid_v),
                usage: wgpu::BufferUsages::VERTEX,
            },
        ));
        self.preview_index_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("preview_idx"),
                contents: bytemuck::cast_slice(&solid.indices),
                usage: wgpu::BufferUsages::INDEX,
            },
        ));
        self.preview_index_count = solid.indices.len() as u32;
        self.preview_wire_vertex_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("preview_wire_vtx"),
                contents: bytemuck::cast_slice(&wire_v),
                usage: wgpu::BufferUsages::VERTEX,
            },
        ));
        self.preview_wire_index_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("preview_wire_idx"),
                contents: bytemuck::cast_slice(&wire.indices),
                usage: wgpu::BufferUsages::INDEX,
            },
        ));
        self.preview_wire_index_count = wire.indices.len() as u32;
    }

    /// Interleave a [`PreviewPrototype`] into a flat `[pos.x, pos.y, pos.z, nx, ny, nz, …]` buffer.
    pub(crate) fn interleaved_from_prototype(proto: &greedy_mesh::PreviewPrototype) -> Vec<f32> {
        let n = proto.positions.len() / 3;
        let mut out = Vec::with_capacity(n * 6);
        for i in 0..n {
            out.push(proto.positions[i * 3]);
            out.push(proto.positions[i * 3 + 1]);
            out.push(proto.positions[i * 3 + 2]);
            out.push(proto.normals[i * 3]);
            out.push(proto.normals[i * 3 + 1]);
            out.push(proto.normals[i * 3 + 2]);
        }
        out
    }

    pub fn upload_preview_mesh_instanced(&mut self, data: &greedy_mesh::PreviewInstancedResult) {
        // Upload instanced data for bulk voxels
        if !data.solid_instances.is_empty() {
            let solid_proto = greedy_mesh::preview_cube_prototype(data.cube_half);
            let solid_v = Self::interleaved_from_prototype(&solid_proto);
            self.preview_solid_proto_vb = Some(self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("preview_inst_solid_proto_vb"),
                    contents: bytemuck::cast_slice(&solid_v),
                    usage: wgpu::BufferUsages::VERTEX,
                },
            ));
            self.preview_solid_proto_ib = Some(self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("preview_inst_solid_proto_ib"),
                    contents: bytemuck::cast_slice(&solid_proto.indices),
                    usage: wgpu::BufferUsages::INDEX,
                },
            ));
            self.preview_solid_proto_idx_count = solid_proto.indices.len() as u32;

            self.preview_solid_instance_buf = Some(self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("preview_inst_solid_instances"),
                    contents: bytemuck::cast_slice(&data.solid_instances),
                    usage: wgpu::BufferUsages::VERTEX,
                },
            ));
            self.preview_solid_instance_count = data.solid_instances.len() as u32;

            let wire_proto = greedy_mesh::preview_wireframe_prototype(data.cube_half);
            let wire_v = Self::interleaved_from_prototype(&wire_proto);
            self.preview_wire_proto_vb = Some(self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("preview_inst_wire_proto_vb"),
                    contents: bytemuck::cast_slice(&wire_v),
                    usage: wgpu::BufferUsages::VERTEX,
                },
            ));
            self.preview_wire_proto_ib = Some(self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("preview_inst_wire_proto_ib"),
                    contents: bytemuck::cast_slice(&wire_proto.indices),
                    usage: wgpu::BufferUsages::INDEX,
                },
            ));
            self.preview_wire_proto_idx_count = wire_proto.indices.len() as u32;

            self.preview_wire_instance_buf = Some(self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("preview_inst_wire_instances"),
                    contents: bytemuck::cast_slice(&data.wire_instances),
                    usage: wgpu::BufferUsages::VERTEX,
                },
            ));
            self.preview_wire_instance_count = data.wire_instances.len() as u32;
        } else {
            self.preview_solid_proto_vb = None;
            self.preview_solid_proto_ib = None;
            self.preview_solid_proto_idx_count = 0;
            self.preview_wire_proto_vb = None;
            self.preview_wire_proto_ib = None;
            self.preview_wire_proto_idx_count = 0;
            self.preview_solid_instance_buf = None;
            self.preview_solid_instance_count = 0;
            self.preview_wire_instance_buf = None;
            self.preview_wire_instance_count = 0;
        }
        // Upload non-instanced extras (gizmos, polygon markers)
        if !data.extra_solid.positions.is_empty() || !data.extra_wire.positions.is_empty() {
            self.upload_preview_mesh(&data.extra_solid, &data.extra_wire);
        } else {
            self.preview_vertex_buffer = None;
            self.preview_index_buffer = None;
            self.preview_index_count = 0;
            self.preview_wire_vertex_buffer = None;
            self.preview_wire_index_buffer = None;
            self.preview_wire_index_count = 0;
        }
    }

    pub fn upload_gen_preview_mesh_instanced(
        &mut self,
        data: &greedy_mesh::PreviewInstancedResult,
    ) {
        if !data.solid_instances.is_empty() {
            let solid_proto = greedy_mesh::preview_cube_prototype(data.cube_half);
            let solid_v = Self::interleaved_from_prototype(&solid_proto);
            self.gen_preview_solid_proto_vb = Some(self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("gen_preview_inst_solid_proto_vb"),
                    contents: bytemuck::cast_slice(&solid_v),
                    usage: wgpu::BufferUsages::VERTEX,
                },
            ));
            self.gen_preview_solid_proto_ib = Some(self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("gen_preview_inst_solid_proto_ib"),
                    contents: bytemuck::cast_slice(&solid_proto.indices),
                    usage: wgpu::BufferUsages::INDEX,
                },
            ));
            self.gen_preview_solid_proto_idx_count = solid_proto.indices.len() as u32;

            self.gen_preview_solid_instance_buf = Some(self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("gen_preview_inst_solid_instances"),
                    contents: bytemuck::cast_slice(&data.solid_instances),
                    usage: wgpu::BufferUsages::VERTEX,
                },
            ));
            self.gen_preview_solid_instance_count = data.solid_instances.len() as u32;

            let wire_proto = greedy_mesh::preview_wireframe_prototype(data.cube_half);
            let wire_v = Self::interleaved_from_prototype(&wire_proto);
            self.gen_preview_wire_proto_vb = Some(self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("gen_preview_inst_wire_proto_vb"),
                    contents: bytemuck::cast_slice(&wire_v),
                    usage: wgpu::BufferUsages::VERTEX,
                },
            ));
            self.gen_preview_wire_proto_ib = Some(self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("gen_preview_inst_wire_proto_ib"),
                    contents: bytemuck::cast_slice(&wire_proto.indices),
                    usage: wgpu::BufferUsages::INDEX,
                },
            ));
            self.gen_preview_wire_proto_idx_count = wire_proto.indices.len() as u32;

            self.gen_preview_wire_instance_buf = Some(self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("gen_preview_inst_wire_instances"),
                    contents: bytemuck::cast_slice(&data.wire_instances),
                    usage: wgpu::BufferUsages::VERTEX,
                },
            ));
            self.gen_preview_wire_instance_count = data.wire_instances.len() as u32;
        } else {
            self.gen_preview_solid_proto_vb = None;
            self.gen_preview_solid_proto_ib = None;
            self.gen_preview_solid_proto_idx_count = 0;
            self.gen_preview_wire_proto_vb = None;
            self.gen_preview_wire_proto_ib = None;
            self.gen_preview_wire_proto_idx_count = 0;
            self.gen_preview_solid_instance_buf = None;
            self.gen_preview_solid_instance_count = 0;
            self.gen_preview_wire_instance_buf = None;
            self.gen_preview_wire_instance_count = 0;
        }
    }

    pub fn clear_preview_mesh(&mut self) {
        self.preview_vertex_buffer = None;
        self.preview_index_buffer = None;
        self.preview_index_count = 0;
        self.preview_wire_vertex_buffer = None;
        self.preview_wire_index_buffer = None;
        self.preview_wire_index_count = 0;
        self.preview_solid_proto_vb = None;
        self.preview_solid_proto_ib = None;
        self.preview_solid_proto_idx_count = 0;
        self.preview_wire_proto_vb = None;
        self.preview_wire_proto_ib = None;
        self.preview_wire_proto_idx_count = 0;
        self.preview_solid_instance_buf = None;
        self.preview_solid_instance_count = 0;
        self.preview_wire_instance_buf = None;
        self.preview_wire_instance_count = 0;
        self.gen_preview_solid_proto_vb = None;
        self.gen_preview_solid_proto_ib = None;
        self.gen_preview_solid_proto_idx_count = 0;
        self.gen_preview_wire_proto_vb = None;
        self.gen_preview_wire_proto_ib = None;
        self.gen_preview_wire_proto_idx_count = 0;
        self.gen_preview_solid_instance_buf = None;
        self.gen_preview_solid_instance_count = 0;
        self.gen_preview_wire_instance_buf = None;
        self.gen_preview_wire_instance_count = 0;
        self.preview_cache_key = None;
    }

    pub fn upload_selection_overlay_solid(&mut self, solid: &MeshBuffers) {
        if solid.indices.is_empty() {
            self.selection_overlay_vertex_buffer = None;
            self.selection_overlay_index_buffer = None;
            self.selection_overlay_index_count = 0;
            return;
        }
        let solid_v = Self::interleaved_from_mesh(solid);
        self.selection_overlay_vertex_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("selection_overlay_vtx"),
                contents: bytemuck::cast_slice(&solid_v),
                usage: wgpu::BufferUsages::VERTEX,
            },
        ));
        self.selection_overlay_index_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("selection_overlay_idx"),
                contents: bytemuck::cast_slice(&solid.indices),
                usage: wgpu::BufferUsages::INDEX,
            },
        ));
        self.selection_overlay_index_count = solid.indices.len() as u32;
    }

    pub fn upload_selection_overlay_lines(&mut self, verts: &[f32]) {
        if verts.is_empty() || !verts.len().is_multiple_of(6) {
            self.selection_overlay_line_vertex_buffer = None;
            self.selection_overlay_line_vertex_count = 0;
            return;
        }
        let n_floats = verts.len();
        let vertex_count = (n_floats / 6) as u32;
        let nbytes = std::mem::size_of_val(verts) as u64;
        if let Some(ref buf) = self.selection_overlay_line_vertex_buffer {
            if buf.size() == nbytes {
                self.queue.write_buffer(buf, 0, bytemuck::cast_slice(verts));
                self.selection_overlay_line_vertex_count = vertex_count;
                return;
            }
        }
        self.selection_overlay_line_vertex_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("selection_overlay_lines_vtx"),
                contents: bytemuck::cast_slice(verts),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            },
        ));
        self.selection_overlay_line_vertex_count = vertex_count;
    }

    pub(crate) fn upload_buf(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buf: &mut Option<wgpu::Buffer>,
        label: &str,
        data: &[f32],
    ) -> u32 {
        if data.is_empty() || !data.len().is_multiple_of(6) {
            *buf = None;
            return 0;
        }
        let nbytes = std::mem::size_of_val(data) as u64;
        if let Some(ref b) = buf {
            if b.size() == nbytes {
                queue.write_buffer(b, 0, bytemuck::cast_slice(data));
                return (data.len() / 6) as u32;
            }
        }
        *buf = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }),
        );
        (data.len() / 6) as u32
    }

    pub fn upload_gizmo_lines(&mut self, verts: &[f32]) {
        self.gizmo_line_vertex_count = Self::upload_buf(
            &self.device,
            &self.queue,
            &mut self.gizmo_line_vertex_buffer,
            "gizmo_lines_vtx",
            verts,
        );
    }

    pub fn upload_gizmo_tris(&mut self, verts: &[f32]) {
        self.gizmo_tri_vertex_count = Self::upload_buf(
            &self.device,
            &self.queue,
            &mut self.gizmo_tri_vertex_buffer,
            "gizmo_tris_vtx",
            verts,
        );
    }

    pub fn upload_grid_border_lines(&mut self, verts: &[f32], indices: &[u32]) {
        if verts.is_empty() || !verts.len().is_multiple_of(6) || indices.is_empty() {
            self.grid_border_line_vertex_buffer = None;
            self.grid_border_line_index_buffer = None;
            self.grid_border_line_index_count = 0;
            return;
        }
        // Vertex buffer
        let vbytes = std::mem::size_of_val(verts) as u64;
        if let Some(ref buf) = self.grid_border_line_vertex_buffer {
            if buf.size() == vbytes {
                self.queue.write_buffer(buf, 0, bytemuck::cast_slice(verts));
            } else {
                self.grid_border_line_vertex_buffer = Some(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("grid_border_lines_vtx"),
                        contents: bytemuck::cast_slice(verts),
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    },
                ));
            }
        } else {
            self.grid_border_line_vertex_buffer = Some(self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("grid_border_lines_vtx"),
                    contents: bytemuck::cast_slice(verts),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                },
            ));
        }
        // Index buffer
        let ibytes = std::mem::size_of_val(indices) as u64;
        if let Some(ref buf) = self.grid_border_line_index_buffer {
            if buf.size() == ibytes {
                self.queue
                    .write_buffer(buf, 0, bytemuck::cast_slice(indices));
            } else {
                self.grid_border_line_index_buffer = Some(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("grid_border_lines_idx"),
                        contents: bytemuck::cast_slice(indices),
                        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    },
                ));
            }
        } else {
            self.grid_border_line_index_buffer = Some(self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("grid_border_lines_idx"),
                    contents: bytemuck::cast_slice(indices),
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                },
            ));
        }
        self.grid_border_line_index_count = indices.len() as u32;
    }

    pub fn clear_grid_border_lines(&mut self) {
        self.grid_border_line_vertex_buffer = None;
        self.grid_border_line_index_buffer = None;
        self.grid_border_line_index_count = 0;
        self.grid_border_cache_key = None;
    }

    pub fn clear_selection_overlay(&mut self) {
        self.selection_overlay_vertex_buffer = None;
        self.selection_overlay_index_buffer = None;
        self.selection_overlay_index_count = 0;
        self.selection_overlay_line_vertex_buffer = None;
        self.selection_overlay_line_vertex_count = 0;
        self.selection_overlay_cache_key = None;
    }

    pub fn clear_collab_peer_lines(&mut self) {
        self.collab_line_vertex_buffer = None;
        self.collab_line_vertex_count = 0;
    }

    pub fn upload_ping_mesh(&mut self, solid: &MeshBuffers, wire: &MeshBuffers) {
        let solid_v = Self::interleaved_from_mesh(solid);
        let wire_v = Self::interleaved_from_mesh(wire);
        self.ping_vertex_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("ping_vtx"),
                contents: bytemuck::cast_slice(&solid_v),
                usage: wgpu::BufferUsages::VERTEX,
            },
        ));
        self.ping_index_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("ping_idx"),
                contents: bytemuck::cast_slice(&solid.indices),
                usage: wgpu::BufferUsages::INDEX,
            },
        ));
        self.ping_index_count = solid.indices.len() as u32;
        self.ping_wire_vertex_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("ping_wire_vtx"),
                contents: bytemuck::cast_slice(&wire_v),
                usage: wgpu::BufferUsages::VERTEX,
            },
        ));
        self.ping_wire_index_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("ping_wire_idx"),
                contents: bytemuck::cast_slice(&wire.indices),
                usage: wgpu::BufferUsages::INDEX,
            },
        ));
        self.ping_wire_index_count = wire.indices.len() as u32;
    }

    pub fn clear_ping_mesh(&mut self) {
        self.ping_vertex_buffer = None;
        self.ping_index_buffer = None;
        self.ping_index_count = 0;
        self.ping_wire_vertex_buffer = None;
        self.ping_wire_index_buffer = None;
        self.ping_wire_index_count = 0;
        self.ping_wave_line_vertex_buffer = None;
        self.ping_wave_line_vertex_count = 0;
    }

    pub fn upload_ping_wave_lines(&mut self, verts: &[f32]) {
        if verts.len() < 6 {
            self.ping_wave_line_vertex_buffer = None;
            self.ping_wave_line_vertex_count = 0;
            return;
        }
        self.ping_wave_line_vertex_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("ping_wave_lines"),
                contents: bytemuck::cast_slice(verts),
                usage: wgpu::BufferUsages::VERTEX,
            },
        ));
        self.ping_wave_line_vertex_count = verts.len() as u32 / 6;
    }

    pub(crate) fn upload_prepared_opaque(&mut self, opaque: PreparedOpaqueUpload) {
        match opaque {
            PreparedOpaqueUpload::Empty => {
                self.vertex_buffer = None;
                self.index_buffer = None;
                self.index_count = 0;
                self.opaque_index_split = 0;
                self.opaque_chunked = false;
                self.opaque_chunks.clear();
                self.pending_chunk_uploads.clear();
                self.spatial_mesh_cache = None;
                self.last_mesh_route = "cpu".to_string();
            }
            PreparedOpaqueUpload::Single(mut mesh) => {
                self.pending_chunk_uploads.clear();
                self.upload_mesh(&mut mesh);
                self.last_mesh_route = "cpu".to_string();
            }
            PreparedOpaqueUpload::Chunked {
                chunk_origin,
                meshes,
                spatial_cache,
            } => {
                self.vertex_buffer = None;
                self.index_buffer = None;
                self.index_count = 0;
                self.opaque_index_split = 0;
                self.opaque_chunks.clear();
                self.pending_chunk_uploads.clear();
                self.chunk_grid_origin = chunk_origin;
                if meshes.is_empty() {
                    self.opaque_chunked = false;
                    self.spatial_mesh_cache = None;
                } else {
                    self.opaque_chunked = true;
                    // Upload first batch immediately; queue rest for progressive per-frame upload.
                    const INITIAL_BATCH: usize = 8;
                    let mut iter = meshes.into_iter();
                    for (key, mut mesh) in iter.by_ref().take(INITIAL_BATCH) {
                        let split = greedy_mesh::partition_indices_by_transparency(&mut mesh);
                        self.opaque_chunks
                            .insert(key, self.opaque_draw_from_mesh(&mesh, split));
                    }
                    self.pending_chunk_uploads.extend(iter);
                    if !self.pending_chunk_uploads.is_empty() {
                        log::info!(
                            target: "voxelle_load",
                            "progressive upload: {} chunks queued, {} uploaded immediately",
                            self.pending_chunk_uploads.len(),
                            self.opaque_chunks.len()
                        );
                    }
                    self.spatial_mesh_cache = Some(spatial_cache);
                    self.last_mesh_route = "cpu_chunked".to_string();
                }
            }
            PreparedOpaqueUpload::ChunkedDeferred { chunk_origin } => {
                self.vertex_buffer = None;
                self.index_buffer = None;
                self.index_count = 0;
                self.opaque_chunks.clear();
                self.pending_chunk_uploads.clear();
                self.chunk_grid_origin = chunk_origin;
                self.opaque_chunked = true;
                // spatial_mesh_cache arrives later via deferred_spatial_cache on ViewerState
                self.spatial_mesh_cache = None;
                self.last_mesh_route = "cpu_chunked_deferred".to_string();
            }
        }
    }
}
