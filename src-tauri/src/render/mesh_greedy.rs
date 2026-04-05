//! GPU greedy mesh compute: dispatch, scratch pool, chunk rebuild, incremental remesh.

use super::*;

/// Reused GPU buffers for greedy mesh compute (grow-only scratch).
#[derive(Default)]
pub(crate) struct MeshGreedyPool {
    pub(crate) counters: Option<wgpu::Buffer>,
    pub(crate) readback: Option<wgpu::Buffer>,
    pub(crate) vtx_scratch: Option<wgpu::Buffer>,
    pub(crate) idx_scratch: Option<wgpu::Buffer>,
    pub(crate) vtx_cap: u64,
    pub(crate) idx_cap: u64,
}

impl MeshGreedyPool {
    pub(crate) fn ensure_counters(&mut self, device: &wgpu::Device) {
        if self.counters.is_none() {
            self.counters = Some(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mesh_atomic_counts"),
                    contents: &[0u8; 8],
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                }),
            );
        }
    }

    pub(crate) fn ensure_readback(&mut self, device: &wgpu::Device) {
        if self.readback.is_none() {
            self.readback = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mesh_counts_rb"),
                size: 8,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
    }

    pub(crate) fn ensure_vtx_out(&mut self, device: &wgpu::Device, need: u64) {
        if self.vtx_scratch.is_none() || self.vtx_cap < need {
            self.vtx_scratch = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mesh_vtx_out"),
                size: need,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.vtx_cap = need;
        }
    }

    pub(crate) fn ensure_idx_out(&mut self, device: &wgpu::Device, need: u64) {
        if self.idx_scratch.is_none() || self.idx_cap < need {
            self.idx_scratch = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mesh_idx_out"),
                size: need,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.idx_cap = need;
        }
    }
}

impl WgpuViewer {
    /// Apply one edit to [`Self::spatial_mesh_cache`] (must match `current_file.voxels` after `apply_edit`).
    pub fn apply_spatial_cache_edit(&mut self, delta: &VoxelEditDelta) {
        let Some(ref mut cache) = self.spatial_mesh_cache else {
            return;
        };
        let cs = greedy_mesh::SPATIAL_CHUNK_SIZE;
        match delta {
            VoxelEditDelta::Added(v) => cache.apply_add(*v, cs),
            VoxelEditDelta::Removed { voxel } => cache.apply_remove(voxel.x, voxel.y, voxel.z, cs),
            VoxelEditDelta::Painted { after, .. } => cache.apply_paint(*after, cs),
        }
    }

    /// Rebuild GPU buffers for `keys` only. Returns `true` if only those chunks were updated; `false` if a full chunked upload ran (origin drift).
    ///
    /// Incremental greedy: [`greedy_mesh::pack_gpu_greedy_slices`] plus GPU greedy compute (same pipeline as [`Self::run_mesh_greedy_compute_with_brick`]) per dirty chunk unless `VOXELLE_CPU_CHUNK_REMESH` is set (any value forces CPU [`greedy_mesh::mesh_buffers_for_chunk_key`] only).
    pub fn remesh_opaque_chunks<R: Runtime>(
        &mut self,
        keys: &[ChunkKey],
        voxels: &[Voxel],
        work_progress: Option<&AppHandle<R>>,
    ) -> (bool, RemeshOpaquePerf) {
        let mut perf = RemeshOpaquePerf::default();
        let cs = greedy_mesh::SPATIAL_CHUNK_SIZE;

        if self.spatial_mesh_cache.is_none() {
            let t_cold = Instant::now();
            if let Some(app) = work_progress {
                crate::emit_work_progress(app, 0.4, "Indexing voxels for mesh…");
            }
            self.spatial_mesh_cache = greedy_mesh::SpatialMeshCache::from_voxels(voxels, cs);
            perf.buckets_ms = t_cold.elapsed().as_secs_f64() * 1000.0;
        }
        let Some(cache_ref) = self.spatial_mesh_cache.as_ref() else {
            self.upload_mesh(&mut MeshBuffers::default());
            return (false, perf);
        };
        let origin_iv = IVec3::new(cache_ref.origin.0, cache_ref.origin.1, cache_ref.origin.2);
        if origin_iv != self.chunk_grid_origin {
            let t_full = Instant::now();
            self.upload_cpu_mesh_chunked_full(voxels, work_progress);
            perf.full_chunked_rebuild_ms = t_full.elapsed().as_secs_f64() * 1000.0;
            return (false, perf);
        }

        let spatial_cache = self.spatial_mesh_cache.take().expect("spatial_mesh_cache");
        let cache = &spatial_cache;

        let use_gpu_chunk = std::env::var("VOXELLE_CPU_CHUNK_REMESH").is_err();

        let halo_pack: Option<(IVec3, (u32, u32, u32), wgpu::Buffer)> = if use_gpu_chunk {
            greedy_mesh::pack_brick_halo_cells(
                &cache.occupancy,
                (
                    self.brick_origin_iv.x,
                    self.brick_origin_iv.y,
                    self.brick_origin_iv.z,
                ),
                self.brick_dims_u,
            )
            .map(|(ho, hd, cells)| {
                let buf = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("mesh_greedy_brick_halo_chunk"),
                        contents: bytemuck::cast_slice(&cells),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
                (IVec3::new(ho.0, ho.1, ho.2), hd, buf)
            })
        } else {
            None
        };

        let t_greedy = Instant::now();
        let mut greedy_gpu_ms = 0.0f64;
        let mut greedy_cpu_ms = 0.0f64;

        let nk = keys.len().max(1) as u32;
        let mut last_emit_permille: u32 = 0;
        for (ki, key) in keys.iter().enumerate() {
            if ki > 0 && ki % 2 == 0 {
                std::thread::yield_now();
            }
            let done = (ki + 1) as u32;
            let frac = done as f32 / nk as f32;
            let permille = (frac * 1000.0).min(1000.0) as u32;
            if let Some(app) = work_progress {
                if permille.saturating_sub(last_emit_permille) >= 40 || done == nk {
                    last_emit_permille = permille;
                    crate::emit_work_progress(
                        app,
                        0.38 + 0.55 * frac,
                        format!("Mesh chunks {done}/{nk}…"),
                    );
                }
            }

            let mut core_vec: Vec<Voxel> = cache
                .buckets
                .get(key)
                .map(|b| b.values().copied().collect())
                .unwrap_or_default();
            core_vec.sort_unstable_by_key(|v| (v.x, v.y, v.z));
            let core: &[Voxel] = &core_vec;

            let mut used_gpu = false;
            if use_gpu_chunk && !core.is_empty() {
                let (mo, md, brick_ref): (IVec3, (u32, u32, u32), &wgpu::Buffer) = match &halo_pack
                {
                    Some((o, d, buf)) => (*o, *d, buf),
                    None => (self.brick_origin_iv, self.brick_dims_u, &self.brick_buffer),
                };

                let t_pack = Instant::now();
                if let Ok((headers, bits)) =
                    greedy_mesh::pack_gpu_greedy_slices(&cache.occupancy, core)
                {
                    greedy_gpu_ms += t_pack.elapsed().as_secs_f64() * 1000.0;
                    if !headers.is_empty() {
                        let t_compute = Instant::now();
                        if let Ok((v_tot, i_tot)) = Self::mesh_greedy_dispatch(
                            &self.device,
                            &self.queue,
                            &mut self.mesh_greedy_pool,
                            &mut self.mesh_greedy_bind_layout,
                            &mut self.mesh_greedy_pipeline,
                            &mut self.mesh_greedy_pl_version,
                            brick_ref,
                            &headers,
                            &bits,
                            mo,
                            md,
                        ) {
                            greedy_gpu_ms += t_compute.elapsed().as_secs_f64() * 1000.0;
                            let t_u = Instant::now();
                            if v_tot == 0 || i_tot == 0 {
                                self.opaque_chunks.remove(key);
                            } else {
                                self.upload_or_replace_chunk_mesh_from_gpu_scratch(
                                    *key, v_tot, i_tot,
                                );
                            }
                            perf.chunk_buffers_ms += t_u.elapsed().as_secs_f64() * 1000.0;
                            used_gpu = true;
                        }
                    }
                } else {
                    greedy_gpu_ms += t_pack.elapsed().as_secs_f64() * 1000.0;
                }
            }

            if !used_gpu {
                let t_cpu = Instant::now();
                let mut mesh =
                    greedy_mesh::mesh_buffers_for_chunk_key(&cache.buckets, &cache.occupancy, *key);
                greedy_cpu_ms += t_cpu.elapsed().as_secs_f64() * 1000.0;
                if mesh.indices.is_empty() {
                    self.opaque_chunks.remove(key);
                } else {
                    let split = greedy_mesh::partition_indices_by_transparency(&mut mesh);
                    let t_u = Instant::now();
                    self.upload_or_replace_chunk_mesh(*key, &mesh, split);
                    perf.chunk_buffers_ms += t_u.elapsed().as_secs_f64() * 1000.0;
                }
            }
        }

        self.spatial_mesh_cache = Some(spatial_cache);

        perf.greedy_ms = t_greedy.elapsed().as_secs_f64() * 1000.0;
        perf.greedy_gpu_ms = greedy_gpu_ms;
        perf.greedy_cpu_ms = greedy_cpu_ms;
        (true, perf)
    }

    /// Run [`gpu::mesh_greedy`] compute via disjoint `&mut self.*` borrows so `brick_storage == self.brick_buffer` works.
    pub(crate) fn mesh_greedy_dispatch(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pool: &mut MeshGreedyPool,
        mesh_greedy_bind_layout: &mut Option<wgpu::BindGroupLayout>,
        mesh_greedy_pipeline: &mut Option<wgpu::ComputePipeline>,
        mesh_greedy_pl_version: &mut u32,
        brick_storage: &wgpu::Buffer,
        headers: &[greedy_mesh::GpuSliceHeader],
        bits: &[u32],
        mesh_brick_origin: IVec3,
        mesh_brick_dims: (u32, u32, u32),
    ) -> Result<(u32, u32), String> {
        if headers.is_empty() {
            return Err("gpu greedy: empty headers".into());
        }

        let mut acc_v: u32 = 0;
        let mut acc_i: u32 = 0;
        for h in headers {
            acc_v = acc_v.saturating_add(4u32.saturating_mul(h.width.saturating_mul(h.height)));
            acc_i = acc_i.saturating_add(6u32.saturating_mul(h.width.saturating_mul(h.height)));
        }
        let max_vertices = acc_v.max(1);
        let max_indices = acc_i.max(1);

        const VTX_STRIDE: u64 = OPAQUE_VERTEX_STRIDE;
        let vtx_storage_size = (max_vertices as u64).saturating_mul(VTX_STRIDE);
        let idx_storage_size = (max_indices as u64).saturating_mul(4);

        let lim = device.limits();
        let max_wg = lim.max_compute_workgroups_per_dimension as usize;
        let max_bind = lim.max_storage_buffer_binding_size as u64;
        if vtx_storage_size > lim.max_buffer_size
            || idx_storage_size > lim.max_buffer_size
            || vtx_storage_size > max_bind
            || idx_storage_size > max_bind
            || headers.len() > max_wg
        {
            return Err("gpu greedy: over device limits".into());
        }

        let params = GpuMeshParams {
            max_vertices,
            max_indices,
            slice_count: headers.len() as u32,
            _pad0: 0,
            brick_ox: mesh_brick_origin.x,
            brick_oy: mesh_brick_origin.y,
            brick_oz: mesh_brick_origin.z,
            _pad1: 0,
            brick_dx: mesh_brick_dims.0,
            brick_dy: mesh_brick_dims.1,
            brick_dz: mesh_brick_dims.2,
            _pad2: 0,
        };

        pool.ensure_counters(device);
        pool.ensure_vtx_out(device, vtx_storage_size);
        pool.ensure_idx_out(device, idx_storage_size);
        pool.ensure_readback(device);
        let counters_buf = pool.counters.as_ref().unwrap();
        queue.write_buffer(counters_buf, 0, &[0u8; 8]);
        let hdr_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_slice_hdr"),
            contents: bytemuck::cast_slice(headers),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bits_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_slice_bits"),
            contents: bytemuck::cast_slice(bits),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let vtx_out = pool.vtx_scratch.as_ref().unwrap();
        let idx_out = pool.idx_scratch.as_ref().unwrap();
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_greedy_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        if *mesh_greedy_pl_version != MESH_GREEDY_PIPELINE_LAYOUT_VERSION {
            *mesh_greedy_bind_layout = None;
            *mesh_greedy_pipeline = None;
            *mesh_greedy_pl_version = MESH_GREEDY_PIPELINE_LAYOUT_VERSION;
        }

        if mesh_greedy_bind_layout.is_none() {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mesh_greedy_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mesh_greedy"),
                source: wgpu::ShaderSource::Wgsl(gpu::mesh_greedy::WGSL.into()),
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mesh_greedy_pl"),
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("mesh_greedy"),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("greedy_slice"),
                compilation_options: Default::default(),
                cache: None,
            });
            *mesh_greedy_bind_layout = Some(layout);
            *mesh_greedy_pipeline = Some(pipeline);
        }

        let layout = mesh_greedy_bind_layout.as_ref().unwrap();
        let pipeline = mesh_greedy_pipeline.as_ref().unwrap();

        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_greedy_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: hdr_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bits_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: vtx_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: idx_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: counters_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: brick_storage.as_entire_binding(),
                },
            ],
        });

        let readback = pool.readback.as_ref().unwrap();

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mesh_greedy_enc"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mesh_greedy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(headers.len() as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(counters_buf, 0, readback, 0, 8);
        queue.submit(std::iter::once(enc.finish()));
        poll_device_yielding_until_queue_empty(device);

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        let map_r = rx.recv().map_err(|_| "mesh count readback channel")?;
        map_r.map_err(|e| e.to_string())?;
        let data = slice.get_mapped_range();
        let v_total = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let i_total = u32::from_le_bytes(data[4..8].try_into().unwrap());
        drop(data);
        readback.unmap();

        if v_total > max_vertices || i_total > max_indices {
            return Err("gpu greedy: counter overflow".into());
        }
        Ok((v_total, i_total))
    }

    /// Copy compact greedy output from [`MeshGreedyPool`] scratch into chunk draw buffers (`VERTEX`/`INDEX` layout matches [`Self::opaque_draw_from_mesh`]).
    /// NOTE: GPU-meshed chunks currently treat all indices as opaque (`transparent_index_count = 0`)
    /// because we don't yet have GPU-side index partitioning.
    pub(crate) fn upload_or_replace_chunk_mesh_from_gpu_scratch(
        &mut self,
        key: ChunkKey,
        v_total: u32,
        i_total: u32,
    ) {
        const VTX_STRIDE: u64 = OPAQUE_VERTEX_STRIDE;
        let vtx_bytes = (v_total as u64).saturating_mul(VTX_STRIDE);
        let idx_bytes = (i_total as u64).saturating_mul(4);
        let vtx_out = self.mesh_greedy_pool.vtx_scratch.as_ref().unwrap();
        let idx_out = self.mesh_greedy_pool.idx_scratch.as_ref().unwrap();

        let can_reuse = self.opaque_chunks.get(&key).map(|d| {
            d.vertex_buffer
                .usage()
                .contains(wgpu::BufferUsages::COPY_DST)
                && d.index_buffer
                    .usage()
                    .contains(wgpu::BufferUsages::COPY_DST)
                && d.vertex_buffer.size() >= vtx_bytes
                && d.index_buffer.size() >= idx_bytes
        }) == Some(true);

        if can_reuse {
            let draw = self.opaque_chunks.get_mut(&key).expect("chunk draw");
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("chunk_mesh_gpu_copy"),
                });
            enc.copy_buffer_to_buffer(vtx_out, 0, &draw.vertex_buffer, 0, vtx_bytes);
            enc.copy_buffer_to_buffer(idx_out, 0, &draw.index_buffer, 0, idx_bytes);
            self.queue.submit(std::iter::once(enc.finish()));
            poll_device_yielding_until_queue_empty(&self.device);
            draw.opaque_index_count = i_total;
            draw.transparent_index_count = 0;
            draw.partitioned = false;
            return;
        }

        let vb = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vtx_chunk_gpu"),
            size: vtx_bytes,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ib = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("idx_chunk_gpu"),
            size: idx_bytes,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("chunk_mesh_gpu_init"),
            });
        enc.copy_buffer_to_buffer(vtx_out, 0, &vb, 0, vtx_bytes);
        enc.copy_buffer_to_buffer(idx_out, 0, &ib, 0, idx_bytes);
        self.queue.submit(std::iter::once(enc.finish()));
        poll_device_yielding_until_queue_empty(&self.device);
        self.opaque_chunks.insert(
            key,
            OpaqueChunkDraw {
                vertex_buffer: vb,
                index_buffer: ib,
                opaque_index_count: i_total,
                transparent_index_count: 0,
                partitioned: false,
            },
        );
    }

    pub(crate) fn apply_prepared_greedy_rebuild(
        &mut self,
        prepared: PreparedGreedyRebuild,
    ) -> Result<MeshBounds, String> {
        match prepared {
            PreparedGreedyRebuild::NoVoxels => {
                self.vertex_buffer = None;
                self.index_buffer = None;
                self.index_count = 0;
                self.opaque_index_split = 0;
                self.opaque_chunked = false;
                self.opaque_chunks.clear();
                self.spatial_mesh_cache = None;
                Err("empty voxels".into())
            }
            PreparedGreedyRebuild::AllHidden => {
                self.vertex_buffer = None;
                self.index_buffer = None;
                self.index_count = 0;
                self.opaque_index_split = 0;
                self.opaque_chunked = false;
                self.opaque_chunks.clear();
                self.spatial_mesh_cache = None;
                self.last_mesh_route = "all_hidden".to_string();
                Err("empty voxels".into())
            }
            PreparedGreedyRebuild::Opaque {
                opaque,
                bounds,
                last_route,
            } => {
                self.upload_prepared_opaque(opaque);
                self.last_mesh_route = last_route;
                Ok(bounds)
            }
            PreparedGreedyRebuild::GpuGreedyPack {
                bounds,
                headers,
                bits,
                fallback_voxels,
                fallback_objects,
            } => self.apply_gpu_greedy_pack(
                bounds,
                &headers,
                &bits,
                &fallback_voxels,
                &fallback_objects,
            ),
        }
    }

    fn apply_gpu_greedy_pack(
        &mut self,
        bounds: MeshBounds,
        headers: &[greedy_mesh::GpuSliceHeader],
        bits: &[u32],
        fallback_voxels: &[Voxel],
        fallback_objects: &[SceneObject],
    ) -> Result<MeshBounds, String> {
        let default_objs = crate::voxelle::default_scene_objects();
        let objs: &[SceneObject] = if fallback_objects.is_empty() {
            default_objs.as_slice()
        } else {
            fallback_objects
        };
        let map = greedy_mesh::voxel_map(fallback_voxels);
        let packed_halo = greedy_mesh::pack_brick_halo_cells(
            &map,
            (
                self.brick_origin_iv.x,
                self.brick_origin_iv.y,
                self.brick_origin_iv.z,
            ),
            self.brick_dims_u,
        );
        let (mesh_brick_origin, mesh_brick_dims, mesh_brick_halo_buf) = match packed_halo {
            Some((ho, hd, cells)) => {
                let buf = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("mesh_greedy_brick_halo"),
                        contents: bytemuck::cast_slice(&cells),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
                (glam::IVec3::new(ho.0, ho.1, ho.2), hd, Some(buf))
            }
            None => (self.brick_origin_iv, self.brick_dims_u, None),
        };
        let mesh_brick_ref: &wgpu::Buffer = mesh_brick_halo_buf
            .as_ref()
            .map(|b| b as &wgpu::Buffer)
            .unwrap_or(&self.brick_buffer);

        let (v_total, i_total) = match Self::mesh_greedy_dispatch(
            &self.device,
            &self.queue,
            &mut self.mesh_greedy_pool,
            &mut self.mesh_greedy_bind_layout,
            &mut self.mesh_greedy_pipeline,
            &mut self.mesh_greedy_pl_version,
            mesh_brick_ref,
            headers,
            bits,
            mesh_brick_origin,
            mesh_brick_dims,
        ) {
            Ok(v) => v,
            Err(_) => {
                self.cpu_mesh_fallback(fallback_voxels, objs);
                return Ok(bounds);
            }
        };

        if v_total == 0 || i_total == 0 {
            self.cpu_mesh_fallback(fallback_voxels, objs);
            return Ok(bounds);
        }

        const VTX_STRIDE: u64 = OPAQUE_VERTEX_STRIDE;
        let vtx_out = self.mesh_greedy_pool.vtx_scratch.as_ref().unwrap();
        let idx_out = self.mesh_greedy_pool.idx_scratch.as_ref().unwrap();

        let vb_final = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_vtx_final"),
            size: (v_total as u64).saturating_mul(VTX_STRIDE),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ib_final = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_idx_final"),
            size: (i_total as u64).saturating_mul(4),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc3 = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mesh_copy_draw_bufs"),
            });
        enc3.copy_buffer_to_buffer(
            vtx_out,
            0,
            &vb_final,
            0,
            (v_total as u64).saturating_mul(VTX_STRIDE),
        );
        enc3.copy_buffer_to_buffer(idx_out, 0, &ib_final, 0, (i_total as u64).saturating_mul(4));
        self.queue.submit(std::iter::once(enc3.finish()));
        poll_device_yielding_until_queue_empty(&self.device);

        self.opaque_chunked = false;
        self.opaque_chunks.clear();
        self.spatial_mesh_cache = None;
        self.vertex_buffer = Some(vb_final);
        self.index_buffer = Some(ib_final);
        self.index_count = i_total;
        // GPU greedy path doesn't partition indices yet — both passes draw 0..total
        // and rely on shader-side mat_kind discard.
        self.opaque_index_split = 0;
        self.last_mesh_route = "gpu_greedy".to_string();
        Ok(bounds)
    }

    /// GPU greedy mesh (WGSL) when slice bitmaps fit 64×64; otherwise CPU [`greedy_mesh::build_greedy_mesh`].
    /// Set `VOXELLE_CPU_MESH=1` to force CPU meshing.
    pub fn rebuild_mesh_gpu_greedy(
        &mut self,
        voxels: &[Voxel],
        objects: &[SceneObject],
        grid_size: i32,
    ) -> Result<MeshBounds, String> {
        let prepared = compute_greedy_rebuild_cpu(voxels, objects, grid_size, None)?;
        self.apply_prepared_greedy_rebuild(prepared)
    }

    /// Drop the spatial mesh cache so the next remesh rebuilds all chunks from scratch.
    /// Call this whenever a baking parameter (e.g. emission lighting) changes.
    pub fn invalidate_spatial_mesh_cache(&mut self) {
        self.spatial_mesh_cache = None;
    }
}
