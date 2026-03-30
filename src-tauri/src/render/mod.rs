//! Multi-pass GPU renderer: shadow map, HDR+MRT, transmission, bloom, composite.

mod gpu {
    pub mod scene {
        pub const WGSL: &str = include_str!("scene.wgsl");
    }
    pub mod shadow {
        pub const WGSL: &str = include_str!("shadow.wgsl");
    }
    pub mod post_bloom_extract {
        pub const WGSL: &str = include_str!("post_bloom_extract.wgsl");
    }
    pub mod post_blur {
        pub const WGSL: &str = include_str!("post_blur.wgsl");
    }
    pub mod post_composite {
        pub const WGSL: &str = include_str!("post_composite.wgsl");
    }
    pub mod sky {
        pub const WGSL: &str = include_str!("sky.wgsl");
    }
    pub mod mesh_greedy {
        pub const WGSL: &str = include_str!("gpu/mesh_greedy.wgsl");
    }
    pub mod collab_peer_lines {
        pub const WGSL: &str = include_str!("collab_peer_lines.wgsl");
    }
}

use crate::camera::OrbitCamera;
use crate::gpu_brick::{BrickCellWrite, GpuVoxelBrick};
use crate::greedy_mesh::{self, ChunkKey, MeshBounds, MeshBuffers};
use crate::render_constants::{BLOOM_STRENGTH, SHADOW_MAP_SIZE};
use crate::voxel_edit::VoxelEditDelta;
use crate::voxelle::Voxel;
use glam::{IVec3, Mat4, Vec3};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::time::Instant;
use wgpu::util::DeviceExt;

/// Bump when [`gpu::mesh_greedy::WGSL`] bind group layout changes.
const MESH_GREEDY_PIPELINE_LAYOUT_VERSION: u32 = 2;

/// Opaque mesh vertex: `vec3 pos, vec3 n, vec3 color, mat_kind, ao` → 11×`f32`.
const OPAQUE_VERTEX_STRIDE: u64 = 44;

/// Timings from [`WgpuViewer::remesh_opaque_chunks`] (incremental CPU chunked path).
#[derive(Clone, Debug, Default)]
pub struct RemeshOpaquePerf {
    /// Cold [`greedy_mesh::SpatialMeshCache::from_voxels`] when cache was missing.
    pub buckets_ms: f64,
    pub greedy_ms: f64,
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
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GlobalState {
    view_proj: [[f32; 4]; 4],
    inv_view: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    light_view_proj: [[f32; 4]; 4],
    light_dir: [f32; 4],
    cam_pos: [f32; 4],
    brick_origin: [f32; 4],
    brick_dims: [f32; 4],
    screen: [f32; 4],
    params: [f32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PostBlurUniform {
    blur_dir: [f32; 4],
}

/// Matches `post_composite.wgsl` `PostCompositeOpts` and Voxelle web tone mapping ids (neutral…reinhard).
#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PostCompositeOpts {
    tone_mode: u32,
    _pad: [u32; 3],
}

pub struct WgpuViewer {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub format: wgpu::TextureFormat,
    pub size: (u32, u32),

    global_buffer: wgpu::Buffer,
    brick_buffer: wgpu::Buffer,
    brick_cell_count: u32,
    brick_origin_iv: IVec3,
    brick_dims_u: (u32, u32, u32),

    scene_bounds: MeshBounds,
    light_dir: Vec3,

    #[allow(dead_code)]
    shadow_texture: wgpu::Texture,
    shadow_view: wgpu::TextureView,

    /// Opaque + glow only — sampled during transmission (never the active color target at the same time).
    hdr_opaque_texture: wgpu::Texture,
    hdr_opaque_view: wgpu::TextureView,
    /// After copy from opaque + transmission pass; bloom/composite use this.
    hdr_texture: wgpu::Texture,
    hdr_view: wgpu::TextureView,
    normal_texture: wgpu::Texture,
    normal_view: wgpu::TextureView,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,

    bloom_a: wgpu::Texture,
    bloom_a_view: wgpu::TextureView,
    bloom_b: wgpu::Texture,
    bloom_b_view: wgpu::TextureView,

    scene_layout0: wgpu::BindGroupLayout,
    scene_layout1: wgpu::BindGroupLayout,
    shadow_vs_layout: wgpu::BindGroupLayout,
    post_bloom_layout: wgpu::BindGroupLayout,
    post_blur_layout: wgpu::BindGroupLayout,
    post_composite_layout: wgpu::BindGroupLayout,

    bind_scene_opaque: wgpu::BindGroup,
    bind_shadow_pass: wgpu::BindGroup,
    bind_bloom_extract: wgpu::BindGroup,
    bind_blur_h: wgpu::BindGroup,
    bind_blur_v: wgpu::BindGroup,
    bind_composite: wgpu::BindGroup,
    bind_trans: Option<wgpu::BindGroup>,

    post_blur_buf: wgpu::Buffer,
    post_composite_opts_buf: wgpu::Buffer,

    pipeline_opaque: wgpu::RenderPipeline,
    /// Web-style ghost: occluded (Greater) then front (Always), unlit + alpha blend; no gbuffer writes.
    pipeline_preview_occluded: wgpu::RenderPipeline,
    pipeline_preview_front: wgpu::RenderPipeline,
    pipeline_collab_lines_occluded: wgpu::RenderPipeline,
    pipeline_collab_lines_front: wgpu::RenderPipeline,
    pipeline_sky: wgpu::RenderPipeline,
    pipeline_trans: wgpu::RenderPipeline,
    pipeline_shadow: wgpu::RenderPipeline,
    pipeline_bloom_extract: wgpu::RenderPipeline,
    pipeline_blur: wgpu::RenderPipeline,
    pipeline_composite: wgpu::RenderPipeline,

    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    index_count: u32,

    /// When set, opaque mesh is drawn from [`Self::opaque_chunks`] (multi-draw).
    pub opaque_chunked: bool,
    /// Chunk bucketing origin (must match [`greedy_mesh::voxel_buckets_by_chunk`]).
    pub chunk_grid_origin: IVec3,
    opaque_chunks: BTreeMap<ChunkKey, OpaqueChunkDraw>,
    /// Incremental occupancy + buckets; rebuilt on full chunked upload, updated O(1) per edit.
    spatial_mesh_cache: Option<greedy_mesh::SpatialMeshCache>,

    preview_vertex_buffer: Option<wgpu::Buffer>,
    preview_index_buffer: Option<wgpu::Buffer>,
    preview_index_count: u32,
    preview_wire_vertex_buffer: Option<wgpu::Buffer>,
    preview_wire_index_buffer: Option<wgpu::Buffer>,
    preview_wire_index_count: u32,
    collab_line_vertex_buffer: Option<wgpu::Buffer>,
    collab_line_vertex_count: u32,
    ping_vertex_buffer: Option<wgpu::Buffer>,
    ping_index_buffer: Option<wgpu::Buffer>,
    ping_index_count: u32,
    ping_wire_vertex_buffer: Option<wgpu::Buffer>,
    ping_wire_index_buffer: Option<wgpu::Buffer>,
    ping_wire_index_count: u32,
    pub(crate) ping_mesh_key: Option<(i32, i32, i32, u32)>,
    /// Dedup CPU mesh rebuild when hover cell unchanged.
    pub preview_cache_key: Option<(i32, i32, i32, u8)>,

    sampler_linear: wgpu::Sampler,
    sampler_comparison: wgpu::Sampler,
    #[allow(dead_code)]
    sampler_nearest: wgpu::Sampler,

    mesh_greedy_pipeline: Option<wgpu::ComputePipeline>,
    mesh_greedy_bind_layout: Option<wgpu::BindGroupLayout>,
    /// Must match [`MESH_GREEDY_PIPELINE_LAYOUT_VERSION`]; clears cached compute pipeline when bumped.
    mesh_greedy_pl_version: u32,
    mesh_greedy_pool: MeshGreedyPool,

    /// Last opaque mesh rebuild path (for perf): `gpu_greedy`, `cpu`, `cpu_chunked`, `clear`, `gpu_no_headers`, etc.
    pub last_mesh_route: String,
}

/// GPU buffers for one spatial chunk of opaque greedy mesh.
struct OpaqueChunkDraw {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

/// Reused GPU buffers for greedy mesh compute (grow-only scratch).
struct MeshGreedyPool {
    counters: Option<wgpu::Buffer>,
    readback: Option<wgpu::Buffer>,
    vtx_scratch: Option<wgpu::Buffer>,
    idx_scratch: Option<wgpu::Buffer>,
    vtx_cap: u64,
    idx_cap: u64,
}

impl Default for MeshGreedyPool {
    fn default() -> Self {
        Self {
            counters: None,
            readback: None,
            vtx_scratch: None,
            idx_scratch: None,
            vtx_cap: 0,
            idx_cap: 0,
        }
    }
}

impl MeshGreedyPool {
    fn ensure_counters(&mut self, device: &wgpu::Device) {
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

    fn ensure_readback(&mut self, device: &wgpu::Device) {
        if self.readback.is_none() {
            self.readback = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mesh_counts_rb"),
                size: 8,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
    }

    fn ensure_vtx_out(&mut self, device: &wgpu::Device, need: u64) {
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

    fn ensure_idx_out(&mut self, device: &wgpu::Device, need: u64) {
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

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuMeshParams {
    max_vertices: u32,
    max_indices: u32,
    slice_count: u32,
    _pad0: u32,
    brick_ox: i32,
    brick_oy: i32,
    brick_oz: i32,
    _pad1: i32,
    brick_dx: u32,
    brick_dy: u32,
    brick_dz: u32,
    _pad2: u32,
}

fn hdr_format() -> wgpu::TextureFormat {
    wgpu::TextureFormat::Rgba16Float
}

/// Uses [`Mat4::orthographic_rh`], which glam documents as \([0,1]\) depth for WebGPU (do not apply an extra OpenGL→wgpu Z remap).
fn light_view_proj(bounds: &MeshBounds, light_dir: Vec3) -> Mat4 {
    let center = bounds.center();
    let r = bounds.radius().max(8.0);
    let ld = light_dir.normalize();
    let up = if ld.cross(Vec3::Y).length() > 0.05 {
        Vec3::Y
    } else {
        Vec3::Z
    };
    // `ld` points from the scene toward the light (e.g. toward the sun). The shadow camera sits on
    // that side of the bounds looking at the scene; the previous `center - ld` put it underground
    // when the sun is above, so +Y faces were back-face culled and the depth map stayed cleared.
    let eye = center + ld * (r * 5.0);
    let view = Mat4::look_at_rh(eye, center, up);
    let he = r * 1.8;
    let proj = Mat4::orthographic_rh(-he, he, -he, he, 1.0, r * 12.0);
    proj * view
}

/// Alpha blend into HDR color; leave destination alpha (bloom/glow mask) unchanged.
fn preview_hdr_blend() -> wgpu::BlendState {
    wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Zero,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
    }
}

fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: OPAQUE_VERTEX_STRIDE,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 12,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 24,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 36,
                shader_location: 3,
                format: wgpu::VertexFormat::Float32,
            },
            wgpu::VertexAttribute {
                offset: 40,
                shader_location: 4,
                format: wgpu::VertexFormat::Float32,
            },
        ],
    }
}

fn vertex_layout_collab_lines() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: (3 + 3) * 4,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 12,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x3,
            },
        ],
    }
}

fn fullscreen_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    fs_entry: &'static str,
    targets: &[Option<wgpu::ColorTargetState>],
    depth_stencil: Option<wgpu::DepthStencilState>,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("fullscreen"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_fullscreen"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fs_entry),
            targets,
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

fn create_shadow_tex(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shadow"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn create_screen_targets(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    hdr_fmt: wgpu::TextureFormat,
) -> (
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
) {
    let w = width.max(1);
    let h = height.max(1);
    let extent = wgpu::Extent3d {
        width: w,
        height: h,
        depth_or_array_layers: 1,
    };
    let color_usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    let hdr_opaque_usage = color_usage | wgpu::TextureUsages::COPY_SRC;
    let hdr_final_usage = color_usage | wgpu::TextureUsages::COPY_DST;
    let depth_usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;

    let hdr_opaque_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hdr_opaque"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: hdr_fmt,
        usage: hdr_opaque_usage,
        view_formats: &[],
    });
    let hdr_opaque_view = hdr_opaque_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let hdr_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hdr"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: hdr_fmt,
        usage: hdr_final_usage,
        view_formats: &[],
    });
    let hdr_view = hdr_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let nrm_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("normal"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: hdr_fmt,
        usage: color_usage,
        view_formats: &[],
    });
    let nrm_view = nrm_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("main_depth"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: depth_usage,
        view_formats: &[],
    });
    let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let bloom_a_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bloom_a"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: hdr_fmt,
        usage: color_usage,
        view_formats: &[],
    });
    let bloom_a_view = bloom_a_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let bloom_b_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bloom_b"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: hdr_fmt,
        usage: color_usage,
        view_formats: &[],
    });
    let bloom_b_view = bloom_b_tex.create_view(&wgpu::TextureViewDescriptor::default());

    (
        hdr_opaque_tex,
        hdr_opaque_view,
        hdr_tex,
        hdr_view,
        nrm_tex,
        nrm_view,
        depth_tex,
        depth_view,
        bloom_a_tex,
        bloom_a_view,
        bloom_b_tex,
        bloom_b_view,
    )
}

impl WgpuViewer {
    pub async fn new(window: impl wgpu::WindowHandle + 'static) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance.create_surface(window).map_err(|e| e.to_string())?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| "no wgpu adapter".to_string())?;
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        // WebGPU defaults cap a single storage buffer at 128 MiB; GPU greedy mesh scratch can exceed that.
        // Ask for the adapter maximum so large scenes can still use the compute path when hardware allows.
        let adapter_limits = adapter.limits();
        let mut required_limits = wgpu::Limits::default();
        required_limits.max_storage_buffer_binding_size =
            adapter_limits.max_storage_buffer_binding_size;
        required_limits.max_buffer_size = adapter_limits.max_buffer_size;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("voxelle-desktop"),
                    required_features: wgpu::Features::empty(),
                    required_limits,
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                },
                None,
            )
            .await
            .map_err(|e| e.to_string())?;

        let size = (800u32, 600u32);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.0.max(1),
            height: size.1.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let scene_layout0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });

        let scene_layout1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene1"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let shadow_vs_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_vs"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let post_bloom_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post_bloom"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let post_blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post_blur"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let post_composite_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("post_composite"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let shader_scene = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene"),
            source: wgpu::ShaderSource::Wgsl(gpu::scene::WGSL.into()),
        });
        let shader_sky = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sky"),
            source: wgpu::ShaderSource::Wgsl(gpu::sky::WGSL.into()),
        });
        let shader_shadow = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow"),
            source: wgpu::ShaderSource::Wgsl(gpu::shadow::WGSL.into()),
        });
        let shader_bloom_ex = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom_ex"),
            source: wgpu::ShaderSource::Wgsl(gpu::post_bloom_extract::WGSL.into()),
        });
        let shader_blur = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blur"),
            source: wgpu::ShaderSource::Wgsl(gpu::post_blur::WGSL.into()),
        });
        let shader_composite = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("composite"),
            source: wgpu::ShaderSource::Wgsl(gpu::post_composite::WGSL.into()),
        });
        let shader_collab_lines = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("collab_peer_lines"),
            source: wgpu::ShaderSource::Wgsl(gpu::collab_peer_lines::WGSL.into()),
        });

        let pl_opaque = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_opaque"),
            bind_group_layouts: &[&scene_layout0],
            push_constant_ranges: &[],
        });
        let pl_trans = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_trans"),
            bind_group_layouts: &[&scene_layout0, &scene_layout1],
            push_constant_ranges: &[],
        });
        let pl_shadow = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_shadow"),
            bind_group_layouts: &[&shadow_vs_layout],
            push_constant_ranges: &[],
        });
        let pl_bloom = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_bloom"),
            bind_group_layouts: &[&post_bloom_layout],
            push_constant_ranges: &[],
        });
        let pl_blur = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_blur"),
            bind_group_layouts: &[&post_blur_layout],
            push_constant_ranges: &[],
        });
        let pl_comp = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_comp"),
            bind_group_layouts: &[&post_composite_layout],
            push_constant_ranges: &[],
        });

        let vf = hdr_format();
        let pipeline_opaque = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("opaque"),
            layout: Some(&pl_opaque),
            vertex: wgpu::VertexState {
                module: &shader_scene,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_scene,
                entry_point: Some("fs_opaque_mrt"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: vf,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: vf,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let preview_targets = &[
            Some(wgpu::ColorTargetState {
                format: vf,
                blend: Some(preview_hdr_blend()),
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: vf,
                blend: None,
                write_mask: wgpu::ColorWrites::empty(),
            }),
        ];
        let pipeline_preview_occluded =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("preview_occluded"),
                layout: Some(&pl_opaque),
                vertex: wgpu::VertexState {
                    module: &shader_scene,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_scene,
                    entry_point: Some("fs_preview_occluded_mrt"),
                    targets: preview_targets,
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Greater,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState {
                        constant: 1,
                        slope_scale: 1.0,
                        clamp: 0.0,
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });
        let pipeline_preview_front =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("preview_front"),
                layout: Some(&pl_opaque),
                vertex: wgpu::VertexState {
                    module: &shader_scene,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_scene,
                    entry_point: Some("fs_preview_front_mrt"),
                    targets: preview_targets,
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Always,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        let pipeline_collab_lines_occluded =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("collab_lines_occluded"),
                layout: Some(&pl_opaque),
                vertex: wgpu::VertexState {
                    module: &shader_collab_lines,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout_collab_lines()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_collab_lines,
                    entry_point: Some("fs_collab_line_occluded"),
                    targets: preview_targets,
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Greater,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState {
                        constant: 1,
                        slope_scale: 1.0,
                        clamp: 0.0,
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });
        let pipeline_collab_lines_front =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("collab_lines_front"),
                layout: Some(&pl_opaque),
                vertex: wgpu::VertexState {
                    module: &shader_collab_lines,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout_collab_lines()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_collab_lines,
                    entry_point: Some("fs_collab_line_front"),
                    targets: preview_targets,
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Always,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        let pipeline_sky = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sky"),
            layout: Some(&pl_opaque),
            vertex: wgpu::VertexState {
                module: &shader_sky,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_sky,
                entry_point: Some("fs_sky_mrt"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: vf,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: vf,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let pipeline_trans = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("trans"),
            layout: Some(&pl_trans),
            vertex: wgpu::VertexState {
                module: &shader_scene,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_scene,
                entry_point: Some("fs_trans"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: vf,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let pipeline_shadow = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow"),
            layout: Some(&pl_shadow),
            vertex: wgpu::VertexState {
                module: &shader_shadow,
                entry_point: Some("vs_shadow"),
                buffers: &[vertex_layout()],
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let pipeline_bloom_extract = fullscreen_pipeline(
            &device,
            &pl_bloom,
            &shader_bloom_ex,
            "fs_bloom_extract",
            &[Some(wgpu::ColorTargetState {
                format: vf,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            None,
        );
        let pipeline_blur = fullscreen_pipeline(
            &device,
            &pl_blur,
            &shader_blur,
            "fs_blur",
            &[Some(wgpu::ColorTargetState {
                format: vf,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            None,
        );
        let pipeline_composite = fullscreen_pipeline(
            &device,
            &pl_comp,
            &shader_composite,
            "fs_composite",
            &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            None,
        );

        let global_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("globals"),
            contents: &vec![0u8; std::mem::size_of::<GlobalState>()],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let brick_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brick"),
            contents: &[0u8; 4],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let post_blur_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blur_u"),
            contents: bytemuck::bytes_of(&PostBlurUniform {
                blur_dir: [1.0, 0.0, 0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let post_composite_opts_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("post_composite_opts"),
            contents: bytemuck::bytes_of(&PostCompositeOpts {
                tone_mode: 0,
                _pad: [0, 0, 0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sampler_linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let sampler_comparison = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_cmp"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let sampler_nearest = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("nearest"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let (shadow_texture, shadow_view) =
            create_shadow_tex(&device, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
        let scene_bounds = MeshBounds {
            min: Vec3::splat(-10.0),
            max: Vec3::splat(10.0),
        };
        let light_dir = Vec3::new(0.35, 0.92, 0.15).normalize();

        let (
            hdr_opaque_texture,
            hdr_opaque_view,
            hdr_texture,
            hdr_view,
            normal_texture,
            normal_view,
            depth_texture,
            depth_view,
            bloom_a,
            bloom_a_view,
            bloom_b,
            bloom_b_view,
        ) = create_screen_targets(&device, size.0, size.1, vf);

        // placeholder bind groups — rebuilt in resize / upload
        let bind_scene_opaque = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_op"),
            layout: &scene_layout0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: global_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: brick_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler_comparison),
                },
            ],
        });
        let bind_shadow_pass = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_pass"),
            layout: &shadow_vs_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: global_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: brick_buffer.as_entire_binding(),
                },
            ],
        });

        let bind_bloom_extract = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom_ex"),
            layout: &post_bloom_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler_linear),
                },
            ],
        });
        let bind_blur_h = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur_h"),
            layout: &post_blur_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&bloom_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler_linear),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: post_blur_buf.as_entire_binding(),
                },
            ],
        });
        let bind_blur_v = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur_v"),
            layout: &post_blur_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&bloom_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler_linear),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: post_blur_buf.as_entire_binding(),
                },
            ],
        });
        let bind_composite = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("comp"),
            layout: &post_composite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&bloom_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler_linear),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: post_composite_opts_buf.as_entire_binding(),
                },
            ],
        });

        let bind_trans = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trans1"),
            layout: &scene_layout1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_opaque_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler_linear),
                },
            ],
        }));

        Ok(Self {
            surface,
            device,
            queue,
            config,
            format,
            size,
            global_buffer,
            brick_buffer,
            brick_cell_count: 1,
            brick_origin_iv: IVec3::ZERO,
            brick_dims_u: (0, 0, 0),
            scene_bounds,
            light_dir,
            shadow_texture,
            shadow_view,
            hdr_opaque_texture,
            hdr_opaque_view,
            hdr_texture,
            hdr_view,
            normal_texture,
            normal_view,
            depth_texture,
            depth_view,
            bloom_a,
            bloom_a_view,
            bloom_b,
            bloom_b_view,
            scene_layout0,
            scene_layout1,
            shadow_vs_layout,
            post_bloom_layout,
            post_blur_layout,
            post_composite_layout,
            bind_scene_opaque,
            bind_shadow_pass,
            bind_bloom_extract,
            bind_blur_h,
            bind_blur_v,
            bind_composite,
            bind_trans,
            post_blur_buf,
            post_composite_opts_buf,
            pipeline_opaque,
            pipeline_preview_occluded,
            pipeline_preview_front,
            pipeline_collab_lines_occluded,
            pipeline_collab_lines_front,
            pipeline_sky,
            pipeline_trans,
            pipeline_shadow,
            pipeline_bloom_extract,
            pipeline_blur,
            pipeline_composite,
            vertex_buffer: None,
            index_buffer: None,
            index_count: 0,
            opaque_chunked: false,
            chunk_grid_origin: IVec3::ZERO,
            opaque_chunks: BTreeMap::new(),
            spatial_mesh_cache: None,
            preview_vertex_buffer: None,
            preview_index_buffer: None,
            preview_index_count: 0,
            preview_wire_vertex_buffer: None,
            preview_wire_index_buffer: None,
            preview_wire_index_count: 0,
            collab_line_vertex_buffer: None,
            collab_line_vertex_count: 0,
            ping_vertex_buffer: None,
            ping_index_buffer: None,
            ping_index_count: 0,
            ping_wire_vertex_buffer: None,
            ping_wire_index_buffer: None,
            ping_wire_index_count: 0,
            ping_mesh_key: None,
            preview_cache_key: None,
            sampler_linear,
            sampler_comparison,
            sampler_nearest,
            mesh_greedy_pipeline: None,
            mesh_greedy_bind_layout: None,
            mesh_greedy_pl_version: 0,
            mesh_greedy_pool: MeshGreedyPool::default(),
            last_mesh_route: String::new(),
        })
    }

    pub fn opaque_index_count(&self) -> u32 {
        if self.opaque_chunked {
            self.opaque_chunks.values().map(|c| c.index_count).sum()
        } else {
            self.index_count
        }
    }

    /// Vertex buffer size / interleaved stride ([`OPAQUE_VERTEX_STRIDE`] bytes per vertex).
    pub fn opaque_vertex_buffer_vertices(&self) -> u32 {
        if self.opaque_chunked {
            self.opaque_chunks
                .values()
                .map(|c| (c.vertex_buffer.size() / OPAQUE_VERTEX_STRIDE) as u32)
                .sum()
        } else {
            self.vertex_buffer
                .as_ref()
                .map(|b| (b.size() / OPAQUE_VERTEX_STRIDE) as u32)
                .unwrap_or(0)
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.size = (width, height);
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        let vf = hdr_format();
        let (
            hdr_opaque_texture,
            hdr_opaque_view,
            hdr_texture,
            hdr_view,
            normal_texture,
            normal_view,
            depth_texture,
            depth_view,
            bloom_a,
            bloom_a_view,
            bloom_b,
            bloom_b_view,
        ) = create_screen_targets(&self.device, width, height, vf);
        self.hdr_opaque_texture = hdr_opaque_texture;
        self.hdr_opaque_view = hdr_opaque_view;
        self.hdr_texture = hdr_texture;
        self.hdr_view = hdr_view;
        self.normal_texture = normal_texture;
        self.normal_view = normal_view;
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
        self.bloom_a = bloom_a;
        self.bloom_a_view = bloom_a_view;
        self.bloom_b = bloom_b;
        self.bloom_b_view = bloom_b_view;
        self.rebuild_bind_groups();
    }

    /// `mode`: 0 neutral … 5 reinhard (see `post_composite.wgsl` / Voxelle web tone mapping ids).
    pub fn set_tone_mapping_mode(&mut self, mode: u32) {
        let mode = mode.min(5);
        let opts = PostCompositeOpts {
            tone_mode: mode,
            _pad: [0, 0, 0],
        };
        self.queue.write_buffer(
            &self.post_composite_opts_buf,
            0,
            bytemuck::bytes_of(&opts),
        );
    }

    fn rebuild_bind_groups(&mut self) {
        self.bind_scene_opaque = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_op"),
            layout: &self.scene_layout0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.global_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.brick_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_comparison),
                },
            ],
        });
        self.bind_shadow_pass = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_pass"),
            layout: &self.shadow_vs_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.global_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.brick_buffer.as_entire_binding(),
                },
            ],
        });
        self.bind_bloom_extract = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom_ex"),
            layout: &self.post_bloom_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_linear),
                },
            ],
        });
        self.bind_blur_h = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur_h"),
            layout: &self.post_blur_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_linear),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.post_blur_buf.as_entire_binding(),
                },
            ],
        });
        self.bind_blur_v = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur_v"),
            layout: &self.post_blur_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_linear),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.post_blur_buf.as_entire_binding(),
                },
            ],
        });
        self.bind_composite = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("comp"),
            layout: &self.post_composite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_linear),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.post_composite_opts_buf.as_entire_binding(),
                },
            ],
        });
        self.bind_trans = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trans1"),
            layout: &self.scene_layout1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.hdr_opaque_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_linear),
                },
            ],
        }));
    }

    fn interleaved_from_mesh(mesh: &MeshBuffers) -> Vec<f32> {
        let n = mesh.positions.len() / 3;
        let mut interleaved: Vec<f32> = Vec::with_capacity(n * 11);
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
        }
        interleaved
    }

    fn opaque_draw_from_mesh(&self, mesh: &MeshBuffers) -> OpaqueChunkDraw {
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
            index_count: mesh.indices.len() as u32,
        }
    }

    /// If existing chunk buffers are large enough, overwrite with [`queue::write_buffer`]; else allocate new.
    fn upload_or_replace_chunk_mesh(&mut self, key: ChunkKey, mesh: &MeshBuffers) {
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
            draw.index_count = mesh.indices.len() as u32;
        } else {
            self.opaque_chunks
                .insert(key, self.opaque_draw_from_mesh(mesh));
        }
    }

    pub fn upload_mesh(&mut self, mesh: &MeshBuffers) {
        self.opaque_chunked = false;
        self.opaque_chunks.clear();
        self.spatial_mesh_cache = None;
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
    }

    /// Full CPU chunked mesh upload (all spatial chunks). Used on load and when chunk origin shifts.
    pub fn upload_cpu_mesh_chunked_full(&mut self, voxels: &[Voxel]) {
        self.vertex_buffer = None;
        self.index_buffer = None;
        self.index_count = 0;
        self.opaque_chunks.clear();
        let Some((origin, meshes, spatial_cache)) =
            greedy_mesh::build_chunk_meshes_and_spatial_cache(voxels, greedy_mesh::SPATIAL_CHUNK_SIZE)
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
        for (key, mesh) in meshes {
            self.opaque_chunks
                .insert(key, self.opaque_draw_from_mesh(&mesh));
        }
        self.spatial_mesh_cache = Some(spatial_cache);
        self.last_mesh_route = "cpu_chunked".to_string();
    }

    /// Apply one edit to [`Self::spatial_mesh_cache`] (must match `current_file.voxels` after `apply_edit`).
    pub fn apply_spatial_cache_edit(&mut self, delta: &VoxelEditDelta) {
        let Some(ref mut cache) = self.spatial_mesh_cache else {
            return;
        };
        let cs = greedy_mesh::SPATIAL_CHUNK_SIZE;
        match delta {
            VoxelEditDelta::Added(v) => cache.apply_add(*v, cs),
            VoxelEditDelta::Removed { voxel } => cache.apply_remove(voxel.x, voxel.y, voxel.z, cs),
        }
    }

    /// Rebuild GPU buffers for `keys` only. Returns `true` if only those chunks were updated; `false` if a full chunked upload ran (origin drift).
    pub fn remesh_opaque_chunks(
        &mut self,
        keys: &[ChunkKey],
        voxels: &[Voxel],
    ) -> (bool, RemeshOpaquePerf) {
        let mut perf = RemeshOpaquePerf::default();
        let cs = greedy_mesh::SPATIAL_CHUNK_SIZE;

        if self.spatial_mesh_cache.is_none() {
            let t_cold = Instant::now();
            self.spatial_mesh_cache = greedy_mesh::SpatialMeshCache::from_voxels(voxels, cs);
            perf.buckets_ms = t_cold.elapsed().as_secs_f64() * 1000.0;
        }
        let Some(cache) = self.spatial_mesh_cache.as_ref() else {
            self.upload_mesh(&MeshBuffers::default());
            return (false, perf);
        };
        let origin_iv = IVec3::new(cache.origin.0, cache.origin.1, cache.origin.2);
        if origin_iv != self.chunk_grid_origin {
            let t_full = Instant::now();
            self.upload_cpu_mesh_chunked_full(voxels);
            perf.full_chunked_rebuild_ms = t_full.elapsed().as_secs_f64() * 1000.0;
            return (false, perf);
        }

        let t_greedy = Instant::now();
        let chunk_meshes: Vec<(ChunkKey, MeshBuffers)> = {
            let cache = self.spatial_mesh_cache.as_ref().unwrap();
            keys.par_iter()
                .map(|key| {
                    let mesh = greedy_mesh::mesh_buffers_for_chunk_key(
                        &cache.buckets,
                        &cache.occupancy,
                        *key,
                    );
                    (*key, mesh)
                })
                .collect()
        };
        perf.greedy_ms = t_greedy.elapsed().as_secs_f64() * 1000.0;

        for (key, mesh) in chunk_meshes {
            if mesh.indices.is_empty() {
                self.opaque_chunks.remove(&key);
            } else {
                let t_u = Instant::now();
                self.upload_or_replace_chunk_mesh(key, &mesh);
                perf.chunk_buffers_ms += t_u.elapsed().as_secs_f64() * 1000.0;
            }
        }
        (true, perf)
    }

    /// CPU greedy mesh, using chunked construction for very large voxel counts.
    pub fn cpu_mesh_fallback(&mut self, voxels: &[Voxel]) {
        if voxels.len() >= greedy_mesh::CHUNKED_CPU_MESH_MIN_VOXELS {
            self.upload_cpu_mesh_chunked_full(voxels);
        } else {
            let (mesh, _) = greedy_mesh::build_greedy_mesh(voxels);
            self.upload_mesh(&mesh);
            self.last_mesh_route = "cpu".to_string();
        }
    }

    /// GPU greedy mesh (WGSL) when slice bitmaps fit 64×64; otherwise CPU [`greedy_mesh::build_greedy_mesh`].
    /// Set `VOXELLE_CPU_MESH=1` to force CPU meshing.
    pub fn rebuild_mesh_gpu_greedy(&mut self, voxels: &[Voxel]) -> Result<MeshBounds, String> {
        if voxels.is_empty() {
            self.vertex_buffer = None;
            self.index_buffer = None;
            self.index_count = 0;
            self.opaque_chunked = false;
            self.opaque_chunks.clear();
            self.spatial_mesh_cache = None;
            return Err("empty voxels".into());
        }

        let bounds = greedy_mesh::mesh_bounds_from_voxels(voxels).ok_or("mesh bounds")?;

        if std::env::var("VOXELLE_CPU_MESH").is_ok() {
            self.cpu_mesh_fallback(voxels);
            return Ok(bounds);
        }

        let map = greedy_mesh::voxel_map(voxels);
        let (headers, bits) = match greedy_mesh::pack_gpu_greedy_slices(&map, voxels) {
            Ok(x) => x,
            Err(()) => {
                self.cpu_mesh_fallback(voxels);
                return Ok(bounds);
            }
        };

        if headers.is_empty() {
            self.vertex_buffer = None;
            self.index_buffer = None;
            self.index_count = 0;
            self.opaque_chunked = false;
            self.opaque_chunks.clear();
            self.spatial_mesh_cache = None;
            self.last_mesh_route = "gpu_no_headers".to_string();
            return Ok(bounds);
        }

        let mut acc_v: u32 = 0;
        let mut acc_i: u32 = 0;
        for h in &headers {
            acc_v = acc_v.saturating_add(4u32.saturating_mul(h.width.saturating_mul(h.height)));
            acc_i = acc_i.saturating_add(6u32.saturating_mul(h.width.saturating_mul(h.height)));
        }
        let max_vertices = acc_v.max(1);
        let max_indices = acc_i.max(1);

        const VTX_STRIDE: u64 = OPAQUE_VERTEX_STRIDE;
        let vtx_storage_size = (max_vertices as u64).saturating_mul(VTX_STRIDE);
        let idx_storage_size = (max_indices as u64).saturating_mul(4);

        // Worst-case scratch scales with slice count × 64×64 per tile; huge scenes exceed max_buffer_size.
        // Default WebGPU `max_storage_buffer_binding_size` is 128 MiB — bind group creation fails if exceeded.
        // `dispatch_workgroups` also has a per-dimension cap (typically 65535).
        let lim = self.device.limits();
        let max_wg = lim.max_compute_workgroups_per_dimension as usize;
        let max_bind = lim.max_storage_buffer_binding_size as u64;
        if vtx_storage_size > lim.max_buffer_size
            || idx_storage_size > lim.max_buffer_size
            || vtx_storage_size > max_bind
            || idx_storage_size > max_bind
            || headers.len() > max_wg
        {
            self.cpu_mesh_fallback(voxels);
            return Ok(bounds);
        }

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
                let buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mesh_greedy_brick_halo"),
                    contents: bytemuck::cast_slice(&cells),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                (
                    glam::IVec3::new(ho.0, ho.1, ho.2),
                    hd,
                    Some(buf),
                )
            }
            None => (self.brick_origin_iv, self.brick_dims_u, None),
        };
        let mesh_brick_binding = mesh_brick_halo_buf
            .as_ref()
            .map(|b| b.as_entire_binding())
            .unwrap_or_else(|| self.brick_buffer.as_entire_binding());

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

        self.mesh_greedy_pool.ensure_counters(&self.device);
        self.mesh_greedy_pool
            .ensure_vtx_out(&self.device, vtx_storage_size);
        self.mesh_greedy_pool
            .ensure_idx_out(&self.device, idx_storage_size);
        self.mesh_greedy_pool.ensure_readback(&self.device);
        let counters_buf = self.mesh_greedy_pool.counters.as_ref().unwrap();
        self.queue.write_buffer(counters_buf, 0, &[0u8; 8]);
        let hdr_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh_slice_hdr"),
                contents: bytemuck::cast_slice(&headers),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let bits_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh_slice_bits"),
                contents: bytemuck::cast_slice(&bits),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let vtx_out = self.mesh_greedy_pool.vtx_scratch.as_ref().unwrap();
        let idx_out = self.mesh_greedy_pool.idx_scratch.as_ref().unwrap();
        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh_greedy_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        if self.mesh_greedy_pl_version != MESH_GREEDY_PIPELINE_LAYOUT_VERSION {
            self.mesh_greedy_bind_layout = None;
            self.mesh_greedy_pipeline = None;
            self.mesh_greedy_pl_version = MESH_GREEDY_PIPELINE_LAYOUT_VERSION;
        }

        if self.mesh_greedy_bind_layout.is_none() {
            let layout = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("mesh_greedy"),
                    source: wgpu::ShaderSource::Wgsl(gpu::mesh_greedy::WGSL.into()),
                });
            let pl = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("mesh_greedy_pl"),
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                });
            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("mesh_greedy"),
                    layout: Some(&pl),
                    module: &shader,
                    entry_point: Some("greedy_slice"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            self.mesh_greedy_bind_layout = Some(layout);
            self.mesh_greedy_pipeline = Some(pipeline);
        }

        let layout = self.mesh_greedy_bind_layout.as_ref().unwrap();
        let pipeline = self.mesh_greedy_pipeline.as_ref().unwrap();

        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                    resource: mesh_brick_binding,
                },
            ],
        });

        let readback = self.mesh_greedy_pool.readback.as_ref().unwrap();

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
        self.queue.submit(std::iter::once(enc.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let map_r = rx.recv().map_err(|_| "mesh count readback channel")?;
        map_r.map_err(|e| e.to_string())?;
        let data = slice.get_mapped_range();
        let v_total = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let i_total = u32::from_le_bytes(data[4..8].try_into().unwrap());
        drop(data);
        readback.unmap();

        if v_total == 0 || i_total == 0 || v_total > max_vertices || i_total > max_indices {
            self.cpu_mesh_fallback(voxels);
            return Ok(bounds);
        }

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
            &vtx_out,
            0,
            &vb_final,
            0,
            (v_total as u64).saturating_mul(VTX_STRIDE),
        );
        enc3.copy_buffer_to_buffer(
            &idx_out,
            0,
            &ib_final,
            0,
            (i_total as u64).saturating_mul(4),
        );
        self.queue.submit(std::iter::once(enc3.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        self.opaque_chunked = false;
        self.opaque_chunks.clear();
        self.spatial_mesh_cache = None;
        self.vertex_buffer = Some(vb_final);
        self.index_buffer = Some(ib_final);
        self.index_count = i_total;
        self.last_mesh_route = "gpu_greedy".to_string();
        Ok(bounds)
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

    pub fn clear_preview_mesh(&mut self) {
        self.preview_vertex_buffer = None;
        self.preview_index_buffer = None;
        self.preview_index_count = 0;
        self.preview_wire_vertex_buffer = None;
        self.preview_wire_index_buffer = None;
        self.preview_wire_index_count = 0;
        self.preview_cache_key = None;
    }

    /// Line list: each vertex is `[x,y,z, r,g,b]` (6 floats); pairs form eye→target segments.
    pub fn upload_collab_peer_lines(&mut self, verts: &[f32]) {
        if verts.is_empty() || verts.len() % 6 != 0 {
            self.collab_line_vertex_buffer = None;
            self.collab_line_vertex_count = 0;
            return;
        }
        let n_floats = verts.len();
        let vertex_count = (n_floats / 6) as u32;
        let nbytes = (n_floats * std::mem::size_of::<f32>()) as u64;
        if let Some(ref buf) = self.collab_line_vertex_buffer {
            if buf.size() == nbytes {
                self.queue.write_buffer(buf, 0, bytemuck::cast_slice(verts));
                self.collab_line_vertex_count = vertex_count;
                return;
            }
        }
        self.collab_line_vertex_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("collab_peer_lines_vtx"),
                contents: bytemuck::cast_slice(verts),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            },
        ));
        self.collab_line_vertex_count = vertex_count;
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
        self.ping_mesh_key = None;
    }

    /// Updates GPU voxel brick. When `patch` is set and matches the existing brick layout, only one cell is written.
    pub fn upload_scene_data(
        &mut self,
        bounds: MeshBounds,
        voxels: &[Voxel],
        patch: Option<BrickCellWrite>,
    ) {
        self.scene_bounds = bounds;
        const MAX_AXIS: u32 = 512;
        if let (Some(layout), Some(p)) =
            (GpuVoxelBrick::layout_from_voxels(voxels, MAX_AXIS), patch)
        {
            if layout.origin == self.brick_origin_iv && layout.dims == self.brick_dims_u {
                if let Some(off) = layout.index_of_world(p.x, p.y, p.z) {
                    self.queue.write_buffer(
                        &self.brick_buffer,
                        (off * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
                        &p.packed.to_le_bytes(),
                    );
                    return;
                }
            }
        }

        let brick = GpuVoxelBrick::from_voxels(voxels, MAX_AXIS).unwrap_or(GpuVoxelBrick {
            origin: IVec3::ZERO,
            dims: (0, 0, 0),
            cells: vec![0u32],
        });
        self.brick_origin_iv = brick.origin;
        self.brick_dims_u = brick.dims;
        self.brick_cell_count = brick.cells.len().max(1) as u32;
        let new_brick = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("brick"),
                contents: bytemuck::cast_slice(&brick.cells),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        self.brick_buffer = new_brick;
        self.rebuild_bind_groups();
    }

    /// Same as [`Self::upload_scene_data`] with `patch == None`, but uses a pre-built brick (CPU work done off-thread).
    pub fn upload_scene_data_from_brick(&mut self, bounds: MeshBounds, brick: GpuVoxelBrick) {
        self.scene_bounds = bounds;
        self.brick_origin_iv = brick.origin;
        self.brick_dims_u = brick.dims;
        self.brick_cell_count = brick.cells.len().max(1) as u32;
        let new_brick = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("brick"),
                contents: bytemuck::cast_slice(&brick.cells),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        self.brick_buffer = new_brick;
        self.rebuild_bind_groups();
    }

    pub(crate) fn upload_prepared_opaque(&mut self, opaque: PreparedOpaqueUpload) {
        match opaque {
            PreparedOpaqueUpload::Empty => {
                self.vertex_buffer = None;
                self.index_buffer = None;
                self.index_count = 0;
                self.opaque_chunked = false;
                self.opaque_chunks.clear();
                self.spatial_mesh_cache = None;
                self.last_mesh_route = "cpu".to_string();
            }
            PreparedOpaqueUpload::Single(mesh) => {
                self.upload_mesh(&mesh);
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
                self.opaque_chunks.clear();
                self.chunk_grid_origin = chunk_origin;
                if meshes.is_empty() {
                    self.opaque_chunked = false;
                    self.spatial_mesh_cache = None;
                } else {
                    self.opaque_chunked = true;
                    for (key, mesh) in meshes {
                        self.opaque_chunks.insert(key, self.opaque_draw_from_mesh(&mesh));
                    }
                    self.spatial_mesh_cache = Some(spatial_cache);
                    self.last_mesh_route = "cpu_chunked".to_string();
                }
            }
        }
    }

    pub fn update_uniforms(&self, camera: &OrbitCamera) {
        let w = self.size.0 as f32;
        let h = self.size.1 as f32;
        let proj = camera.proj_matrix(w, h);
        let view = camera.view_matrix();
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
            params: [0.0, 0.0, BLOOM_STRENGTH, camera.near],
        };
        self.queue
            .write_buffer(&self.global_buffer, 0, bytemuck::bytes_of(&gs));
    }

    fn draw_indexed_mesh(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.opaque_chunked {
            for ch in self.opaque_chunks.values() {
                if ch.index_count == 0 {
                    continue;
                }
                pass.set_vertex_buffer(0, ch.vertex_buffer.slice(..));
                pass.set_index_buffer(ch.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..ch.index_count, 0, 0..1);
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

    fn draw_indexed_preview(&self, pass: &mut wgpu::RenderPass<'_>) {
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
                // Triangle edges (not LineList): same pipelines as solid preview.
                pass.set_pipeline(&self.pipeline_preview_occluded);
                pass.draw_indexed(0..self.preview_wire_index_count, 0, 0..1);
                pass.set_pipeline(&self.pipeline_preview_front);
                pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
                pass.draw_indexed(0..self.preview_wire_index_count, 0, 0..1);
            }
        }
    }

    fn draw_indexed_ping(&self, pass: &mut wgpu::RenderPass<'_>) {
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
                pass.set_pipeline(&self.pipeline_preview_occluded);
                pass.draw_indexed(0..self.ping_wire_index_count, 0, 0..1);
                pass.set_pipeline(&self.pipeline_preview_front);
                pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
                pass.draw_indexed(0..self.ping_wire_index_count, 0, 0..1);
            }
        }
    }

    fn draw_collab_peer_lines(&self, pass: &mut wgpu::RenderPass<'_>) {
        let Some(ref vb) = self.collab_line_vertex_buffer else {
            return;
        };
        if self.collab_line_vertex_count < 2 {
            return;
        }
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
        pass.set_pipeline(&self.pipeline_collab_lines_occluded);
        pass.draw(0..self.collab_line_vertex_count, 0..1);
        pass.set_pipeline(&self.pipeline_collab_lines_front);
        pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
        pass.draw(0..self.collab_line_vertex_count, 0..1);
    }

    pub fn render(&self) -> Result<(), String> {
        let frame = self
            .surface
            .get_current_texture()
            .map_err(|e| e.to_string())?;
        let swap_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

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
            self.draw_indexed_mesh(&mut pass);
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
            pass.set_pipeline(&self.pipeline_sky);
            pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
            pass.draw(0..3, 0..1);
            pass.set_pipeline(&self.pipeline_opaque);
            pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
            self.draw_indexed_mesh(&mut pass);
            self.draw_indexed_preview(&mut pass);
            self.draw_collab_peer_lines(&mut pass);
            self.draw_indexed_ping(&mut pass);
        }

        let ext = wgpu::Extent3d {
            width: self.size.0.max(1),
            height: self.size.1.max(1),
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

        if let Some(ref trans_bg) = self.bind_trans {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("trans"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
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
            pass.set_pipeline(&self.pipeline_trans);
            pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
            pass.set_bind_group(1, trans_bg, &[]);
            self.draw_indexed_mesh(&mut pass);
        }

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

        let blur_h = PostBlurUniform {
            blur_dir: [1.0, 0.0, 0.0, 0.0],
        };
        self.queue
            .write_buffer(&self.post_blur_buf, 0, bytemuck::bytes_of(&blur_h));
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blur_h"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_b_view,
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
            pass.set_bind_group(0, &self.bind_blur_h, &[]);
            pass.draw(0..3, 0..1);
        }

        let blur_v = PostBlurUniform {
            blur_dir: [0.0, 1.0, 0.0, 0.0],
        };
        self.queue
            .write_buffer(&self.post_blur_buf, 0, bytemuck::bytes_of(&blur_v));
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blur_v"),
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
            pass.set_pipeline(&self.pipeline_blur);
            pass.set_bind_group(0, &self.bind_blur_v, &[]);
            pass.draw(0..3, 0..1);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &swap_view,
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

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }
}

#[cfg(test)]
mod global_state_tests {
    use super::GlobalState;

    #[test]
    fn global_state_storage_aligned() {
        assert_eq!(std::mem::size_of::<GlobalState>() % 16, 0);
        assert!(std::mem::size_of::<GlobalState>() >= 256);
    }
}
