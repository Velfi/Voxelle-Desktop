//! Multi-pass GPU renderer: shadow map, HDR+MRT, transmission, bloom, composite.
//!
//! Synchronization is WebGPU-style: the implementation inserts layout transitions and barriers;
//! we only need valid `TextureUsages` / pass ordering and copies where a texture cannot be both
//! written and sampled in one pass (e.g. depth snapshot for SSR).

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
    pub mod meter_luminance {
        pub const WGSL: &str = include_str!("meter_luminance.wgsl");
    }
    pub mod post_blur {
        pub const WGSL: &str = include_str!("post_blur.wgsl");
    }
    pub mod post_blit {
        pub const WGSL: &str = include_str!("post_blit.wgsl");
    }
    pub mod post_composite {
        pub const WGSL: &str = include_str!("post_composite.wgsl");
    }
    pub mod sky {
        pub const WGSL: &str = include_str!("sky.wgsl");
    }
    pub mod start_screen_bg {
        pub const WGSL: &str = include_str!("start_screen_bg.wgsl");
    }
    pub mod mesh_greedy {
        pub const WGSL: &str = include_str!("gpu/mesh_greedy.wgsl");
    }
    pub mod preview_fill_occ {
        pub const WGSL: &str = include_str!("gpu/preview_fill_occ.wgsl");
    }
    pub mod preview_shell_emit {
        pub const WGSL: &str = include_str!("gpu/preview_shell_emit.wgsl");
    }
    pub mod collab_peer_lines {
        pub const WGSL: &str = include_str!("collab_peer_lines.wgsl");
    }
    pub mod avatar {
        pub const WGSL: &str = include_str!("avatar.wgsl");
    }
    pub mod ray_trace {
        pub const WGSL: &str = include_str!("ray_trace.wgsl");
    }
    pub mod oit_composite {
        pub const WGSL: &str = include_str!("oit_composite.wgsl");
    }
    pub mod post_ssr {
        pub const WGSL: &str = include_str!("post_ssr.wgsl");
    }
    pub mod mascot {
        pub const WGSL: &str = include_str!("mascot.wgsl");
    }
    pub mod speech_bubble {
        pub const WGSL: &str = include_str!("speech_bubble.wgsl");
    }
}

mod mood;
use mood::hex_to_linear_rgb;
pub use mood::MoodParams;

mod uniforms;
use uniforms::*;

mod speech_bubble_impl;
pub use speech_bubble_impl::SpeechBubble;

mod mesh_upload;
pub(crate) use mesh_upload::{
    compute_greedy_rebuild_cpu, OpaqueChunkDraw, PreparedGreedyRebuild, PreparedOpaqueUpload,
    RemeshOpaquePerf,
};

mod mesh_greedy;
pub(crate) use mesh_greedy::MeshGreedyPool;

mod gpu_resources;
pub(crate) use gpu_resources::*;

mod frame;
pub use frame::RaytraceBenchmarkResult;

mod overlays;

mod pipelines;
use pipelines::*;

mod avatar_impl;
pub use avatar_impl::{AvatarMeshData, AvatarPeerEntry};

mod text;
pub use text::GpuPeerLabel;

mod mascot_impl;
pub use mascot_impl::{LogoOverlay, MascotEntry};

use crate::camera::OrbitCamera;
use crate::gpu_brick::{BrickCellWrite, GpuVoxelBrick};
use crate::greedy_mesh::{self, ChunkKey, MeshBounds, MeshBuffers};
use crate::render_constants::{BLOOM_STRENGTH, SHADOW_MAP_SIZE};
use crate::voxel_edit::VoxelEditDelta;
use crate::voxelle::{SceneObject, Voxel};
use glam::{IVec3, Mat4, Vec3};
use glyphon::{
    Attrs, Buffer as GlyphonBuffer, Cache as GlyphonCache, Color as GlyphonColor, Family,
    FontSystem, Metrics, Resolution, Shaping, SwashCache, TextArea, TextAtlas, TextBounds,
    TextRenderer, Viewport as GlyphonViewport,
};
use serde_json::json;
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::SystemTime;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Runtime};
use wgpu::util::DeviceExt;

/// Bump when [`gpu::mesh_greedy::WGSL`] bind group layout changes.
const MESH_GREEDY_PIPELINE_LAYOUT_VERSION: u32 = 2;

fn debug_log(hypothesis_id: &str, location: &str, message: &str, data: serde_json::Value) {
    let ts = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let payload = json!({
        "sessionId": "373ecd",
        "runId": "run-pre-fix-1",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": ts
    });
    if let Ok(mut f) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("C:\\Users\\zelda\\Documents\\Voxelle-Desktop\\debug-373ecd.log")
    {
        let _ = writeln!(f, "{}", payload);
    }
}

pub struct WgpuViewer {
    pub(crate) surface: wgpu::Surface<'static>,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) config: wgpu::SurfaceConfiguration,
    pub(crate) format: wgpu::TextureFormat,
    /// The SDR surface format (sRGB) chosen at init.
    pub(crate) sdr_format: wgpu::TextureFormat,
    /// HDR surface format (Rgba16Float) if the display supports it.
    pub(crate) hdr_surface_format: Option<wgpu::TextureFormat>,
    /// Whether HDR output is currently active.
    pub(crate) hdr_output: bool,
    /// Swapchain / Metal drawable size (full webview — must match window or macOS stretches the image).
    pub(crate) surface_size: (u32, u32),
    /// `.viewport` div in physical pixels (same space as [`Self::surface_size`]).
    pub(crate) viewport_x: u32,
    pub(crate) viewport_y: u32,
    pub(crate) viewport_width: u32,
    pub(crate) viewport_height: u32,

    pub(crate) global_buffer: wgpu::Buffer,
    pub(crate) brick_buffer: wgpu::Buffer,
    pub(crate) brick_cell_count: u32,
    pub(crate) brick_origin_iv: IVec3,
    pub(crate) brick_dims_u: (u32, u32, u32),

    pub(crate) scene_bounds: MeshBounds,
    pub(crate) light_dir: Vec3,
    pub(crate) light_ambient: f32,
    pub(crate) light_sun: f32,
    pub(crate) sun_color_linear: Vec3,
    pub(crate) bg_color_linear: Vec3,
    pub(crate) shadows_enabled: bool,
    pub(crate) soft_shadows: bool,
    pub(crate) sky_enabled: bool,

    #[allow(dead_code)]
    pub(crate) shadow_texture: wgpu::Texture,
    pub(crate) shadow_view: wgpu::TextureView,

    /// Opaque + glow only — sampled during transmission (never the active color target at the same time).
    pub(crate) hdr_opaque_texture: wgpu::Texture,
    pub(crate) hdr_opaque_view: wgpu::TextureView,
    /// After copy from opaque + transmission pass; bloom/composite use this.
    pub(crate) hdr_texture: wgpu::Texture,
    pub(crate) hdr_view: wgpu::TextureView,
    pub(crate) normal_texture: wgpu::Texture,
    pub(crate) normal_view: wgpu::TextureView,
    pub(crate) depth_texture: wgpu::Texture,
    pub(crate) depth_view: wgpu::TextureView,
    /// Snapshot of the opaque depth, copied before the trans pass so fs_trans can
    /// read depth for SSR without conflicting with the depth-stencil write attachment.
    pub(crate) depth_snapshot_texture: wgpu::Texture,
    pub(crate) depth_snapshot_view: wgpu::TextureView,

    /// OIT accumulation texture (Rgba16Float): weighted color + alpha accumulation.
    pub(crate) oit_accum_texture: wgpu::Texture,
    pub(crate) oit_accum_view: wgpu::TextureView,
    /// OIT revealage texture (R16Float): product of (1 − alpha) across all transparent layers.
    pub(crate) oit_revealage_texture: wgpu::Texture,
    pub(crate) oit_revealage_view: wgpu::TextureView,

    pub(crate) bloom_a: wgpu::Texture,
    pub(crate) bloom_a_view: wgpu::TextureView,
    pub(crate) bloom_b: wgpu::Texture,
    pub(crate) bloom_b_view: wgpu::TextureView,
    /// Pyramid mip levels for bloom: index 0 = 1/2 res, 1 = 1/4, …, 4 = 1/32.
    pub(crate) bloom_pyramid_a: Vec<wgpu::Texture>,
    pub(crate) bloom_pyramid_a_views: Vec<wgpu::TextureView>,
    pub(crate) bloom_pyramid_b: Vec<wgpu::Texture>,
    pub(crate) bloom_pyramid_b_views: Vec<wgpu::TextureView>,

    pub(crate) present_texture: wgpu::Texture,
    pub(crate) present_view: wgpu::TextureView,

    pub(crate) scene_layout0: wgpu::BindGroupLayout,
    pub(crate) scene_layout1: wgpu::BindGroupLayout,
    pub(crate) shadow_vs_layout: wgpu::BindGroupLayout,
    pub(crate) post_bloom_layout: wgpu::BindGroupLayout,
    pub(crate) post_blur_layout: wgpu::BindGroupLayout,
    pub(crate) post_composite_layout: wgpu::BindGroupLayout,
    pub(crate) oit_composite_layout: wgpu::BindGroupLayout,

    pub(crate) bind_scene_opaque: wgpu::BindGroup,
    pub(crate) bind_shadow_pass: wgpu::BindGroup,
    pub(crate) bind_bloom_extract: wgpu::BindGroup,
    pub(crate) bind_blur_h: wgpu::BindGroup,
    pub(crate) bind_blur_v: wgpu::BindGroup,
    pub(crate) bind_composite: wgpu::BindGroup,
    /// Pyramid bloom bind groups — rebuilt on resize via rebuild_bind_groups.
    pub(crate) bind_blit_down: Vec<wgpu::BindGroup>,
    pub(crate) bind_blit_up: Vec<wgpu::BindGroup>,
    /// Weighted upsample bind groups (use post_blur_layout + post_blit_weight_buf).
    pub(crate) bind_blit_up_weighted: Vec<wgpu::BindGroup>,
    pub(crate) bind_blit_final: wgpu::BindGroup,
    pub(crate) bind_blur_pyr_h: Vec<wgpu::BindGroup>,
    pub(crate) bind_blur_pyr_v: Vec<wgpu::BindGroup>,
    pub(crate) bind_trans: Option<wgpu::BindGroup>,
    pub(crate) bind_oit_composite: wgpu::BindGroup,

    pub(crate) post_blur_buf: wgpu::Buffer,
    /// Feeds exposure_ev into the bloom extract shader for physical threshold scaling.
    pub(crate) bloom_extract_buf: wgpu::Buffer,
    /// Constant weight (0.75) for the weighted bloom upsample pyramid.
    pub(crate) post_blit_weight_buf: wgpu::Buffer,
    pub(crate) post_composite_opts_buf: wgpu::Buffer,
    pub(crate) post_composite_opts: PostCompositeOpts,

    pub(crate) pipeline_opaque: wgpu::RenderPipeline,
    /// Web-style ghost: occluded (Greater) then front (Always), unlit + alpha blend; no gbuffer writes.
    pub(crate) pipeline_preview_occluded: wgpu::RenderPipeline,
    pub(crate) pipeline_preview_front: wgpu::RenderPipeline,
    /// Same as [`pipeline_preview_front`] but **no** depth bias — wireframe edges use true depth
    /// so they are occluded by scene geometry when embedded in solids.
    pub(crate) pipeline_preview_front_wire: wgpu::RenderPipeline,
    // GPU-instanced preview pipelines
    pub(crate) pipeline_preview_inst_occluded: wgpu::RenderPipeline,
    pub(crate) pipeline_preview_inst_front: wgpu::RenderPipeline,
    pub(crate) pipeline_preview_inst_front_wire: wgpu::RenderPipeline,
    // Lit generator preview pipelines (opaque, self-shadowing)
    pub(crate) pipeline_gen_preview_inst_front: wgpu::RenderPipeline,
    pub(crate) pipeline_gen_preview_inst_occluded: wgpu::RenderPipeline,
    pub(crate) pipeline_gen_preview_inst_front_wire: wgpu::RenderPipeline,
    pub(crate) pipeline_collab_lines_occluded: wgpu::RenderPipeline,
    pub(crate) pipeline_collab_lines_front: wgpu::RenderPipeline,
    pub(crate) pipeline_avatar: wgpu::RenderPipeline,
    pub(crate) avatar_bind_layout: wgpu::BindGroupLayout,
    /// Shared GPU meshes keyed by avatar name; `""` = default glow dot.
    pub(crate) avatar_mesh_cache: std::collections::HashMap<String, AvatarMeshData>,
    /// One entry per visible remote peer.
    pub(crate) avatar_peers: Vec<AvatarPeerEntry>,
    /// Voxel grid borders: depth-tested only (no occluded ghost pass), semi-transparent.
    pub(crate) pipeline_grid_border_lines: wgpu::RenderPipeline,
    /// Selection transform gizmo (move arrows + rotation rings).
    pub(crate) pipeline_gizmo_lines_front: wgpu::RenderPipeline,
    pub(crate) pipeline_gizmo_lines_occluded: wgpu::RenderPipeline,
    pub(crate) pipeline_gizmo_tris_front: wgpu::RenderPipeline,
    pub(crate) pipeline_gizmo_tris_occluded: wgpu::RenderPipeline,
    /// Gizmo pipelines with depth-compare: Always — used when "always on top" is enabled.
    pub(crate) pipeline_gizmo_lines_always: wgpu::RenderPipeline,
    pub(crate) pipeline_gizmo_tris_always: wgpu::RenderPipeline,
    /// When true, gizmo is drawn over all geometry regardless of depth.
    pub(crate) gizmo_on_top: bool,
    pub(crate) pipeline_sky: wgpu::RenderPipeline,
    pub(crate) pipeline_start_screen_bg: wgpu::RenderPipeline,
    pub(crate) pipeline_oit_accum: wgpu::RenderPipeline,
    pub(crate) pipeline_oit_composite: wgpu::RenderPipeline,
    pub(crate) pipeline_shadow: wgpu::RenderPipeline,
    pub(crate) pipeline_bloom_extract: wgpu::RenderPipeline,
    pub(crate) pipeline_blur: wgpu::RenderPipeline,
    pub(crate) pipeline_blit: wgpu::RenderPipeline,
    /// Weighted additive blit for bloom upsample: multiplies by a scalar uniform before adding.
    pub(crate) pipeline_blit_weighted_add: wgpu::RenderPipeline,
    pub(crate) pipeline_composite: wgpu::RenderPipeline,
    /// Stored for pipeline rebuild on HDR toggle.
    pub(crate) shader_composite: wgpu::ShaderModule,
    /// Stored for pipeline rebuild on HDR toggle.
    pub(crate) composite_pipeline_layout: wgpu::PipelineLayout,
    /// 1×1 average linear luminance (HDR, pre-exposure) for auto exposure.
    pub(crate) pipeline_meter: wgpu::RenderPipeline,
    // ── Progressive ray tracer ────────────────────────────────────────────
    pub(crate) raytrace_enabled: bool,
    /// Ping-pong accumulation textures (Rgba16Float, viewport size).
    /// Index 0/1 alternate as read/write each frame.
    pub(crate) rt_accum_textures: [wgpu::Texture; 2],
    pub(crate) rt_accum_views: [wgpu::TextureView; 2],
    /// Which accumulator index is currently written to this frame.
    pub(crate) rt_accum_flip: bool,
    /// Running count of accumulated samples; 0 = reset / first frame.
    pub(crate) rt_sample_n: u32,
    pub(crate) rt_uniform_buf: wgpu::Buffer,
    /// Layout for group 0 (global + brick only — no shadow sampler).
    pub(crate) rt_scene_layout: wgpu::BindGroupLayout,
    /// Layout for group 1 (accum_prev + sampler + rt_uniform).
    pub(crate) rt_accum_layout: wgpu::BindGroupLayout,
    /// Bind group for group 0: global + brick.
    pub(crate) rt_scene_bg: wgpu::BindGroup,
    /// Two bind groups for group 1 — one per ping-pong side.
    /// rt_accum_bgs[i] reads from rt_accum_textures[1-i] (the non-written side).
    pub(crate) rt_accum_bgs: Vec<wgpu::BindGroup>,
    /// Half-resolution texture used as the raytrace render target in fast_preview mode.
    /// Upscaled to full-res after the pass so the rest of the pipeline is unchanged.
    pub(crate) rt_preview_tex: wgpu::Texture,
    pub(crate) rt_preview_view: wgpu::TextureView,
    /// Bind group (post_bloom_layout: tex + sampler) for the upscale blit.
    pub(crate) rt_preview_bg: wgpu::BindGroup,
    pub(crate) pipeline_raytrace: wgpu::RenderPipeline,
    /// Last camera eye used for accumulation-reset detection.
    pub(crate) rt_prev_eye: [f32; 3],
    pub(crate) rt_prev_inv_view: [[f32; 4]; 4],
    /// True when the camera moved this frame; shader uses cheap shading path.
    pub(crate) rt_fast_preview: bool,
    /// Surface normal mode for ray tracing: 0=blocky, 1=smooth, 2=puffy.
    pub(crate) rt_surface_mode: u32,

    // ── Screen-space reflections ───────────────────────────────────────────
    pub(crate) ssr_opts_buf: wgpu::Buffer,
    pub(crate) ssr_opts: SsrOpts,
    /// SSR fullscreen pass output texture (Rgba16Float, rgb=reflected colour, a=confidence).
    pub(crate) ssr_texture: wgpu::Texture,
    pub(crate) ssr_view: wgpu::TextureView,
    pub(crate) ssr_layout: wgpu::BindGroupLayout,
    pub(crate) bind_ssr: wgpu::BindGroup,
    pub(crate) pipeline_ssr_fullscreen: wgpu::RenderPipeline,
    pub(crate) meter_texture: wgpu::Texture,
    pub(crate) meter_view: wgpu::TextureView,
    pub(crate) meter_staging: wgpu::Buffer,
    pub(crate) bind_meter: wgpu::BindGroup,
    /// Pending async luminance readback from the *previous* frame's meter pass.
    /// `Some` while the GPU copy is in-flight; `None` after it has been consumed.
    pub(crate) meter_pending_rx:
        Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    /// User EV slider (−5…5); with auto exposure, added as bias on metered EV.
    pub(crate) exposure_user_ev: f32,
    pub(crate) auto_exposure_enabled: bool,
    /// Smoothed \( \log_2(\text{target}/\bar{L}) \) from metering.
    pub(crate) auto_exposure_smoothed: f32,
    /// Monotonic clock for animated shader effects (grain, atmosphere drift).
    pub(crate) creation_instant: std::time::Instant,

    /// When true, draw [`pipeline_start_screen_bg`] instead of sky (default true until a scene load sets [`ViewerState::start_screen_logo_transparent`] false).
    pub(crate) start_screen_transparent: bool,
    /// 0 = dark cold-start gradient, 1 = light (paper) — passed in [`GlobalState::params`].x for `start_screen_bg.wgsl`.
    pub(crate) start_screen_appearance: f32,

    pub(crate) vertex_buffer: Option<wgpu::Buffer>,
    pub(crate) index_buffer: Option<wgpu::Buffer>,
    pub(crate) index_count: u32,
    /// Split point in the non-chunked index buffer: indices `0..opaque_index_split` are opaque,
    /// `opaque_index_split..index_count` are transparent.
    pub(crate) opaque_index_split: u32,

    /// When set, opaque mesh is drawn from [`Self::opaque_chunks`] (multi-draw).
    pub(crate) opaque_chunked: bool,
    /// Chunk bucketing origin (must match [`greedy_mesh::voxel_buckets_by_chunk`]).
    pub(crate) chunk_grid_origin: IVec3,
    pub(crate) opaque_chunks: BTreeMap<ChunkKey, OpaqueChunkDraw>,
    /// Chunks waiting to be uploaded to GPU (progressive loading after initial batch).
    pub(crate) pending_chunk_uploads: VecDeque<(ChunkKey, MeshBuffers)>,
    /// Incremental occupancy + buckets; rebuilt on full chunked upload, updated O(1) per edit.
    pub(crate) spatial_mesh_cache: Option<greedy_mesh::SpatialMeshCache>,

    pub(crate) preview_vertex_buffer: Option<wgpu::Buffer>,
    pub(crate) preview_index_buffer: Option<wgpu::Buffer>,
    pub(crate) preview_index_count: u32,
    pub(crate) preview_wire_vertex_buffer: Option<wgpu::Buffer>,
    pub(crate) preview_wire_index_buffer: Option<wgpu::Buffer>,
    pub(crate) preview_wire_index_count: u32,
    // GPU-instanced preview buffers
    pub(crate) preview_solid_proto_vb: Option<wgpu::Buffer>,
    pub(crate) preview_solid_proto_ib: Option<wgpu::Buffer>,
    pub(crate) preview_solid_proto_idx_count: u32,
    pub(crate) preview_wire_proto_vb: Option<wgpu::Buffer>,
    pub(crate) preview_wire_proto_ib: Option<wgpu::Buffer>,
    pub(crate) preview_wire_proto_idx_count: u32,
    pub(crate) preview_solid_instance_buf: Option<wgpu::Buffer>,
    pub(crate) preview_solid_instance_count: u32,
    pub(crate) preview_wire_instance_buf: Option<wgpu::Buffer>,
    pub(crate) preview_wire_instance_count: u32,
    // Generator preview buffers (lit, opaque)
    pub(crate) gen_preview_solid_proto_vb: Option<wgpu::Buffer>,
    pub(crate) gen_preview_solid_proto_ib: Option<wgpu::Buffer>,
    pub(crate) gen_preview_solid_proto_idx_count: u32,
    pub(crate) gen_preview_wire_proto_vb: Option<wgpu::Buffer>,
    pub(crate) gen_preview_wire_proto_ib: Option<wgpu::Buffer>,
    pub(crate) gen_preview_wire_proto_idx_count: u32,
    pub(crate) gen_preview_solid_instance_buf: Option<wgpu::Buffer>,
    pub(crate) gen_preview_solid_instance_count: u32,
    pub(crate) gen_preview_wire_instance_buf: Option<wgpu::Buffer>,
    pub(crate) gen_preview_wire_instance_count: u32,
    // GPU compute preview (large-stroke shell filter)
    /// Raw packed voxel positions uploaded from CPU; STORAGE | COPY_DST.
    pub(crate) preview_compute_raw_buf: Option<wgpu::Buffer>,
    /// Object matrices (up to 16 × mat4x4); STORAGE | COPY_DST.
    pub(crate) preview_compute_obj_matrix_buf: Option<wgpu::Buffer>,
    /// Flat u32 bitfield occupancy grid; STORAGE | COPY_DST.
    pub(crate) preview_compute_occupancy_buf: Option<wgpu::Buffer>,
    /// Output PreviewInstance array for solid cubes; STORAGE | VERTEX.
    pub(crate) preview_compute_solid_instance_buf: Option<wgpu::Buffer>,
    /// Output PreviewInstance array for wireframe cubes; STORAGE | VERTEX.
    pub(crate) preview_compute_wire_instance_buf: Option<wgpu::Buffer>,
    /// Two DrawIndexedIndirect structs (solid @ offset 0, wire @ offset 20); INDIRECT | STORAGE | COPY_DST.
    pub(crate) preview_compute_indirect_buf: Option<wgpu::Buffer>,
    /// PreviewUniforms uniform buffer (112 bytes); UNIFORM | COPY_DST.
    pub(crate) preview_compute_uniform_buf: Option<wgpu::Buffer>,
    /// Compute bind groups: [0] = fill_occ pass, [1] = shell_emit pass.
    pub(crate) preview_compute_bgs: Option<[wgpu::BindGroup; 2]>,
    /// Prototype VB for the GPU compute path (unit solid cube at half=0.5).
    pub(crate) preview_compute_solid_proto_vb: Option<wgpu::Buffer>,
    pub(crate) preview_compute_solid_proto_ib: Option<wgpu::Buffer>,
    pub(crate) preview_compute_solid_proto_idx_count: u32,
    /// Prototype VB for wireframe cubes.
    pub(crate) preview_compute_wire_proto_vb: Option<wgpu::Buffer>,
    pub(crate) preview_compute_wire_proto_ib: Option<wgpu::Buffer>,
    pub(crate) preview_compute_wire_proto_idx_count: u32,
    /// Number of voxels in the current compute upload (for dispatch sizing).
    pub(crate) preview_compute_voxel_count: u32,
    /// Allocated capacity (in voxels) of `preview_compute_raw_buf`.
    pub(crate) preview_compute_capacity: u32,
    /// Allocated capacity (in instances) of the solid instance buffer
    /// (capped at [`greedy_mesh::MAX_PREVIEW_SHELL_INSTANCES`]).
    pub(crate) preview_compute_shell_capacity: u32,
    /// Allocated capacity (in instances) of the wire instance buffer.
    /// Shrinks to 1 when skip-wire mode is active.
    pub(crate) preview_compute_wire_capacity: u32,
    /// Whether the wire instance buffer was last allocated in skip-wire mode.
    pub(crate) preview_compute_is_skip_wire: bool,
    /// Allocated occupancy word count (ceil(bbox_vol / 32)).
    pub(crate) preview_compute_occ_word_count: u32,
    /// Set to `true` when new raw voxel data has been uploaded; cleared after the
    /// first frame that dispatches the compute passes.
    pub(crate) preview_needs_compute: bool,

    pub(crate) collab_line_vertex_buffer: Option<wgpu::Buffer>,
    pub(crate) collab_line_vertex_count: u32,

    pub(crate) ping_wave_line_vertex_buffer: Option<wgpu::Buffer>,
    pub(crate) ping_wave_line_vertex_count: u32,
    pub(crate) ping_vertex_buffer: Option<wgpu::Buffer>,
    pub(crate) ping_index_buffer: Option<wgpu::Buffer>,
    pub(crate) ping_index_count: u32,
    pub(crate) ping_wire_vertex_buffer: Option<wgpu::Buffer>,
    pub(crate) ping_wire_index_buffer: Option<wgpu::Buffer>,
    pub(crate) ping_wire_index_count: u32,
    pub(crate) selection_overlay_vertex_buffer: Option<wgpu::Buffer>,
    pub(crate) selection_overlay_index_buffer: Option<wgpu::Buffer>,
    pub(crate) selection_overlay_index_count: u32,
    pub(crate) selection_overlay_line_vertex_buffer: Option<wgpu::Buffer>,
    pub(crate) selection_overlay_line_vertex_count: u32,
    /// Selection transform gizmo geometry (lines: shafts + rings; tris: arrowheads).
    pub(crate) gizmo_line_vertex_buffer: Option<wgpu::Buffer>,
    pub(crate) gizmo_line_vertex_count: u32,
    pub(crate) gizmo_tri_vertex_buffer: Option<wgpu::Buffer>,
    pub(crate) gizmo_tri_vertex_count: u32,
    /// Full-scene grid wireframe (View → Show borders); indexed line-list.
    pub(crate) grid_border_line_vertex_buffer: Option<wgpu::Buffer>,
    pub(crate) grid_border_line_index_buffer: Option<wgpu::Buffer>,
    pub(crate) grid_border_line_index_count: u32,
    /// Fingerprint of visible voxel set + mesh generation for grid overlay rebuilds.
    pub(crate) grid_border_cache_key: Option<u64>,
    /// Dedup CPU mesh rebuild when hover cell unchanged.
    pub(crate) preview_cache_key: Option<u64>,
    /// `(selection fingerprint, mesh_refresh_generation)` — invalidates when selection or voxels change.
    pub(crate) selection_overlay_cache_key: Option<u64>,

    pub(crate) sampler_linear: wgpu::Sampler,
    pub(crate) sampler_depth: wgpu::Sampler,
    pub(crate) sampler_comparison: wgpu::Sampler,
    #[allow(dead_code)]
    pub(crate) sampler_nearest: wgpu::Sampler,

    pub(crate) mesh_greedy_pipeline: Option<wgpu::ComputePipeline>,
    pub(crate) mesh_greedy_bind_layout: Option<wgpu::BindGroupLayout>,
    /// Compute pipeline for pass 1: fill occupancy bitfield from raw voxels.
    pub(crate) pipeline_preview_fill_occ: wgpu::ComputePipeline,
    /// Compute pipeline for pass 2: shell emit → write PreviewInstances.
    pub(crate) pipeline_preview_shell_emit: wgpu::ComputePipeline,
    /// Bind group layouts for the two compute passes.
    pub(crate) preview_compute_fill_layout: wgpu::BindGroupLayout,
    pub(crate) preview_compute_emit_layout: wgpu::BindGroupLayout,
    /// Must match [`MESH_GREEDY_PIPELINE_LAYOUT_VERSION`]; clears cached compute pipeline when bumped.
    pub(crate) mesh_greedy_pl_version: u32,
    pub(crate) mesh_greedy_pool: MeshGreedyPool,

    /// Last opaque mesh rebuild path (for perf): `gpu_greedy`, `cpu`, `cpu_chunked`, `clear`, `gpu_no_headers`, etc.
    pub(crate) last_mesh_route: String,

    // ── Glyphon text rendering (peer labels) ──
    pub(crate) glyphon_font_system: FontSystem,
    pub(crate) glyphon_swash_cache: SwashCache,
    #[allow(dead_code)]
    pub(crate) glyphon_cache: GlyphonCache,
    pub(crate) glyphon_atlas: TextAtlas,
    pub(crate) glyphon_text_renderer: TextRenderer,
    pub(crate) glyphon_viewport: GlyphonViewport,
    /// Per-frame peer label data uploaded via [`Self::upload_peer_labels`].
    pub(crate) peer_label_data: Vec<GpuPeerLabel>,
    /// Active ping label (at most one).
    pub(crate) ping_label_data: Option<GpuPeerLabel>,
    /// Gizmo move-drag coordinate delta label (e.g. "+0, +3, +0"). None when no drag.
    pub(crate) gizmo_delta_label: Option<GpuPeerLabel>,

    // ── Logo overlay (start-screen logo, rendered as mascot-style overlay) ────
    pub(crate) logo_overlay: Option<LogoOverlay>,

    // ── Mascot (start-screen floating model views) ────────────────────────────
    pub(crate) mascots: Vec<MascotEntry>,
    pub(crate) mascot_pipeline: wgpu::RenderPipeline,
    pub(crate) mascot_bind_layout: wgpu::BindGroupLayout,
    /// Shared depth buffer for all mascot render passes; sized to the swapchain.
    pub(crate) mascot_depth_view: wgpu::TextureView,
    /// Tracks the surface size at which `mascot_depth_view` was last created.
    pub(crate) mascot_depth_size: (u32, u32),
    /// Wall-clock instant of the last frame that ran mascot animation.
    pub(crate) mascot_last_tick: std::time::Instant,

    // ── Speech bubbles (GPU-rendered floating notes / dialogue) ───────────────
    pub(crate) speech_bubbles: Vec<SpeechBubble>,
    pub(crate) speech_bubble_pipeline: wgpu::RenderPipeline,
    pub(crate) speech_bubble_bind_layout: wgpu::BindGroupLayout,
    pub(crate) speech_bubble_last_tick: std::time::Instant,
    /// Bubble ids that completed their dismiss animation since the last drain.
    /// The event loop in lib.rs drains this and emits `speech-bubble-dismissed`.
    pub(crate) pending_dismissed_bubble_ids: Vec<u32>,
    /// Separate glyphon atlas targeting sdr_format (swapchain surface).
    #[allow(dead_code)]
    pub(crate) speech_bubble_glyphon_cache: GlyphonCache,
    pub(crate) speech_bubble_glyphon_atlas: TextAtlas,
    pub(crate) speech_bubble_text_renderer: TextRenderer,
    pub(crate) speech_bubble_glyphon_viewport: GlyphonViewport,
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
    let eye = center + ld * (r * 5.0);
    let view = Mat4::look_at_rh(eye, center, up);
    let he = r * 1.8;
    let proj = Mat4::orthographic_rh(-he, he, -he, he, 1.0, r * 12.0);
    proj * view
}

/// Horizontal angle (degrees, CCW from +X in XZ) and elevation above XZ (degrees). Y-up.
pub fn light_dir_from_azimuth_elevation_deg(azimuth_deg: f32, elevation_deg: f32) -> Vec3 {
    let az = azimuth_deg.to_radians();
    let el = elevation_deg.clamp(0.5, 89.5).to_radians();
    let ce = el.cos();
    Vec3::new(ce * az.cos(), el.sin(), ce * az.sin()).normalize()
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

impl WgpuViewer {
    pub async fn new(window: impl wgpu::WindowHandle + 'static) -> Result<Self, String> {
        // Prefer Vulkan on Windows to avoid D3D12 swapchain state-validation churn
        // seen on some systems. Keep all backends elsewhere.
        let backend_mask = if cfg!(target_os = "windows") {
            wgpu::Backends::VULKAN
        } else {
            wgpu::Backends::all()
        };

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: backend_mask,
            // Enables backend API validation in debug builds (Vulkan layers, D3D12 debug, etc.).
            // `with_env` honors WGPU_VALIDATION / WGPU_DEBUG so developers can disable if needed.
            flags: wgpu::InstanceFlags::from_build_config().with_env(),
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
        let adapter_info = adapter.get_info();
        // #region agent log
        debug_log(
            "H1",
            "src-tauri/src/render/mod.rs:new",
            "adapter-selected",
            json!({
                "backend_mask": format!("{:?}", backend_mask),
                "adapter_backend": format!("{:?}", adapter_info.backend),
                "adapter_name": adapter_info.name,
                "adapter_driver": adapter_info.driver,
                "adapter_device_type": format!("{:?}", adapter_info.device_type)
            }),
        );
        // #endregion
        let caps = surface.get_capabilities(&adapter);
        // #region agent log
        debug_log(
            "H2",
            "src-tauri/src/render/mod.rs:new",
            "surface-caps",
            json!({
                "present_modes": format!("{:?}", caps.present_modes),
                "alpha_modes": format!("{:?}", caps.alpha_modes),
                "formats_count": caps.formats.len()
            }),
        );
        // #endregion
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let sdr_format = format;
        let hdr_surface_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| *f == wgpu::TextureFormat::Rgba16Float);

        // WebGPU defaults cap a single storage buffer at 128 MiB; GPU greedy mesh scratch can exceed that.
        // Ask for the adapter maximum so large scenes can still use the compute path when hardware allows.
        let adapter_limits = adapter.limits();
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_buffer_size: adapter_limits.max_buffer_size,
            ..Default::default()
        };

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
        // On Windows, prefer Opaque when available. Transparent swapchains can trigger
        // D3D12 debug-layer state warnings in mixed compositor scenarios (wgpu + webview).
        // Other platforms keep preferring composited alpha for native-under-webview blending.
        let alpha_mode = if cfg!(target_os = "windows")
            && caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque)
        {
            wgpu::CompositeAlphaMode::Opaque
        } else if caps
            .alpha_modes
            .contains(&wgpu::CompositeAlphaMode::PreMultiplied)
        {
            wgpu::CompositeAlphaMode::PreMultiplied
        } else if caps
            .alpha_modes
            .contains(&wgpu::CompositeAlphaMode::PostMultiplied)
        {
            wgpu::CompositeAlphaMode::PostMultiplied
        } else if caps
            .alpha_modes
            .contains(&wgpu::CompositeAlphaMode::Inherit)
        {
            wgpu::CompositeAlphaMode::Inherit
        } else {
            caps.alpha_modes[0]
        };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
            format,
            width: size.0.max(1),
            height: size.1.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        if matches!(config.alpha_mode, wgpu::CompositeAlphaMode::Opaque) {
            debug_log(
                "H3",
                "src-tauri/src/render/mod.rs:new",
                "opaque-alpha-mode-selected",
                json!({
                    "alpha_modes": format!("{:?}", caps.alpha_modes),
                    "note": "Opaque swapchain can place viewport above webview UI on some systems"
                }),
            );
        }
        surface.configure(&device, &config);

        let BindGroupLayouts {
            scene_layout0,
            scene_layout1,
            shadow_vs_layout,
            post_bloom_layout,
            post_blur_layout,
            post_composite_layout,
        } = create_bind_group_layouts(&device);

        // ── Pipeline creation (delegated to pipelines.rs) ────────────────────
        let ScenePipelines {
            pipeline_opaque,
            pipeline_preview_occluded,
            pipeline_preview_front,
            pipeline_preview_front_wire,
            pipeline_preview_inst_occluded,
            pipeline_preview_inst_front,
            pipeline_preview_inst_front_wire,
            pipeline_gen_preview_inst_front,
            pipeline_gen_preview_inst_occluded,
            pipeline_gen_preview_inst_front_wire,
        } = create_scene_pipelines(&device, &scene_layout0);

        let PreviewComputePipelines {
            pipeline_preview_fill_occ,
            pipeline_preview_shell_emit,
            preview_compute_fill_layout,
            preview_compute_emit_layout,
        } = create_preview_compute_pipelines(&device);

        let OverlayPipelines {
            pipeline_collab_lines_occluded,
            pipeline_collab_lines_front,
            pipeline_grid_border_lines,
            pipeline_gizmo_lines_front,
            pipeline_gizmo_lines_occluded,
            pipeline_gizmo_tris_front,
            pipeline_gizmo_tris_occluded,
            pipeline_gizmo_lines_always,
            pipeline_gizmo_tris_always,
        } = create_overlay_pipelines(&device, &scene_layout0);

        let AvatarPipeline {
            pipeline_avatar,
            avatar_bind_layout,
        } = create_avatar_pipeline(&device);

        let SkyPipelines {
            pipeline_sky,
            pipeline_start_screen_bg,
        } = create_sky_pipelines(&device, &scene_layout0);

        let OitPipelines {
            pipeline_oit_accum,
            pipeline_oit_composite,
            oit_composite_layout,
        } = create_oit_pipelines(&device, &scene_layout0, &scene_layout1);

        let pipeline_shadow = create_shadow_pipeline(&device, &shadow_vs_layout);

        let PostPipelines {
            pipeline_bloom_extract,
            pipeline_blur,
            pipeline_blit,
            pipeline_blit_weighted_add,
            pipeline_composite,
            shader_composite,
            composite_pipeline_layout: pl_comp,
            pipeline_meter,
        } = create_post_pipelines(
            &device,
            &post_bloom_layout,
            &post_blur_layout,
            &post_composite_layout,
            format,
        );

        let meter_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("meter_lum"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let meter_view = meter_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let meter_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("meter_staging"),
            size: 256,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let default_lit = crate::voxelle::LightingSettings::default();
        let light_dir = light_dir_from_azimuth_elevation_deg(
            default_lit.light_angle_deg,
            default_lit.light_elevation_deg,
        );
        let sun_c = crate::voxelle::hex_srgb_to_linear_rgb3(&default_lit.light_color)
            .map(|c| Vec3::new(c[0], c[1], c[2]))
            .unwrap_or(Vec3::ONE);
        let bg_c = crate::voxelle::hex_srgb_to_linear_rgb3(&default_lit.background_color)
            .map(|c| Vec3::new(c[0], c[1], c[2]))
            .unwrap_or(Vec3::new(0.04, 0.045, 0.055));

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
        let bloom_extract_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bloom_extract_u"),
            contents: bytemuck::bytes_of(&BloomExtractUniform {
                exposure_ev: 0.0,
                _pad: [0.0; 3],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Constant weight written once: each bloom pyramid level contributes 0.75× the next finer
        // level, giving a geometric falloff instead of a flat equal-weight accumulation.
        let post_blit_weight_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blit_weight_u"),
            contents: bytemuck::bytes_of(&BloomExtractUniform {
                exposure_ev: 0.75,
                _pad: [0.0; 3],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut post_composite_opts: PostCompositeOpts = bytemuck::Zeroable::zeroed();
        post_composite_opts.tone_mode = 0;
        post_composite_opts.exposure_ev = default_lit.exposure_ev.clamp(-5.0, 5.0);
        post_composite_opts.ss_soft = 1.0; // soft sunshafts on by default
        let post_composite_opts_buf =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("post_composite_opts"),
                contents: bytemuck::bytes_of(&post_composite_opts),
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
        let sampler_depth = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("depth_non_filter"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        // #region agent log
        debug_log(
            "H7",
            "src-tauri/src/render/mod.rs:new",
            "samplers-created",
            json!({
                "linear_label": "linear",
                "comparison_label": "shadow_cmp",
                "depth_label": "depth_non_filter"
            }),
        );
        // #endregion

        let (shadow_texture, shadow_view) =
            create_shadow_tex(&device, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
        let scene_bounds = MeshBounds {
            min: Vec3::splat(-10.0),
            max: Vec3::splat(10.0),
        };

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
        ) = create_screen_targets(&device, size.0, size.1, vf);

        let (present_texture, present_view) =
            create_present_texture(&device, size.0, size.1, format);
        let (depth_snapshot_texture, depth_snapshot_view) =
            create_depth_snapshot(&device, size.0, size.1);
        let (oit_accum_texture, oit_accum_view, oit_revealage_texture, oit_revealage_view) =
            create_oit_textures(&device, size.0, size.1);

        let (bloom_pyramid_a, bloom_pyramid_a_views, bloom_pyramid_b, bloom_pyramid_b_views) =
            create_bloom_pyramid(&device, size.0, size.1, vf);

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
            layout: &post_blur_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler_linear),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bloom_extract_buf.as_entire_binding(),
                },
            ],
        });
        let bind_meter = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("meter_lum"),
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
        let (
            bind_blit_down,
            bind_blit_up,
            bind_blit_up_weighted,
            bind_blit_final,
            bind_blur_pyr_h,
            bind_blur_pyr_v,
        ) = build_bloom_pyramid_bind_groups(
            &device,
            &post_bloom_layout,
            &post_blur_layout,
            &bloom_a_view,
            &bloom_pyramid_a_views,
            &bloom_pyramid_b_views,
            &sampler_linear,
            &post_blur_buf,
            &post_blit_weight_buf,
        );
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
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&sampler_depth),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: global_buffer.as_entire_binding(),
                },
            ],
        });

        // ── SSR uniform buffer ──────────────────────────────────────────────
        let ssr_opts: SsrOpts = SsrOpts {
            strength: 0.8,
            max_steps: 32.0,
            thickness: 2.0,
            enabled: 0.0,
        };
        let ssr_opts_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ssr_opts"),
            contents: bytemuck::bytes_of(&ssr_opts),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&depth_snapshot_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler_depth),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: ssr_opts_buf.as_entire_binding(),
                },
            ],
        }));

        // ── SSR fullscreen pass (opaque metals) ─────────────────────────────
        let (ssr_texture, ssr_view) = create_ssr_texture(&device, size.0, size.1);
        let SsrPipeline {
            pipeline_ssr_fullscreen,
            ssr_layout,
        } = create_ssr_pipeline(&device);
        let bind_ssr = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssr_fullscreen"),
            layout: &ssr_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: global_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&depth_snapshot_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&hdr_opaque_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler_linear),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&sampler_depth),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: ssr_opts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&normal_view),
                },
            ],
        });

        let bind_oit_composite = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("oit_composite"),
            layout: &oit_composite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&oit_accum_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&oit_revealage_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&hdr_opaque_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler_linear),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&ssr_view),
                },
            ],
        });

        // ── Progressive raytracer setup ──────────────────────────────────────
        let RaytracePipeline {
            pipeline_raytrace,
            rt_scene_layout,
            rt_accum_layout,
        } = create_raytrace_pipeline(&device);

        let rt_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rt_uniform"),
            contents: bytemuck::bytes_of(&RtUniform {
                frame_seed: 0,
                sample_n: 0,
                fast_preview: 0,
                surface_mode: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let (rt_accum_tex0, rt_accum_view0) = create_rt_accum_tex(&device, size.0, size.1);
        let (rt_accum_tex1, rt_accum_view1) = create_rt_accum_tex(&device, size.0, size.1);
        let (rt_preview_tex, rt_preview_view) =
            create_rt_accum_tex(&device, (size.0 / 2).max(1), (size.1 / 2).max(1));
        let rt_preview_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rt_preview_bg"),
            layout: &post_bloom_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&rt_preview_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler_linear),
                },
            ],
        });

        let rt_scene_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rt_scene_bg"),
            layout: &rt_scene_layout,
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
        // rt_accum_bgs[i] reads from the side that is NOT written this frame:
        //   flip=false -> write to [0], so bg[0] reads from [1]
        //   flip=true  -> write to [1], so bg[1] reads from [0]
        let rt_accum_bgs: Vec<wgpu::BindGroup> = vec![
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rt_accum_bg0"),
                layout: &rt_accum_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&rt_accum_view1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler_linear),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: rt_uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: post_composite_opts_buf.as_entire_binding(),
                    },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rt_accum_bg1"),
                layout: &rt_accum_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&rt_accum_view0),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler_linear),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: rt_uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: post_composite_opts_buf.as_entire_binding(),
                    },
                ],
            }),
        ];

        let rt_accum_textures = [rt_accum_tex0, rt_accum_tex1];
        let rt_accum_views = [rt_accum_view0, rt_accum_view1];

        // ── Glyphon text rendering setup ──
        let mut glyphon_font_system = FontSystem::new();
        #[cfg(target_os = "macos")]
        glyphon_font_system
            .db_mut()
            .set_sans_serif_family("Helvetica Neue");
        #[cfg(not(target_os = "macos"))]
        glyphon_font_system
            .db_mut()
            .set_sans_serif_family("Segoe UI");
        glyphon_font_system.db_mut().load_font_data(
            include_bytes!("../../../public/fonts/ZeldaSans-Regular-v1.otf").to_vec(),
        );
        let glyphon_swash_cache = SwashCache::new();
        let glyphon_cache = GlyphonCache::new(&device);
        let mut glyphon_atlas = TextAtlas::new(&device, &queue, &glyphon_cache, format);
        let glyphon_text_renderer = TextRenderer::new(
            &mut glyphon_atlas,
            &device,
            wgpu::MultisampleState::default(),
            None,
        );
        let glyphon_viewport = GlyphonViewport::new(&device, &glyphon_cache);
        // #region agent log
        debug_log(
            "H9",
            "src-tauri/src/render/mod.rs:new",
            "viewer-construction-ready",
            json!({
                "glyphon_ready": true,
                "surface_size": [size.0, size.1]
            }),
        );
        // #endregion

        // ── Mascot pipeline (start-screen floating model views) ──────────────
        let MascotPipelineResult {
            mascot_pipeline,
            mascot_bind_layout,
        } = create_mascot_pipeline(&device, sdr_format);

        let mascot_depth_view = Self::make_mascot_depth(&device, size.0.max(1), size.1.max(1)).1;

        // ── Speech bubble pipeline ────────────────────────────────────────────
        let SpeechBubblePipelineResult {
            speech_bubble_pipeline,
            speech_bubble_bind_layout,
        } = create_speech_bubble_pipeline(&device, sdr_format);

        // Glyphon text renderer targeting sdr_format (swapchain surface).
        let speech_bubble_glyphon_cache = GlyphonCache::new(&device);
        let mut speech_bubble_glyphon_atlas =
            TextAtlas::new(&device, &queue, &speech_bubble_glyphon_cache, sdr_format);
        let speech_bubble_text_renderer = TextRenderer::new(
            &mut speech_bubble_glyphon_atlas,
            &device,
            wgpu::MultisampleState::default(),
            None,
        );
        let mut speech_bubble_glyphon_viewport =
            GlyphonViewport::new(&device, &speech_bubble_glyphon_cache);
        speech_bubble_glyphon_viewport.update(
            &queue,
            Resolution {
                width: size.0.max(1),
                height: size.1.max(1),
            },
        );

        Ok(Self {
            surface,
            device,
            queue,
            config,
            format,
            sdr_format,
            hdr_surface_format,
            hdr_output: false,
            surface_size: size,
            viewport_x: 0,
            viewport_y: 0,
            viewport_width: size.0,
            viewport_height: size.1,
            global_buffer,
            brick_buffer,
            brick_cell_count: 1,
            brick_origin_iv: IVec3::ZERO,
            brick_dims_u: (0, 0, 0),
            scene_bounds,
            light_dir,
            light_ambient: default_lit.ambient_intensity,
            light_sun: default_lit.sunlight_intensity,
            sun_color_linear: sun_c,
            bg_color_linear: bg_c,
            shadows_enabled: default_lit.enable_shadows,
            soft_shadows: true,
            sky_enabled: default_lit.enable_sky,
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
            depth_snapshot_texture,
            depth_snapshot_view,
            oit_accum_texture,
            oit_accum_view,
            oit_revealage_texture,
            oit_revealage_view,
            bloom_a,
            bloom_a_view,
            bloom_b,
            bloom_b_view,
            bloom_pyramid_a,
            bloom_pyramid_a_views,
            bloom_pyramid_b,
            bloom_pyramid_b_views,
            present_texture,
            present_view,
            scene_layout0,
            scene_layout1,
            shadow_vs_layout,
            post_bloom_layout,
            post_blur_layout,
            post_composite_layout,
            oit_composite_layout,
            bind_scene_opaque,
            bind_shadow_pass,
            bind_bloom_extract,
            bind_meter,
            bind_blur_h,
            bind_blur_v,
            bind_composite,
            bind_trans,
            bind_oit_composite,
            bind_blit_down,
            bind_blit_up,
            bind_blit_up_weighted,
            bind_blit_final,
            bind_blur_pyr_h,
            bind_blur_pyr_v,
            post_blur_buf,
            bloom_extract_buf,
            post_blit_weight_buf,
            post_composite_opts_buf,
            post_composite_opts,
            pipeline_opaque,
            pipeline_preview_occluded,
            pipeline_preview_front,
            pipeline_preview_front_wire,
            pipeline_preview_inst_occluded,
            pipeline_preview_inst_front,
            pipeline_preview_inst_front_wire,
            pipeline_gen_preview_inst_front,
            pipeline_gen_preview_inst_occluded,
            pipeline_gen_preview_inst_front_wire,
            pipeline_collab_lines_occluded,
            pipeline_collab_lines_front,
            pipeline_avatar,
            avatar_bind_layout,
            avatar_mesh_cache: std::collections::HashMap::new(),
            avatar_peers: Vec::new(),
            pipeline_grid_border_lines,
            pipeline_gizmo_lines_front,
            pipeline_gizmo_lines_occluded,
            pipeline_gizmo_tris_front,
            pipeline_gizmo_tris_occluded,
            pipeline_gizmo_lines_always,
            pipeline_gizmo_tris_always,
            gizmo_on_top: true,
            pipeline_sky,
            pipeline_start_screen_bg,
            pipeline_oit_accum,
            pipeline_oit_composite,
            pipeline_shadow,
            pipeline_bloom_extract,
            pipeline_blur,
            pipeline_blit,
            pipeline_blit_weighted_add,
            pipeline_composite,
            shader_composite,
            composite_pipeline_layout: pl_comp,
            pipeline_meter,
            meter_texture,
            meter_view,
            meter_staging,
            ssr_opts_buf,
            ssr_opts,
            ssr_texture,
            ssr_view,
            ssr_layout,
            bind_ssr,
            pipeline_ssr_fullscreen,
            raytrace_enabled: false,
            rt_accum_textures,
            rt_accum_views,
            rt_accum_flip: false,
            rt_sample_n: 0,
            rt_uniform_buf,
            rt_scene_layout,
            rt_accum_layout,
            rt_scene_bg,
            rt_accum_bgs,
            rt_preview_tex,
            rt_preview_view,
            rt_preview_bg,
            pipeline_raytrace,
            rt_prev_eye: [0.0; 3],
            rt_prev_inv_view: [[0.0; 4]; 4],
            rt_fast_preview: false,
            rt_surface_mode: 0,
            exposure_user_ev: default_lit.exposure_ev.clamp(-5.0, 5.0),
            auto_exposure_enabled: default_lit.auto_exposure,
            auto_exposure_smoothed: 0.0,
            meter_pending_rx: None,
            // Match default `ViewerState::start_screen_logo_transparent`: gradient until a real scene load turns it off.
            start_screen_transparent: true,
            start_screen_appearance: 0.0,
            vertex_buffer: None,
            index_buffer: None,
            index_count: 0,
            opaque_index_split: 0,
            opaque_chunked: false,
            chunk_grid_origin: IVec3::ZERO,
            opaque_chunks: BTreeMap::new(),
            pending_chunk_uploads: VecDeque::new(),
            spatial_mesh_cache: None,
            preview_vertex_buffer: None,
            preview_index_buffer: None,
            preview_index_count: 0,
            preview_wire_vertex_buffer: None,
            preview_wire_index_buffer: None,
            preview_wire_index_count: 0,
            preview_solid_proto_vb: None,
            preview_solid_proto_ib: None,
            preview_solid_proto_idx_count: 0,
            preview_wire_proto_vb: None,
            preview_wire_proto_ib: None,
            preview_wire_proto_idx_count: 0,
            preview_solid_instance_buf: None,
            preview_solid_instance_count: 0,
            preview_wire_instance_buf: None,
            preview_wire_instance_count: 0,
            gen_preview_solid_proto_vb: None,
            gen_preview_solid_proto_ib: None,
            gen_preview_solid_proto_idx_count: 0,
            gen_preview_wire_proto_vb: None,
            gen_preview_wire_proto_ib: None,
            gen_preview_wire_proto_idx_count: 0,
            gen_preview_solid_instance_buf: None,
            gen_preview_solid_instance_count: 0,
            gen_preview_wire_instance_buf: None,
            gen_preview_wire_instance_count: 0,
            preview_compute_raw_buf: None,
            preview_compute_obj_matrix_buf: None,
            preview_compute_occupancy_buf: None,
            preview_compute_solid_instance_buf: None,
            preview_compute_wire_instance_buf: None,
            preview_compute_indirect_buf: None,
            preview_compute_uniform_buf: None,
            preview_compute_bgs: None,
            preview_compute_solid_proto_vb: None,
            preview_compute_solid_proto_ib: None,
            preview_compute_solid_proto_idx_count: 0,
            preview_compute_wire_proto_vb: None,
            preview_compute_wire_proto_ib: None,
            preview_compute_wire_proto_idx_count: 0,
            preview_compute_voxel_count: 0,
            preview_compute_capacity: 0,
            preview_compute_shell_capacity: 0,
            preview_compute_wire_capacity: 0,
            preview_compute_is_skip_wire: false,
            preview_compute_occ_word_count: 0,
            preview_needs_compute: false,
            collab_line_vertex_buffer: None,
            collab_line_vertex_count: 0,

            ping_wave_line_vertex_buffer: None,
            ping_wave_line_vertex_count: 0,
            ping_vertex_buffer: None,
            ping_index_buffer: None,
            ping_index_count: 0,
            ping_wire_vertex_buffer: None,
            ping_wire_index_buffer: None,
            ping_wire_index_count: 0,
            selection_overlay_vertex_buffer: None,
            selection_overlay_index_buffer: None,
            selection_overlay_index_count: 0,
            selection_overlay_line_vertex_buffer: None,
            selection_overlay_line_vertex_count: 0,
            gizmo_line_vertex_buffer: None,
            gizmo_line_vertex_count: 0,
            gizmo_tri_vertex_buffer: None,
            gizmo_tri_vertex_count: 0,
            grid_border_line_vertex_buffer: None,
            grid_border_line_index_buffer: None,
            grid_border_line_index_count: 0,
            grid_border_cache_key: None,
            preview_cache_key: None,
            selection_overlay_cache_key: None,
            sampler_linear,
            sampler_depth,
            sampler_comparison,
            sampler_nearest,
            creation_instant: std::time::Instant::now(),
            mesh_greedy_pipeline: None,
            mesh_greedy_bind_layout: None,
            mesh_greedy_pl_version: 0,
            mesh_greedy_pool: MeshGreedyPool::default(),
            last_mesh_route: String::new(),
            pipeline_preview_fill_occ,
            pipeline_preview_shell_emit,
            preview_compute_fill_layout,
            preview_compute_emit_layout,

            // ── Glyphon (initialized below) ──
            glyphon_font_system,
            glyphon_swash_cache,
            glyphon_cache,
            glyphon_atlas,
            glyphon_text_renderer,
            glyphon_viewport,
            peer_label_data: Vec::new(),
            ping_label_data: None,
            gizmo_delta_label: None,

            logo_overlay: None,

            mascots: Vec::new(),
            mascot_pipeline,
            mascot_bind_layout,
            mascot_depth_view,
            mascot_depth_size: size,
            mascot_last_tick: std::time::Instant::now(),

            speech_bubbles: Vec::new(),
            speech_bubble_pipeline,
            speech_bubble_bind_layout,
            speech_bubble_last_tick: std::time::Instant::now(),
            pending_dismissed_bubble_ids: Vec::new(),
            speech_bubble_glyphon_cache,
            speech_bubble_glyphon_atlas,
            speech_bubble_text_renderer,
            speech_bubble_glyphon_viewport,
        })
    }

    pub fn opaque_index_count(&self) -> u32 {
        if self.opaque_chunked {
            self.opaque_chunks
                .values()
                .map(|c| c.opaque_index_count + c.transparent_index_count)
                .sum()
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

    /// `surface_*` = full webview drawable (must match the native window). Viewport = `.viewport` div in the same pixel space.
    pub fn resize(
        &mut self,
        surface_w: u32,
        surface_h: u32,
        mut viewport_x: u32,
        mut viewport_y: u32,
        mut viewport_width: u32,
        mut viewport_height: u32,
    ) {
        if surface_w == 0 || surface_h == 0 {
            return;
        }
        viewport_x = viewport_x.min(surface_w.saturating_sub(1));
        viewport_y = viewport_y.min(surface_h.saturating_sub(1));
        viewport_width = viewport_width.max(1).min(surface_w - viewport_x);
        viewport_height = viewport_height.max(1).min(surface_h - viewport_y);

        self.surface_size = (surface_w, surface_h);
        self.viewport_x = viewport_x;
        self.viewport_y = viewport_y;
        self.viewport_width = viewport_width;
        self.viewport_height = viewport_height;

        self.config.width = surface_w;
        self.config.height = surface_h;
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
        ) = create_screen_targets(&self.device, viewport_width, viewport_height, vf);
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

        let (pyr_a, pyr_a_views, pyr_b, pyr_b_views) =
            create_bloom_pyramid(&self.device, viewport_width, viewport_height, vf);
        self.bloom_pyramid_a = pyr_a;
        self.bloom_pyramid_a_views = pyr_a_views;
        self.bloom_pyramid_b = pyr_b;
        self.bloom_pyramid_b_views = pyr_b_views;

        let (present_texture, present_view) =
            create_present_texture(&self.device, viewport_width, viewport_height, self.format);
        self.present_texture = present_texture;
        self.present_view = present_view;
        let (depth_snapshot_texture, depth_snapshot_view) =
            create_depth_snapshot(&self.device, viewport_width, viewport_height);
        self.depth_snapshot_texture = depth_snapshot_texture;
        self.depth_snapshot_view = depth_snapshot_view;
        self.mascot_depth_view =
            Self::make_mascot_depth(&self.device, surface_w.max(1), surface_h.max(1)).1;
        self.mascot_depth_size = (surface_w.max(1), surface_h.max(1));

        let (oit_accum_texture, oit_accum_view, oit_revealage_texture, oit_revealage_view) =
            create_oit_textures(&self.device, viewport_width, viewport_height);
        self.oit_accum_texture = oit_accum_texture;
        self.oit_accum_view = oit_accum_view;
        self.oit_revealage_texture = oit_revealage_texture;
        self.oit_revealage_view = oit_revealage_view;

        let (ssr_texture, ssr_view) =
            create_ssr_texture(&self.device, viewport_width, viewport_height);
        self.ssr_texture = ssr_texture;
        self.ssr_view = ssr_view;

        // Reallocate raytracer accumulation textures for new viewport size.
        let (rt_a_tex, rt_a_view) =
            create_rt_accum_tex(&self.device, viewport_width, viewport_height);
        let (rt_b_tex, rt_b_view) =
            create_rt_accum_tex(&self.device, viewport_width, viewport_height);
        self.rt_accum_textures = [rt_a_tex, rt_b_tex];
        self.rt_accum_views = [rt_a_view, rt_b_view];
        let (rt_preview_tex, rt_preview_view) = create_rt_accum_tex(
            &self.device,
            (viewport_width / 2).max(1),
            (viewport_height / 2).max(1),
        );
        self.rt_preview_tex = rt_preview_tex;
        self.rt_preview_view = rt_preview_view;
        self.rt_sample_n = 0;

        self.rebuild_bind_groups();
    }

    pub fn viewport_size(&self) -> (u32, u32) {
        (self.viewport_width.max(1), self.viewport_height.max(1))
    }

    /// Swapchain drawable in physical pixels (matches [`Self::surface_size`]; may differ slightly from webview `inner* × dpr` after configure).
    pub fn surface_pixel_size(&self) -> (u32, u32) {
        (self.surface_size.0.max(1), self.surface_size.1.max(1))
    }

    /// Update world-space AABB used for lighting / shadow frusta (call when the opaque mesh changes without a voxel brick upload).
    pub fn set_scene_bounds(&mut self, bounds: MeshBounds) {
        self.scene_bounds = bounds;
    }

    /// Whether the display supports HDR output (Rgba16Float surface format).
    pub fn hdr_available(&self) -> bool {
        self.hdr_surface_format.is_some()
    }

    /// Toggle HDR output.  Reconfigures the surface, present texture, composite
    /// pipeline and glyphon text atlas for the new format.  Caller should also
    /// call `set_tone_mapping_mode(6)` (HDR) or restore the SDR preference.
    pub fn set_hdr_output(&mut self, enabled: bool) {
        if enabled && self.hdr_surface_format.is_none() {
            return;
        }
        if enabled == self.hdr_output {
            return;
        }
        self.hdr_output = enabled;
        self.format = if enabled {
            self.hdr_surface_format.unwrap()
        } else {
            self.sdr_format
        };
        self.config.format = self.format;
        self.surface.configure(&self.device, &self.config);

        // Recreate present texture with new format.
        let (present_texture, present_view) = create_present_texture(
            &self.device,
            self.viewport_width,
            self.viewport_height,
            self.format,
        );
        self.present_texture = present_texture;
        self.present_view = present_view;

        // Rebuild composite pipeline — target format is baked into the pipeline.
        self.pipeline_composite = fullscreen_pipeline(
            &self.device,
            &self.composite_pipeline_layout,
            &self.shader_composite,
            "fs_composite",
            &[Some(wgpu::ColorTargetState {
                format: self.format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            None,
        );

        // Recreate glyphon text atlas + renderer for new surface format.
        let glyphon_cache = GlyphonCache::new(&self.device);
        let mut glyphon_atlas =
            TextAtlas::new(&self.device, &self.queue, &glyphon_cache, self.format);
        let glyphon_text_renderer = TextRenderer::new(
            &mut glyphon_atlas,
            &self.device,
            wgpu::MultisampleState::default(),
            None,
        );
        self.glyphon_cache = glyphon_cache;
        self.glyphon_atlas = glyphon_atlas;
        self.glyphon_text_renderer = glyphon_text_renderer;

        // Recreate speech-bubble glyphon (targets sdr_format / swapchain).
        let speech_bubble_glyphon_cache = GlyphonCache::new(&self.device);
        let mut speech_bubble_glyphon_atlas = TextAtlas::new(
            &self.device,
            &self.queue,
            &speech_bubble_glyphon_cache,
            self.sdr_format,
        );
        let speech_bubble_text_renderer = TextRenderer::new(
            &mut speech_bubble_glyphon_atlas,
            &self.device,
            wgpu::MultisampleState::default(),
            None,
        );
        let mut speech_bubble_glyphon_viewport =
            GlyphonViewport::new(&self.device, &speech_bubble_glyphon_cache);
        speech_bubble_glyphon_viewport.update(
            &self.queue,
            Resolution {
                width: self.config.width.max(1),
                height: self.config.height.max(1),
            },
        );
        self.speech_bubble_glyphon_cache = speech_bubble_glyphon_cache;
        self.speech_bubble_glyphon_atlas = speech_bubble_glyphon_atlas;
        self.speech_bubble_text_renderer = speech_bubble_text_renderer;
        self.speech_bubble_glyphon_viewport = speech_bubble_glyphon_viewport;
    }

    pub fn set_tone_mapping_mode(&mut self, mode: u32) {
        let mode = mode.min(6);
        self.post_composite_opts.tone_mode = mode;
        self.queue.write_buffer(
            &self.post_composite_opts_buf,
            0,
            bytemuck::bytes_of(&self.post_composite_opts),
        );
    }

    pub fn set_soft_sunshafts(&mut self, enabled: bool) {
        self.post_composite_opts.ss_soft = if enabled { 1.0 } else { 0.0 };
        self.flush_composite_opts();
    }

    /// Update all mood/post-processing parameters and push to GPU.
    pub fn set_mood_params(&mut self, p: &MoodParams) {
        let o = &mut self.post_composite_opts;
        // Vignette
        o.vignette_strength = p.vignette.clamp(0.0, 1.0);
        // Grain
        o.grain_enabled = if p.grain_enabled { 1.0 } else { 0.0 };
        o.grain_strength = p.grain_strength.clamp(0.0, 0.5);
        o.grain_animated = if p.grain_animated { 1.0 } else { 0.0 };
        o.grain_speed = p.grain_speed.clamp(0.0, 4.0);
        o.grain_colorful = if p.grain_colorful { 1.0 } else { 0.0 };
        // Atmosphere
        o.atm_enabled = if p.atm_enabled { 1.0 } else { 0.0 };
        o.atm_thickness = p.atm_thickness.max(0.1);
        o.atm_density = p.atm_density.clamp(0.0, 1.0);
        o.atm_spatial_mode = if p.atm_aerial { 1.0 } else { 0.0 };
        let (ar, ag, ab) = hex_to_linear_rgb(&p.atm_color);
        o.atm_color_r = ar;
        o.atm_color_g = ag;
        o.atm_color_b = ab;
        o.atm_mode = if p.atm_positive_side { 1.0 } else { 0.0 };
        o.atm_plane_nx = p.atm_plane_nx;
        o.atm_plane_ny = p.atm_plane_ny;
        o.atm_plane_nz = p.atm_plane_nz;
        o.atm_plane_c = p.atm_plane_c;
        o.atm_height_bias = p.atm_height_bias;
        o.atm_height_falloff = p.atm_height_falloff.max(1.0);
        o.atm_drift_enabled = if p.atm_drift_enabled { 1.0 } else { 0.0 };
        o.atm_drift_amount = p.atm_drift_amount.clamp(0.0, 1.0);
        o.atm_drift_scale = p.atm_drift_scale;
        o.atm_drift_speed = p.atm_drift_speed;
        // Distance tint
        o.dt_enabled = if p.dt_enabled { 1.0 } else { 0.0 };
        let (nr, ng, nb) = hex_to_linear_rgb(&p.dt_near_color);
        o.dt_near_r = nr;
        o.dt_near_g = ng;
        o.dt_near_b = nb;
        let (mr, mg, mb) = hex_to_linear_rgb(&p.dt_mid_color);
        o.dt_mid_r = mr;
        o.dt_mid_g = mg;
        o.dt_mid_b = mb;
        let (fr, fg, fb) = hex_to_linear_rgb(&p.dt_far_color);
        o.dt_far_r = fr;
        o.dt_far_g = fg;
        o.dt_far_b = fb;
        o.dt_near_dist = p.dt_near_dist.max(0.0);
        o.dt_far_dist = p.dt_far_dist.max(0.0);
        o.dt_strength = p.dt_strength.clamp(0.0, 1.0);
        // Sun shafts
        o.ss_enabled = if p.ss_enabled { 1.0 } else { 0.0 };
        o.ss_strength = p.ss_strength.clamp(0.0, 10.0);
        o.ss_decay = p.ss_decay.clamp(0.5, 0.99);
        o.ss_density = p.ss_density.clamp(0.1, 1.5);
        o.ss_weight = p.ss_weight.clamp(0.0, 1.5);
        o.ss_samples = p.ss_samples.clamp(20.0, 56.0);
        // sun_uv computed per-frame from light direction
        o.bloom_strength = p.bloom_strength.clamp(0.0, 3.0);
        self.flush_composite_opts();
        // Screen-space reflections
        self.ssr_opts.enabled = if p.ssr_enabled { 1.0 } else { 0.0 };
        self.ssr_opts.strength = p.ssr_strength.clamp(0.0, 1.0);
        self.queue
            .write_buffer(&self.ssr_opts_buf, 0, bytemuck::bytes_of(&self.ssr_opts));
        // Mood changes invalidate the ray-trace accumulation buffer.
        // Reset to sample_n=0 with fast_preview so the next frame shows a quick
        // low-noise preview before the full progressive convergence resumes.
        if self.raytrace_enabled {
            self.rt_sample_n = 0;
            self.rt_fast_preview = true;
        }
    }

    /// Push current composite opts to GPU.
    fn flush_composite_opts(&self) {
        self.queue.write_buffer(
            &self.post_composite_opts_buf,
            0,
            bytemuck::bytes_of(&self.post_composite_opts),
        );
    }

    fn sync_composite_exposure_ev(&mut self) {
        // Positive bias compensates for tonemapper compression that darkens the
        // mid-range; without it, autoexposure at user-bias +0 renders too dark.
        const AUTO_EV_BIAS: f32 = 1.0;
        self.post_composite_opts.exposure_ev = if self.auto_exposure_enabled {
            (self.auto_exposure_smoothed + self.exposure_user_ev + AUTO_EV_BIAS).clamp(-5.0, 5.0)
        } else {
            self.exposure_user_ev
        };
    }

    /// Initiate an async readback of the 1×1 meter texture that was just copied to
    /// `meter_staging` in this frame's command encoder.  The result will be consumed
    /// on the *next* frame by `try_collect_meter_luminance` — no blocking poll here.
    pub(crate) fn begin_meter_readback(&mut self) {
        // If a previous mapping is still in flight, abandon it (don't start a second one).
        if self.meter_pending_rx.is_some() {
            return;
        }
        let slice = self.meter_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.meter_pending_rx = Some(rx);
    }

    /// Non-blockingly collect the luminance value started by `begin_meter_readback` on the
    /// previous frame.  If the GPU hasn't finished yet, the value is silently skipped this
    /// frame (exposure adapts on the next available result — imperceptible at 60 fps).
    pub(crate) fn try_collect_meter_luminance(&mut self) {
        let rx = match self.meter_pending_rx.take() {
            Some(r) => r,
            None => return,
        };
        // Poll the device once (non-blocking) to give wgpu a chance to mark the mapping done.
        self.device.poll(wgpu::Maintain::Poll);
        match rx.try_recv() {
            Ok(Ok(())) => {
                let slice = self.meter_staging.slice(..);
                let lum = {
                    let view = slice.get_mapped_range();
                    let arr: [u8; 4] = view[0..4].try_into().unwrap_or([0; 4]);
                    let v = f32::from_le_bytes(arr);
                    drop(view);
                    v
                };
                self.meter_staging.unmap();
                let target = 0.18_f32;
                let ev_inst = (target / lum.max(1e-5)).log2();
                // Faster adaptation so toggling auto is visibly responsive (still stable).
                self.auto_exposure_smoothed = self.auto_exposure_smoothed * 0.82 + ev_inst * 0.18;
            }
            Ok(Err(_)) => {
                // Mapping failed — unmap and discard.
                self.meter_staging.unmap();
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                // GPU not done yet — put the receiver back so we retry next frame.
                self.meter_pending_rx = Some(rx);
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                // Channel dropped unexpectedly — nothing to unmap.
            }
        }
    }

    pub fn apply_lighting_settings(&mut self, s: &crate::voxelle::LightingSettings) {
        self.light_ambient = s.ambient_intensity.max(0.0);
        self.light_sun = s.sunlight_intensity.max(0.0);
        self.shadows_enabled = s.enable_shadows;
        self.sky_enabled = s.enable_sky;
        self.light_dir =
            light_dir_from_azimuth_elevation_deg(s.light_angle_deg, s.light_elevation_deg);
        self.sun_color_linear = crate::voxelle::hex_srgb_to_linear_rgb3(&s.light_color)
            .map(|c| Vec3::new(c[0], c[1], c[2]))
            .unwrap_or(Vec3::ONE);
        self.bg_color_linear = crate::voxelle::hex_srgb_to_linear_rgb3(&s.background_color)
            .map(|c| Vec3::new(c[0], c[1], c[2]))
            .unwrap_or(Vec3::new(0.04, 0.045, 0.055));
        self.exposure_user_ev = s.exposure_ev.clamp(-5.0, 5.0);
        let was_auto = self.auto_exposure_enabled;
        self.auto_exposure_enabled = s.auto_exposure;
        if s.auto_exposure && !was_auto {
            self.auto_exposure_smoothed = 0.0;
        }
        self.sync_composite_exposure_ev();
        self.queue.write_buffer(
            &self.post_composite_opts_buf,
            0,
            bytemuck::bytes_of(&self.post_composite_opts),
        );
        // Lighting change invalidates accumulated RT samples.
        self.rt_sample_n = 0;
    }

    pub fn set_raytrace_mode(&mut self, enabled: bool) {
        self.raytrace_enabled = enabled;
        self.rt_sample_n = 0;
    }

    /// Update the surface-normal style used by the ray tracer.
    /// 0 = blocky (greedy), 1 = smooth (marching cubes), 2 = puffy (dual contour).
    pub fn set_rt_surface_mode(&mut self, mode: u32) {
        if self.rt_surface_mode != mode {
            self.rt_surface_mode = mode;
            self.rt_sample_n = 0;
        }
    }

    pub fn set_start_screen_transparent(&mut self, v: bool) {
        self.start_screen_transparent = v;
    }

    pub fn set_start_screen_appearance(&mut self, t: f32) {
        self.start_screen_appearance = t.clamp(0.0, 1.0);
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
            layout: &self.post_blur_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_linear),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.bloom_extract_buf.as_entire_binding(),
                },
            ],
        });
        self.bind_meter = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("meter_lum"),
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
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_depth),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.global_buffer.as_entire_binding(),
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.depth_snapshot_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_depth),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.ssr_opts_buf.as_entire_binding(),
                },
            ],
        }));
        self.bind_ssr = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssr_fullscreen"),
            layout: &self.ssr_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.global_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.depth_snapshot_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.hdr_opaque_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_linear),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_depth),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.ssr_opts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&self.normal_view),
                },
            ],
        });
        self.bind_oit_composite = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("oit_composite"),
            layout: &self.oit_composite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.oit_accum_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.oit_revealage_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.hdr_opaque_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_linear),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.ssr_view),
                },
            ],
        });

        let (
            bind_blit_down,
            bind_blit_up,
            bind_blit_up_weighted,
            bind_blit_final,
            bind_blur_pyr_h,
            bind_blur_pyr_v,
        ) = build_bloom_pyramid_bind_groups(
            &self.device,
            &self.post_bloom_layout,
            &self.post_blur_layout,
            &self.bloom_a_view,
            &self.bloom_pyramid_a_views,
            &self.bloom_pyramid_b_views,
            &self.sampler_linear,
            &self.post_blur_buf,
            &self.post_blit_weight_buf,
        );
        self.bind_blit_down = bind_blit_down;
        self.bind_blit_up = bind_blit_up;
        self.bind_blit_up_weighted = bind_blit_up_weighted;
        self.bind_blit_final = bind_blit_final;
        self.bind_blur_pyr_h = bind_blur_pyr_h;
        self.bind_blur_pyr_v = bind_blur_pyr_v;

        // Raytracer bind groups — must be rebuilt on brick change (rt_scene_bg) and on
        // viewport resize (rt_accum_bgs, because rt_accum_views are reallocated).
        self.rt_scene_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rt_scene_bg"),
            layout: &self.rt_scene_layout,
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
        // bg[0]: write→accum[0], read from accum[1]
        // bg[1]: write→accum[1], read from accum[0]
        self.rt_accum_bgs = vec![
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rt_accum_bg0"),
                layout: &self.rt_accum_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.rt_accum_views[1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler_linear),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.rt_uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.post_composite_opts_buf.as_entire_binding(),
                    },
                ],
            }),
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rt_accum_bg1"),
                layout: &self.rt_accum_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.rt_accum_views[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler_linear),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.rt_uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.post_composite_opts_buf.as_entire_binding(),
                    },
                ],
            }),
        ];
        self.rt_preview_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rt_preview_bg"),
            layout: &self.post_bloom_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.rt_preview_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_linear),
                },
            ],
        });
    }

    // -- mesh upload, mesh greedy, overlay upload, text, avatar methods moved to sub-modules --

    pub fn set_gizmo_on_top(&mut self, v: bool) {
        self.gizmo_on_top = v;
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
        self.rt_sample_n = 0;
        self.rebuild_bind_groups();
    }

    /// Multiple single-cell brick writes; falls back to full brick rebuild if layout mismatches.
    pub fn upload_scene_data_patches(
        &mut self,
        bounds: MeshBounds,
        voxels: &[Voxel],
        patches: &[crate::gpu_brick::BrickCellWrite],
    ) {
        self.scene_bounds = bounds;
        const MAX_AXIS: u32 = 512;
        if let Some(layout) = GpuVoxelBrick::layout_from_voxels(voxels, MAX_AXIS) {
            if layout.origin == self.brick_origin_iv && layout.dims == self.brick_dims_u {
                let mut ok = true;
                for p in patches {
                    if layout.index_of_world(p.x, p.y, p.z).is_none() {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    for p in patches {
                        if let Some(off) = layout.index_of_world(p.x, p.y, p.z) {
                            self.queue.write_buffer(
                                &self.brick_buffer,
                                (off * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
                                &p.packed.to_le_bytes(),
                            );
                        }
                    }
                    self.rt_sample_n = 0;
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

    // update_uniforms(), render(), and run_raytrace_benchmark() moved to frame.rs
    // ── Mascot helpers ────────────────────────────────────────────────────────

    fn make_mascot_depth(
        device: &wgpu::Device,
        w: u32,
        h: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mascot_depth"),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
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
