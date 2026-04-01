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
    pub mod collab_peer_lines {
        pub const WGSL: &str = include_str!("collab_peer_lines.wgsl");
    }
    pub mod collab_frustum {
        pub const WGSL: &str = include_str!("collab_frustum.wgsl");
    }
    pub mod ray_trace {
        pub const WGSL: &str = include_str!("ray_trace.wgsl");
    }
    pub mod oit_composite {
        pub const WGSL: &str = include_str!("oit_composite.wgsl");
    }
}

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
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Runtime};
use wgpu::util::DeviceExt;

/// Bump when [`gpu::mesh_greedy::WGSL`] bind group layout changes.
const MESH_GREEDY_PIPELINE_LAYOUT_VERSION: u32 = 2;
/// Number of mip levels in the bloom downsample/upsample pyramid.
const BLOOM_LEVELS: usize = 5;

/// Opaque mesh vertex: `vec3 pos, vec3 n, vec3 color, mat_kind, ao, vec3 emission_tint` → 14×`f32`.
const OPAQUE_VERTEX_STRIDE: u64 = 56;

/// [`Maintain::Wait`] can starve other threads while the GPU drains; use a short [`Maintain::Poll`]
/// loop with yields during heavy mesh rebuild so the app stays responsive.
#[inline]
fn poll_device_yielding_until_queue_empty(device: &wgpu::Device) {
    loop {
        if device.poll(wgpu::Maintain::Poll).is_queue_empty() {
            break;
        }
        std::thread::yield_now();
    }
}

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

fn cpu_mesh_fallback_prepare(
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
        Ok((PreparedOpaqueUpload::Single(mesh), bounds, "cpu".to_string()))
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
    /// x: ambient scale, y: sun scale, z: shadows on (0/1), w: sky gradient on (0/1)
    light_params: [f32; 4],
    sun_color: [f32; 4],
    bg_color: [f32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PostBlurUniform {
    blur_dir: [f32; 4],
}

/// Matches `post_bloom_extract.wgsl` `BloomExtractU`.
#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BloomExtractUniform {
    exposure_ev: f32,
    _pad: [f32; 3],
}

/// Matches `scene.wgsl` `SsrOpts` (inline SSR in `fs_trans`).
#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsrOpts {
    /// Scales the final confidence → controls overall reflection intensity.
    strength: f32,
    /// Number of world-space ray-march steps (clamped 8..64 in the shader).
    max_steps: f32,
    /// Maximum distance between ray hit and scene surface before ignoring hit.
    thickness: f32,
    /// 1.0 = SSR enabled, 0.0 = disabled (shader early-outs).
    enabled: f32,
}

/// sRGB hex (#rrggbb or rrggbb) → linear RGB (0–1).
fn hex_to_linear_rgb(hex: &str) -> (f32, f32, f32) {
    let s = hex.trim_start_matches('#');
    let n = u32::from_str_radix(s, 16).unwrap_or(0xffffff);
    let srgb = |c: u32| -> f32 {
        let v = (c & 0xff) as f32 / 255.0;
        v.powf(2.2)
    };
    (srgb(n >> 16), srgb(n >> 8), srgb(n))
}

/// All mood effect parameters sent from the React frontend.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MoodParams {
    // vignette (desktop-only)
    #[serde(default)]
    pub vignette: f32,
    // grain
    #[serde(default)]
    pub grain_enabled: bool,
    #[serde(default = "default_grain_strength")]
    pub grain_strength: f32,
    #[serde(default = "default_true")]
    pub grain_animated: bool,
    #[serde(default = "default_one")]
    pub grain_speed: f32,
    #[serde(default = "default_true")]
    pub grain_colorful: bool,
    // atmosphere
    #[serde(default)]
    pub atm_enabled: bool,
    #[serde(default = "default_atm_color")]
    pub atm_color: String,
    #[serde(default = "default_atm_thickness")]
    pub atm_thickness: f32,
    #[serde(default = "default_atm_density")]
    pub atm_density: f32,
    #[serde(default = "default_true")]
    pub atm_aerial: bool,
    #[serde(default)]
    pub atm_positive_side: bool,
    #[serde(default)]
    pub atm_plane_nx: f32,
    #[serde(default)]
    pub atm_plane_ny: f32,
    #[serde(default)]
    pub atm_plane_nz: f32,
    #[serde(default)]
    pub atm_plane_c: f32,
    #[serde(default)]
    pub atm_height_bias: f32,
    #[serde(default = "default_atm_height_falloff")]
    pub atm_height_falloff: f32,
    #[serde(default)]
    pub atm_drift_enabled: bool,
    #[serde(default = "default_drift_amount")]
    pub atm_drift_amount: f32,
    #[serde(default = "default_drift_scale")]
    pub atm_drift_scale: f32,
    #[serde(default = "default_drift_speed")]
    pub atm_drift_speed: f32,
    // distance tint
    #[serde(default)]
    pub dt_enabled: bool,
    #[serde(default = "default_dt_near_color")]
    pub dt_near_color: String,
    #[serde(default = "default_dt_mid_color")]
    pub dt_mid_color: String,
    #[serde(default = "default_dt_far_color")]
    pub dt_far_color: String,
    #[serde(default = "default_dt_near_dist")]
    pub dt_near_dist: f32,
    #[serde(default = "default_dt_far_dist")]
    pub dt_far_dist: f32,
    #[serde(default = "default_dt_strength")]
    pub dt_strength: f32,
    // sun shafts
    #[serde(default)]
    pub ss_enabled: bool,
    #[serde(default = "default_ss_strength")]
    pub ss_strength: f32,
    #[serde(default = "default_ss_decay")]
    pub ss_decay: f32,
    #[serde(default = "default_ss_density")]
    pub ss_density: f32,
    #[serde(default = "default_ss_weight")]
    pub ss_weight: f32,
    #[serde(default = "default_ss_samples")]
    pub ss_samples: f32,
    // screen-space reflections
    #[serde(default)]
    pub ssr_enabled: bool,
    #[serde(default = "default_ssr_strength")]
    pub ssr_strength: f32,
    // bloom
    #[serde(default = "default_bloom_strength")]
    pub bloom_strength: f32,
}

fn default_true() -> bool { true }
fn default_one() -> f32 { 1.0 }
fn default_grain_strength() -> f32 { 0.12 }
fn default_atm_color() -> String { "#c8d4e0".into() }
fn default_atm_thickness() -> f32 { 28.0 }
fn default_atm_density() -> f32 { 0.85 }
fn default_atm_height_falloff() -> f32 { 120.0 }
fn default_drift_amount() -> f32 { 0.2 }
fn default_drift_scale() -> f32 { 0.02 }
fn default_drift_speed() -> f32 { 0.2 }
fn default_dt_near_color() -> String { "#ffffff".into() }
fn default_dt_mid_color() -> String { "#c8d4e0".into() }
fn default_dt_far_color() -> String { "#8fa3bf".into() }
fn default_dt_near_dist() -> f32 { 16.0 }
fn default_dt_far_dist() -> f32 { 140.0 }
fn default_dt_strength() -> f32 { 0.6 }
fn default_ss_strength() -> f32 { 0.7 }
fn default_ss_decay() -> f32 { 0.92 }
fn default_ss_density() -> f32 { 0.8 }
fn default_ss_weight() -> f32 { 0.6 }
fn default_ss_samples() -> f32 { 32.0 }
fn default_ssr_strength() -> f32 { 0.8 }
fn default_bloom_strength() -> f32 { 0.1 }

impl Default for MoodParams {
    fn default() -> Self {
        Self {
            vignette: 0.0,
            grain_enabled: false,
            grain_strength: default_grain_strength(),
            grain_animated: true,
            grain_speed: 1.0,
            grain_colorful: true,
            atm_enabled: false,
            atm_color: default_atm_color(),
            atm_thickness: default_atm_thickness(),
            atm_density: default_atm_density(),
            atm_aerial: true,
            atm_positive_side: false,
            atm_plane_nx: 0.0,
            atm_plane_ny: 0.0,
            atm_plane_nz: 0.0,
            atm_plane_c: 0.0,
            atm_height_bias: 0.0,
            atm_height_falloff: default_atm_height_falloff(),
            atm_drift_enabled: false,
            atm_drift_amount: default_drift_amount(),
            atm_drift_scale: default_drift_scale(),
            atm_drift_speed: default_drift_speed(),
            dt_enabled: false,
            dt_near_color: default_dt_near_color(),
            dt_mid_color: default_dt_mid_color(),
            dt_far_color: default_dt_far_color(),
            dt_near_dist: default_dt_near_dist(),
            dt_far_dist: default_dt_far_dist(),
            dt_strength: default_dt_strength(),
            ss_enabled: false,
            ss_strength: default_ss_strength(),
            ss_decay: default_ss_decay(),
            ss_density: default_ss_density(),
            ss_weight: default_ss_weight(),
            ss_samples: default_ss_samples(),
            ssr_enabled: false,
            ssr_strength: default_ssr_strength(),
            bloom_strength: default_bloom_strength(),
        }
    }
}

/// Matches `post_composite.wgsl` `PostCompositeOpts` and Voxelle web tone mapping ids (neutral…reinhard).
/// Layout: 14 vec4 rows = 224 bytes.
#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PostCompositeOpts {
    // --- Row 0 ---
    tone_mode: u32,
    /// 1 = premultiplied alpha from scene energy (cold-start logo over webview).
    transparent_bg: f32,
    /// Exposure in EV stops (`rgb *= 2^ev` before tone mapping).
    exposure_ev: f32,
    /// Monotonic seconds since viewer creation (for animated effects).
    time_seconds: f32,
    // --- Row 1: vignette + grain basics ---
    vignette_strength: f32,
    grain_enabled: f32,
    grain_strength: f32,
    grain_animated: f32,
    // --- Row 2: grain continued ---
    grain_speed: f32,
    /// 1.0 = colorful (per-channel noise), 0.0 = monochrome.
    grain_colorful: f32,
    _pad2a: f32,
    _pad2b: f32,
    // --- Row 3: atmosphere controls ---
    atm_enabled: f32,
    atm_thickness: f32,
    atm_density: f32,
    /// 0 = plane, 1 = aerial.
    atm_spatial_mode: f32,
    // --- Row 4: atmosphere color + mode ---
    atm_color_r: f32,
    atm_color_g: f32,
    atm_color_b: f32,
    /// 0 = slab, 1 = positiveSide.
    atm_mode: f32,
    // --- Row 5: atmosphere plane ---
    atm_plane_nx: f32,
    atm_plane_ny: f32,
    atm_plane_nz: f32,
    atm_plane_c: f32,
    // --- Row 6: atmosphere height + drift ---
    atm_height_bias: f32,
    atm_height_falloff: f32,
    atm_drift_enabled: f32,
    atm_drift_amount: f32,
    // --- Row 7: drift continued ---
    atm_drift_scale: f32,
    atm_drift_speed: f32,
    _pad7a: f32,
    _pad7b: f32,
    // --- Row 8: distance tint controls ---
    dt_enabled: f32,
    dt_near_dist: f32,
    dt_far_dist: f32,
    dt_strength: f32,
    // --- Row 9-11: distance tint colors ---
    dt_near_r: f32,
    dt_near_g: f32,
    dt_near_b: f32,
    _pad9: f32,
    dt_mid_r: f32,
    dt_mid_g: f32,
    dt_mid_b: f32,
    _pad10: f32,
    dt_far_r: f32,
    dt_far_g: f32,
    dt_far_b: f32,
    _pad11: f32,
    // --- Row 12: sun shafts ---
    ss_enabled: f32,
    ss_strength: f32,
    ss_decay: f32,
    ss_density: f32,
    // --- Row 13: sun shafts continued ---
    ss_weight: f32,
    ss_samples: f32,
    ss_sun_uv_x: f32,
    ss_sun_uv_y: f32,
    // --- Row 14: bloom ---
    bloom_strength: f32,
    _pad14a: f32,
    _pad14b: f32,
    _pad14c: f32,
}

pub struct WgpuViewer {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub format: wgpu::TextureFormat,
    /// Swapchain / Metal drawable size (full webview — must match window or macOS stretches the image).
    pub surface_size: (u32, u32),
    /// `.viewport` div in physical pixels (same space as [`Self::surface_size`]).
    pub viewport_x: u32,
    pub viewport_y: u32,
    pub viewport_width: u32,
    pub viewport_height: u32,

    global_buffer: wgpu::Buffer,
    brick_buffer: wgpu::Buffer,
    brick_cell_count: u32,
    brick_origin_iv: IVec3,
    brick_dims_u: (u32, u32, u32),

    scene_bounds: MeshBounds,
    light_dir: Vec3,
    light_ambient: f32,
    light_sun: f32,
    sun_color_linear: Vec3,
    bg_color_linear: Vec3,
    shadows_enabled: bool,
    pub(crate) soft_shadows: bool,
    sky_enabled: bool,

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
    /// Snapshot of the opaque depth, copied before the trans pass so fs_trans can
    /// read depth for SSR without conflicting with the depth-stencil write attachment.
    depth_snapshot_texture: wgpu::Texture,
    depth_snapshot_view: wgpu::TextureView,

    /// OIT accumulation texture (Rgba16Float): weighted color + alpha accumulation.
    oit_accum_texture: wgpu::Texture,
    oit_accum_view: wgpu::TextureView,
    /// OIT revealage texture (R16Float): product of (1 − alpha) across all transparent layers.
    oit_revealage_texture: wgpu::Texture,
    oit_revealage_view: wgpu::TextureView,

    bloom_a: wgpu::Texture,
    bloom_a_view: wgpu::TextureView,
    bloom_b: wgpu::Texture,
    bloom_b_view: wgpu::TextureView,
    /// Pyramid mip levels for bloom: index 0 = 1/2 res, 1 = 1/4, …, 4 = 1/32.
    bloom_pyramid_a: Vec<wgpu::Texture>,
    bloom_pyramid_a_views: Vec<wgpu::TextureView>,
    bloom_pyramid_b: Vec<wgpu::Texture>,
    bloom_pyramid_b_views: Vec<wgpu::TextureView>,

    present_texture: wgpu::Texture,
    present_view: wgpu::TextureView,

    scene_layout0: wgpu::BindGroupLayout,
    scene_layout1: wgpu::BindGroupLayout,
    shadow_vs_layout: wgpu::BindGroupLayout,
    post_bloom_layout: wgpu::BindGroupLayout,
    post_blur_layout: wgpu::BindGroupLayout,
    post_composite_layout: wgpu::BindGroupLayout,
    oit_composite_layout: wgpu::BindGroupLayout,

    bind_scene_opaque: wgpu::BindGroup,
    bind_shadow_pass: wgpu::BindGroup,
    bind_bloom_extract: wgpu::BindGroup,
    bind_blur_h: wgpu::BindGroup,
    bind_blur_v: wgpu::BindGroup,
    bind_composite: wgpu::BindGroup,
    /// Pyramid bloom bind groups — rebuilt on resize via rebuild_bind_groups.
    bind_blit_down: Vec<wgpu::BindGroup>,
    bind_blit_up: Vec<wgpu::BindGroup>,
    /// Weighted upsample bind groups (use post_blur_layout + post_blit_weight_buf).
    bind_blit_up_weighted: Vec<wgpu::BindGroup>,
    bind_blit_final: wgpu::BindGroup,
    bind_blur_pyr_h: Vec<wgpu::BindGroup>,
    bind_blur_pyr_v: Vec<wgpu::BindGroup>,
    bind_trans: Option<wgpu::BindGroup>,
    bind_oit_composite: wgpu::BindGroup,

    post_blur_buf: wgpu::Buffer,
    /// Feeds exposure_ev into the bloom extract shader for physical threshold scaling.
    bloom_extract_buf: wgpu::Buffer,
    /// Constant weight (0.75) for the weighted bloom upsample pyramid.
    post_blit_weight_buf: wgpu::Buffer,
    post_composite_opts_buf: wgpu::Buffer,
    post_composite_opts: PostCompositeOpts,

    pipeline_opaque: wgpu::RenderPipeline,
    /// Web-style ghost: occluded (Greater) then front (Always), unlit + alpha blend; no gbuffer writes.
    pipeline_preview_occluded: wgpu::RenderPipeline,
    pipeline_preview_front: wgpu::RenderPipeline,
    /// Same as [`pipeline_preview_front`] but **no** depth bias — wireframe edges use true depth
    /// so they are occluded by scene geometry when embedded in solids.
    pipeline_preview_front_wire: wgpu::RenderPipeline,
    // GPU-instanced preview pipelines
    pipeline_preview_inst_occluded: wgpu::RenderPipeline,
    pipeline_preview_inst_front: wgpu::RenderPipeline,
    pipeline_preview_inst_front_wire: wgpu::RenderPipeline,
    pipeline_collab_lines_occluded: wgpu::RenderPipeline,
    pipeline_collab_lines_front: wgpu::RenderPipeline,
    pipeline_collab_frustum_occluded: wgpu::RenderPipeline,
    pipeline_collab_frustum_front: wgpu::RenderPipeline,
    pipeline_collab_frustum_tri_occluded: wgpu::RenderPipeline,
    pipeline_collab_frustum_tri_front: wgpu::RenderPipeline,
    /// Voxel grid borders: depth-tested only (no occluded ghost pass), semi-transparent.
    pipeline_grid_border_lines: wgpu::RenderPipeline,
    /// Selection transform gizmo (move arrows + rotation rings).
    pipeline_gizmo_lines_front: wgpu::RenderPipeline,
    pipeline_gizmo_lines_occluded: wgpu::RenderPipeline,
    pipeline_gizmo_tris_front: wgpu::RenderPipeline,
    pipeline_gizmo_tris_occluded: wgpu::RenderPipeline,
    pipeline_sky: wgpu::RenderPipeline,
    pipeline_start_screen_bg: wgpu::RenderPipeline,
    pipeline_oit_accum: wgpu::RenderPipeline,
    pipeline_oit_composite: wgpu::RenderPipeline,
    pipeline_shadow: wgpu::RenderPipeline,
    pipeline_bloom_extract: wgpu::RenderPipeline,
    pipeline_blur: wgpu::RenderPipeline,
    pipeline_blit: wgpu::RenderPipeline,
    /// Weighted additive blit for bloom upsample: multiplies by a scalar uniform before adding.
    pipeline_blit_weighted_add: wgpu::RenderPipeline,
    pipeline_composite: wgpu::RenderPipeline,
    /// 1×1 average linear luminance (HDR, pre-exposure) for auto exposure.
    pipeline_meter: wgpu::RenderPipeline,
    // ── Progressive ray tracer ────────────────────────────────────────────
    pub raytrace_enabled: bool,
    /// Ping-pong accumulation textures (Rgba16Float, viewport size).
    /// Index 0/1 alternate as read/write each frame.
    rt_accum_textures: [wgpu::Texture; 2],
    rt_accum_views: [wgpu::TextureView; 2],
    /// Which accumulator index is currently written to this frame.
    rt_accum_flip: bool,
    /// Running count of accumulated samples; 0 = reset / first frame.
    rt_sample_n: u32,
    rt_uniform_buf: wgpu::Buffer,
    /// Layout for group 0 (global + brick only — no shadow sampler).
    rt_scene_layout: wgpu::BindGroupLayout,
    /// Layout for group 1 (accum_prev + sampler + rt_uniform).
    rt_accum_layout: wgpu::BindGroupLayout,
    /// Bind group for group 0: global + brick.
    rt_scene_bg: wgpu::BindGroup,
    /// Two bind groups for group 1 — one per ping-pong side.
    /// rt_accum_bgs[i] reads from rt_accum_textures[1-i] (the non-written side).
    rt_accum_bgs: Vec<wgpu::BindGroup>,
    /// Half-resolution texture used as the raytrace render target in fast_preview mode.
    /// Upscaled to full-res after the pass so the rest of the pipeline is unchanged.
    rt_preview_tex:  wgpu::Texture,
    rt_preview_view: wgpu::TextureView,
    /// Bind group (post_bloom_layout: tex + sampler) for the upscale blit.
    rt_preview_bg:   wgpu::BindGroup,
    pipeline_raytrace: wgpu::RenderPipeline,
    /// Last camera eye used for accumulation-reset detection.
    rt_prev_eye: [f32; 3],
    rt_prev_inv_view: [[f32; 4]; 4],
    /// True when the camera moved this frame; shader uses cheap shading path.
    rt_fast_preview: bool,

    // ── Screen-space reflections (inline in fs_trans) ─────────────────────
    ssr_opts_buf: wgpu::Buffer,
    ssr_opts: SsrOpts,
    meter_texture: wgpu::Texture,
    meter_view: wgpu::TextureView,
    meter_staging: wgpu::Buffer,
    bind_meter: wgpu::BindGroup,
    /// User EV slider (−5…5); with auto exposure, added as bias on metered EV.
    exposure_user_ev: f32,
    auto_exposure_enabled: bool,
    /// Smoothed \( \log_2(\text{target}/\bar{L}) \) from metering.
    auto_exposure_smoothed: f32,
    /// Monotonic clock for animated shader effects (grain, atmosphere drift).
    creation_instant: std::time::Instant,

    /// When true, draw [`pipeline_start_screen_bg`] instead of sky (default true until a scene load sets [`ViewerState::start_screen_logo_transparent`] false).
    start_screen_transparent: bool,
    /// 0 = dark cold-start gradient, 1 = light (paper) — passed in [`GlobalState::params`].x for `start_screen_bg.wgsl`.
    start_screen_appearance: f32,

    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    index_count: u32,
    /// Split point in the non-chunked index buffer: indices `0..opaque_index_split` are opaque,
    /// `opaque_index_split..index_count` are transparent.
    opaque_index_split: u32,

    /// When set, opaque mesh is drawn from [`Self::opaque_chunks`] (multi-draw).
    pub opaque_chunked: bool,
    /// Chunk bucketing origin (must match [`greedy_mesh::voxel_buckets_by_chunk`]).
    pub chunk_grid_origin: IVec3,
    opaque_chunks: BTreeMap<ChunkKey, OpaqueChunkDraw>,
    /// Chunks waiting to be uploaded to GPU (progressive loading after initial batch).
    pending_chunk_uploads: VecDeque<(ChunkKey, MeshBuffers)>,
    /// Incremental occupancy + buckets; rebuilt on full chunked upload, updated O(1) per edit.
    spatial_mesh_cache: Option<greedy_mesh::SpatialMeshCache>,

    preview_vertex_buffer: Option<wgpu::Buffer>,
    preview_index_buffer: Option<wgpu::Buffer>,
    preview_index_count: u32,
    preview_wire_vertex_buffer: Option<wgpu::Buffer>,
    preview_wire_index_buffer: Option<wgpu::Buffer>,
    preview_wire_index_count: u32,
    // GPU-instanced preview buffers
    preview_solid_proto_vb: Option<wgpu::Buffer>,
    preview_solid_proto_ib: Option<wgpu::Buffer>,
    preview_solid_proto_idx_count: u32,
    preview_wire_proto_vb: Option<wgpu::Buffer>,
    preview_wire_proto_ib: Option<wgpu::Buffer>,
    preview_wire_proto_idx_count: u32,
    preview_solid_instance_buf: Option<wgpu::Buffer>,
    preview_solid_instance_count: u32,
    preview_wire_instance_buf: Option<wgpu::Buffer>,
    preview_wire_instance_count: u32,
    collab_line_vertex_buffer: Option<wgpu::Buffer>,
    collab_line_vertex_count: u32,
    collab_frustum_vertex_buffer: Option<wgpu::Buffer>,
    collab_frustum_vertex_count: u32,
    collab_frustum_tri_vertex_buffer: Option<wgpu::Buffer>,
    collab_frustum_tri_vertex_count: u32,
    ping_wave_line_vertex_buffer: Option<wgpu::Buffer>,
    ping_wave_line_vertex_count: u32,
    ping_vertex_buffer: Option<wgpu::Buffer>,
    ping_index_buffer: Option<wgpu::Buffer>,
    ping_index_count: u32,
    ping_wire_vertex_buffer: Option<wgpu::Buffer>,
    ping_wire_index_buffer: Option<wgpu::Buffer>,
    ping_wire_index_count: u32,
    selection_overlay_vertex_buffer: Option<wgpu::Buffer>,
    selection_overlay_index_buffer: Option<wgpu::Buffer>,
    selection_overlay_index_count: u32,
    selection_overlay_line_vertex_buffer: Option<wgpu::Buffer>,
    selection_overlay_line_vertex_count: u32,
    /// Selection transform gizmo geometry (lines: shafts + rings; tris: arrowheads).
    gizmo_line_vertex_buffer: Option<wgpu::Buffer>,
    gizmo_line_vertex_count: u32,
    gizmo_tri_vertex_buffer: Option<wgpu::Buffer>,
    gizmo_tri_vertex_count: u32,
    /// Full-scene grid wireframe (View → Show borders); indexed line-list.
    grid_border_line_vertex_buffer: Option<wgpu::Buffer>,
    grid_border_line_index_buffer: Option<wgpu::Buffer>,
    grid_border_line_index_count: u32,
    /// Fingerprint of visible voxel set + mesh generation for grid overlay rebuilds.
    pub(crate) grid_border_cache_key: Option<u64>,
    /// Dedup CPU mesh rebuild when hover cell unchanged.
    pub preview_cache_key: Option<u64>,
    /// `(selection fingerprint, mesh_refresh_generation)` — invalidates when selection or voxels change.
    pub selection_overlay_cache_key: Option<u64>,

    sampler_linear: wgpu::Sampler,
    sampler_depth: wgpu::Sampler,
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

    // ── Glyphon text rendering (peer labels) ──
    glyphon_font_system: FontSystem,
    glyphon_swash_cache: SwashCache,
    #[allow(dead_code)]
    glyphon_cache: GlyphonCache,
    glyphon_atlas: TextAtlas,
    glyphon_text_renderer: TextRenderer,
    glyphon_viewport: GlyphonViewport,
    /// Per-frame peer label data uploaded via [`Self::upload_peer_labels`].
    peer_label_data: Vec<GpuPeerLabel>,
    /// Active ping label (at most one).
    ping_label_data: Option<GpuPeerLabel>,
}

/// Screen-space peer label for GPU text rendering.
pub struct GpuPeerLabel {
    pub name: String,
    pub color_rgb: u32,
    /// Pixel X in viewport space.
    pub x: f32,
    /// Pixel Y in viewport space.
    pub y: f32,
}

/// GPU buffers for one spatial chunk of opaque greedy mesh.
struct OpaqueChunkDraw {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    opaque_index_count: u32,
    transparent_index_count: u32,
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RtUniform {
    frame_seed: u32,
    sample_n: u32,
    /// 1 when the camera moved this frame: use cheap shading (1 shadow ray, no bounces).
    fast_preview: u32,
    _pad1: u32,
}

fn create_rt_accum_tex(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rt_accum"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: hdr_format(),
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// Clamp a point so the shadow frustum center stays within (or near) the scene bounds.
fn clamp_point_to_bounds(p: Vec3, bounds: &MeshBounds, margin: f32) -> Vec3 {
    Vec3::new(
        p.x.clamp(bounds.min.x - margin, bounds.max.x + margin),
        p.y.clamp(bounds.min.y - margin, bounds.max.y + margin),
        p.z.clamp(bounds.min.z - margin, bounds.max.z + margin),
    )
}

/// Uses [`Mat4::orthographic_rh`], which glam documents as \([0,1]\) depth for WebGPU (do not apply an extra OpenGL→wgpu Z remap).
///
/// The shadow frustum is fitted to the camera's visible area so that close-up views
/// get dramatically higher shadow texel density without changing the map resolution.
fn light_view_proj(
    bounds: &MeshBounds,
    light_dir: Vec3,
    camera: &OrbitCamera,
    viewport_w: f32,
    viewport_h: f32,
) -> Mat4 {
    let scene_r = bounds.radius().max(8.0);

    // Visible radius at the camera target plane.
    let cam_dist = camera.smooth_spherical.radius;
    let aspect = (viewport_w / viewport_h.max(1.0)).max(1e-4);
    let visible_r = if camera.perspective {
        let v_half = cam_dist * (camera.fov_y * 0.5).tan();
        let h_half = v_half * aspect;
        (v_half * v_half + h_half * h_half).sqrt()
    } else {
        let hh = camera.ortho_half_height;
        let hw = hh * aspect;
        (hh * hh + hw * hw).sqrt()
    };

    // 1.5× padding keeps shadow casters just outside the view.
    // Clamp to scene radius (no point covering more) and a minimum of 2 voxels.
    let effective_r = (visible_r * 1.5).clamp(2.0, scene_r);

    // Centre the shadow frustum on the camera target, clamped to the scene.
    let shadow_center = clamp_point_to_bounds(camera.smooth_target, bounds, effective_r);

    let ld = light_dir.normalize();
    let up = if ld.cross(Vec3::Y).length() > 0.05 {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let eye = shadow_center + ld * (scene_r * 5.0);
    let view = Mat4::look_at_rh(eye, shadow_center, up);

    // Texel-snap the shadow centre in light-space so the sampling grid stays fixed
    // when the camera pans — prevents shadow swimming / shimmer.
    let he = effective_r * 1.8;
    let texel_world = (he * 2.0) / SHADOW_MAP_SIZE as f32;
    let ls = view.transform_point3(shadow_center);
    let snap = Vec3::new(
        (ls.x / texel_world).floor() * texel_world - ls.x,
        (ls.y / texel_world).floor() * texel_world - ls.y,
        0.0,
    );
    let view = Mat4::from_translation(snap) * view;

    let proj = Mat4::orthographic_rh(-he, he, -he, he, 1.0, scene_r * 12.0);
    proj * view
}

/// Horizontal angle (degrees, CCW from +X in XZ) and elevation above XZ (degrees). Y-up.
fn light_dir_from_azimuth_elevation_deg(azimuth_deg: f32, elevation_deg: f32) -> Vec3 {
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
            wgpu::VertexAttribute {
                offset: 44,
                shader_location: 5,
                format: wgpu::VertexFormat::Float32x3,
            },
        ],
    }
}

/// Prototype vertex for instanced preview: position (vec3) + normal (vec3) = 24 bytes.
fn preview_proto_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: 24, // 2 × vec3<f32>
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

/// Per-instance data for instanced preview: model matrix (4×vec4) + color (vec3) + mat_kind (f32) = 80 bytes.
fn preview_instance_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: 80, // 4×16 + 12 + 4
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            // model_c0
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x4,
            },
            // model_c1
            wgpu::VertexAttribute {
                offset: 16,
                shader_location: 3,
                format: wgpu::VertexFormat::Float32x4,
            },
            // model_c2
            wgpu::VertexAttribute {
                offset: 32,
                shader_location: 4,
                format: wgpu::VertexFormat::Float32x4,
            },
            // model_c3
            wgpu::VertexAttribute {
                offset: 48,
                shader_location: 5,
                format: wgpu::VertexFormat::Float32x4,
            },
            // inst_color
            wgpu::VertexAttribute {
                offset: 64,
                shader_location: 6,
                format: wgpu::VertexFormat::Float32x3,
            },
            // inst_mat_kind
            wgpu::VertexAttribute {
                offset: 76,
                shader_location: 7,
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

fn vertex_layout_collab_frustum() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: (3 + 3 + 1) * 4, // 28 bytes: pos + color + alpha
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
                format: wgpu::VertexFormat::Float32,
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


/// Read-only depth snapshot for SSR in the trans pass (COPY_DST | TEXTURE_BINDING only).
fn create_depth_snapshot(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth_snapshot"),
        size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// OIT accumulation (Rgba16Float) + revealage (R16Float) textures for WBOIT.
fn create_oit_textures(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::Texture, wgpu::TextureView) {
    let w = width.max(1);
    let h = height.max(1);
    let extent = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
    let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;

    let accum_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("oit_accum"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage,
        view_formats: &[],
    });
    let accum_view = accum_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let reveal_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("oit_revealage"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R16Float,
        usage,
        view_formats: &[],
    });
    let reveal_view = reveal_tex.create_view(&wgpu::TextureViewDescriptor::default());

    (accum_tex, accum_view, reveal_tex, reveal_view)
}

/// Tonemapped sRGB output at **viewport** resolution before [`copy_texture_to_texture`] into the swapchain.
fn create_present_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let w = width.max(1);
    let h = height.max(1);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("present"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
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
    let depth_usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC;

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

fn create_bloom_pyramid(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    hdr_fmt: wgpu::TextureFormat,
) -> (
    Vec<wgpu::Texture>,
    Vec<wgpu::TextureView>,
    Vec<wgpu::Texture>,
    Vec<wgpu::TextureView>,
) {
    let color_usage =
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    let mut a_textures = Vec::with_capacity(BLOOM_LEVELS);
    let mut a_views = Vec::with_capacity(BLOOM_LEVELS);
    let mut b_textures = Vec::with_capacity(BLOOM_LEVELS);
    let mut b_views = Vec::with_capacity(BLOOM_LEVELS);
    for i in 0..BLOOM_LEVELS {
        let w = (width >> (i + 1)).max(1);
        let h = (height >> (i + 1)).max(1);
        let extent = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
        let a = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("bloom_pyr_a_{i}")),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: hdr_fmt,
            usage: color_usage,
            view_formats: &[],
        });
        let a_view = a.create_view(&wgpu::TextureViewDescriptor::default());
        let b = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("bloom_pyr_b_{i}")),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: hdr_fmt,
            usage: color_usage,
            view_formats: &[],
        });
        let b_view = b.create_view(&wgpu::TextureViewDescriptor::default());
        a_textures.push(a);
        a_views.push(a_view);
        b_textures.push(b);
        b_views.push(b_view);
    }
    (a_textures, a_views, b_textures, b_views)
}

#[allow(clippy::too_many_arguments)]
fn build_bloom_pyramid_bind_groups(
    device: &wgpu::Device,
    post_bloom_layout: &wgpu::BindGroupLayout,
    post_blur_layout: &wgpu::BindGroupLayout,
    bloom_a_view: &wgpu::TextureView,
    pyr_a_views: &[wgpu::TextureView],
    pyr_b_views: &[wgpu::TextureView],
    sampler_linear: &wgpu::Sampler,
    post_blur_buf: &wgpu::Buffer,
    post_blit_weight_buf: &wgpu::Buffer,
) -> (
    Vec<wgpu::BindGroup>,
    Vec<wgpu::BindGroup>,
    Vec<wgpu::BindGroup>,
    wgpu::BindGroup,
    Vec<wgpu::BindGroup>,
    Vec<wgpu::BindGroup>,
) {
    let blit_bg = |label: &str, view: &wgpu::TextureView| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: post_bloom_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler_linear),
                },
            ],
        })
    };

    // bind_blit_down[0] reads bloom_a (full-res extract); [i] reads pyr_a[i-1].
    let bind_blit_down = (0..BLOOM_LEVELS)
        .map(|i| {
            let src = if i == 0 { bloom_a_view } else { &pyr_a_views[i - 1] };
            blit_bg(&format!("blit_down_{i}"), src)
        })
        .collect();

    // bind_blit_up[i] reads pyr_a[i+1] for additive upsample into pyr_a[i].
    let bind_blit_up = (0..BLOOM_LEVELS - 1)
        .map(|i| blit_bg(&format!("blit_up_{i}"), &pyr_a_views[i + 1]))
        .collect();

    // bind_blit_up_weighted[i]: same source textures as bind_blit_up but using post_blur_layout
    // so fs_blit_weighted can read the per-level weight uniform before the additive blend.
    let bind_blit_up_weighted: Vec<wgpu::BindGroup> = (0..BLOOM_LEVELS - 1)
        .map(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("blit_up_w_{i}")),
                layout: post_blur_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&pyr_a_views[i + 1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler_linear),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: post_blit_weight_buf.as_entire_binding(),
                    },
                ],
            })
        })
        .collect();

    // Final replace-blit: pyr_a[0] → bloom_a.
    let bind_blit_final = blit_bg("blit_final", &pyr_a_views[0]);

    let blur_bg = |label: &str, view: &wgpu::TextureView| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: post_blur_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler_linear),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: post_blur_buf.as_entire_binding(),
                },
            ],
        })
    };

    let bind_blur_pyr_h = (0..BLOOM_LEVELS)
        .map(|i| blur_bg(&format!("blur_pyr_h_{i}"), &pyr_a_views[i]))
        .collect();
    let bind_blur_pyr_v = (0..BLOOM_LEVELS)
        .map(|i| blur_bg(&format!("blur_pyr_v_{i}"), &pyr_b_views[i]))
        .collect();

    (bind_blit_down, bind_blit_up, bind_blit_up_weighted, bind_blit_final, bind_blur_pyr_h, bind_blur_pyr_v)
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
            format,
            width: size.0.max(1),
            height: size.1.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::PreMultiplied) {
                wgpu::CompositeAlphaMode::PreMultiplied
            } else {
                caps.alpha_modes[0]
            },
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
                // binding 2: depth_for_ssr — opaque scene depth for SSR ray marching
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
                // binding 3: samp_depth — non-filtering sampler for depth
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // binding 4: ssr_opts — SSR parameters uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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
                    // depth texture for world-space mood effects
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    // GlobalState (storage) for camera matrices
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
        let shader_start_screen_bg = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("start_screen_bg"),
            source: wgpu::ShaderSource::Wgsl(gpu::start_screen_bg::WGSL.into()),
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
        let shader_meter = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("meter_lum"),
            source: wgpu::ShaderSource::Wgsl(gpu::meter_luminance::WGSL.into()),
        });
        let shader_collab_lines = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("collab_peer_lines"),
            source: wgpu::ShaderSource::Wgsl(gpu::collab_peer_lines::WGSL.into()),
        });
        let shader_collab_frustum = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("collab_frustum"),
            source: wgpu::ShaderSource::Wgsl(gpu::collab_frustum::WGSL.into()),
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
                    // In-front preview only (complements `Greater` occluded pass). `Always` was
                    // blending bright preview over occluded/x-ray regions and through solids.
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    // Nudge slightly toward the camera so preview wins coplanar depth ties with mesh.
                    bias: wgpu::DepthBiasState {
                        constant: -2,
                        slope_scale: -0.5,
                        clamp: 0.0,
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });
        let pipeline_preview_front_wire =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("preview_front_wire"),
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
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        // GPU-instanced preview pipelines (same fragment shaders, new vertex shader + buffer layout)
        let inst_bufs = &[preview_proto_vertex_layout(), preview_instance_layout()];
        let pipeline_preview_inst_occluded =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("preview_inst_occluded"),
                layout: Some(&pl_opaque),
                vertex: wgpu::VertexState {
                    module: &shader_scene,
                    entry_point: Some("vs_preview_instanced"),
                    buffers: inst_bufs,
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
        let pipeline_preview_inst_front =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("preview_inst_front"),
                layout: Some(&pl_opaque),
                vertex: wgpu::VertexState {
                    module: &shader_scene,
                    entry_point: Some("vs_preview_instanced"),
                    buffers: inst_bufs,
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
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState {
                        constant: -2,
                        slope_scale: -0.5,
                        clamp: 0.0,
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });
        let pipeline_preview_inst_front_wire =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("preview_inst_front_wire"),
                layout: Some(&pl_opaque),
                vertex: wgpu::VertexState {
                    module: &shader_scene,
                    entry_point: Some("vs_preview_instanced"),
                    buffers: inst_bufs,
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
                    depth_compare: wgpu::CompareFunction::LessEqual,
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
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState {
                        constant: -2,
                        slope_scale: -0.5,
                        clamp: 0.0,
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        let pipeline_grid_border_lines =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("grid_border_lines"),
                layout: Some(&pl_opaque),
                vertex: wgpu::VertexState {
                    module: &shader_collab_lines,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout_collab_lines()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_collab_lines,
                    entry_point: Some("fs_grid_border_line"),
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
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState {
                        constant: -2,
                        slope_scale: -0.5,
                        clamp: 0.0,
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        let pipeline_gizmo_lines_front =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("gizmo_lines_front"),
                layout: Some(&pl_opaque),
                vertex: wgpu::VertexState {
                    module: &shader_collab_lines,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout_collab_lines()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_collab_lines,
                    entry_point: Some("fs_gizmo_front"),
                    targets: preview_targets,
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState { constant: -2, slope_scale: -0.5, clamp: 0.0 },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });
        let pipeline_gizmo_lines_occluded =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("gizmo_lines_occluded"),
                layout: Some(&pl_opaque),
                vertex: wgpu::VertexState {
                    module: &shader_collab_lines,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout_collab_lines()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_collab_lines,
                    entry_point: Some("fs_gizmo_occluded"),
                    targets: preview_targets,
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Greater,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState { constant: 1, slope_scale: 1.0, clamp: 0.0 },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });
        let pipeline_gizmo_tris_front =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("gizmo_tris_front"),
                layout: Some(&pl_opaque),
                vertex: wgpu::VertexState {
                    module: &shader_collab_lines,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout_collab_lines()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_collab_lines,
                    entry_point: Some("fs_gizmo_front"),
                    targets: preview_targets,
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState { constant: -4, slope_scale: -1.0, clamp: 0.0 },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });
        let pipeline_gizmo_tris_occluded =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("gizmo_tris_occluded"),
                layout: Some(&pl_opaque),
                vertex: wgpu::VertexState {
                    module: &shader_collab_lines,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout_collab_lines()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_collab_lines,
                    entry_point: Some("fs_gizmo_occluded"),
                    targets: preview_targets,
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Greater,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState { constant: 1, slope_scale: 1.0, clamp: 0.0 },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        let pipeline_collab_frustum_occluded =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("collab_frustum_occluded"),
                layout: Some(&pl_opaque),
                vertex: wgpu::VertexState {
                    module: &shader_collab_frustum,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout_collab_frustum()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_collab_frustum,
                    entry_point: Some("fs_frustum_occluded"),
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
        let pipeline_collab_frustum_front =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("collab_frustum_front"),
                layout: Some(&pl_opaque),
                vertex: wgpu::VertexState {
                    module: &shader_collab_frustum,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout_collab_frustum()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_collab_frustum,
                    entry_point: Some("fs_frustum_front"),
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
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState {
                        constant: -2,
                        slope_scale: -0.5,
                        clamp: 0.0,
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        let pipeline_collab_frustum_tri_occluded =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("collab_frustum_tri_occluded"),
                layout: Some(&pl_opaque),
                vertex: wgpu::VertexState {
                    module: &shader_collab_frustum,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout_collab_frustum()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_collab_frustum,
                    entry_point: Some("fs_frustum_occluded"),
                    targets: preview_targets,
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
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
        let pipeline_collab_frustum_tri_front =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("collab_frustum_tri_front"),
                layout: Some(&pl_opaque),
                vertex: wgpu::VertexState {
                    module: &shader_collab_frustum,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout_collab_frustum()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_collab_frustum,
                    entry_point: Some("fs_frustum_front"),
                    targets: preview_targets,
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState {
                        constant: -2,
                        slope_scale: -0.5,
                        clamp: 0.0,
                    },
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

        let pipeline_start_screen_bg = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("start_screen_bg"),
            layout: Some(&pl_opaque),
            vertex: wgpu::VertexState {
                module: &shader_start_screen_bg,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_start_screen_bg,
                entry_point: Some("fs_start_screen_mrt"),
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

        let pipeline_oit_accum = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("oit_accum"),
            layout: Some(&pl_trans),
            vertex: wgpu::VertexState {
                module: &shader_scene,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_scene,
                entry_point: Some("fs_oit_accum"),
                targets: &[
                    // Target 0: accumulation (additive blend)
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // Target 1: revealage (multiplicative: dst * (1 - src))
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R16Float,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Zero,
                                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Zero,
                                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None, // Glass visible from both sides
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                // Small bias pulls transparent faces toward camera to prevent
                // z-fighting where water faces are coplanar with opaque depth.
                bias: wgpu::DepthBiasState {
                    constant: -2,
                    slope_scale: -1.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let shader_oit_composite = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("oit_composite"),
            source: wgpu::ShaderSource::Wgsl(gpu::oit_composite::WGSL.into()),
        });
        let oit_composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("oit_composite"),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pl_oit_composite = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_oit_composite"),
            bind_group_layouts: &[&oit_composite_layout],
            push_constant_ranges: &[],
        });
        let pipeline_oit_composite = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("oit_composite"),
            layout: Some(&pl_oit_composite),
            vertex: wgpu::VertexState {
                module: &shader_oit_composite,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_oit_composite,
                entry_point: Some("fs_oit_composite"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: vf,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
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
                bias: wgpu::DepthBiasState {
                        constant: 2,
                        slope_scale: 2.0,
                        clamp: 0.0,
                    },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let pipeline_bloom_extract = fullscreen_pipeline(
            &device,
            &pl_blur,
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
        let shader_blit = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(gpu::post_blit::WGSL.into()),
        });
        let pipeline_blit = fullscreen_pipeline(
            &device,
            &pl_bloom,
            &shader_blit,
            "fs_blit",
            &[Some(wgpu::ColorTargetState {
                format: vf,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            None,
        );
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        // Weighted additive blit: uses post_blur_layout (texture + sampler + uniform) so the
        // fragment shader can multiply each sample by a scalar weight before the additive blend.
        let pipeline_blit_weighted_add = fullscreen_pipeline(
            &device,
            &pl_blur,
            &shader_blit,
            "fs_blit_weighted",
            &[Some(wgpu::ColorTargetState {
                format: vf,
                blend: Some(additive),
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
        let pipeline_meter = fullscreen_pipeline(
            &device,
            &pl_bloom,
            &shader_meter,
            "fs_meter",
            &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::R32Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            None,
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
            contents: bytemuck::bytes_of(&BloomExtractUniform { exposure_ev: 0.0, _pad: [0.0; 3] }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Constant weight written once: each bloom pyramid level contributes 0.75× the next finer
        // level, giving a geometric falloff instead of a flat equal-weight accumulation.
        let post_blit_weight_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blit_weight_u"),
            contents: bytemuck::bytes_of(&BloomExtractUniform { exposure_ev: 0.75, _pad: [0.0; 3] }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut post_composite_opts: PostCompositeOpts = bytemuck::Zeroable::zeroed();
        post_composite_opts.tone_mode = 0;
        post_composite_opts.exposure_ev = default_lit.exposure_ev.clamp(-5.0, 5.0);
        let post_composite_opts_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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

        let (shadow_texture, shadow_view) =
            create_shadow_tex(&device, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
        let scene_bounds = MeshBounds {
            min: Vec3::splat(-10.0),
            max: Vec3::splat(10.0),
        };

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
        let (bind_blit_down, bind_blit_up, bind_blit_up_weighted, bind_blit_final, bind_blur_pyr_h, bind_blur_pyr_v) =
            build_bloom_pyramid_bind_groups(
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

        let bind_oit_composite = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("oit_composite"),
            layout: &oit_composite_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&oit_accum_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&oit_revealage_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&hdr_opaque_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&sampler_linear) },
            ],
        });

        // ── Progressive raytracer setup ──────────────────────────────────────
        let rt_scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rt_scene"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let rt_accum_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rt_accum"),
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

        let rt_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rt_uniform"),
            contents: bytemuck::bytes_of(&RtUniform { frame_seed: 0, sample_n: 0, fast_preview: 0, _pad1: 0 }),
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
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&rt_preview_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler_linear) },
            ],
        });

        let rt_scene_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rt_scene_bg"),
            layout: &rt_scene_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: global_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: brick_buffer.as_entire_binding() },
            ],
        });
        // rt_accum_bgs[i] reads from the side that is NOT written this frame:
        //   flip=false → write to [0], so bg[0] reads from [1]
        //   flip=true  → write to [1], so bg[1] reads from [0]
        let rt_accum_bgs: Vec<wgpu::BindGroup> = vec![
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rt_accum_bg0"),
                layout: &rt_accum_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&rt_accum_view1) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler_linear) },
                    wgpu::BindGroupEntry { binding: 2, resource: rt_uniform_buf.as_entire_binding() },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rt_accum_bg1"),
                layout: &rt_accum_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&rt_accum_view0) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler_linear) },
                    wgpu::BindGroupEntry { binding: 2, resource: rt_uniform_buf.as_entire_binding() },
                ],
            }),
        ];

        let rt_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rt_pl"),
            bind_group_layouts: &[&rt_scene_layout, &rt_accum_layout],
            push_constant_ranges: &[],
        });
        let shader_rt = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ray_trace"),
            source: wgpu::ShaderSource::Wgsl(gpu::ray_trace::WGSL.into()),
        });
        let pipeline_raytrace = fullscreen_pipeline(
            &device,
            &rt_pl_layout,
            &shader_rt,
            "fs_trace",
            &[Some(wgpu::ColorTargetState {
                format: vf,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            None,
        );

        let rt_accum_textures = [rt_accum_tex0, rt_accum_tex1];
        let rt_accum_views    = [rt_accum_view0, rt_accum_view1];

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

        Ok(Self {
            surface,
            device,
            queue,
            config,
            format,
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
            pipeline_collab_lines_occluded,
            pipeline_collab_lines_front,
            pipeline_collab_frustum_occluded,
            pipeline_collab_frustum_front,
            pipeline_collab_frustum_tri_occluded,
            pipeline_collab_frustum_tri_front,
            pipeline_grid_border_lines,
            pipeline_gizmo_lines_front,
            pipeline_gizmo_lines_occluded,
            pipeline_gizmo_tris_front,
            pipeline_gizmo_tris_occluded,
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
            pipeline_meter,
            meter_texture,
            meter_view,
            meter_staging,
            ssr_opts_buf,
            ssr_opts,
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
            exposure_user_ev: default_lit.exposure_ev.clamp(-5.0, 5.0),
            auto_exposure_enabled: default_lit.auto_exposure,
            auto_exposure_smoothed: 0.0,
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
            collab_line_vertex_buffer: None,
            collab_line_vertex_count: 0,
            collab_frustum_vertex_buffer: None,
            collab_frustum_vertex_count: 0,
            collab_frustum_tri_vertex_buffer: None,
            collab_frustum_tri_vertex_count: 0,
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

            // ── Glyphon (initialized below) ──
            glyphon_font_system: glyphon_font_system,
            glyphon_swash_cache: glyphon_swash_cache,
            glyphon_cache: glyphon_cache,
            glyphon_atlas: glyphon_atlas,
            glyphon_text_renderer: glyphon_text_renderer,
            glyphon_viewport: glyphon_viewport,
            peer_label_data: Vec::new(),
            ping_label_data: None,
        })
    }

    pub fn opaque_index_count(&self) -> u32 {
        if self.opaque_chunked {
            self.opaque_chunks.values().map(|c| c.opaque_index_count + c.transparent_index_count).sum()
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

        let (oit_accum_texture, oit_accum_view, oit_revealage_texture, oit_revealage_view) =
            create_oit_textures(&self.device, viewport_width, viewport_height);
        self.oit_accum_texture = oit_accum_texture;
        self.oit_accum_view = oit_accum_view;
        self.oit_revealage_texture = oit_revealage_texture;
        self.oit_revealage_view = oit_revealage_view;

        // Reallocate raytracer accumulation textures for new viewport size.
        let (rt_a_tex, rt_a_view) = create_rt_accum_tex(&self.device, viewport_width, viewport_height);
        let (rt_b_tex, rt_b_view) = create_rt_accum_tex(&self.device, viewport_width, viewport_height);
        self.rt_accum_textures = [rt_a_tex, rt_b_tex];
        self.rt_accum_views    = [rt_a_view, rt_b_view];
        let (rt_preview_tex, rt_preview_view) =
            create_rt_accum_tex(&self.device, (viewport_width / 2).max(1), (viewport_height / 2).max(1));
        self.rt_preview_tex  = rt_preview_tex;
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

    /// `mode`: 0 neutral … 5 reinhard (see `post_composite.wgsl` / Voxelle web tone mapping ids).
    /// Drop the spatial mesh cache so the next remesh rebuilds all chunks from scratch.
    /// Call this whenever a baking parameter (e.g. emission lighting) changes.
    pub fn invalidate_spatial_mesh_cache(&mut self) {
        self.spatial_mesh_cache = None;
    }

    pub fn set_tone_mapping_mode(&mut self, mode: u32) {
        let mode = mode.min(5);
        self.post_composite_opts.tone_mode = mode;
        self.queue.write_buffer(
            &self.post_composite_opts_buf,
            0,
            bytemuck::bytes_of(&self.post_composite_opts),
        );
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
        self.queue.write_buffer(
            &self.ssr_opts_buf,
            0,
            bytemuck::bytes_of(&self.ssr_opts),
        );
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

    /// Read 1×1 meter from the previous submit; updates smoothed EV for the next frame.
    fn read_meter_luminance_and_update_auto_exposure(&mut self) {
        let slice = self.meter_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let Ok(map_ok) = rx.recv() else {
            return;
        };
        if map_ok.is_err() {
            return;
        }
        let lum = {
            let view = slice.get_mapped_range();
            let arr: [u8; 4] = view[0..4].try_into().unwrap_or([0; 4]);
            let v = f32::from_le_bytes(arr);
            drop(view);
            self.meter_staging.unmap();
            v
        };
        let target = 0.18_f32;
        let ev_inst = (target / lum.max(1e-5)).log2();
        // Faster adaptation so toggling auto is visibly responsive (still stable).
        self.auto_exposure_smoothed = self.auto_exposure_smoothed * 0.82 + ev_inst * 0.18;
    }

    pub fn apply_lighting_settings(&mut self, s: &crate::voxelle::LightingSettings) {
        self.light_ambient = s.ambient_intensity.max(0.0);
        self.light_sun = s.sunlight_intensity.max(0.0);
        self.shadows_enabled = s.enable_shadows;
        self.sky_enabled = s.enable_sky;
        self.light_dir = light_dir_from_azimuth_elevation_deg(s.light_angle_deg, s.light_elevation_deg);
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
                _pad1: 0,
            };
            self.queue.write_buffer(&self.rt_uniform_buf, 0, bytemuck::bytes_of(&rt_u));

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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

    /// Reset progressive accumulation (call on camera move, scene change, lighting change).
    pub fn reset_rt_accumulation(&mut self) {
        self.rt_sample_n = 0;
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
        self.bind_oit_composite = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("oit_composite"),
            layout: &self.oit_composite_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.oit_accum_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.oit_revealage_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.hdr_opaque_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler_linear) },
            ],
        });

        let (bind_blit_down, bind_blit_up, bind_blit_up_weighted, bind_blit_final, bind_blur_pyr_h, bind_blur_pyr_v) =
            build_bloom_pyramid_bind_groups(
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
                wgpu::BindGroupEntry { binding: 0, resource: self.global_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.brick_buffer.as_entire_binding() },
            ],
        });
        // bg[0]: write→accum[0], read from accum[1]
        // bg[1]: write→accum[1], read from accum[0]
        self.rt_accum_bgs = vec![
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rt_accum_bg0"),
                layout: &self.rt_accum_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.rt_accum_views[1]) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler_linear) },
                    wgpu::BindGroupEntry { binding: 2, resource: self.rt_uniform_buf.as_entire_binding() },
                ],
            }),
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rt_accum_bg1"),
                layout: &self.rt_accum_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.rt_accum_views[0]) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler_linear) },
                    wgpu::BindGroupEntry { binding: 2, resource: self.rt_uniform_buf.as_entire_binding() },
                ],
            }),
        ];
        self.rt_preview_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rt_preview_bg"),
            layout: &self.post_bloom_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.rt_preview_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler_linear) },
            ],
        });
    }

    fn interleaved_from_mesh(mesh: &MeshBuffers) -> Vec<f32> {
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

    fn opaque_draw_from_mesh(&self, mesh: &MeshBuffers, opaque_split: u32) -> OpaqueChunkDraw {
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
        }
    }

    /// If existing chunk buffers are large enough, overwrite with [`queue::write_buffer`]; else allocate new.
    fn upload_or_replace_chunk_mesh(&mut self, key: ChunkKey, mesh: &MeshBuffers, opaque_split: u32) {
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
    pub(crate) fn enqueue_chunk_uploads(
        &mut self,
        inbox: &mut VecDeque<(ChunkKey, MeshBuffers)>,
    ) {
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
                self.opaque_chunks.insert(key, self.opaque_draw_from_mesh(&mesh, split));
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
            crate::emit_work_progress(
                app,
                0.38,
                "Indexing voxels for mesh…",
            );
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
                crate::emit_work_progress(
                    app,
                    0.4,
                    "Indexing voxels for mesh…",
                );
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

        let spatial_cache = self
            .spatial_mesh_cache
            .take()
            .expect("spatial_mesh_cache");
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
                let buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
                let (mo, md, brick_ref): (IVec3, (u32, u32, u32), &wgpu::Buffer) =
                    match &halo_pack {
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
                        match Self::mesh_greedy_dispatch(
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
                            Ok((v_tot, i_tot)) => {
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
                            Err(_) => {}
                        }
                    }
                } else {
                    greedy_gpu_ms += t_pack.elapsed().as_secs_f64() * 1000.0;
                }
            }

            if !used_gpu {
                let t_cpu = Instant::now();
                let mut mesh = greedy_mesh::mesh_buffers_for_chunk_key(
                    &cache.buckets,
                    &cache.occupancy,
                    *key,
                );
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

    /// Run [`gpu::mesh_greedy`] compute: fills [`MeshGreedyPool`] scratch; read back vertex/index counts.
    /// Does not copy to draw buffers — see [`Self::upload_or_replace_chunk_mesh_from_gpu_scratch`] or full rebuild path.
    ///
    /// For the scene voxel brick buffer, call sites use [`Self::mesh_greedy_dispatch`] so `brick_storage` can be `&self.brick_buffer` with disjoint field borrows.
    pub fn run_mesh_greedy_compute_with_brick(
        &mut self,
        headers: &[greedy_mesh::GpuSliceHeader],
        bits: &[u32],
        mesh_brick_origin: IVec3,
        mesh_brick_dims: (u32, u32, u32),
        brick_storage: &wgpu::Buffer,
    ) -> Result<(u32, u32), String> {
        Self::mesh_greedy_dispatch(
            &self.device,
            &self.queue,
            &mut self.mesh_greedy_pool,
            &mut self.mesh_greedy_bind_layout,
            &mut self.mesh_greedy_pipeline,
            &mut self.mesh_greedy_pl_version,
            brick_storage,
            headers,
            bits,
            mesh_brick_origin,
            mesh_brick_dims,
        )
    }

    /// Same as [`Self::run_mesh_greedy_compute_with_brick`], but allows `brick_storage == self.brick_buffer` via disjoint `&mut self.*` borrows at the call site.
    fn mesh_greedy_dispatch(
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
    fn upload_or_replace_chunk_mesh_from_gpu_scratch(&mut self, key: ChunkKey, v_total: u32, i_total: u32) {
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
            PreparedGreedyRebuild::AllHidden { .. } => {
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
        poll_device_yielding_until_queue_empty(&self.device);

        self.opaque_chunked = false;
        self.opaque_chunks.clear();
        self.spatial_mesh_cache = None;
        self.vertex_buffer = Some(vb_final);
        self.index_buffer = Some(ib_final);
        self.index_count = i_total;
        // GPU greedy path doesn't partition indices yet — treat all as opaque.
        self.opaque_index_split = i_total;
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
    fn interleaved_from_prototype(proto: &greedy_mesh::PreviewPrototype) -> Vec<f32> {
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
        if verts.is_empty() || verts.len() % 6 != 0 {
            self.selection_overlay_line_vertex_buffer = None;
            self.selection_overlay_line_vertex_count = 0;
            return;
        }
        let n_floats = verts.len();
        let vertex_count = (n_floats / 6) as u32;
        let nbytes = (n_floats * std::mem::size_of::<f32>()) as u64;
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

    fn upload_buf(device: &wgpu::Device, queue: &wgpu::Queue, buf: &mut Option<wgpu::Buffer>, label: &str, data: &[f32]) -> u32 {
        if data.is_empty() || data.len() % 6 != 0 {
            *buf = None;
            return 0;
        }
        let nbytes = (data.len() * std::mem::size_of::<f32>()) as u64;
        if let Some(ref b) = buf {
            if b.size() == nbytes {
                queue.write_buffer(b, 0, bytemuck::cast_slice(data));
                return (data.len() / 6) as u32;
            }
        }
        *buf = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        }));
        (data.len() / 6) as u32
    }

    pub fn upload_gizmo_lines(&mut self, verts: &[f32]) {
        self.gizmo_line_vertex_count =
            Self::upload_buf(&self.device, &self.queue, &mut self.gizmo_line_vertex_buffer, "gizmo_lines_vtx", verts);
    }

    pub fn upload_gizmo_tris(&mut self, verts: &[f32]) {
        self.gizmo_tri_vertex_count =
            Self::upload_buf(&self.device, &self.queue, &mut self.gizmo_tri_vertex_buffer, "gizmo_tris_vtx", verts);
    }

    pub fn upload_grid_border_lines(&mut self, verts: &[f32], indices: &[u32]) {
        if verts.is_empty() || verts.len() % 6 != 0 || indices.is_empty() {
            self.grid_border_line_vertex_buffer = None;
            self.grid_border_line_index_buffer = None;
            self.grid_border_line_index_count = 0;
            return;
        }
        // Vertex buffer
        let vbytes = (verts.len() * std::mem::size_of::<f32>()) as u64;
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
        let ibytes = (indices.len() * std::mem::size_of::<u32>()) as u64;
        if let Some(ref buf) = self.grid_border_line_index_buffer {
            if buf.size() == ibytes {
                self.queue.write_buffer(buf, 0, bytemuck::cast_slice(indices));
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

    /// Replace the set of peer labels to render as GPU text this frame.
    pub fn upload_peer_labels(&mut self, labels: Vec<GpuPeerLabel>) {
        self.peer_label_data = labels;
    }

    pub fn clear_peer_labels(&mut self) {
        self.peer_label_data.clear();
    }

    pub fn upload_ping_label(&mut self, label: GpuPeerLabel) {
        self.ping_label_data = Some(label);
    }

    pub fn clear_ping_label(&mut self) {
        self.ping_label_data = None;
    }

    /// Frustum wireframe: each vertex is `[x,y,z, r,g,b, a]` (7 floats); line-list pairs.
    pub fn upload_collab_frustum_lines(&mut self, verts: &[f32]) {
        if verts.is_empty() || verts.len() % 7 != 0 {
            self.collab_frustum_vertex_buffer = None;
            self.collab_frustum_vertex_count = 0;
            return;
        }
        let n_floats = verts.len();
        let vertex_count = (n_floats / 7) as u32;
        let nbytes = (n_floats * std::mem::size_of::<f32>()) as u64;
        if let Some(ref buf) = self.collab_frustum_vertex_buffer {
            if buf.size() == nbytes {
                self.queue.write_buffer(buf, 0, bytemuck::cast_slice(verts));
                self.collab_frustum_vertex_count = vertex_count;
                return;
            }
        }
        self.collab_frustum_vertex_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("collab_frustum_vtx"),
                contents: bytemuck::cast_slice(verts),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            },
        ));
        self.collab_frustum_vertex_count = vertex_count;
    }

    pub fn clear_collab_frustum_lines(&mut self) {
        self.collab_frustum_vertex_buffer = None;
        self.collab_frustum_vertex_count = 0;
        self.collab_frustum_tri_vertex_buffer = None;
        self.collab_frustum_tri_vertex_count = 0;
    }

    /// Frustum side faces: each vertex is `[x,y,z, r,g,b, a]` (7 floats); triangle-list.
    pub fn upload_collab_frustum_tris(&mut self, verts: &[f32]) {
        if verts.is_empty() || verts.len() % 7 != 0 {
            self.collab_frustum_tri_vertex_buffer = None;
            self.collab_frustum_tri_vertex_count = 0;
            return;
        }
        let n_floats = verts.len();
        let vertex_count = (n_floats / 7) as u32;
        let nbytes = (n_floats * std::mem::size_of::<f32>()) as u64;
        if let Some(ref buf) = self.collab_frustum_tri_vertex_buffer {
            if buf.size() == nbytes {
                self.queue.write_buffer(buf, 0, bytemuck::cast_slice(verts));
                self.collab_frustum_tri_vertex_count = vertex_count;
                return;
            }
        }
        self.collab_frustum_tri_vertex_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("collab_frustum_tri_vtx"),
                contents: bytemuck::cast_slice(verts),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            },
        ));
        self.collab_frustum_tri_vertex_count = vertex_count;
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
                        self.opaque_chunks.insert(key, self.opaque_draw_from_mesh(&mesh, split));
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
        let lvp = light_view_proj(&self.scene_bounds, self.light_dir, camera, w, h);
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

    /// Draw only the opaque index range.
    fn draw_indexed_mesh(&self, pass: &mut wgpu::RenderPass<'_>) {
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
            if self.opaque_index_split > 0 {
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.opaque_index_split, 0, 0..1);
            }
        }
    }

    /// Draw only the transparent index range for the OIT accumulation pass.
    fn draw_indexed_oit_mesh(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.opaque_chunked {
            for ch in self.opaque_chunks.values() {
                if ch.transparent_index_count == 0 {
                    continue;
                }
                pass.set_vertex_buffer(0, ch.vertex_buffer.slice(..));
                pass.set_index_buffer(ch.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(
                    ch.opaque_index_count..ch.opaque_index_count + ch.transparent_index_count,
                    0,
                    0..1,
                );
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
    fn draw_indexed_mesh_all(&self, pass: &mut wgpu::RenderPass<'_>) {
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

    fn draw_indexed_preview(&self, pass: &mut wgpu::RenderPass<'_>) {
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
                pass.draw_indexed(0..self.preview_solid_proto_idx_count, 0, 0..self.preview_solid_instance_count);
                pass.set_pipeline(&self.pipeline_preview_inst_front);
                pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
                pass.draw_indexed(0..self.preview_solid_proto_idx_count, 0, 0..self.preview_solid_instance_count);
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
                pass.draw_indexed(0..self.preview_wire_proto_idx_count, 0, 0..self.preview_wire_instance_count);
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
    }

    fn draw_indexed_selection_overlay_solid(&self, pass: &mut wgpu::RenderPass<'_>) {
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

    fn draw_selection_overlay_lines(&self, pass: &mut wgpu::RenderPass<'_>) {
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

    fn draw_grid_border_lines(&self, pass: &mut wgpu::RenderPass<'_>) {
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

    fn draw_gizmo(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
        if let Some(ref vb) = self.gizmo_line_vertex_buffer {
            if self.gizmo_line_vertex_count >= 3 {
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_pipeline(&self.pipeline_gizmo_lines_occluded);
                pass.draw(0..self.gizmo_line_vertex_count, 0..1);
                pass.set_pipeline(&self.pipeline_gizmo_lines_front);
                pass.draw(0..self.gizmo_line_vertex_count, 0..1);
            }
        }
        if let Some(ref tb) = self.gizmo_tri_vertex_buffer {
            if self.gizmo_tri_vertex_count >= 3 {
                pass.set_vertex_buffer(0, tb.slice(..));
                pass.set_pipeline(&self.pipeline_gizmo_tris_occluded);
                pass.draw(0..self.gizmo_tri_vertex_count, 0..1);
                pass.set_pipeline(&self.pipeline_gizmo_tris_front);
                pass.draw(0..self.gizmo_tri_vertex_count, 0..1);
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
                pass.set_pipeline(&self.pipeline_preview_front_wire);
                pass.draw_indexed(0..self.ping_wire_index_count, 0, 0..1);
            }
        }
    }

    fn draw_collab_frustum_lines(&self, pass: &mut wgpu::RenderPass<'_>) {
        // Draw filled side faces first (triangles behind wireframe)
        if let Some(ref tb) = self.collab_frustum_tri_vertex_buffer {
            if self.collab_frustum_tri_vertex_count >= 3 {
                pass.set_vertex_buffer(0, tb.slice(..));
                pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
                pass.set_pipeline(&self.pipeline_collab_frustum_tri_occluded);
                pass.draw(0..self.collab_frustum_tri_vertex_count, 0..1);
                pass.set_pipeline(&self.pipeline_collab_frustum_tri_front);
                pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
                pass.draw(0..self.collab_frustum_tri_vertex_count, 0..1);
            }
        }
        // Draw wireframe edges on top
        let Some(ref vb) = self.collab_frustum_vertex_buffer else {
            return;
        };
        if self.collab_frustum_vertex_count < 2 {
            return;
        }
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
        pass.set_pipeline(&self.pipeline_collab_frustum_occluded);
        pass.draw(0..self.collab_frustum_vertex_count, 0..1);
        pass.set_pipeline(&self.pipeline_collab_frustum_front);
        pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
        pass.draw(0..self.collab_frustum_vertex_count, 0..1);
    }

    fn draw_ping_wave_lines(&self, pass: &mut wgpu::RenderPass<'_>) {
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

    pub fn render(&mut self) -> Result<(), String> {
        let frame = self
            .surface
            .get_current_texture()
            .map_err(|e| e.to_string())?;
        let tex_size = frame.texture.size();
        // Keep CPU-side surface size in sync with the actual swapchain (configure can differ slightly).
        self.surface_size = (tex_size.width.max(1), tex_size.height.max(1));
        let swap_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

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
                _pad1: 0,
            };
            self.queue.write_buffer(&self.rt_uniform_buf, 0, bytemuck::bytes_of(&rt_u));

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
                self.draw_collab_frustum_lines(&mut pass);
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
            self.draw_collab_frustum_lines(&mut pass);
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
        // Snapshot depth for SSR reads in the trans pass (avoids DEPTH_STENCIL_WRITE conflict).
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
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 1.0, g: 0.0, b: 0.0, a: 0.0 }),
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
        let blur_h_u = PostBlurUniform { blur_dir: [1.0, 0.0, 1.0, 0.0] };
        let blur_v_u = PostBlurUniform { blur_dir: [0.0, 1.0, 1.0, 0.0] };
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
        // Sync exposure into the bloom extract uniform so the threshold scales with EV.
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
                let mut buffer =
                    GlyphonBuffer::new(&mut self.glyphon_font_system, Metrics::new(font_size, line_height));
                buffer.set_size(&mut self.glyphon_font_system, Some(400.0), Some(line_height));

                let r = ((label.color_rgb >> 16) & 0xff) as u8;
                let g = ((label.color_rgb >> 8) & 0xff) as u8;
                let b = (label.color_rgb & 0xff) as u8;

                buffer.set_text(
                    &mut self.glyphon_font_system,
                    &label.name,
                    Attrs::new().family(Family::SansSerif).color(GlyphonColor::rgb(r, g, b)),
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

        {
            let _clear_swap = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear_swap"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &swap_view,
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

        self.queue.submit(std::iter::once(encoder.finish()));
        if self.auto_exposure_enabled {
            self.read_meter_luminance_and_update_auto_exposure();
        }
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
