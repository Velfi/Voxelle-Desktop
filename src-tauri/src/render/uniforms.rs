//! `#[repr(C)]` bytemuck GPU uniform structs shared across render passes.

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GlobalState {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view: [[f32; 4]; 4],
    pub inv_proj: [[f32; 4]; 4],
    pub light_view_proj: [[f32; 4]; 4],
    pub light_dir: [f32; 4],
    pub cam_pos: [f32; 4],
    pub brick_origin: [f32; 4],
    pub brick_dims: [f32; 4],
    pub screen: [f32; 4],
    pub params: [f32; 4],
    /// x: ambient scale, y: sun scale, z: shadows on (0/1), w: sky gradient on (0/1)
    pub light_params: [f32; 4],
    pub sun_color: [f32; 4],
    pub bg_color: [f32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PostBlurUniform {
    pub blur_dir: [f32; 4],
}

/// Matches `post_bloom_extract.wgsl` `BloomExtractU`.
#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BloomExtractUniform {
    pub exposure_ev: f32,
    pub _pad: [f32; 3],
}

/// Matches `scene.wgsl` `SsrOpts` (inline SSR in `fs_trans`).
#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SsrOpts {
    /// Scales the final confidence → controls overall reflection intensity.
    pub strength: f32,
    /// Number of world-space ray-march steps (clamped 8..64 in the shader).
    pub max_steps: f32,
    /// Maximum distance between ray hit and scene surface before ignoring hit.
    pub thickness: f32,
    /// 1.0 = SSR enabled, 0.0 = disabled (shader early-outs).
    pub enabled: f32,
}

/// Matches `post_composite.wgsl` `PostCompositeOpts` and Voxelle web tone mapping ids (neutral…reinhard).
/// Layout: 14 vec4 rows = 224 bytes.
#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PostCompositeOpts {
    // --- Row 0 ---
    pub tone_mode: u32,
    /// 1 = premultiplied alpha from scene energy (cold-start logo over webview).
    pub transparent_bg: f32,
    /// Exposure in EV stops (`rgb *= 2^ev` before tone mapping).
    pub exposure_ev: f32,
    /// Monotonic seconds since viewer creation (for animated effects).
    pub time_seconds: f32,
    // --- Row 1: vignette + grain basics ---
    pub vignette_strength: f32,
    pub grain_enabled: f32,
    pub grain_strength: f32,
    pub grain_animated: f32,
    // --- Row 2: grain continued ---
    pub grain_speed: f32,
    /// 1.0 = colorful (per-channel noise), 0.0 = monochrome.
    pub grain_colorful: f32,
    pub _pad2a: f32,
    pub _pad2b: f32,
    // --- Row 3: atmosphere controls ---
    pub atm_enabled: f32,
    pub atm_thickness: f32,
    pub atm_density: f32,
    /// 0 = plane, 1 = aerial.
    pub atm_spatial_mode: f32,
    // --- Row 4: atmosphere color + mode ---
    pub atm_color_r: f32,
    pub atm_color_g: f32,
    pub atm_color_b: f32,
    /// 0 = slab, 1 = positiveSide.
    pub atm_mode: f32,
    // --- Row 5: atmosphere plane ---
    pub atm_plane_nx: f32,
    pub atm_plane_ny: f32,
    pub atm_plane_nz: f32,
    pub atm_plane_c: f32,
    // --- Row 6: atmosphere height + drift ---
    pub atm_height_bias: f32,
    pub atm_height_falloff: f32,
    pub atm_drift_enabled: f32,
    pub atm_drift_amount: f32,
    // --- Row 7: drift continued ---
    pub atm_drift_scale: f32,
    pub atm_drift_speed: f32,
    pub _pad7a: f32,
    pub _pad7b: f32,
    // --- Row 8: distance tint controls ---
    pub dt_enabled: f32,
    pub dt_near_dist: f32,
    pub dt_far_dist: f32,
    pub dt_strength: f32,
    // --- Row 9-11: distance tint colors ---
    pub dt_near_r: f32,
    pub dt_near_g: f32,
    pub dt_near_b: f32,
    pub _pad9: f32,
    pub dt_mid_r: f32,
    pub dt_mid_g: f32,
    pub dt_mid_b: f32,
    pub _pad10: f32,
    pub dt_far_r: f32,
    pub dt_far_g: f32,
    pub dt_far_b: f32,
    pub _pad11: f32,
    // --- Row 12: sun shafts ---
    pub ss_enabled: f32,
    pub ss_strength: f32,
    pub ss_decay: f32,
    pub ss_density: f32,
    // --- Row 13: sun shafts continued ---
    pub ss_weight: f32,
    pub ss_samples: f32,
    pub ss_sun_uv_x: f32,
    pub ss_sun_uv_y: f32,
    // --- Row 14: bloom + soft sunshafts ---
    pub bloom_strength: f32,
    pub ss_soft: f32,
    pub _pad14b: f32,
    pub _pad14c: f32,
}

/// Matches `mascot.wgsl` `MascotUniforms`.
#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MascotUniforms {
    pub mvp: [[f32; 4]; 4],
    pub light_dir: [f32; 4],
    pub ambient: f32,
    pub sun: f32,
    pub explode_radius: f32,
    pub explode_strength: f32,
    pub mouse_ndc: [f32; 2],
    pub mouse_active: f32,
    pub time_seconds: f32,
}

/// Matches `avatar.wgsl` `AvatarUniforms`.
#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct AvatarUniforms {
    pub mvp: [[f32; 4]; 4],
    pub light_dir: [f32; 4],
    /// Per-peer color tint (xyz); w unused. [1,1,1,0] = preserve mesh colors.
    pub color_tint: [f32; 4],
    pub ambient: f32,
    pub sun: f32,
    pub _pad: [f32; 2],
    /// Rotation part of the model matrix, column-major with std140 padding (each column is
    /// stored as [x, y, z, 0.0]).  Transforms mesh-local normals into world space.
    pub normal_mat: [[f32; 4]; 3],
}

/// Matches `speech_bubble.wgsl` `BubbleUniforms`. Must stay 16-byte aligned.
#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SpeechBubbleUniforms {
    /// x, y (top-left), w, h — swapchain pixels.
    pub rect: [f32; 4],
    /// Tail tip in swapchain pixels.
    pub tail_tip: [f32; 2],
    /// Horizontal shake offset.
    pub shake_x: f32,
    pub corner_r: f32,
    pub bg_color: [f32; 4],
    pub border_color: [f32; 4],
    pub border_w: f32,
    pub _pad: [f32; 3],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuMeshParams {
    pub max_vertices: u32,
    pub max_indices: u32,
    pub slice_count: u32,
    pub _pad0: u32,
    pub brick_ox: i32,
    pub brick_oy: i32,
    pub brick_oz: i32,
    pub _pad1: i32,
    pub brick_dx: u32,
    pub brick_dy: u32,
    pub brick_dz: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct RtUniform {
    pub frame_seed: u32,
    pub sample_n: u32,
    /// 1 when the camera moved this frame: use cheap shading (1 shadow ray, no bounces).
    pub fast_preview: u32,
    pub _pad1: u32,
}
