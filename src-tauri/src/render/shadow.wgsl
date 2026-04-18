struct GlobalState {
    view_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    light_dir: vec4<f32>,
    cam_pos: vec4<f32>,
    brick_origin: vec4<f32>,
    brick_dims: vec4<f32>,
    screen: vec4<f32>,
    params: vec4<f32>,
    light_params: vec4<f32>,
    sun_color: vec4<f32>,
    bg_color: vec4<f32>,
}

@group(0) @binding(0)
var<storage, read> g: GlobalState;

@group(0) @binding(1)
var<storage, read> brick_cells: array<u32>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) mat_kind: f32,
    @location(4) vertex_ao: f32,
    /// Must match main scene mesh layout (`vertex_layout`); unused for shadow depth.
    @location(5) emission_tint: vec3<f32>,
}

struct ShadowVOut {
    @builtin(position) clip: vec4<f32>,
}

@vertex
fn vs_shadow(in: VertexInput) -> ShadowVOut {
    var o: ShadowVOut;
    // Transparent materials (glass ≥ 2.0, water ≥ 2.5) skip the depth map entirely.
    // Their light attenuation is applied in the opaque fragment shader via a
    // voxel march toward the sun (see `glass_tint_to_sun` in scene.wgsl).
    if (in.mat_kind > 1.95) {
        o.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return o;
    }
    o.clip = g.light_view_proj * vec4<f32>(in.position, 1.0);
    return o;
}
