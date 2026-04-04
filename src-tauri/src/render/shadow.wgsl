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

fn unpack_mat(packed: u32) -> u32 {
    return (packed >> 24u) & 0xFu;
}

fn brick_fetch(ix: vec3<i32>) -> u32 {
    let o = vec3<i32>(i32(g.brick_origin.x), i32(g.brick_origin.y), i32(g.brick_origin.z));
    let rel = ix - o;
    let dx = i32(g.brick_dims.x);
    let dy = i32(g.brick_dims.y);
    let dz = i32(g.brick_dims.z);
    if (rel.x < 0 || rel.y < 0 || rel.z < 0) { return 0u; }
    if (rel.x >= dx || rel.y >= dy || rel.z >= dz) { return 0u; }
    let idx = u32(rel.x) + u32(rel.y) * u32(dx) + u32(rel.z) * u32(dx) * u32(dy);
    return brick_cells[idx];
}

fn is_occupied(packed: u32) -> bool {
    return (packed & (1u << 31u)) != 0u;
}

fn march_slab_thickness(world: vec3<f32>, outward_normal: vec3<f32>) -> f32 {
    let inward = -normalize(outward_normal);
    var acc = 0.0;
    for (var i = 1; i < 48; i++) {
        let p = world + inward * f32(i);
        let ix = vec3<i32>(floor(p + vec3<f32>(0.5)));
        let cell = brick_fetch(ix);
        if (!is_occupied(cell)) { break; }
        let mat = unpack_mat(cell);
        if (mat != 3u && mat != 4u) { break; }
        acc += 1.0;
    }
    return max(acc, 1.0);
}

fn glass_shadow_push(world: vec3<f32>, n: vec3<f32>, mat_kind: f32) -> f32 {
    if (mat_kind < 1.95) { return 0.0; }
    let slab = march_slab_thickness(world, n);
    let abs_v = 0.16;
    let min_tv = 0.35;
    let d_slab = max(slab, 1.0);
    var raw_aov = 1.0;
    if (d_slab > 1.0) {
        raw_aov = clamp(max(min_tv, exp(-abs_v * (d_slab - 1.0))), 0.0, 1.0);
    }
    let vertex_ao = clamp(pow(raw_aov, 1.65) * 1.0, 0.0, 1.0);
    let net_t = clamp(0.96 * exp(-(0.65 * mix(1.5, 0.72, raw_aov)) / 2.5), 0.0, 1.0);
    return 0.02 * net_t * vertex_ao;
}

@vertex
fn vs_shadow(in: VertexInput) -> ShadowVOut {
    var o: ShadowVOut;
    var clip = g.light_view_proj * vec4<f32>(in.position, 1.0);
    let n = normalize(in.normal);
    let push = glass_shadow_push(in.position, n, in.mat_kind);
    let dz = 2.0 * push * clip.w;
    clip.z = clip.z + dz;
    let wlim = max(abs(clip.w), 1e-6);
    clip.z = clamp(clip.z, 1e-4 * wlim, wlim - 1e-4);
    o.clip = clip;
    return o;
}
