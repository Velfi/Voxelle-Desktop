// Mascot model shader for start-screen floating views.
// Vertex layout mirrors scene.wgsl: pos3 + normal3 + color3 + mat_kind + ao + emission3 = 56 bytes.
// No shadow map, no MRT — single opaque render target.

struct MascotUniforms {
    mvp: mat4x4<f32>,
    /// World-space direction toward the light source.
    light_dir: vec4<f32>,
    ambient: f32,
    sun: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0)
var<uniform> u: MascotUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) mat_kind: f32,
    @location(4) vertex_ao: f32,
    @location(5) emission_tint: vec3<f32>,
}

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) vertex_ao: f32,
    @location(3) emission_tint: vec3<f32>,
}

@vertex
fn vs_mascot(v: VertexInput) -> VertexOut {
    var o: VertexOut;
    o.clip_pos = u.mvp * vec4<f32>(v.position, 1.0);
    o.color = v.color;
    o.normal = v.normal;
    o.vertex_ao = v.vertex_ao;
    o.emission_tint = v.emission_tint;
    return o;
}

@fragment
fn fs_mascot(i: VertexOut) -> @location(0) vec4<f32> {
    let n = normalize(i.normal);
    let l = normalize(u.light_dir.xyz);
    let n_dot_l = max(dot(n, l), 0.0);
    let ao = pow(i.vertex_ao, 0.9);
    // Hemisphere ambient: sky above, ground below.
    let sky_col    = vec3<f32>(0.722, 0.831, 0.910);
    let ground_col = vec3<f32>(0.290, 0.333, 0.408);
    let hemi = mix(ground_col, sky_col, n.y * 0.5 + 0.5);
    let diffuse = i.color * (hemi * u.ambient * ao + u.sun * n_dot_l);
    let rgb = clamp(diffuse + i.emission_tint, vec3<f32>(0.0), vec3<f32>(1.5));
    return vec4<f32>(rgb, 1.0);
}
