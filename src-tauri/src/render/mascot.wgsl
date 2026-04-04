// Mascot model shader for start-screen floating views.
// Vertex layout mirrors scene.wgsl: pos3 + normal3 + color3 + mat_kind + ao + emission3 = 56 bytes.
// No shadow map, no MRT — single opaque render target.

struct MascotUniforms {
    mvp: mat4x4<f32>,
    /// World-space direction toward the light source.
    light_dir: vec4<f32>,
    ambient: f32,
    sun: f32,
    explode_radius: f32,
    explode_strength: f32,
    mouse_ndc: vec2<f32>,
    mouse_active: f32,
    time_seconds: f32,
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
    /// Center of the voxel this face belongs to (all faces of the same voxel share this value).
    @location(6) voxel_center: vec3<f32>,
}

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) vertex_ao: f32,
    @location(3) emission_tint: vec3<f32>,
    @location(4) explode_t: f32,
}

// Simple position-based hash for per-vertex scatter variation.
fn hash_pos(p: vec3<f32>) -> f32 {
    let q = fract(p * vec3<f32>(127.1, 311.7, 74.7));
    let d = dot(q, vec3<f32>(269.5, 183.3, 246.1));
    return fract(sin(d) * 43758.5453);
}

@vertex
fn vs_mascot(v: VertexInput) -> VertexOut {
    var o: VertexOut;

    // 1) Compute undisplaced clip position for screen-space distance check.
    let base_clip = u.mvp * vec4<f32>(v.position, 1.0);
    let ndc = base_clip.xy / base_clip.w;

    // 2) Screen-space distance to mouse cursor (NDC space).
    let d = length(ndc - u.mouse_ndc);

    // 3) Smooth falloff: 1.0 at mouse, 0.0 at radius edge.
    let t = smoothstep(u.explode_radius, 0.0, d) * u.mouse_active;

    // 4) Displacement: hash-based scatter derived from voxel center so that
    //    every face of the same voxel gets the same seed, direction, and
    //    magnitude → whole voxels move as rigid cubes.
    let seed = hash_pos(v.voxel_center);
    let scatter_dir = normalize(vec3<f32>(
        sin(seed * 6.283),
        cos(seed * 4.17),
        sin(seed * 2.91 + 1.0),
    ));
    // Gentle wobble while hovering so fragments feel alive.
    let wobble = 1.0 + 0.08 * sin(u.time_seconds * 4.0 + seed * 6.283);
    let displacement = scatter_dir * t * u.explode_strength * (0.5 + seed * 0.5) * wobble;

    // 5) Apply displacement in model space and recompute clip position.
    let displaced = v.position + displacement;
    o.clip_pos = u.mvp * vec4<f32>(displaced, 1.0);

    o.color = v.color;
    o.normal = v.normal;
    o.vertex_ao = v.vertex_ao;
    o.emission_tint = v.emission_tint;
    o.explode_t = t;

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
