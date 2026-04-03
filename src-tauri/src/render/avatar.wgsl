// World-space avatar shader for collab peer voxel models.
// Vertex layout mirrors scene.wgsl: pos3 + normal3 + color3 + mat_kind + ao + emission3 = 56 bytes.
// Outputs HDR color to @location(0) and packed world-space normal to @location(1) (same
// encoding as scene.wgsl's gbuf_n) so SSR does not sample stale scene geometry at avatar pixels.

struct AvatarUniforms {
    mvp: mat4x4<f32>,
    /// World-space direction toward the light source.
    light_dir: vec4<f32>,
    /// Per-peer color tint (xyz). Use [1,1,1] for named avatars, peer accent color for the default glow dot.
    color_tint: vec4<f32>,
    ambient: f32,
    sun: f32,
    _pad: vec2<f32>,
    /// Rotation matrix (upper-left 3×3 of the model matrix) to transform mesh-local normals
    /// into world space.  Each column is padded to vec4 alignment per WGSL std140 rules.
    normal_mat: mat3x3<f32>,
}

@group(0) @binding(0)
var<uniform> u: AvatarUniforms;

struct VertexInput {
    @location(0) position:      vec3<f32>,
    @location(1) normal:        vec3<f32>,
    @location(2) color:         vec3<f32>,
    @location(3) mat_kind:      f32,
    @location(4) vertex_ao:     f32,
    @location(5) emission_tint: vec3<f32>,
}

struct VertexOut {
    @builtin(position) clip_pos:      vec4<f32>,
    @location(0)       color:         vec3<f32>,
    @location(1)       world_normal:  vec3<f32>,
    @location(2)       vertex_ao:     f32,
    @location(3)       emission_tint: vec3<f32>,
}

@vertex
fn vs_avatar(v: VertexInput) -> VertexOut {
    var o: VertexOut;
    o.clip_pos      = u.mvp * vec4<f32>(v.position, 1.0);
    o.color         = v.color * u.color_tint.xyz;
    o.world_normal  = normalize(u.normal_mat * v.normal);
    o.vertex_ao     = v.vertex_ao;
    o.emission_tint = v.emission_tint * u.color_tint.xyz;
    return o;
}

struct AvatarOut {
    @location(0) color:  vec4<f32>,
    /// Packed world-space normal + metalness=0, matching scene.wgsl gbuf_n encoding.
    @location(1) gbuf_n: vec4<f32>,
}

@fragment
fn fs_avatar(i: VertexOut) -> AvatarOut {
    let n = normalize(i.world_normal);
    let l = normalize(u.light_dir.xyz);
    let n_dot_l = max(dot(n, l), 0.0);
    let ao = pow(i.vertex_ao, 0.9);
    // Hemisphere ambient: sky above, ground below.
    let sky_col    = vec3<f32>(0.722, 0.831, 0.910);
    let ground_col = vec3<f32>(0.290, 0.333, 0.408);
    let hemi = mix(ground_col, sky_col, n.y * 0.5 + 0.5);
    let diffuse = i.color * (hemi * u.ambient * ao + u.sun * n_dot_l);
    let rgb = clamp(diffuse + i.emission_tint, vec3<f32>(0.0), vec3<f32>(1.5));
    var out: AvatarOut;
    out.color  = vec4<f32>(rgb, 1.0);
    out.gbuf_n = vec4<f32>(n * 0.5 + 0.5, 0.0);
    return out;
}
