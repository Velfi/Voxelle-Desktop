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

@group(0) @binding(2)
var shadow_map: texture_depth_2d;

@group(0) @binding(3)
var shadow_cmp: sampler_comparison;

const HEMI_SKY: vec3<f32> = vec3<f32>(0.722, 0.831, 0.910);
const HEMI_GROUND: vec3<f32> = vec3<f32>(0.290, 0.333, 0.408);
/// Slightly lifts dark baked AO creases after vertex interpolation (`vertex_ao`).
const VERTEX_AO_GAMMA: f32 = 0.9;

@group(1) @binding(0)
var hdr_bg: texture_2d<f32>;

@group(1) @binding(1)
var samp_linear: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) mat_kind: f32,
    /// Baked Minecraft-style corner AO; multiplies ambient hemisphere only.
    @location(4) vertex_ao: f32,
}

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) mat_kind: f32,
    @location(4) uv: vec2<f32>,
    @location(5) vertex_ao: f32,
}

fn unpack_mat(packed: u32) -> u32 {
    return (packed >> 24u) & 7u;
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

/// Parity: `render_constants` (`SHADOW_DEPTH_BIAS_*`, `SHADOW_NORMAL_BIAS`).
const SHADOW_DEPTH_BIAS_BASE: f32 = 0.0015;
const SHADOW_DEPTH_BIAS_SLOPE: f32 = 0.003;
const SHADOW_NORMAL_BIAS_WORLD: f32 = 0.012;

fn shadow_visibility(world: vec3<f32>, n: vec3<f32>) -> f32 {
    let l = normalize(g.light_dir.xyz);
    let nn = normalize(n);
    let ndl = max(dot(nn, l), 0.0);
    let biased_world = world + nn * (SHADOW_NORMAL_BIAS_WORLD * ndl);
    let lp = g.light_view_proj * vec4<f32>(biased_world, 1.0);
    let ndc = lp.xyz / max(lp.w, 1e-6);
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { return 1.0; }
    let depth_bias = SHADOW_DEPTH_BIAS_BASE + SHADOW_DEPTH_BIAS_SLOPE * (1.0 - ndl);
    let sh = textureSampleCompare(shadow_map, shadow_cmp, uv, ndc.z - depth_bias);
    return select(1.0, sh, g.light_params.z > 0.5);
}

fn schlick_f0(n: vec3<f32>, v: vec3<f32>, f0: vec3<f32>) -> vec3<f32> {
    let h = max(dot(n, v), 0.0);
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - h, 5.0);
}

fn transmission_shade(
    base: vec3<f32>,
    world: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    is_water: bool,
    bg: vec3<f32>,
    vertex_ao: f32,
) -> vec3<f32> {
    let slab = march_slab_thickness(world, n);
    let transmission = select(0.96, 0.998, is_water);
    let thickness = select(0.65, 0.9, is_water);
    let att_dist = select(2.5, 32.0, is_water);
    let ior = select(1.5, 1.333, is_water);
    let absorb = exp(-(thickness * slab * 1.2) / max(att_dist, 1e-4));
    let net_t = transmission * absorb;
    let f0 = vec3<f32>(pow((ior - 1.0) / (ior + 1.0), 2.0));
    let fresnel = schlick_f0(n, v, f0);
    let l = normalize(g.light_dir.xyz);
    let ndl = max(dot(n, l), 0.0);
    let hemi = mix(HEMI_GROUND, HEMI_SKY, n.y * 0.5 + 0.5);
    let amb = g.light_params.x;
    let sun = g.light_params.y;
    let sc = g.sun_color.xyz;
    let ao_h = pow(max(vertex_ao, 0.001), VERTEX_AO_GAMMA);
    let lit = base * (hemi * 0.28 * ao_h * amb + 0.55 * ndl * shadow_visibility(world, n) * sun * sc);
    let refr = bg * base * net_t * (vec3<f32>(1.0) - fresnel);
    let spec = pow(max(dot(normalize(l + v), n), 0.0), 48.0) * 0.2 * sun;
    return mix(refr, lit + sc * spec, 0.15);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOut {
    var o: VertexOut;
    o.clip_pos = g.view_proj * vec4<f32>(in.position, 1.0);
    o.world_pos = in.position;
    o.normal = in.normal;
    o.color = in.color;
    o.mat_kind = in.mat_kind;
    o.vertex_ao = in.vertex_ao;
    let h = o.clip_pos.w;
    o.uv = vec2<f32>(0.5, 0.5) + vec2<f32>(0.5, -0.5) * (o.clip_pos.xy / vec2<f32>(h, h));
    return o;
}

// ---------------------------------------------------------------------------
// GPU-instanced preview: prototype vertex (position + normal) + per-instance
// model matrix, color, mat_kind.
// ---------------------------------------------------------------------------

struct PreviewProtoVertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

struct PreviewInstanceVertex {
    @location(2) model_c0: vec4<f32>,
    @location(3) model_c1: vec4<f32>,
    @location(4) model_c2: vec4<f32>,
    @location(5) model_c3: vec4<f32>,
    @location(6) inst_color: vec3<f32>,
    @location(7) inst_mat_kind: f32,
}

@vertex
fn vs_preview_instanced(proto: PreviewProtoVertex, inst: PreviewInstanceVertex) -> VertexOut {
    let model = mat4x4<f32>(inst.model_c0, inst.model_c1, inst.model_c2, inst.model_c3);
    let world = model * vec4<f32>(proto.position, 1.0);
    var o: VertexOut;
    o.clip_pos = g.view_proj * world;
    o.world_pos = world.xyz;
    let normal_mat = mat3x3<f32>(model[0].xyz, model[1].xyz, model[2].xyz);
    o.normal = normalize(normal_mat * proto.normal);
    o.color = inst.inst_color;
    o.mat_kind = inst.inst_mat_kind;
    o.vertex_ao = 1.0;
    let h = o.clip_pos.w;
    o.uv = vec2<f32>(0.5, 0.5) + vec2<f32>(0.5, -0.5) * (o.clip_pos.xy / vec2<f32>(h, h));
    return o;
}

struct OpaqueOut {
    @location(0) color: vec4<f32>,
    @location(1) gbuf_n: vec4<f32>,
}

@fragment
fn fs_opaque_mrt(in: VertexOut) -> OpaqueOut {
    if (in.mat_kind > 1.6) {
        discard;
    }
    var out: OpaqueOut;
    let n = normalize(in.normal);
    let l = normalize(g.light_dir.xyz);
    let v = normalize(g.cam_pos.xyz - in.world_pos);
    let base = in.color;
    let h = normalize(l + v);
    let ndl = max(dot(n, l), 0.0);
    let sh = shadow_visibility(in.world_pos, n);
    let hemi = mix(HEMI_GROUND, HEMI_SKY, n.y * 0.5 + 0.5);
    let amb = g.light_params.x;
    let sun = g.light_params.y;
    let sc = g.sun_color.xyz;
    let ao_h = pow(max(in.vertex_ao, 0.001), VERTEX_AO_GAMMA);
    let ndh = max(dot(n, h), 0.0);
    let ndv = max(dot(n, v), 0.0);

    let is_metal = in.mat_kind > 0.25 && in.mat_kind < 0.75;
    let is_glow  = in.mat_kind > 0.75 && in.mat_kind < 1.25;

    var rgb: vec3<f32>;
    var glow_mask = 0.0;

    if (is_metal) {
        // Metallic PBR: tinted specular from base color, sharp highlight, strong Fresnel.
        let f0 = base * 0.96 + vec3<f32>(0.04);
        let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - ndv, 5.0);
        let spec_power = pow(ndh, 96.0);
        let spec = fresnel * spec_power * 1.8 * sh * sun;
        // Metallic diffuse is minimal (energy absorbed); ambient reflections via hemi.
        let ambient_refl = base * hemi * 0.72 * ao_h * amb;
        let direct = base * 0.15 * ndl * sh * sun * sc;
        rgb = ambient_refl + direct + spec * sc;
    } else if (is_glow) {
        // Self-illuminated: emissive color + subtle ambient for shape readability.
        let emissive = base * 4.0;
        let shape = base * (hemi * 0.12 * ao_h * amb + 0.18 * ndl * sh * sun * sc);
        let spec_glow = pow(ndh, 24.0) * 0.06 * sh * sun;
        rgb = emissive + shape + sc * spec_glow;
        glow_mask = 1.0;
    } else {
        // Plastic / rubber.
        let spec_blinn = pow(ndh, 32.0) * 0.12 * sh * sun;
        rgb = base * (hemi * 0.30 * ao_h * amb + 0.78 * ndl * sh * sun * sc) + sc * spec_blinn;
    }

    out.color = vec4<f32>(rgb, glow_mask);
    let vn = (g.inv_view * vec4<f32>(n, 0.0)).xyz;
    let nn = normalize(vn) * 0.5 + 0.5;
    out.gbuf_n = vec4<f32>(nn, 1.0);
    return out;
}

/// Hover/stroke preview voxels (see product spec):
/// - Slightly transparent in-front pass; darker when depth-occluded or when overlapping an occupied cell.
/// - Vertex color comes from Rust: palette (add/paint/sculpt), fixed red (remove), fixed blue (select).
/// - Wireframe uses the same tint; drawn with an unbiased depth pipeline so it can be occluded by scene mesh.
/// `mat_kind` > 1.5 marks edge outline geometry (`preview_cube_wireframe_mesh`); use **dark** vertex colors.
const PREVIEW_ALPHA_FRONT: f32 = 0.82;
const PREVIEW_ALPHA_OCCLUDED: f32 = 0.70;
/// Edge lines: slightly more opaque than fill so the cage reads; still blends as glass.
const PREVIEW_ALPHA_EDGE: f32 = 0.90;
const PREVIEW_ALPHA_EDGE_OCCLUDED: f32 = 0.82;
/// Darkening for x-ray / overlap — **neutral** (hue-preserving), not a fixed blue tint.
const PREVIEW_OCCLUDED_DIM: f32 = 0.46;
const PREVIEW_OVERLAP_DIM: f32 = 0.52;
/// Rust tags Paint/Remove empty-footprint preview with `mat_kind` in these bands (see `lib.rs`).
fn is_preview_ghost_fill_paint_remove(mat: f32) -> bool {
    return mat > 1.02 && mat < 1.14;
}
fn is_preview_ghost_wire_paint_remove(mat: f32) -> bool {
    return mat > 1.55 && mat < 1.85;
}
/// 75% transparent (25% opaque) vs default preview glass.
const PREVIEW_GHOST_PR_ALPHA_MUL: f32 = 0.25;

fn preview_tonemap(rgb: vec3<f32>) -> vec3<f32> {
    return rgb / (rgb * vec3<f32>(0.42) + vec3<f32>(1.0));
}

/// Dark outline: lift slightly for HDR without turning into mint fill.
fn preview_edge_rgb(in_color: vec3<f32>) -> vec3<f32> {
    let boosted = saturate(in_color * 5.0);
    return preview_tonemap(boosted);
}

/// In-front preview (`LessEqual` depth): unlit, semi-transparent. When the preview cell overlaps an
/// existing voxel (`occ`), darken the tint (same hue as vertex color).
@fragment
fn fs_preview_front_mrt(in: VertexOut) -> OpaqueOut {
    var out: OpaqueOut;
    let is_edge = in.mat_kind > 1.5;
    let cell = vec3<i32>(floor(in.world_pos + vec3<f32>(0.5)));
    let occ = is_occupied(brick_fetch(cell));
    if (is_edge) {
        let rgb = preview_edge_rgb(in.color * select(1.0, 0.80, occ));
        var a = select(PREVIEW_ALPHA_EDGE, PREVIEW_ALPHA_EDGE_OCCLUDED, occ);
        if (is_preview_ghost_wire_paint_remove(in.mat_kind)) {
            a = a * PREVIEW_GHOST_PR_ALPHA_MUL;
        }
        out.color = vec4<f32>(rgb, a);
    } else if (occ) {
        var a = PREVIEW_ALPHA_OCCLUDED;
        if (is_preview_ghost_fill_paint_remove(in.mat_kind)) {
            a = a * PREVIEW_GHOST_PR_ALPHA_MUL;
        }
        let rgb = preview_tonemap(in.color * PREVIEW_OVERLAP_DIM);
        out.color = vec4<f32>(rgb, a);
    } else {
        var a = PREVIEW_ALPHA_FRONT;
        if (is_preview_ghost_fill_paint_remove(in.mat_kind)) {
            a = a * PREVIEW_GHOST_PR_ALPHA_MUL;
        }
        let rgb = preview_tonemap(saturate(in.color * 1.06));
        out.color = vec4<f32>(rgb, a);
    }
    out.gbuf_n = vec4<f32>(0.0);
    return out;
}

/// X-ray pass: only where preview is behind scene geometry (`depthCompare` Greater).
@fragment
fn fs_preview_occluded_mrt(in: VertexOut) -> OpaqueOut {
    var out: OpaqueOut;
    let is_edge = in.mat_kind > 1.5;
    if (is_edge) {
        let rgb = preview_edge_rgb(in.color * 0.80);
        var a = PREVIEW_ALPHA_EDGE_OCCLUDED;
        if (is_preview_ghost_wire_paint_remove(in.mat_kind)) {
            a = a * PREVIEW_GHOST_PR_ALPHA_MUL;
        }
        out.color = vec4<f32>(rgb, a);
    } else {
        let rgb = preview_tonemap(in.color * PREVIEW_OCCLUDED_DIM);
        var a = PREVIEW_ALPHA_OCCLUDED;
        if (is_preview_ghost_fill_paint_remove(in.mat_kind)) {
            a = a * PREVIEW_GHOST_PR_ALPHA_MUL;
        }
        out.color = vec4<f32>(rgb, a);
    }
    out.gbuf_n = vec4<f32>(0.0);
    return out;
}

@fragment
fn fs_trans(in: VertexOut) -> @location(0) vec4<f32> {
    if (in.mat_kind < 1.6) {
        discard;
    }
    let n = normalize(in.normal);
    let v = normalize(g.cam_pos.xyz - in.world_pos);
    let is_water = in.mat_kind > 2.2;
    let bg = textureSample(hdr_bg, samp_linear, in.uv).rgb;
    let rgb = transmission_shade(in.color, in.world_pos, n, v, is_water, bg, in.vertex_ao);
    let a = 0.94;
    return vec4<f32>(rgb * a, a);
}
