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

/// Opaque scene depth — read-only in fs_trans for SSR ray marching.
@group(1) @binding(2)
var depth_for_ssr: texture_depth_2d;

@group(1) @binding(3)
var samp_depth: sampler;

struct SsrOpts { strength: f32, max_steps: f32, thickness: f32, enabled: f32, }
@group(1) @binding(4)
var<uniform> ssr_opts: SsrOpts;

/// Inline screen-space reflection. Returns vec4(rgb, confidence).
/// Uses true glass-surface world_pos and normal from vertex interpolation —
/// no depth-buffer ambiguity since glass fragments are discarded in the opaque pass.
fn compute_ssr(world_pos: vec3<f32>, n: vec3<f32>, v: vec3<f32>) -> vec4<f32> {
    if (ssr_opts.enabled < 0.5) { return vec4<f32>(0.0); }
    // Incident direction (camera → surface) reflected off the surface normal.
    let incident = -v;
    let refl = normalize(reflect(incident, n));
    // If the reflection goes behind the camera immediately we can't see it —
    // but let the clip.w guard catch that; don't reject based on dot(refl,n)
    // since normals may be inconsistent across back/front faces.
    let max_dist = 48.0;
    let steps = clamp(ssr_opts.max_steps, 8.0, 64.0);
    let step_dist = max_dist / steps;
    for (var i = 1; i <= i32(steps); i++) {
        let t = f32(i) * step_dist;
        let p = world_pos + refl * t;
        let clip = g.view_proj * vec4<f32>(p, 1.0);
        if (clip.w <= 0.0) { break; }
        let ndc = clip.xyz / clip.w;
        // Off-screen: continue (don't break) — ray may re-enter the frustum.
        if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) { continue; }
        let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
        let scene_depth = textureSampleLevel(depth_for_ssr, samp_depth, uv, 0.0);
        if (ndc.z > scene_depth) {
            // Accept hit; use a generous thickness so reconstruction precision
            // doesn't silently swallow valid hits.
            let sv4 = g.inv_proj * vec4<f32>(ndc.xy, scene_depth, 1.0);
            let sw4 = g.inv_view * vec4<f32>(sv4.xyz / sv4.w, 1.0);
            let behind_dist = length(p - sw4.xyz / sw4.w);
            if (behind_dist < ssr_opts.thickness) {
                let ef = 1.0 - smoothstep(0.8, 1.0, max(abs(ndc.x), abs(ndc.y)));
                let df = 1.0 - t / max_dist;
                let col = textureSampleLevel(hdr_bg, samp_linear, uv, 0.0).rgb;
                return vec4<f32>(col, ef * df * ssr_opts.strength);
            }
        }
    }
    // Sky fallback ─────────────────────────────────────────────────────────
    // When the reflection ray climbs above the scene (typical for water viewed
    // from above) it misses all opaque geometry.  Sample the background colour
    // texture at the projected reflection direction so the result changes with
    // camera movement and shows the actual sky / distant scene colour.
    // Threshold is raised (0.30) to avoid side-facing glass walls triggering
    // sky fallback and appearing as flat blue mirrors.
    if (refl.y > 0.30) {
        // Project the reflection direction as a very distant point.
        let sky_p = world_pos + refl * 900.0;
        let sky_clip = g.view_proj * vec4<f32>(sky_p, 1.0);
        var sky_col = g.bg_color.rgb;
        if (sky_clip.w > 0.0) {
            let sky_ndc = sky_clip.xyz / sky_clip.w;
            // Clamp to [−1,1] so off-frustum directions sample the screen edge.
            let sky_uv = clamp(sky_ndc.xy, vec2<f32>(-1.0), vec2<f32>(1.0))
                         * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
            sky_col = textureSampleLevel(hdr_bg, samp_linear, sky_uv, 0.0).rgb;
        }
        let sky_w = smoothstep(0.30, 0.8, refl.y) * ssr_opts.strength * 0.45;
        return vec4<f32>(sky_col, sky_w);
    }
    return vec4<f32>(0.0);
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) mat_kind: f32,
    /// Baked Minecraft-style corner AO; multiplies ambient hemisphere only.
    @location(4) vertex_ao: f32,
    /// Baked RGB irradiance from nearby glow voxels (point-light accumulation at mesh-gen time).
    @location(5) emission_tint: vec3<f32>,
}

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) mat_kind: f32,
    @location(4) uv: vec2<f32>,
    @location(5) vertex_ao: f32,
    @location(6) emission_tint: vec3<f32>,
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

/// World-space shadow bias (in voxel units). Converted to NDC using the light frustum depth range
/// extracted from `light_view_proj`, so bias is consistent regardless of scene size.
const SHADOW_BIAS_WORLD_BASE:  f32 = 0.04;   // flat surfaces facing the light
const SHADOW_BIAS_WORLD_SLOPE: f32 = 0.15;   // extra for surfaces at grazing angles
const SHADOW_NORMAL_OFFSET:    f32 = 0.08;   // world-space offset along surface normal
const SHADOW_PCF_SPREAD:       f32 = 1.5;    // texel radius for PCF soft shadows

/// Interleaved gradient noise — gives a unique-looking pseudo-random value per pixel
/// that is stable across frames.  Used to rotate the PCF kernel so the regular 3×3
/// grid doesn't produce visible striping artefacts.
fn ign(screen_xy: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(screen_xy, vec2<f32>(0.06711056, 0.00583715))));
}

fn shadow_visibility(world: vec3<f32>, n: vec3<f32>, screen_pos: vec2<f32>) -> f32 {
    let l = normalize(g.light_dir.xyz);
    let nn = normalize(n);
    let ndl = max(dot(nn, l), 0.0);
    // Push the sample point away from the surface along its normal to avoid self-shadowing.
    let offset_pos = world + nn * SHADOW_NORMAL_OFFSET;
    let lp = g.light_view_proj * vec4<f32>(offset_pos, 1.0);
    let ndc = lp.xyz / max(lp.w, 1e-6);
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { return 1.0; }
    // Extract NDC-per-world-unit from the Z row of light_view_proj (= 1/(far-near) for ortho).
    let z_grad = vec3<f32>(g.light_view_proj[0][2], g.light_view_proj[1][2], g.light_view_proj[2][2]);
    let ndc_per_world = length(z_grad);
    let world_bias = SHADOW_BIAS_WORLD_BASE + SHADOW_BIAS_WORLD_SLOPE * (1.0 - ndl);
    let cmp_depth = ndc.z - world_bias * ndc_per_world;
    // g.params.y: 1 = soft (3x3 PCF), 0 = hard (single tap).
    var sh: f32;
    if (g.params.y > 0.5) {
        let texel = SHADOW_PCF_SPREAD / vec2<f32>(textureDimensions(shadow_map));
        // Per-pixel rotation angle breaks up the regular grid pattern that causes striping.
        let angle = ign(screen_pos) * 6.2831853;
        let rot_c = cos(angle);
        let rot_s = sin(angle);
        var sum = 0.0;
        for (var y = -1i; y <= 1i; y++) {
            for (var x = -1i; x <= 1i; x++) {
                let off = vec2<f32>(f32(x), f32(y));
                let rotated = vec2<f32>(off.x * rot_c - off.y * rot_s,
                                        off.x * rot_s + off.y * rot_c);
                sum += textureSampleCompare(shadow_map, shadow_cmp, uv + rotated * texel, cmp_depth);
            }
        }
        sh = sum / 9.0;
    } else {
        sh = textureSampleCompare(shadow_map, shadow_cmp, uv, cmp_depth);
    }
    return select(1.0, sh, g.light_params.z > 0.5);
}

fn schlick_f0(n: vec3<f32>, v: vec3<f32>, f0: vec3<f32>) -> vec3<f32> {
    let h = max(dot(n, v), 0.0);
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - h, 5.0);
}

/// Returns `vec4(color, alpha)` where alpha is the physically-derived glass opacity.
/// Alpha is low for thin clear glass (most light passes through) and high for
/// thick / heavily absorbed glass.  This allows multi-layer transparency to
/// accumulate correctly via premultiplied alpha blending.
fn transmission_shade(
    base: vec3<f32>,
    world: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    is_water: bool,
    screen_pos: vec2<f32>,
    bg: vec3<f32>,
    vertex_ao: f32,
) -> vec4<f32> {
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
    let hemi = mix(HEMI_GROUND, HEMI_SKY, n.y * 0.5 + 0.5);
    let amb = g.light_params.x;
    let sun = g.light_params.y;
    let sc = g.sun_color.xyz;
    let ao_h = pow(max(vertex_ao, 0.001), VERTEX_AO_GAMMA);
    // Transmission: background tinted and absorbed through glass/water thickness.
    // Fresnel controls transmitted vs reflected energy split.
    let refr = bg * base * net_t * (vec3<f32>(1.0) - fresnel);
    // Specular reflection: Fresnel-weighted Blinn-Phong from sun.
    // Dielectrics have no diffuse scattering — only specular — so the highlight
    // scales with Fresnel: bright at grazing incidence, near-zero head-on.
    let h_spec = normalize(l + v);
    let spec = fresnel * pow(max(dot(h_spec, n), 0.0), 96.0) * shadow_visibility(world, n, screen_pos) * sun * sc;
    // Small hemisphere ambient keeps silhouettes readable in dark environments.
    let ambient = base * hemi * 0.06 * ao_h * amb;
    let rgb = refr + spec + ambient;
    // Physical alpha: 1 − transmittance.  Thin clear glass at normal incidence
    // is mostly transparent (alpha ~0.3); thick glass or grazing Fresnel pushes
    // alpha toward 1.  Minimum 0.12 keeps glass edges readable.
    let avg_fresnel = max(fresnel.x, max(fresnel.y, fresnel.z));
    let a = max(0.12, 1.0 - net_t * (1.0 - avg_fresnel));
    return vec4<f32>(rgb, a);
}

/// OIT variant: returns glass self-contribution (specular, ambient, absorption tint)
/// WITHOUT baking in the background.  The transmitted background is handled by the
/// revealage term in the OIT composite pass.
fn transmission_shade_oit(
    base: vec3<f32>,
    world: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    is_water: bool,
    screen_pos: vec2<f32>,
    vertex_ao: f32,
) -> vec4<f32> {
    let slab_raw = march_slab_thickness(world, n);
    let transmission = select(0.96, 0.998, is_water);
    let thickness = select(0.65, 0.9, is_water);
    let att_dist = select(2.5, 32.0, is_water);
    let ior = select(1.5, 1.333, is_water);
    // Full-depth absorption drives the voxel-color tint: deeper bodies show
    // more of their color.
    let full_absorb = exp(-(thickness * slab_raw * 1.2) / max(att_dist, 1e-4));
    let full_net_t = transmission * full_absorb;
    // Thin-slab absorption for alpha only: OIT compositing can dim the
    // background but cannot tint it, so thick absorption would replace the
    // scene behind with the material's own color instead of tinting through.
    let thin_absorb = exp(-(thickness * min(slab_raw, 1.0) * 1.2) / max(att_dist, 1e-4));
    let net_t = transmission * thin_absorb;
    let f0 = vec3<f32>(pow((ior - 1.0) / (ior + 1.0), 2.0));
    let fresnel = schlick_f0(n, v, f0);
    let l = normalize(g.light_dir.xyz);
    let hemi = mix(HEMI_GROUND, HEMI_SKY, n.y * 0.5 + 0.5);
    let amb = g.light_params.x;
    let sun = g.light_params.y;
    let sc = g.sun_color.xyz;
    let ao_h = pow(max(vertex_ao, 0.001), VERTEX_AO_GAMMA);
    // Self-tint from full depth so the voxel color is visible.
    let self_tint = base * (1.0 - full_net_t) * 0.5;
    // Specular reflection from sun (Fresnel-weighted Blinn-Phong).
    let h_spec = normalize(l + v);
    let spec = fresnel * pow(max(dot(h_spec, n), 0.0), 96.0) * shadow_visibility(world, n, screen_pos) * sun * sc;
    // Small hemisphere ambient for silhouette readability.
    let ambient = base * hemi * 0.06 * ao_h * amb;
    let rgb = self_tint + spec + ambient;
    // Physical alpha: 1 - transmittance.
    let avg_fresnel = max(fresnel.x, max(fresnel.y, fresnel.z));
    let a = max(0.12, 1.0 - net_t * (1.0 - avg_fresnel));
    return vec4<f32>(rgb, a);
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
    o.emission_tint = in.emission_tint;
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
    o.emission_tint = vec3<f32>(0.0);
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
    let sh = shadow_visibility(in.world_pos, n, in.clip_pos.xy);
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

    // Add baked emission irradiance from nearby glow voxels (non-emissive surfaces only).
    if (!is_glow) {
        rgb += base * in.emission_tint;
    }

    out.color = vec4<f32>(rgb, glow_mask);
    let nn = n * 0.5 + 0.5;
    let metalness = select(0.0, 1.0, is_metal);
    out.gbuf_n = vec4<f32>(nn, metalness);
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
    let v = normalize(g.cam_pos.xyz - in.world_pos);
    let n_raw = normalize(in.normal);
    // Flip back-facing normals toward the camera so the Fresnel term doesn't
    // drive single-thickness glass panes to full opacity.
    let n = select(-n_raw, n_raw, dot(n_raw, v) >= 0.0);
    let is_water = in.mat_kind > 2.2;
    // IOR-based refraction: offset the background sample UV by the angular
    // deviation of the refracted ray (Snell's law). Without this, glass and water
    // tint the background but don't visually displace it, which looks flat.
    // refract(I, N, eta): I = incident direction (-v), eta = 1/IOR (air→material).
    let ior_fs = select(1.5, 1.333, is_water);
    let refract_dir = refract(-v, n, 1.0 / ior_fs);
    // (refract_dir + v) is the tangential deviation from the straight-through ray.
    // Guard against total internal reflection (refract() returns zero vector).
    let tangent_shift = select(
        vec2<f32>(0.0),
        (refract_dir.xy + v.xy) * 0.08,
        dot(refract_dir, refract_dir) > 0.5
    );
    let refr_uv = clamp(in.uv + tangent_shift, vec2<f32>(0.001), vec2<f32>(0.999));
    let bg = textureSample(hdr_bg, samp_linear, refr_uv).rgb;
    let shade = transmission_shade(in.color, in.world_pos, n, v, is_water, in.clip_pos.xy, bg, in.vertex_ao);
    var rgb = shade.rgb;
    let a = shade.a;

    // ── Screen-space reflection blend ───────────────────────────────────────
    // Weight SSR by the Fresnel term at the glass/water surface: reflections
    // are physically strongest at grazing angles and nearly invisible head-on.
    // This prevents side-facing glass walls from appearing as opaque mirrors
    // and keeps reflections on horizontal water surfaces where they belong.
    let ssr_sample = compute_ssr(in.world_pos, n, v);
    let ndv_ssr = max(dot(n, v), 0.0);
    let fresnel_ssr = pow(1.0 - ndv_ssr, 4.0);
    rgb = mix(rgb, ssr_sample.rgb, saturate(ssr_sample.a * fresnel_ssr));
    // ────────────────────────────────────────────────────────────────────────

    return vec4<f32>(rgb * a, a);
}

struct OitOut {
    @location(0) accum: vec4<f32>,
    @location(1) revealage: vec4<f32>,
}

@fragment
fn fs_oit_accum(in: VertexOut) -> OitOut {
    if (in.mat_kind < 1.6) {
        discard;
    }
    let v = normalize(g.cam_pos.xyz - in.world_pos);
    let n_raw = normalize(in.normal);
    // Flip back-facing normals toward the camera so the Fresnel term doesn't
    // drive single-thickness glass panes to full opacity.
    let n = select(-n_raw, n_raw, dot(n_raw, v) >= 0.0);
    let is_water = in.mat_kind > 2.2;

    // Discard back-facing water: the bottom face of a water body is coplanar
    // with the opaque surface beneath it, causing z-fighting.  Glass keeps
    // both faces (cull_mode is None) but water's underside is redundant.
    if (is_water && dot(n, v) < 0.0) {
        discard;
    }

    let shade = transmission_shade_oit(in.color, in.world_pos, n, v, is_water, in.clip_pos.xy, in.vertex_ao);
    var rgb = shade.rgb;
    let alpha = shade.a;

    // Screen-space reflections (same as fs_trans).
    let ssr_sample = compute_ssr(in.world_pos, n, v);
    let ndv_ssr = max(dot(n, v), 0.0);
    let fresnel_ssr = pow(1.0 - ndv_ssr, 4.0);
    rgb = mix(rgb, ssr_sample.rgb, saturate(ssr_sample.a * fresnel_ssr));

    // WBOIT depth weight (McGuire & Bavoil 2013).
    // Clip-space z is in [0,1] with 0=near.  The weight biases toward nearer
    // fragments to approximate correct ordering.
    let z = in.clip_pos.z;
    let w = alpha * max(1e-2, min(3e3,
        10.0 / (1e-5 + pow(z / 5.0, 2.0) + pow(z / 200.0, 6.0))
    ));

    var out: OitOut;
    out.accum = vec4<f32>(rgb * alpha * w, alpha * w);
    out.revealage = vec4<f32>(alpha, 0.0, 0.0, 0.0);
    return out;
}
