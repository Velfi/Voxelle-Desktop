/// Progressive voxel ray tracer.
///
/// Each frame casts one jittered primary ray per pixel.  Results are blended
/// into `accum_prev` via a running arithmetic mean so the image converges as
/// the camera is held still:
///
///   accum_new = mix(accum_prev, sample, 1.0 / f32(sample_n + 1))
///
/// sample_n == 0  →  discard accum_prev (first frame after a reset).
///
/// Bind-group 0  : GlobalState (global + brick storage)
/// Bind-group 1  : accum_prev texture + sampler + RtUniform
///
/// WGSL does not support forward declarations or recursion.  Bounces are
/// unrolled into two layers:
///   shade_secondary  — terminal; no further bounces.
///   shade_metal / shade_transmissive  — call shade_secondary.
///   shade (primary)  — calls shade_metal / shade_transmissive.

// ─────────────────────────────────────────────────────────────────────────────
// Bind groups
// ─────────────────────────────────────────────────────────────────────────────

struct GlobalState {
    view_proj:       mat4x4<f32>,
    inv_view:        mat4x4<f32>,
    inv_proj:        mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    light_dir:       vec4<f32>,
    cam_pos:         vec4<f32>,
    brick_origin:    vec4<f32>,
    brick_dims:      vec4<f32>,
    screen:          vec4<f32>,
    params:          vec4<f32>,
    light_params:    vec4<f32>,   // x=ambient, y=sun, z=shadows_on, w=sky_on
    sun_color:       vec4<f32>,
    bg_color:        vec4<f32>,
}

@group(0) @binding(0) var<storage, read> g:           GlobalState;
@group(0) @binding(1) var<storage, read> brick_cells: array<u32>;

struct RtUniform {
    frame_seed:   u32,  // different every frame for decorrelated jitter
    sample_n:     u32,  // accumulated samples so far (0 = first frame / reset)
    fast_preview: u32,  // 1 when camera moved this frame: 1 shadow ray, no bounces
    _pad1:        u32,
}

@group(1) @binding(0) var accum_prev: texture_2d<f32>;
@group(1) @binding(1) var samp:       sampler;
@group(1) @binding(2) var<uniform>   rt: RtUniform;

// ─────────────────────────────────────────────────────────────────────────────
// Voxel access  (bit layout: occupied=31, mat=26:24, B=23:16, G=15:8, R=7:0)
// ─────────────────────────────────────────────────────────────────────────────

fn brick_fetch(ix: vec3<i32>) -> u32 {
    let o   = vec3<i32>(i32(g.brick_origin.x), i32(g.brick_origin.y), i32(g.brick_origin.z));
    let rel = ix - o;
    let dx  = i32(g.brick_dims.x);
    let dy  = i32(g.brick_dims.y);
    let dz  = i32(g.brick_dims.z);
    if (rel.x < 0 || rel.y < 0 || rel.z < 0)          { return 0u; }
    if (rel.x >= dx || rel.y >= dy || rel.z >= dz)     { return 0u; }
    let idx = u32(rel.x) + u32(rel.y) * u32(dx) + u32(rel.z) * u32(dx) * u32(dy);
    return brick_cells[idx];
}

fn is_occupied(packed: u32) -> bool { return (packed >> 31u) != 0u; }
fn unpack_mat(packed: u32) -> u32   { return (packed >> 24u) & 7u; }

// Approximate sRGB → linear (gamma 2.2)
fn srgb_to_linear(c: f32) -> f32 { return c * c; }

fn unpack_rgb(packed: u32) -> vec3<f32> {
    let r = f32( packed        & 0xFFu) / 255.0;
    let gv= f32((packed >> 8u) & 0xFFu) / 255.0;
    let b = f32((packed >>16u) & 0xFFu) / 255.0;
    return vec3<f32>(srgb_to_linear(r), srgb_to_linear(gv), srgb_to_linear(b));
}

const MAT_PLASTIC: u32 = 0u;
const MAT_METAL:   u32 = 1u;
const MAT_RUBBER:  u32 = 2u;
const MAT_GLASS:   u32 = 3u;
const MAT_WATER:   u32 = 4u;
const MAT_GLOW:    u32 = 5u;

fn is_transmissive(mat: u32) -> bool { return mat == MAT_GLASS || mat == MAT_WATER; }

// ─────────────────────────────────────────────────────────────────────────────
// Random / hash
// ─────────────────────────────────────────────────────────────────────────────

fn hash_u(a: u32) -> u32 {
    var x = a;
    x ^= x >> 17u; x *= 0xbf324c81u;
    x ^= x >> 11u; x *= 0x68e31da4u;
    x ^= x >> 14u;
    return x;
}
fn rand_f(seed: u32) -> f32 { return f32(hash_u(seed)) / 4294967296.0; }

// Vogel-disk sample index i of n, with angular offset `phi` (radians).
const GOLDEN_ANGLE: f32 = 2.3999632297286535; // π(3−√5)
fn vogel_disk(i: u32, n: u32, phi: f32) -> vec2<f32> {
    let r     = sqrt(f32(i) / f32(max(n, 1u) - 1u + 1u));
    let theta = f32(i) * GOLDEN_ANGLE + phi;
    return vec2<f32>(r * cos(theta), r * sin(theta));
}

// ─────────────────────────────────────────────────────────────────────────────
// Sky (full atmosphere — ported from sky.wgsl)
// ─────────────────────────────────────────────────────────────────────────────

fn sun_air_mass(sun_y: f32) -> f32 { return 1.0 / max(sun_y, 0.032); }

fn extinction_rgb(air_mass: f32) -> vec3<f32> {
    return vec3<f32>(exp(-0.052 * air_mass), exp(-0.082 * air_mass), exp(-0.128 * air_mass));
}

fn rayleigh_phase(cos_theta: f32) -> f32 { return 0.65 * (1.0 + cos_theta * cos_theta); }

fn day_sky_blend(amb: f32, sun_lvl: f32) -> f32 {
    let k = clamp(amb + sun_lvl, 0.0, 8.0);
    return smoothstep(0.1, 0.82, pow(k, 1.42));
}

fn hash31(p: vec3<f32>) -> f32 {
    var q = fract(p * vec3<f32>(443.897, 441.423, 437.195));
    q += dot(q, q.yzx + 19.19);
    return fract((q.x + q.y) * q.z);
}

fn stars(dir: vec3<f32>) -> vec3<f32> {
    let hf = smoothstep(-0.06, 0.12, dir.y);
    if (hf < 0.001) { return vec3<f32>(0.0); }
    let p  = dir * 95.0;
    let i  = floor(p);
    let f  = fract(p) - 0.5;
    let h  = hash31(i);
    if (h < 0.972) { return vec3<f32>(0.0); }
    let off = hash31(i + vec3<f32>(31.0, 17.0, 43.0)) - 0.5;
    let dist = length(f - off * 0.82);
    let br   = hash31(i + vec3<f32>(11.0, 59.0, 23.0));
    let sp   = smoothstep(0.22, 0.0, dist) * (0.35 + br * 0.95);
    let tint = mix(vec3<f32>(0.75, 0.82, 1.0), vec3<f32>(1.0, 0.95, 0.88), br);
    return tint * sp * 3.8 * hf;
}

fn night_sky(dir: vec3<f32>, sun_dir: vec3<f32>, sc: vec3<f32>, amb: f32, sun_lvl: f32) -> vec3<f32> {
    let up  = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    let af  = 0.035 + amb * 0.55;
    let zen = vec3<f32>(0.006, 0.008, 0.022) * af;
    let hor = vec3<f32>(0.02,  0.024, 0.045) * af;
    var rgb = mix(hor, zen, pow(up, 0.65));
    let mu  = max(dot(dir, sun_dir), 0.0);
    rgb    += sc * pow(mu, 2800.0) * sun_lvl * 6.5;
    rgb    += sc * pow(mu, 12.0)   * sun_lvl * 0.045;
    return rgb;
}

fn day_sky(dir: vec3<f32>, sun_dir: vec3<f32>, sc: vec3<f32>) -> vec3<f32> {
    let sy          = clamp(sun_dir.y, 0.0, 1.0);
    let cos_theta   = clamp(dot(dir, sun_dir), -1.0, 1.0);
    let mu          = max(cos_theta, 0.0);
    let ext         = extinction_rgb(sun_air_mass(sy));
    let sunlight    = sc * ext;
    let vy          = dir.y;
    let horiz_path  = 1.0 / max(abs(vy) + 0.1, 0.06);
    let haze_w      = clamp(horiz_path * 0.038, 0.0, 1.6);
    let up          = clamp(vy * 0.5 + 0.5, 0.0, 1.0);
    let zen = mix(
        vec3<f32>(0.05, 0.07, 0.14),
        mix(vec3<f32>(0.18, 0.38, 0.82), vec3<f32>(0.35, 0.58, 0.96), sy),
        smoothstep(0.04, 0.92, sy)
    );
    let hor = mix(
        sunlight * vec3<f32>(0.85, 0.62, 0.42),
        mix(vec3<f32>(0.48, 0.62, 0.82), vec3<f32>(0.62, 0.82, 0.95), sy),
        smoothstep(0.08, 0.55, sy)
    );
    let base     = mix(hor, zen, pow(up, 0.52));
    let pr       = rayleigh_phase(cos_theta);
    let rayleigh = vec3<f32>(0.2, 0.42, 0.92) * sunlight * pr * (0.1 + 0.26 * sy);
    let haze_rgb = vec3<f32>(0.82, 0.76, 0.68) * sunlight * haze_w * (0.28 + 0.72 * (1.0 - sy));
    var rgb      = base * (0.42 + 0.58 * sy) + rayleigh + haze_rgb * 0.62;
    let mie_exp  = mix(5.5, 90.0, sy);
    rgb         += sunlight * pow(mu, mie_exp) * (0.28 + 0.55 * (1.0 - sy));
    let disk     = smoothstep(0.99925, 0.99982, cos_theta);
    rgb         += sunlight * disk * (3.2 + 2.8 * sy);
    return min(rgb, vec3<f32>(14.0));
}

fn sky_color(dir: vec3<f32>) -> vec3<f32> {
    let amb     = max(g.light_params.x, 0.0);
    let sun_lvl = max(g.light_params.y, 0.0);
    let sc      = max(g.sun_color.xyz, vec3<f32>(1e-4));
    let sun_dir = normalize(g.light_dir.xyz);
    // sky_on flag — if off return flat bg_color
    if (g.light_params.w < 0.5) { return g.bg_color.xyz; }
    let w     = day_sky_blend(amb, sun_lvl);
    let day   = day_sky(dir, sun_dir, sc) * clamp(0.12 + 0.88 * smoothstep(0.04, 1.35, amb * 0.48 + sun_lvl * 0.52), 0.08, 1.35);
    let night = night_sky(dir, sun_dir, sc, amb, sun_lvl);
    let sky0  = mix(night, day, w);
    let sw    = (1.0 - smoothstep(0.20, 0.25, amb)) * (1.0 - smoothstep(0.20, 0.25, sun_lvl));
    return sky0 + stars(dir) * sw;
}

// ─────────────────────────────────────────────────────────────────────────────
// DDA ray traversal
// ─────────────────────────────────────────────────────────────────────────────

struct DdaHit {
    hit:               bool,
    cell:              vec3<i32>,
    normal:            vec3<f32>,
    packed:            u32,
    t:                 f32,
    medium_dist_glass: f32,  // distance through MAT_GLASS voxels
    medium_dist_water: f32,  // distance through MAT_WATER voxels
}

/// Generic DDA.  When skip_trans=true the ray passes through glass/water,
/// accumulating their path lengths in medium_dist_glass / medium_dist_water
/// separately so each can be Beer-Lambert attenuated at its own rate.
/// The first opaque/glow voxel is returned.
/// When skip_trans=false the first voxel of any kind is returned.
/// max_steps caps the grid-cell loop independently of max_dist.
fn dda(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32, skip_trans: bool, max_steps: i32) -> DdaHit {
    var h: DdaHit;
    h.hit = false; h.medium_dist_glass = 0.0; h.medium_dist_water = 0.0;

    // Prevent NaN from 1/0 — nudge near-zero components
    let EPS = 1e-7;
    let dx = select(dir.x, select(-EPS, EPS, dir.x >= 0.0), abs(dir.x) < EPS);
    let dy = select(dir.y, select(-EPS, EPS, dir.y >= 0.0), abs(dir.y) < EPS);
    let dz = select(dir.z, select(-EPS, EPS, dir.z >= 0.0), abs(dir.z) < EPS);

    let step = vec3<i32>(i32(sign(dx)), i32(sign(dy)), i32(sign(dz)));
    let inv  = vec3<f32>(1.0 / dx, 1.0 / dy, 1.0 / dz);
    let tD   = abs(inv);

    var cell = vec3<i32>(i32(floor(origin.x)), i32(floor(origin.y)), i32(floor(origin.z)));

    var tMax = vec3<f32>(
        select((f32(cell.x)   - origin.x) * inv.x, (f32(cell.x+1) - origin.x) * inv.x, dx > 0.0),
        select((f32(cell.y)   - origin.y) * inv.y, (f32(cell.y+1) - origin.y) * inv.y, dy > 0.0),
        select((f32(cell.z)   - origin.z) * inv.z, (f32(cell.z+1) - origin.z) * inv.z, dz > 0.0),
    );

    var t      = 0.0;
    var normal = vec3<f32>(0.0, 1.0, 0.0);

    for (var i = 0; i < max_steps; i++) {
        if (t >= max_dist) { break; }

        let t_next = min(tMax.x, min(tMax.y, tMax.z));
        let packed = brick_fetch(cell);

        if (is_occupied(packed)) {
            let mat = unpack_mat(packed);
            if (is_transmissive(mat) && skip_trans) {
                let seg = min(t_next, max_dist) - t;
                if (mat == MAT_GLASS) { h.medium_dist_glass += seg; }
                if (mat == MAT_WATER) { h.medium_dist_water += seg; }
            } else {
                h.hit = true; h.cell = cell; h.normal = normal;
                h.packed = packed; h.t = t;
                return h;
            }
        }

        // Advance to next cell boundary
        if (tMax.x < tMax.y) {
            if (tMax.x < tMax.z) {
                t = tMax.x; tMax.x += tD.x; cell.x += step.x;
                normal = vec3<f32>(-f32(step.x), 0.0, 0.0);
            } else {
                t = tMax.z; tMax.z += tD.z; cell.z += step.z;
                normal = vec3<f32>(0.0, 0.0, -f32(step.z));
            }
        } else {
            if (tMax.y < tMax.z) {
                t = tMax.y; tMax.y += tD.y; cell.y += step.y;
                normal = vec3<f32>(0.0, -f32(step.y), 0.0);
            } else {
                t = tMax.z; tMax.z += tD.z; cell.z += step.z;
                normal = vec3<f32>(0.0, 0.0, -f32(step.z));
            }
        }
    }
    return h;
}

// ─────────────────────────────────────────────────────────────────────────────
// Lighting / shading
// ─────────────────────────────────────────────────────────────────────────────

const HEMI_SKY:    vec3<f32> = vec3<f32>(0.722, 0.831, 0.910);
const HEMI_GROUND: vec3<f32> = vec3<f32>(0.290, 0.333, 0.408);

/// Hemisphere ambient term (matches scene.wgsl).
fn hemisphere_ambient(n: vec3<f32>) -> vec3<f32> {
    let t = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(HEMI_GROUND, HEMI_SKY, t);
}

/// Build an orthonormal basis (tangent, bitangent) from a normal.
fn tangent_basis(n: vec3<f32>) -> mat3x3<f32> {
    let up  = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.x) > 0.9);
    let t   = normalize(cross(up, n));
    let b   = cross(n, t);
    return mat3x3<f32>(t, b, n);
}

/// Soft-shadow:  Vogel-disk shadow rays toward the sun.
/// Returns 1.0 (fully lit) down to 0.0 (fully shadowed).
fn soft_shadow(world_pos: vec3<f32>, n: vec3<f32>, seed: u32) -> f32 {
    if (g.light_params.z < 0.5) { return 1.0; }  // shadows disabled

    let sun_dir = normalize(g.light_dir.xyz);
    let ndl     = dot(n, sun_dir);
    if (ndl <= 0.0) { return 0.0; }

    // Cone half-angle ≈ 4°.  In fast-preview mode use a single hard shadow ray.
    let TAN_HALF = 0.07;
    let NUM_SAMPLES = select(4u, 1u, rt.fast_preview != 0u);
    let phi = rand_f(seed ^ 0xF1E2D3C4u) * 6.28318;

    let basis = tangent_basis(sun_dir);
    var lit   = 0.0;

    for (var i = 0u; i < NUM_SAMPLES; i++) {
        let d2    = vogel_disk(i, NUM_SAMPLES, phi);
        let jit   = normalize(sun_dir + (basis[0] * d2.x + basis[1] * d2.y) * TAN_HALF);
        let bias  = world_pos + n * 0.06;
        let hit   = dda(bias, jit, 256.0, true, select(512, 128, rt.fast_preview != 0u));
        if (!hit.hit) { lit += 1.0; }
    }
    return lit / f32(NUM_SAMPLES);
}

/// Shade a diffuse (plastic / rubber) surface.
fn shade_diffuse(world_pos: vec3<f32>, n: vec3<f32>, color: vec3<f32>, seed: u32) -> vec3<f32> {
    let amb     = g.light_params.x;
    let sun_lvl = g.light_params.y;
    let sc      = g.sun_color.xyz;
    let sun_dir = normalize(g.light_dir.xyz);

    let ndl     = max(dot(n, sun_dir), 0.0);
    let shadow  = soft_shadow(world_pos, n, seed);
    let ambient = hemisphere_ambient(n) * amb * 0.28;
    let direct  = sc * ndl * shadow * sun_lvl * 0.9;
    return color * (ambient + direct);
}

/// Schlick Fresnel.
fn schlick(cos_i: f32, ior: f32) -> f32 {
    let r0 = pow((1.0 - ior) / (1.0 + ior), 2.0);
    return r0 + (1.0 - r0) * pow(1.0 - cos_i, 5.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Bounce layers (unrolled to avoid recursion)
//
//   shade_secondary  — terminal bounce: DDA + diffuse/glow/sky, no further
//                      bounces for metal or transmissive (sky fallback only).
//   shade_metal / shade_transmissive  — first-bounce helpers; call
//                      shade_secondary for their own secondary rays.
//   shade (primary)  — full dispatch; calls shade_metal / shade_transmissive.
// ─────────────────────────────────────────────────────────────────────────────

/// Terminal bounce: cast one DDA ray and shade with at most diffuse/glow/sky.
/// Metal reflects to sky; transmissive reflects/refracts to sky or diffuse.
fn shade_secondary(origin: vec3<f32>, dir: vec3<f32>, seed: u32) -> vec3<f32> {
    // In fast_preview: half the distance, quarter the steps — coarse but shows geometry.
    let sec_steps = select(1024, 256, rt.fast_preview != 0u);
    let sec_dist  = select(1024.0, 512.0, rt.fast_preview != 0u);
    let h = dda(origin, dir, sec_dist, false, sec_steps);
    if (!h.hit) { return sky_color(dir); }

    let mat   = unpack_mat(h.packed);
    let color = unpack_rgb(h.packed);
    let wp    = origin + dir * h.t;

    if (mat == MAT_GLOW) { return color * 4.0; }

    if (mat == MAT_METAL) {
        // Reflect to sky — no further DDA
        let refl_dir = reflect(dir, h.normal);
        let refl_col = sky_color(refl_dir);
        let f0       = color;
        let cos_i    = abs(dot(-dir, h.normal));
        let fresnel  = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cos_i, 5.0);
        let shadow   = soft_shadow(wp, h.normal, seed ^ 0x1234u);
        let sun_dir  = normalize(g.light_dir.xyz);
        let spec_d   = max(dot(normalize(sun_dir - dir), h.normal), 0.0);
        let spec_hi  = g.sun_color.xyz * pow(spec_d, 32.0) * g.light_params.y * shadow * 0.6;
        let ambient  = hemisphere_ambient(h.normal) * g.light_params.x * 0.08;
        return refl_col * fresnel + spec_hi * color + color * ambient;
    }

    if (is_transmissive(mat)) {
        let is_water = mat == MAT_WATER;
        let ior      = select(1.5, 1.333, is_water);
        let eta      = 1.0 / ior;
        let cos_i    = abs(dot(-dir, h.normal));
        let fresnel  = schlick(cos_i, ior);
        let refr_raw = refract(dir, h.normal, eta);
        var refr_col = vec3<f32>(0.0);
        let tir      = length(refr_raw) < 0.5;
        if (!tir) {
            let refr_dir = normalize(refr_raw);
            let enter    = wp - h.normal * 0.06;
            // Trace through all transmissive media; dual Beer-Lambert so each
            // material is absorbed at its own rate (glass 2.5, water 24.0).
            let hit2 = dda(enter, refr_dir, 128.0, true, 512);
            if (hit2.hit) {
                let hmat = unpack_mat(hit2.packed);
                if (hmat == MAT_GLOW) {
                    refr_col = unpack_rgb(hit2.packed) * 4.0;
                } else {
                    let hw = enter + refr_dir * hit2.t;
                    refr_col = shade_diffuse(hw, hit2.normal, unpack_rgb(hit2.packed), seed ^ 0x7777u);
                }
            } else {
                refr_col = sky_color(refr_dir);
            }
            let tint_strength = select(0.25, 0.65, is_water);
            let absorb        = exp(-hit2.medium_dist_glass / 2.5)
                              * exp(-hit2.medium_dist_water / 24.0);
            let tint          = mix(vec3<f32>(1.0), color, tint_strength);
            refr_col         *= absorb * tint;
        }
        // Reflection also goes to sky at this terminal level
        let refl_dir = reflect(dir, h.normal);
        let refl_col = sky_color(refl_dir);
        let f        = select(fresnel, clamp(fresnel * 2.5, 0.0, 1.0), is_water);
        return mix(refr_col, refl_col, select(f, 1.0, tir));
    }

    // Plastic / rubber
    return shade_diffuse(wp, h.normal, color, seed);
}

/// Shade a metallic surface (primary bounce).  Calls shade_secondary for the
/// reflected ray; in fast_preview shade_secondary uses reduced DDA params.
fn shade_metal(world_pos: vec3<f32>, n: vec3<f32>, color: vec3<f32>,
               incident: vec3<f32>, seed: u32) -> vec3<f32> {
    let sun_dir  = normalize(g.light_dir.xyz);
    let sun_lvl  = g.light_params.y;
    let sc       = g.sun_color.xyz;

    let refl_dir = reflect(incident, n);
    let refl_col = shade_secondary(world_pos + n * 0.06, refl_dir, seed ^ 0xABCDu);

    let f0      = color;
    let cos_i   = abs(dot(-incident, n));
    let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cos_i, 5.0);

    let shadow   = soft_shadow(world_pos, n, seed ^ 0x1234u);
    let spec_d   = max(dot(normalize(sun_dir - incident), n), 0.0);
    let spec_hi  = sc * pow(spec_d, 32.0) * sun_lvl * shadow * 0.6;
    let ambient  = hemisphere_ambient(n) * g.light_params.x * 0.08;

    return refl_col * fresnel + spec_hi * color + color * ambient;
}

/// Shade a glass or water surface (primary bounce).
///
/// Strategy:
///   1. Compute refracted direction (Snell, air→medium).
///   2. Single DDA (skip_trans=true) traces through ALL transmissive media,
///      accumulating glass and water distances separately for Beer-Lambert.
///      This avoids the double-absorption that occurred when shade_secondary
///      was called from inside the primary medium and re-found the same voxel.
///   3. Shade the first opaque hit (diffuse / glow) or sky.
///   4. Apply dual Beer-Lambert: exp(-glass/2.5) * exp(-water/24).
///   5. Call shade_secondary for the reflected colour.
///   6. Fresnel blend.
fn shade_transmissive(world_pos: vec3<f32>, n: vec3<f32>, color: vec3<f32>,
                      incident: vec3<f32>, mat: u32, seed: u32) -> vec3<f32> {
    let is_water = mat == MAT_WATER;
    let ior      = select(1.5, 1.333, is_water);
    let eta      = 1.0 / ior;  // air → medium

    let cos_i   = abs(dot(-incident, n));
    let fresnel = schlick(cos_i, ior);

    // ── Refracted ray ────────────────────────────────────────────────────────
    let refr_raw = refract(incident, n, eta);
    var refr_col = vec3<f32>(0.0);

    // If refract() returns zero-length (total internal reflection) fall back.
    let tir = length(refr_raw) < 0.5;
    if (!tir) {
        let refr_dir = normalize(refr_raw);
        let enter    = world_pos - n * 0.06;

        // One DDA pass skips all transmissive media and tracks glass + water
        // distances separately. This avoids starting shade_secondary from
        // inside the medium (which caused it to re-process the same surface
        // and apply Beer-Lambert a second time).
        let sec_steps = select(1024, 256, rt.fast_preview != 0u);
        let sec_dist  = select(1024.0, 512.0, rt.fast_preview != 0u);
        let thr = dda(enter, refr_dir, sec_dist, true, sec_steps);

        if (thr.hit) {
            let hmat = unpack_mat(thr.packed);
            let hw   = enter + refr_dir * thr.t;
            if (hmat == MAT_GLOW) {
                refr_col = unpack_rgb(thr.packed) * 4.0;
            } else {
                refr_col = shade_diffuse(hw, thr.normal, unpack_rgb(thr.packed), seed ^ 0x9F2Bu);
            }
        } else {
            refr_col = sky_color(refr_dir);
        }

        // Dual Beer-Lambert: each material absorbed at its own rate.
        let tint_strength = select(0.25, 0.65, is_water);
        let absorb        = exp(-thr.medium_dist_glass / 2.5)
                          * exp(-thr.medium_dist_water / 24.0);
        let tint          = mix(vec3<f32>(1.0), color, tint_strength);
        refr_col         *= absorb * tint;
    }

    // ── Reflected ray ────────────────────────────────────────────────────────
    let refl_dir = reflect(incident, n);
    let refl_col = shade_secondary(world_pos + n * 0.06, refl_dir, seed ^ 0x5A5Au);

    // Boost Fresnel on water for a more mirror-like surface at grazing.
    let f = select(fresnel, clamp(fresnel * 2.5, 0.0, 1.0), is_water);
    return mix(refr_col, refl_col, select(f, 1.0, tir));
}

/// Primary shade: full material dispatch.
fn shade(origin: vec3<f32>, dir: vec3<f32>, seed: u32) -> vec3<f32> {
    let max_steps = select(2048, 1024, rt.fast_preview != 0u);
    let h = dda(origin, dir, 4096.0, false, max_steps);
    if (!h.hit) { return sky_color(dir); }

    let mat   = unpack_mat(h.packed);
    let color = unpack_rgb(h.packed);
    let wp    = origin + dir * h.t;

    if (mat == MAT_GLOW) {
        return color * 4.0;
    }
    if (mat == MAT_METAL) {
        return shade_metal(wp, h.normal, color, dir, seed);
    }
    if (is_transmissive(mat)) {
        return shade_transmissive(wp, h.normal, color, dir, mat, seed);
    }
    // Plastic / rubber
    return shade_diffuse(wp, h.normal, color, seed);
}

// ─────────────────────────────────────────────────────────────────────────────
// Vertex shader (fullscreen triangle)
// ─────────────────────────────────────────────────────────────────────────────

struct FullscreenOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> FullscreenOut {
    var o: FullscreenOut;
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    o.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    o.uv  = vec2<f32>(x, y);
    return o;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fragment shader
// ─────────────────────────────────────────────────────────────────────────────

@fragment
fn fs_trace(in: FullscreenOut) -> @location(0) vec4<f32> {
    let pixel = vec2<u32>(u32(in.pos.x), u32(in.pos.y));
    let pid   = pixel.x + pixel.y * u32(g.screen.x);

    // Per-pixel, per-frame seed
    let seed = hash_u(pid ^ hash_u(rt.frame_seed));

    // Sub-pixel jitter for anti-aliasing
    let jx = rand_f(seed)               - 0.5;
    let jy = rand_f(seed ^ 0x80000001u) - 0.5;

    // Reconstruct world-space ray from (jittered) NDC
    let ndc = vec2<f32>(
        (in.uv.x + jx * g.screen.z) * 2.0 - 1.0,
        1.0 - (in.uv.y + jy * g.screen.w) * 2.0
    );
    let clip      = vec4<f32>(ndc, 1.0, 1.0);
    var view_dir  = g.inv_proj * clip;
    view_dir      = vec4<f32>(view_dir.xyz / max(view_dir.w, 1e-6), 1.0);
    let world_dir = normalize((g.inv_view * vec4<f32>(view_dir.xyz, 0.0)).xyz);
    let origin    = g.cam_pos.xyz;

    // Trace primary ray
    let sample_rgb = shade(origin, world_dir, seed);
    let sample_v   = vec4<f32>(sample_rgb, 1.0);

    // Progressive accumulation
    if (rt.sample_n == 0u) {
        return sample_v;
    }
    let prev   = textureSample(accum_prev, samp, in.uv);
    let weight = 1.0 / f32(rt.sample_n + 1u);
    return mix(prev, sample_v, weight);
}
