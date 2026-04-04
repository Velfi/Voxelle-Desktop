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

// Atmosphere / fog params — mirrors PostCompositeOpts in mod.rs (14 vec4 rows = 224 bytes).
// Only the atmosphere fields (rows 3–7) are used here; the rest exist for layout correctness.
struct PostCompositeOpts {
    // Row 0
    tone_mode:        u32,
    transparent_bg:   f32,
    exposure_ev:      f32,
    time_seconds:     f32,
    // Row 1
    vignette_strength: f32,
    grain_enabled:    f32,
    grain_strength:   f32,
    grain_animated:   f32,
    // Row 2
    grain_speed:      f32,
    grain_colorful:   f32,
    _pad2a:           f32,
    _pad2b:           f32,
    // Row 3: atmosphere controls
    atm_enabled:      f32,
    atm_thickness:    f32,
    atm_density:      f32,
    atm_spatial_mode: f32,  // 0 = plane, 1 = aerial
    // Row 4: atmosphere color + mode
    atm_color_r:      f32,
    atm_color_g:      f32,
    atm_color_b:      f32,
    atm_mode:         f32,  // 0 = slab, 1 = positiveSide (plane modes)
    // Row 5: atmosphere plane
    atm_plane_nx:     f32,
    atm_plane_ny:     f32,
    atm_plane_nz:     f32,
    atm_plane_c:      f32,
    // Row 6: atmosphere height + drift
    atm_height_bias:   f32,
    atm_height_falloff: f32,
    atm_drift_enabled: f32,
    atm_drift_amount:  f32,
    // Row 7: drift continued
    atm_drift_scale:  f32,
    atm_drift_speed:  f32,
    _pad7a:           f32,
    _pad7b:           f32,
    // Row 8–14: unused by ray tracer (distance tint, sun shafts, bloom)
    _row8:  vec4<f32>,
    _row9:  vec4<f32>,
    _row10: vec4<f32>,
    _row11: vec4<f32>,
    _row12: vec4<f32>,
    _row13: vec4<f32>,
    _row14: vec4<f32>,
}

@group(1) @binding(3) var<uniform> fog: PostCompositeOpts;

// ─────────────────────────────────────────────────────────────────────────────
// Voxel access  (bit layout: occupied=31, mat=27:24, B=23:16, G=15:8, R=7:0)
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
fn unpack_mat(packed: u32) -> u32   { return (packed >> 24u) & 0xFu; }

// Proper sRGB → linear (piecewise curve)
fn srgb_to_linear(c: f32) -> f32 {
    if (c <= 0.04045) { return c / 12.92; }
    return pow((c + 0.055) / 1.055, 2.4);
}

fn unpack_rgb(packed: u32) -> vec3<f32> {
    let r = f32( packed        & 0xFFu) / 255.0;
    let gv= f32((packed >> 8u) & 0xFFu) / 255.0;
    let b = f32((packed >>16u) & 0xFFu) / 255.0;
    return vec3<f32>(srgb_to_linear(r), srgb_to_linear(gv), srgb_to_linear(b));
}

const MAT_PLASTIC:     u32 = 0u;
const MAT_METAL:       u32 = 1u;
const MAT_RUBBER:      u32 = 2u;
const MAT_GLASS:       u32 = 3u;
const MAT_WATER:       u32 = 4u;
const MAT_GLOW:        u32 = 5u;
const MAT_VELVET:      u32 = 6u;
const MAT_WAX:         u32 = 7u;
const MAT_HOLOGRAPHIC: u32 = 8u;

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

    var cell = vec3<i32>(i32(floor(origin.x + 0.5)), i32(floor(origin.y + 0.5)), i32(floor(origin.z + 0.5)));

    var tMax = vec3<f32>(
        select((f32(cell.x) - 0.5 - origin.x) * inv.x, (f32(cell.x) + 0.5 - origin.x) * inv.x, dx > 0.0),
        select((f32(cell.y) - 0.5 - origin.y) * inv.y, (f32(cell.y) + 0.5 - origin.y) * inv.y, dy > 0.0),
        select((f32(cell.z) - 0.5 - origin.z) * inv.z, (f32(cell.z) + 0.5 - origin.z) * inv.z, dz > 0.0),
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

/// Irradiance from nearby glow voxels onto a surface point.
/// Casts short hemisphere rays (Vogel-disk cosine-weighted) and collects
/// glow hits — smooth falloff, no voxel-grid blockiness.
const SURFACE_GLOW_STRENGTH: f32 = 5.0;
const GLOW_REACH: f32            = 64.0;

fn surface_glow_at(p: vec3<f32>, n: vec3<f32>, seed: u32) -> vec3<f32> {
    let NUM_RAYS = select(12u, 4u, rt.fast_preview != 0u);
    let phi      = rand_f(seed ^ 0xE41Au) * 6.28318;
    let basis    = tangent_basis(n);
    let origin   = p + n * 0.06;
    var acc      = vec3<f32>(0.0);

    for (var i = 0u; i < NUM_RAYS; i++) {
        // Cosine-weighted hemisphere sample via Vogel disk.
        let d2        = vogel_disk(i, NUM_RAYS, phi);
        let cos_theta = sqrt(1.0 - dot(d2, d2));
        let local_dir = vec3<f32>(d2.x, d2.y, cos_theta);
        let dir       = normalize(basis * local_dir);

        let hit = dda(origin, dir, GLOW_REACH, false, select(512, 96, rt.fast_preview != 0u));
        if (hit.hit && unpack_mat(hit.packed) == MAT_GLOW) {
            let gc  = unpack_rgb(hit.packed);
            // Smooth quadratic falloff: full brightness at contact, zero at GLOW_REACH.
            let norm_t = hit.t / GLOW_REACH;
            let att    = (1.0 - norm_t) * (1.0 - norm_t);
            acc       += gc * att * SURFACE_GLOW_STRENGTH;
        }
    }
    return acc / f32(NUM_RAYS);
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
    let glow    = surface_glow_at(world_pos, n, seed ^ 0xE10Bu);
    return color * (ambient + direct + glow);
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

    if (mat == MAT_GLOW) { return color * 8.0; }

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
                    refr_col = unpack_rgb(hit2.packed) * 8.0;
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

    if (mat == MAT_VELVET) {
        // Terminal-bounce velvet: wrap diffuse + rim, no further bounce.
        let sun_dir2  = normalize(g.light_dir.xyz);
        let wrap2     = 0.45;
        let ndl_wrap2 = max((dot(h.normal, sun_dir2) + wrap2) / (1.0 + wrap2), 0.0);
        let shadow2   = soft_shadow(wp, h.normal, seed ^ 0x5E5Eu);
        let amb2      = g.light_params.x;
        let sun2      = g.light_params.y;
        let sc2       = g.sun_color.xyz;
        let ambient2  = hemisphere_ambient(h.normal) * amb2 * 0.26;
        let direct2   = sc2 * ndl_wrap2 * shadow2 * sun2 * 0.60;
        let cos_i2    = abs(dot(-dir, h.normal));
        let rim2      = (1.0 - cos_i2);
        let rim2_sq   = rim2 * rim2;
        let rim_col2  = mix(color, min(color * 1.5 + vec3<f32>(0.06), vec3<f32>(1.0)), rim2_sq * 0.55);
        let rim_lit2  = rim_col2 * rim2_sq * 0.35 * (amb2 * 0.5 + shadow2 * sun2 * 0.5);
        return color * (ambient2 + direct2) + rim_lit2;
    }

    if (mat == MAT_HOLOGRAPHIC) {
        // Terminal-bounce holographic: iridescent reflection to sky, no further DDA.
        let holo_cos  = abs(dot(-dir, h.normal));
        let holo_spec = thin_film_iridescence(holo_cos, color);
        let refl_dir2 = reflect(dir, h.normal);
        let refl_sky  = sky_color(refl_dir2);
        let shadow2   = soft_shadow(wp, h.normal, seed ^ 0xA0A0u);
        let sun_dir2  = normalize(g.light_dir.xyz);
        let spec_d2   = max(dot(normalize(sun_dir2 - dir), h.normal), 0.0);
        let spec_hi2  = g.sun_color.xyz * pow(spec_d2, 64.0) * g.light_params.y * shadow2 * 1.0;
        let ambient2  = hemisphere_ambient(h.normal) * g.light_params.x * 0.06;
        return refl_sky * holo_spec + spec_hi2 * holo_spec + color * ambient2 * holo_spec;
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

/// Thin-film interference: compute the iridescent spectral color for a given
/// cos(theta_i) at the film surface.  Models a dielectric thin film (IOR 1.38,
/// thickness ~550 nm) on a metallic substrate — the same physics as oil slicks,
/// soap bubbles, and holographic foil.
fn thin_film_iridescence(cos_i: f32, base_color: vec3<f32>) -> vec3<f32> {
    let n_film = 1.38;
    let d_nm   = 550.0;
    let sin_t2 = (1.0 - cos_i * cos_i) / (n_film * n_film);
    let cos_t  = sqrt(max(1.0 - sin_t2, 0.0));
    let opd    = 2.0 * n_film * d_nm * cos_t;

    // Per-channel phase from optical path difference vs wavelength.
    let phase_r = opd / 630.0 * 6.28318530;
    let phase_g = opd / 530.0 * 6.28318530;
    let phase_b = opd / 460.0 * 6.28318530;

    // Simplified Airy reflectance for the thin film.
    let r_film = 0.20;
    let denom  = 1.0 + r_film * r_film + 2.0 * r_film;
    let irid = vec3<f32>(
        (r_film * r_film + r_film * 2.0 * cos(phase_r) + 1.0) / denom,
        (r_film * r_film + r_film * 2.0 * cos(phase_g) + 1.0) / denom,
        (r_film * r_film + r_film * 2.0 * cos(phase_b) + 1.0) / denom,
    );

    // Tint by base color and blend with metallic Fresnel.
    let f0      = base_color * 0.75 + vec3<f32>(0.04);
    let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cos_i, 5.0);
    return mix(fresnel, irid, 0.85);
}

/// Shade a holographic surface (primary bounce).  Thin-film diffraction grating
/// on a metallic substrate: reflects environment with angle-dependent spectral
/// color shifts.  The reflected ray uses shade_secondary.
fn shade_holographic(world_pos: vec3<f32>, n: vec3<f32>, color: vec3<f32>,
                     incident: vec3<f32>, seed: u32) -> vec3<f32> {
    let sun_dir = normalize(g.light_dir.xyz);
    let sun_lvl = g.light_params.y;
    let sc      = g.sun_color.xyz;
    let cos_i   = abs(dot(-incident, n));

    // ── Thin-film spectral Fresnel ──────────────────────────────────────────
    let holo_fresnel = thin_film_iridescence(cos_i, color);

    // ── Grating dispersion: add lateral rainbow shift per face direction ─────
    let abs_n = abs(n);
    var grating_dir: vec3<f32>;
    if (abs_n.y > max(abs_n.x, abs_n.z)) {
        grating_dir = vec3<f32>(1.0, 0.0, 0.0);
    } else if (abs_n.x > abs_n.z) {
        grating_dir = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        grating_dir = vec3<f32>(1.0, 0.0, 0.0);
    }
    let grating_dot = dot(grating_dir, -incident);
    let grating_shift = grating_dot * 120.0;
    let sin_t2_g = (1.0 - cos_i * cos_i) / (1.38 * 1.38);
    let cos_t_g  = sqrt(max(1.0 - sin_t2_g, 0.0));
    let opd_g    = 2.0 * 1.38 * 550.0 * cos_t_g + grating_shift;
    let grating_col = vec3<f32>(
        0.5 + 0.5 * cos(opd_g / 630.0 * 6.28318530),
        0.5 + 0.5 * cos(opd_g / 530.0 * 6.28318530),
        0.5 + 0.5 * cos(opd_g / 460.0 * 6.28318530),
    );
    let spectral = mix(holo_fresnel, grating_col, 0.4);

    // ── Reflected environment (secondary bounce) ────────────────────────────
    let refl_dir = reflect(incident, n);
    let refl_col = shade_secondary(world_pos + n * 0.06, refl_dir, seed ^ 0xA010u);

    // ── Sun specular ────────────────────────────────────────────────────────
    let shadow   = soft_shadow(world_pos, n, seed ^ 0x1234u);
    let spec_d   = max(dot(normalize(sun_dir - incident), n), 0.0);
    let spec_hi  = sc * pow(spec_d, 64.0) * sun_lvl * shadow * 1.2;

    // ── Combine: iridescent reflection + specular + ambient ─────────────────
    let ambient = hemisphere_ambient(n) * g.light_params.x * 0.06;
    return refl_col * spectral + spec_hi * spectral + color * ambient * spectral;
}

/// Christensen-Burley normalized diffusion profile (two-exponential fit).
/// `r` = distance from entry point, `d` = mean free path (diffuse scale).
/// Returns the radial falloff weight; energy is already normalized.
/// Reference: Christensen & Burley, "Approximate Reflectance Profiles for
/// Efficient Subsurface Scattering" (BSSRDF), Pixar 2015.
fn burley_profile(r: f32, d: f32) -> f32 {
    let rd = r / max(d, 0.001);
    return (exp(-rd) + exp(-rd / 3.0)) / (8.0 * 3.14159265 * d);
}

/// Sample a cosine-weighted hemisphere direction using two uniform random values.
/// Returns a direction in the tangent-space basis defined by `tangent_basis(n)`.
fn cosine_hemisphere_sample(n: vec3<f32>, u1: f32, u2: f32) -> vec3<f32> {
    let r   = sqrt(u1);
    let phi = 6.28318530 * u2;
    let x   = r * cos(phi);
    let y   = r * sin(phi);
    let z   = sqrt(max(1.0 - u1, 0.0));
    let basis = tangent_basis(n);
    return normalize(basis[0] * x + basis[1] * y + basis[2] * z);
}

/// Sample a uniform direction on the full sphere.
fn uniform_sphere_sample(u1: f32, u2: f32) -> vec3<f32> {
    let z   = 1.0 - 2.0 * u1;
    let r   = sqrt(max(1.0 - z * z, 0.0));
    let phi = 6.28318530 * u2;
    return vec3<f32>(r * cos(phi), r * sin(phi), z);
}

/// Result from `march_to_exit`: distance to the exit point and the face normal
/// at the cell boundary where the ray left the wax body.
struct ExitHit {
    dist:   f32,
    normal: vec3<f32>,
}

/// March through wax voxels (mat == MAT_WAX) from inside a solid body to find
/// where the ray exits.  Returns the distance to the first non-wax cell and
/// the face normal at that boundary.  Stops at empty space, non-wax material,
/// or `max_dist`.
fn march_to_exit(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32, max_steps: i32) -> ExitHit {
    var result: ExitHit;
    result.dist = max_dist;
    result.normal = vec3<f32>(0.0, 1.0, 0.0);

    let EPS = 1e-7;
    let dx = select(dir.x, select(-EPS, EPS, dir.x >= 0.0), abs(dir.x) < EPS);
    let dy = select(dir.y, select(-EPS, EPS, dir.y >= 0.0), abs(dir.y) < EPS);
    let dz = select(dir.z, select(-EPS, EPS, dir.z >= 0.0), abs(dir.z) < EPS);

    let step = vec3<i32>(i32(sign(dx)), i32(sign(dy)), i32(sign(dz)));
    let inv  = vec3<f32>(1.0 / dx, 1.0 / dy, 1.0 / dz);
    let tD   = abs(inv);

    var cell = vec3<i32>(i32(floor(origin.x + 0.5)), i32(floor(origin.y + 0.5)), i32(floor(origin.z + 0.5)));
    var tMax = vec3<f32>(
        select((f32(cell.x) - 0.5 - origin.x) * inv.x, (f32(cell.x) + 0.5 - origin.x) * inv.x, dx > 0.0),
        select((f32(cell.y) - 0.5 - origin.y) * inv.y, (f32(cell.y) + 0.5 - origin.y) * inv.y, dy > 0.0),
        select((f32(cell.z) - 0.5 - origin.z) * inv.z, (f32(cell.z) + 0.5 - origin.z) * inv.z, dz > 0.0),
    );
    var t      = 0.0;
    var normal = vec3<f32>(0.0, 1.0, 0.0);

    for (var i = 0; i < max_steps; i++) {
        if (t >= max_dist) { break; }

        let packed = brick_fetch(cell);
        if (!is_occupied(packed) || unpack_mat(packed) != MAT_WAX) {
            // Exited wax: either empty space or a different material.
            result.dist = t;
            result.normal = normal;
            return result;
        }

        // Advance to next cell boundary, tracking the face normal.
        if (tMax.x < tMax.y) {
            if (tMax.x < tMax.z) {
                t = tMax.x; tMax.x += tD.x;
                normal = vec3<f32>(-f32(step.x), 0.0, 0.0);
                cell.x += step.x;
            } else {
                t = tMax.z; tMax.z += tD.z;
                normal = vec3<f32>(0.0, 0.0, -f32(step.z));
                cell.z += step.z;
            }
        } else {
            if (tMax.y < tMax.z) {
                t = tMax.y; tMax.y += tD.y;
                normal = vec3<f32>(0.0, -f32(step.y), 0.0);
                cell.y += step.y;
            } else {
                t = tMax.z; tMax.z += tD.z;
                normal = vec3<f32>(0.0, 0.0, -f32(step.z));
                cell.z += step.z;
            }
        }
    }
    return result;
}

/// Per-channel Beer-Lambert absorption for wax.
/// Wax absorbs blue fastest, green moderately, red least — giving the
/// characteristic warm amber glow in thin sections and deep reddish hue in
/// thick ones.  Values tuned to beeswax / paraffin spectral data.
fn wax_absorption(thickness: f32, base_color: vec3<f32>) -> vec3<f32> {
    // Mean free path per channel (voxel units).  Higher = light travels further.
    // These ratios model wax: red penetrates ~3× deeper than blue.
    let mfp = vec3<f32>(2.8, 1.6, 0.9) * (base_color * 0.5 + vec3<f32>(0.5));
    return exp(-vec3<f32>(thickness) / max(mfp, vec3<f32>(0.01)));
}

/// Shade a wax surface with physically-based subsurface scattering.
///
/// Combines five components for realistic translucent wax:
///   1. Dielectric Fresnel sheen (IOR 1.46) — subtle surface reflection.
///   2. Wrap diffuse — soft, round shading that extends past the terminator.
///   3. Direct SSS (Christensen-Burley) — light entering from the sun side,
///      diffusing through the interior, and exiting at this point.  Multi-
///      direction thickness sampling captures both backlit translucency and
///      ambient light bleeding through thin geometry.
///   4. Random-walk SSS — 2-3 short random bounces inside the wax volume
///      to find actual exit points and shade them, capturing true subsurface
///      light transport including color bleeding and internal scattering.
///   5. Per-channel spectral absorption — red travels furthest, blue least,
///      producing warm amber glow in thin sections.
fn shade_wax(world_pos: vec3<f32>, n: vec3<f32>, color: vec3<f32>,
             incident: vec3<f32>, seed: u32) -> vec3<f32> {
    let sun_dir = normalize(g.light_dir.xyz);
    let sun_lvl = g.light_params.y;
    let sc      = g.sun_color.xyz;
    let amb     = g.light_params.x;
    let v       = -incident;
    let cos_i   = max(dot(n, v), 0.0);
    let fast    = rt.fast_preview != 0u;

    // ── 1. Dielectric Fresnel (IOR 1.46 — paraffin wax) ─────────────────────
    let ior  = 1.46;
    let r0   = pow((ior - 1.0) / (ior + 1.0), 2.0);
    let fres = r0 + (1.0 - r0) * pow(1.0 - cos_i, 5.0);

    // Reflected environment for the Fresnel sheen.
    let refl_dir = reflect(incident, n);
    let refl_col = shade_secondary(world_pos + n * 0.06, refl_dir, seed ^ 0xCA01u);

    // ── 2. Wrap diffuse ─────────────────────────────────────────────────────
    let wax_wrap  = 0.55;
    let ndl_wrap  = max((dot(n, sun_dir) + wax_wrap) / (1.0 + wax_wrap), 0.0);
    let shadow    = soft_shadow(world_pos, n, seed ^ 0xCA02u);
    let hemi      = hemisphere_ambient(n);
    let diffuse   = color * (hemi * amb * 0.24 + sc * ndl_wrap * shadow * sun_lvl * 0.60);

    // ── 3. Multi-direction thickness SSS (Christensen-Burley) ────────────────
    // Sample thickness from several directions to capture both direct (sun)
    // and ambient subsurface light transmission.
    let sss_budget = select(12, 6, fast);
    let num_dirs   = select(4u, 2u, fast);  // 1 sun + N hemisphere samples
    let mean_free  = 1.8;  // Burley diffuse scale parameter (voxel units)

    // Enter just inside the surface for all interior marches.
    let sss_enter = world_pos - n * 0.3;

    // Sun direction: primary SSS contributor (backlit translucency).
    let sun_exit  = march_to_exit(sss_enter, sun_dir, 12.0, sss_budget);
    let sun_abs   = wax_absorption(sun_exit.dist, color);
    let sun_prof  = burley_profile(sun_exit.dist, mean_free);
    var sss_accum = sun_abs * sun_prof * sc * sun_lvl * shadow;

    // Hemisphere samples: ambient SSS from scattered environment light.
    // Each direction probes how thin the wax is in that direction; light from
    // that hemisphere contributes proportional to the diffusion profile.
    for (var i = 0u; i < num_dirs; i++) {
        let u1  = rand_f(seed ^ (0xCA10u + i * 3u));
        let u2  = rand_f(seed ^ (0xCA11u + i * 3u));
        let dir = cosine_hemisphere_sample(-n, u1, u2);  // inward hemisphere
        let exit_i = march_to_exit(sss_enter, dir, 8.0, sss_budget);
        let abs_c  = wax_absorption(exit_i.dist, color);
        let prof   = burley_profile(exit_i.dist, mean_free);
        // Approximate incoming radiance from that direction as hemisphere ambient.
        let incoming = hemisphere_ambient(-dir) * amb;
        sss_accum += abs_c * prof * incoming;
    }
    // Normalize by sample count and scale.
    // Note: wax_absorption already encodes the spectral character via color-
    // dependent mean free path, so we do NOT multiply by `color` again here.
    let sss_direct = sss_accum * (mean_free * 12.0 / f32(1u + num_dirs));

    // ── 4. Random-walk interior scattering ───────────────────────────────────
    // Trace a short random walk inside the wax: each step scatters in a random
    // direction.  If the walk exits the surface, shade that exit point —
    // capturing true volumetric color bleeding and internal light transport.
    let walk_steps  = select(3u, 1u, fast);
    let step_length = 1.2;  // voxels per step (≈ mean free path)
    var walk_col    = vec3<f32>(0.0);
    var walk_weight = 0.0;
    var walk_dist_accum = 0.0;

    // Start just inside the surface.
    var walk_pos  = world_pos - n * 0.5;
    var walk_seed = seed ^ 0xCA20u;

    for (var s = 0u; s < walk_steps; s++) {
        // Random scatter direction (uniform sphere — isotropic scattering).
        let r1 = rand_f(walk_seed); walk_seed = hash_u(walk_seed);
        let r2 = rand_f(walk_seed); walk_seed = hash_u(walk_seed);
        let scatter_dir = uniform_sphere_sample(r1, r2);

        // March through wax to find the exit surface.
        let walk_exit = march_to_exit(walk_pos, scatter_dir, 6.0, select(16, 8, fast));
        walk_dist_accum += walk_exit.dist;

        if (walk_exit.dist < 5.9) {
            // Exited the wax body — shade the exit point.
            let exit_pos = walk_pos + scatter_dir * (walk_exit.dist + 0.1);
            // Use the tracked face normal from march_to_exit.
            let exit_n = walk_exit.normal;
            // Spectral absorption for the total walk distance.
            let walk_abs = wax_absorption(walk_dist_accum, color);
            // Shade the exit point with wrap diffuse.
            let exit_ndl  = max((dot(exit_n, sun_dir) + 0.3) / 1.3, 0.0);
            let exit_sh   = soft_shadow(exit_pos, exit_n, walk_seed);
            let exit_hemi = hemisphere_ambient(exit_n);
            let exit_light = sc * exit_ndl * exit_sh * sun_lvl + exit_hemi * amb;
            walk_col += walk_abs * exit_light;
            walk_weight += 1.0;
            break;  // One successful exit is enough per pixel.
        }
        // Still deep inside: advance to next scatter point (step_length into the scatter).
        walk_pos = walk_pos + scatter_dir * min(walk_exit.dist * 0.5, step_length);
    }
    let sss_walk = select(vec3<f32>(0.0), walk_col / max(walk_weight, 1.0), walk_weight > 0.0);

    // ── 5. Broad specular highlight ──────────────────────────────────────────
    // Wax has a soft, wide specular lobe (low roughness but spread by
    // micro-surface undulations).
    let h_spec = normalize(sun_dir + v);
    let ndh    = max(dot(n, h_spec), 0.0);
    let spec   = sc * pow(ndh, 16.0) * 0.10 * shadow * sun_lvl;

    // ── Combine ──────────────────────────────────────────────────────────────
    // Surface = Fresnel-weighted reflection + specular; subsurface underneath.
    let surface    = (refl_col + spec) * fres;
    let subsurface = diffuse * 0.55 + sss_direct * 0.30 + sss_walk * 0.15;
    return surface + subsurface * (1.0 - fres);
}

/// Shade a velvet / felt surface.
///
/// Strategy:
///   1. Wrap diffuse for soft, rounded shading (tighter wrap than wax).
///   2. Inverse Fresnel rim: fabric brightens at grazing angles as fibers
///      scatter light toward the viewer.
///   3. Anisotropic sheen from axis-derived tangent: approximates directional
///      shimmer from aligned microfibers.
///   4. Secondary bounce for environment color bleeding onto the soft surface.
fn shade_velvet(world_pos: vec3<f32>, n: vec3<f32>, color: vec3<f32>,
                incident: vec3<f32>, seed: u32) -> vec3<f32> {
    let sun_dir = normalize(g.light_dir.xyz);
    let sun_lvl = g.light_params.y;
    let sc      = g.sun_color.xyz;
    let amb     = g.light_params.x;

    // Wrap diffuse: softer than plastic, tighter than wax.
    let wrap     = 0.45;
    let ndl_wrap = max((dot(n, sun_dir) + wrap) / (1.0 + wrap), 0.0);
    let shadow   = soft_shadow(world_pos, n, seed);
    let ambient  = hemisphere_ambient(n) * amb * 0.26;
    let direct   = sc * ndl_wrap * shadow * sun_lvl * 0.60;
    let diffuse  = color * (ambient + direct);

    // Inverse Fresnel rim: velvet brightens at grazing angles.
    let cos_i = abs(dot(-incident, n));
    let rim   = 1.0 - cos_i;
    let rim2  = rim * rim;
    let rim_color = mix(color, min(color * 1.5 + vec3<f32>(0.06), vec3<f32>(1.0)), rim2 * 0.55);
    let rim_light = rim_color * rim2 * 0.40 * (amb * 0.5 + shadow * sun_lvl * 0.5);

    // Anisotropic sheen: derive tangent from face axis for directional shimmer.
    let abs_n = abs(n);
    var tangent: vec3<f32>;
    if (abs_n.y > max(abs_n.x, abs_n.z)) {
        tangent = vec3<f32>(1.0, 0.0, 0.0); // Y-face: fibers along X
    } else if (abs_n.x > abs_n.z) {
        tangent = vec3<f32>(0.0, 1.0, 0.0); // X-face: fibers along Y
    } else {
        tangent = vec3<f32>(0.0, 1.0, 0.0); // Z-face: fibers along Y
    }
    let h_vec = normalize(sun_dir - incident);
    let tdh   = dot(tangent, h_vec);
    let sheen = pow(max(1.0 - tdh * tdh, 0.0), 4.0) * 0.35 * shadow * sun_lvl * ndl_wrap;

    // Secondary bounce: environment color bleeding onto the soft surface.
    // Cast a hemisphere-scattered ray biased toward grazing angles to simulate
    // fiber scattering (real fibers scatter more at rim than at normal).
    let basis = tangent_basis(n);
    let r1    = rand_f(seed ^ 0xBE1BE1u);
    let r2    = rand_f(seed ^ 0xBE2BE2u);
    // Bias toward grazing vs standard cosine hemisphere: pow(r1, 0.8) flattens
    // the distribution so more rays go sideways than pow(r1, 0.5).
    let cos_theta = pow(r1, 0.8);
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    let phi       = r2 * 6.28318;
    let local_dir = vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    let bounce_dir = normalize(basis * local_dir);
    let bounce_origin = world_pos + n * 0.06;
    let bounce_steps  = select(512, 128, rt.fast_preview != 0u);
    let bounce_hit    = dda(bounce_origin, bounce_dir, 128.0, false, bounce_steps);
    var indirect = vec3<f32>(0.0);
    if (bounce_hit.hit) {
        let bmat  = unpack_mat(bounce_hit.packed);
        let bcol  = unpack_rgb(bounce_hit.packed);
        if (bmat == MAT_GLOW) {
            indirect = bcol * 4.0;
        } else {
            let bwp = bounce_origin + bounce_dir * bounce_hit.t;
            indirect = shade_diffuse(bwp, bounce_hit.normal, bcol, seed ^ 0x4321u);
        }
    } else {
        indirect = sky_color(bounce_dir);
    }
    // Weight indirect by rim — fibers scatter more environment light at grazing.
    let indirect_contrib = indirect * color * (0.12 + rim2 * 0.18);

    return diffuse + rim_light + sc * sheen + indirect_contrib;
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
                refr_col = unpack_rgb(thr.packed) * 8.0;
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

    // t_hit for fog march: actual hit distance, or capped sky distance.
    let t_hit = select(256.0, h.t, h.hit);

    var rgb: vec3<f32>;
    if (!h.hit) {
        rgb = sky_color(dir);
    } else {
        let mat   = unpack_mat(h.packed);
        let color = unpack_rgb(h.packed);
        let wp    = origin + dir * h.t;

        if (mat == MAT_GLOW) {
            rgb = color * 8.0;
        } else if (mat == MAT_METAL) {
            rgb = shade_metal(wp, h.normal, color, dir, seed);
        } else if (is_transmissive(mat)) {
            rgb = shade_transmissive(wp, h.normal, color, dir, mat, seed);
        } else if (mat == MAT_WAX) {
            rgb = shade_wax(wp, h.normal, color, dir, seed);
        } else if (mat == MAT_VELVET) {
            rgb = shade_velvet(wp, h.normal, color, dir, seed);
        } else if (mat == MAT_HOLOGRAPHIC) {
            rgb = shade_holographic(wp, h.normal, color, dir, seed);
        } else {
            rgb = shade_diffuse(wp, h.normal, color, seed);
        }
    }

    return apply_volumetric_fog(rgb, origin, dir, t_hit);
}

// ─────────────────────────────────────────────────────────────────────────────
// Volumetric fog
// ─────────────────────────────────────────────────────────────────────────────

/// Per-voxel extinction coefficient (sigma_t) at world point `p`.
/// The spatial shape uses the same formulae as post_composite.wgsl atmosphere,
/// but returns an extinction rate (1/voxel) rather than a cumulative lerp factor,
/// so it can be integrated correctly along a ray via Beer-Lambert.
fn fog_extinction_at(p: vec3<f32>) -> f32 {
    let thickness = max(fog.atm_thickness, 0.1);
    // Base extinction: at distance `thickness` total OD ≈ 1 → transmittance ≈ 1/e.
    let sigma_base = fog.atm_density / thickness;

    // Spatial shape factor in [0, ∞) — 1.0 for aerial (uniform), varied for plane modes.
    var shape: f32 = 1.0;
    if (fog.atm_spatial_mode < 0.5) {
        let n  = vec3<f32>(fog.atm_plane_nx, fog.atm_plane_ny, fog.atm_plane_nz);
        let sd = dot(n, p) + fog.atm_plane_c;
        if (fog.atm_mode > 0.5) {
            // Above-face: half-space with soft edge
            let softness = thickness * 0.5;
            shape = smoothstep(-softness, 0.0, sd) * exp(-max(0.0, sd) / thickness);
        } else {
            // Slab: Gaussian falloff around plane
            shape = exp(-(sd * sd) / (thickness * thickness));
        }
    }

    // Height modulation
    let falloff = max(fog.atm_height_falloff, 1.0);
    shape *= 0.65 + 0.35 * exp(-abs(p.y - fog.atm_height_bias) / falloff);

    // Drift (animated sine displacement)
    if (fog.atm_drift_enabled > 0.5) {
        let drift = sin((p.x + p.z) * fog.atm_drift_scale + fog.time_seconds * fog.atm_drift_speed)
                  * fog.atm_drift_amount * 0.35;
        shape = max(0.0, shape + drift);
    }

    return sigma_base * shape;
}

/// Light scattered into the fog from nearby glow voxels at world point `p`.
/// Searches within GLOW_FOG_RADIUS voxels using the live brick grid.
const GLOW_FOG_STRENGTH: f32 = 0.55;

fn fog_glow_at(p: vec3<f32>) -> vec3<f32> {
    let radius = select(5, 3, rt.fast_preview != 0u);
    let r2     = radius * radius;
    let cell   = vec3<i32>(i32(floor(p.x + 0.5)), i32(floor(p.y + 0.5)), i32(floor(p.z + 0.5)));
    var acc    = vec3<f32>(0.0);
    for (var dx = -radius; dx <= radius; dx++) {
        for (var dy = -radius; dy <= radius; dy++) {
            for (var dz = -radius; dz <= radius; dz++) {
                let d2 = dx * dx + dy * dy + dz * dz;
                if (d2 > r2) { continue; }
                let pk = brick_fetch(cell + vec3<i32>(dx, dy, dz));
                if (is_occupied(pk) && unpack_mat(pk) == MAT_GLOW) {
                    let gc  = unpack_rgb(pk);
                    let att = 1.0 / (f32(d2) + 1.0);
                    acc    += gc * att * GLOW_FOG_STRENGTH;
                }
            }
        }
    }
    return acc;
}

/// Apply volumetric fog along the ray [origin, origin + dir * t_hit].
/// Uses Beer-Lambert transmittance + in-scattering accumulation.
/// t_hit should be capped (e.g. 256 for sky) to limit the march range.
fn apply_volumetric_fog(color: vec3<f32>, origin: vec3<f32>, dir: vec3<f32>, t_hit: f32) -> vec3<f32> {
    if (fog.atm_enabled < 0.5) { return color; }

    let N         = select(8, 4, rt.fast_preview != 0u);
    let step_size = t_hit / f32(N);
    var transmit  = 1.0;
    var in_scatter = vec3<f32>(0.0);
    let fog_color  = vec3<f32>(fog.atm_color_r, fog.atm_color_g, fog.atm_color_b);

    for (var i = 0; i < N; i++) {
        let t            = (f32(i) + 0.5) * step_size;
        let p            = origin + dir * t;
        let sigma        = fog_extinction_at(p);
        let optical_depth = sigma * step_size;
        if (optical_depth < 1e-6) { continue; }
        let step_transmit = exp(-optical_depth);
        // In-scatter: (1 - exp(-od)) is the fraction of light scattered into the ray this step.
        let scatter_w    = 1.0 - step_transmit;
        let glow_light   = fog_glow_at(p);
        let scatter_color = fog_color + glow_light;
        in_scatter += transmit * scatter_w * scatter_color;
        transmit   *= step_transmit;
    }

    return color * transmit + in_scatter;
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
