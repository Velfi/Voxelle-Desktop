/// Fullscreen sky: analytic atmosphere vs sun elevation (path length / Rayleigh + Mie).
/// `light_dir` points toward the sun; `sun_color` tints transmitted and scattered light.
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
    o.uv = vec2<f32>(x, y);
    return o;
}

fn world_ray_dir(uv: vec2<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let clip = vec4<f32>(ndc, 1.0, 1.0);
    var view = g.inv_proj * clip;
    view = vec4<f32>(view.xyz / max(view.w, 1e-6), 1.0);
    let world = (g.inv_view * vec4<f32>(view.xyz, 0.0)).xyz;
    return normalize(world);
}

/// Relative optical air mass toward the sun (Y-up). Longer paths when the sun is low.
fn sun_air_mass(sun_y: f32) -> f32 {
    return 1.0 / max(sun_y, 0.032);
}

/// Chromatic extinction: blue scatters / absorbs more along a long slant path (redder sun).
fn extinction_rgb(air_mass: f32) -> vec3<f32> {
    return vec3<f32>(
        exp(-0.052 * air_mass),
        exp(-0.082 * air_mass),
        exp(-0.128 * air_mass)
    );
}

/// Rayleigh phase (normalized to ~[0,1] for shading).
fn rayleigh_phase(cos_theta: f32) -> f32 {
    return 0.65 * (1.0 + cos_theta * cos_theta);
}

/// Blend toward full analytic day sky as ambient+sun increase (moonlight stays night).
fn day_sky_blend_weight(amb: f32, sun_lvl: f32) -> f32 {
    let k = clamp(amb + sun_lvl, 0.0, 8.0);
    return smoothstep(0.1, 0.82, pow(k, 1.42));
}

/// Dims the day model when either channel is low (matches scene lighting energy).
fn day_sky_brightness(amb: f32, sun_lvl: f32) -> f32 {
    let e = amb * 0.48 + sun_lvl * 0.52;
    return clamp(0.12 + 0.88 * smoothstep(0.04, 1.35, e), 0.08, 1.35);
}

/// 1 only when both ambient and sunlight are effectively off (Total darkness preset).
fn star_field_weight(amb: f32, sun_lvl: f32) -> f32 {
    let a = 1.0 - smoothstep(0.0, 0.028, amb);
    let b = 1.0 - smoothstep(0.0, 0.028, sun_lvl);
    return a * b;
}

fn hash31(p: vec3<f32>) -> f32 {
    var q = fract(p * vec3<f32>(443.897, 441.423, 437.195));
    q += dot(q, q.yzx + 19.19);
    return fract((q.x + q.y) * q.z);
}

/// Sparse procedural stars (stable in world direction; fades toward horizon).
fn stars_radiance(dir: vec3<f32>) -> vec3<f32> {
    let d = normalize(dir);
    let horiz_fade = smoothstep(-0.06, 0.12, d.y);
    if (horiz_fade < 0.001) {
        return vec3<f32>(0.0);
    }
    let s = 95.0;
    let p = d * s;
    let i = floor(p);
    let f = fract(p) - vec3<f32>(0.5);
    let h = hash31(i);
    if (h < 0.972) {
        return vec3<f32>(0.0);
    }
    let off = hash31(i + vec3<f32>(31.0, 17.0, 43.0)) - vec3<f32>(0.5);
    let dist = length(f - off * 0.82);
    let br = hash31(i + vec3<f32>(11.0, 59.0, 23.0));
    let sparkle = smoothstep(0.22, 0.0, dist) * (0.35 + br * 0.95);
    let tint = mix(vec3<f32>(0.75, 0.82, 1.0), vec3<f32>(1.0, 0.95, 0.88), br);
    return tint * sparkle * 3.8 * horiz_fade;
}

/// Moonlit / starlit hemisphere when scene lighting is minimal.
fn night_sky_radiance(
    dir: vec3<f32>,
    sun_dir: vec3<f32>,
    sun_color: vec3<f32>,
    amb: f32,
    sun_lvl: f32,
) -> vec3<f32> {
    let vy = dir.y;
    let up = clamp(vy * 0.5 + 0.5, 0.0, 1.0);
    // Dark zenith → slightly lighter horizon (airglow); scales with ambient fill
    let amb_floor = 0.035 + amb * 0.55;
    let zen = vec3<f32>(0.006, 0.008, 0.022) * amb_floor;
    let hor = vec3<f32>(0.02, 0.024, 0.045) * amb_floor;
    var rgb = mix(hor, zen, pow(up, 0.65));

    // Tiny moon / sun body from directional light (uses scene sun color)
    let mu = max(dot(dir, sun_dir), 0.0);
    let disk = pow(mu, 2800.0) * sun_lvl * 6.5;
    rgb = rgb + sun_color * disk;

    // Broad forward glow when moon is up (very subtle)
    let glow = sun_color * pow(mu, 12.0) * sun_lvl * 0.045;
    rgb = rgb + glow;

    return rgb;
}

fn sky_atmosphere(dir: vec3<f32>, sun_dir: vec3<f32>, sun_color: vec3<f32>) -> vec3<f32> {
    let sy = clamp(sun_dir.y, 0.0, 1.0);
    let cos_theta = clamp(dot(dir, sun_dir), -1.0, 1.0);
    let mu = max(cos_theta, 0.0);

    let am_sun = sun_air_mass(sy);
    let ext = extinction_rgb(am_sun);
    let sunlight = sun_color * ext;

    // Viewer ray: more air toward the horizon (|dir.y| small) → brighter, desaturated haze band
    let vy = dir.y;
    let horiz_path = 1.0 / max(abs(vy) + 0.1, 0.06);
    let haze_w = clamp(horiz_path * 0.038, 0.0, 1.6);

    // Zenith vs horizon base gradient (before scattering terms)
    let up = clamp(vy * 0.5 + 0.5, 0.0, 1.0);
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
    var base = mix(hor, zen, pow(up, 0.52));

    // Rayleigh-like skylight: shorter wavelengths dominate away from the sun lobe
    let pr = rayleigh_phase(cos_theta);
    let rayleigh_rgb = vec3<f32>(0.2, 0.42, 0.92) * sunlight * pr * (0.1 + 0.26 * sy);

    // Horizontally long view path picks up forward-scattered / aerosol light
    let haze_rgb = vec3<f32>(0.82, 0.76, 0.68) * sunlight * haze_w * (0.28 + 0.72 * (1.0 - sy));

    var rgb = base * (0.42 + 0.58 * sy) + rayleigh_rgb + haze_rgb * 0.62;

    // Mie / corona (forward scattering), sharper when the sun is high
    let mie_exp = mix(5.5, 90.0, sy);
    let mie = sunlight * pow(mu, mie_exp) * (0.28 + 0.55 * (1.0 - sy));
    rgb = rgb + mie;

    // Sun disk (small, bright)
    let disk = smoothstep(0.99925, 0.99982, cos_theta);
    rgb = rgb + sunlight * disk * (3.2 + 2.8 * sy);

    // Clamp: scene-referred HDR but avoid runaway bloom on the disk
    return min(rgb, vec3<f32>(14.0));
}

fn sky_with_scene_lighting(
    dir: vec3<f32>,
    sun_dir: vec3<f32>,
    sun_color: vec3<f32>,
    amb: f32,
    sun_lvl: f32,
) -> vec3<f32> {
    let w = day_sky_blend_weight(amb, sun_lvl);
    let day = sky_atmosphere(dir, sun_dir, sun_color) * day_sky_brightness(amb, sun_lvl);
    let night = night_sky_radiance(dir, sun_dir, sun_color, amb, sun_lvl);
    let sky0 = mix(night, day, w);
    let sw = star_field_weight(amb, sun_lvl);
    return sky0 + stars_radiance(dir) * sw;
}

struct SkyOut {
    @location(0) color: vec4<f32>,
    @location(1) gbuf_n: vec4<f32>,
}

@fragment
fn fs_sky_mrt(in: FullscreenOut) -> SkyOut {
    if (g.light_params.w < 0.5) {
        var out_solid: SkyOut;
        out_solid.color = vec4<f32>(g.bg_color.xyz, 0.0);
        let vn = normalize((g.inv_view * vec4<f32>(0.0, 1.0, 0.0, 0.0)).xyz);
        out_solid.gbuf_n = vec4<f32>(vn * 0.5 + 0.5, 1.0);
        return out_solid;
    }
    let dir = world_ray_dir(in.uv);
    let sun_dir = normalize(g.light_dir.xyz);
    let sc = max(g.sun_color.xyz, vec3<f32>(1e-4));
    let amb = max(g.light_params.x, 0.0);
    let sun_lvl = max(g.light_params.y, 0.0);
    let rgb = sky_with_scene_lighting(dir, sun_dir, sc, amb, sun_lvl);
    var out: SkyOut;
    out.color = vec4<f32>(rgb, 0.0);
    let vn = normalize((g.inv_view * vec4<f32>(0.0, 1.0, 0.0, 0.0)).xyz);
    out.gbuf_n = vec4<f32>(vn * 0.5 + 0.5, 1.0);
    return out;
}
