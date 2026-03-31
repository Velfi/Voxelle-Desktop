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

@group(0) @binding(0)
var t_hdr: texture_2d<f32>;

@group(0) @binding(1)
var t_bloom: texture_2d<f32>;

@group(0) @binding(2)
var samp_linear: sampler;

struct PostCompositeOpts {
    // Row 0
    tone_mode: u32,
    transparent_bg: f32,
    exposure_ev: f32,
    time_seconds: f32,
    // Row 1: vignette + grain
    vignette_strength: f32,
    grain_enabled: f32,
    grain_strength: f32,
    grain_animated: f32,
    // Row 2: grain continued
    grain_speed: f32,
    grain_colorful: f32,
    _pad2a: f32,
    _pad2b: f32,
    // Row 3: atmosphere
    atm_enabled: f32,
    atm_thickness: f32,
    atm_density: f32,
    atm_spatial_mode: f32,
    // Row 4: atm color + mode
    atm_color_r: f32,
    atm_color_g: f32,
    atm_color_b: f32,
    atm_mode: f32,
    // Row 5: atm plane
    atm_plane_nx: f32,
    atm_plane_ny: f32,
    atm_plane_nz: f32,
    atm_plane_c: f32,
    // Row 6: atm height + drift
    atm_height_bias: f32,
    atm_height_falloff: f32,
    atm_drift_enabled: f32,
    atm_drift_amount: f32,
    // Row 7: drift continued
    atm_drift_scale: f32,
    atm_drift_speed: f32,
    _pad7a: f32,
    _pad7b: f32,
    // Row 8: distance tint
    dt_enabled: f32,
    dt_near_dist: f32,
    dt_far_dist: f32,
    dt_strength: f32,
    // Row 9-11: dt colors
    dt_near_r: f32,
    dt_near_g: f32,
    dt_near_b: f32,
    _pad9: f32,
    dt_mid_r: f32,
    dt_mid_g: f32,
    dt_mid_b: f32,
    _pad10: f32,
    dt_far_r: f32,
    dt_far_g: f32,
    dt_far_b: f32,
    _pad11: f32,
    // Row 12: sun shafts
    ss_enabled: f32,
    ss_strength: f32,
    ss_decay: f32,
    ss_density: f32,
    // Row 13: sun shafts continued
    ss_weight: f32,
    ss_samples: f32,
    ss_sun_uv_x: f32,
    ss_sun_uv_y: f32,
}

@group(0) @binding(3)
var<uniform> po: PostCompositeOpts;

@group(0) @binding(4)
var t_depth: texture_depth_2d;

@group(0) @binding(5)
var samp_depth: sampler;

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

@group(0) @binding(6)
var<storage, read> g: GlobalState;

// ── Tone mapping ────────────────────────────────────────────────────

fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn reinhard_v3(c: vec3<f32>) -> vec3<f32> { return c / (c + vec3<f32>(1.0)); }

fn neutral_tonemap(c: vec3<f32>) -> vec3<f32> { return aces_tonemap(c * 0.94); }

fn linear_to_display_srgb(c: vec3<f32>) -> vec3<f32> {
    return pow(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
}

fn agx_like_tonemap(c: vec3<f32>) -> vec3<f32> {
    let x = max(c, vec3<f32>(1e-4));
    let t = x / (x + vec3<f32>(0.155));
    return pow(clamp(t, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.4));
}

fn apply_tone(mode: u32, rgb: vec3<f32>) -> vec3<f32> {
    switch mode {
        case 0u: { return neutral_tonemap(rgb); }
        case 1u: { return aces_tonemap(rgb); }
        case 2u: { return linear_to_display_srgb(rgb); }
        case 3u: { return clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)); }
        case 4u: { return agx_like_tonemap(rgb); }
        case 5u: { return linear_to_display_srgb(reinhard_v3(rgb)); }
        default: { return aces_tonemap(rgb); }
    }
}

// ── Hashing / noise ─────────────────────────────────────────────────

fn hash12(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn hash12b(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(269.5, 183.3))) * 43758.5453);
}

fn hash12c(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(419.2, 371.9))) * 43758.5453);
}

// ── World position from depth ───────────────────────────────────────

fn world_pos_from_uv(uv: vec2<f32>) -> vec4<f32> {
    let depth = textureSample(t_depth, samp_depth, uv);
    if depth >= 0.9999 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0); // sky
    }
    // wgpu depth is 0..1 (reversed-Z: near=1, far=0 by convention, or near=0, far=1).
    // NDC: x,y in [-1,1], z in [0,1].
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let view_h = g.inv_proj * ndc;
    let view = view_h.xyz / view_h.w;
    let world = (g.inv_view * vec4<f32>(view, 1.0)).xyz;
    return vec4<f32>(world, 1.0); // w=1 means valid geometry
}

// ── Atmosphere fog ──────────────────────────────────────────────────

fn apply_atmosphere(color: vec3<f32>, wp: vec4<f32>) -> vec3<f32> {
    if po.atm_enabled < 0.5 || wp.w < 0.5 { return color; }

    let fog_color = vec3<f32>(po.atm_color_r, po.atm_color_g, po.atm_color_b);
    let cam = g.cam_pos.xyz;
    let thickness = max(po.atm_thickness, 0.1);
    let density = po.atm_density;
    var fog_factor: f32 = 0.0;

    if po.atm_spatial_mode > 0.5 {
        // Aerial mode: exponential distance fog
        let view_dist = length(wp.xyz - cam);
        fog_factor = 1.0 - exp(-view_dist / thickness);
    } else {
        // Plane mode
        let n = vec3<f32>(po.atm_plane_nx, po.atm_plane_ny, po.atm_plane_nz);
        let n_len = length(n);
        if n_len < 0.001 {
            return color; // no plane set
        }
        let sd = dot(n, wp.xyz) + po.atm_plane_c;
        let softness = max(0.5, thickness * 0.15);
        if po.atm_mode > 0.5 {
            // positiveSide
            fog_factor = smoothstep(-softness, 0.0, sd) * (1.0 - exp(-max(0.0, sd) / thickness));
        } else {
            // slab
            fog_factor = exp(-(sd * sd) / (thickness * thickness));
        }
    }

    fog_factor = fog_factor * density;

    // Height modulation
    let hf = max(1.0, po.atm_height_falloff);
    let height_mod = 0.65 + 0.35 * exp(-abs(wp.y - po.atm_height_bias) / hf);
    fog_factor = fog_factor * height_mod;

    // Drift noise
    if po.atm_drift_enabled > 0.5 {
        let drift = sin((wp.x + wp.z) * po.atm_drift_scale + po.time_seconds * po.atm_drift_speed)
                  * po.atm_drift_amount * density * 0.35;
        fog_factor = fog_factor + drift;
    }

    fog_factor = clamp(fog_factor, 0.0, 1.0);
    return mix(color, fog_color, fog_factor * 1.12);
}

// ── Distance tint ───────────────────────────────────────────────────

fn apply_distance_tint(color: vec3<f32>, wp: vec4<f32>) -> vec3<f32> {
    if po.dt_enabled < 0.5 || wp.w < 0.5 { return color; }

    let cam = g.cam_pos.xyz;
    let view_dist = length(wp.xyz - cam);
    let near_t = clamp(view_dist / max(po.dt_near_dist, 0.01), 0.0, 1.0);
    let far_span = max(1.0, po.dt_far_dist - po.dt_near_dist);
    let far_t = clamp((view_dist - po.dt_near_dist) / far_span, 0.0, 1.0);
    let far_t_grade = pow(far_t, 1.28);

    let near_col = vec3<f32>(po.dt_near_r, po.dt_near_g, po.dt_near_b);
    let mid_col = vec3<f32>(po.dt_mid_r, po.dt_mid_g, po.dt_mid_b);
    let far_col = vec3<f32>(po.dt_far_r, po.dt_far_g, po.dt_far_b);

    let tint_a = mix(near_col, mid_col, near_t);
    let tint_b = mix(mid_col, far_col, far_t_grade);
    let tint = mix(tint_a, tint_b, far_t_grade);

    return mix(color, tint, po.dt_strength);
}

// ── Grain ───────────────────────────────────────────────────────────

fn apply_grain(color: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    if po.grain_enabled < 0.5 { return color; }

    var seed_uv = uv * vec2<f32>(1920.0, 1080.0);
    if po.grain_animated > 0.5 {
        seed_uv = seed_uv + vec2<f32>(po.time_seconds * po.grain_speed * 137.0, po.time_seconds * po.grain_speed * 59.0);
    }

    let strength = po.grain_strength * 1.14;
    if po.grain_colorful > 0.5 {
        let nr = hash12(seed_uv) - 0.5;
        let ng = hash12b(seed_uv) - 0.5;
        let nb = hash12c(seed_uv) - 0.5;
        return color + vec3<f32>(nr, ng, nb) * strength;
    } else {
        let n = hash12(seed_uv) - 0.5;
        return color + vec3<f32>(n, n, n) * strength;
    }
}

// ── Sun shafts (ray-marched) ────────────────────────────────────────

fn apply_sun_shafts(color: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    if po.ss_enabled < 0.5 { return color; }

    let sun_uv = vec2<f32>(po.ss_sun_uv_x, po.ss_sun_uv_y);
    let num_samples = i32(po.ss_samples);
    let delta_uv = (sun_uv - uv) / f32(num_samples);

    var sample_uv = uv;
    var illumination: f32 = 0.0;
    var decay_accum: f32 = 1.0;
    var current_density = po.ss_density;

    for (var i = 0; i < num_samples; i = i + 1) {
        sample_uv = sample_uv + delta_uv;
        let clamped = clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(1.0));
        if any(clamped != sample_uv) { break; }

        let d = textureSample(t_depth, samp_depth, sample_uv);
        // Sky pixels (depth near far plane) contribute light
        let is_sky = select(0.0, 1.0, d >= 0.9992);

        illumination = illumination + is_sky * decay_accum * po.ss_weight * current_density;
        decay_accum = decay_accum * po.ss_decay;
        current_density = current_density * po.ss_decay;
    }

    // Radial falloff from sun position
    let dist_to_sun = distance(uv, sun_uv);
    let radial_falloff = exp(-dist_to_sun * 1.38);

    let shaft_color = vec3<f32>(0.35, 0.32, 0.22);
    return color + shaft_color * illumination * po.ss_strength * radial_falloff;
}

// ── Vignette ────────────────────────────────────────────────────────

fn apply_vignette(color: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    let vig = po.vignette_strength;
    if vig < 0.001 { return color; }
    let d = distance(uv, vec2<f32>(0.5, 0.5)) * 1.414;
    let factor = 1.0 - smoothstep(0.2, 0.95, d) * vig;
    return color * factor;
}

// ── Main fragment ───────────────────────────────────────────────────

@fragment
fn fs_composite(i: FullscreenOut) -> @location(0) vec4<f32> {
    let hdr = textureSample(t_hdr, samp_linear, i.uv).rgb;
    let blo = textureSample(t_bloom, samp_linear, i.uv).rgb;
    var rgb0 = (hdr + blo * 0.42) * exp2(po.exposure_ev);
    let pre_energy = max(max(rgb0.r, rgb0.g), rgb0.b);
    var mapped = apply_tone(po.tone_mode, rgb0);

    // Reconstruct world position for depth-aware effects
    let wp = world_pos_from_uv(i.uv);

    // Mood effects (order matches web)
    mapped = apply_atmosphere(mapped, wp);
    mapped = apply_distance_tint(mapped, wp);
    mapped = apply_grain(mapped, i.uv);
    mapped = apply_sun_shafts(mapped, i.uv);
    mapped = apply_vignette(mapped, i.uv);

    if po.transparent_bg > 0.5 {
        let a = clamp(pre_energy * 14.0, 0.0, 1.0);
        return vec4<f32>(mapped * a, a);
    }
    return vec4<f32>(mapped, 1.0);
}
