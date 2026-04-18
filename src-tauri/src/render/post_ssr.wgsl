/// Screen-Space Reflections pass (opaque metals).
///
/// Runs after the opaque pass and texture copy, before the OIT pass.
/// Outputs (rgb = reflected colour, a = confidence 0..1) into `ssr_texture`.
/// The OIT composite reads this and blends it onto opaque pixels.
///
/// Algorithm:
///   1. Read world normal + metalness from the GBuffer.
///   2. Skip non-metallic pixels (early-out).
///   3. Reconstruct world position from the depth buffer.
///   4. Reflect the view ray around the surface normal.
///   5. March the reflected ray in world space, projecting each step back to
///      screen-space UV to compare against the depth buffer.
///   6. On hit: sample `hdr_opaque`, fade confidence near screen edges and at
///      large march distances to hide the miss boundary.

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

// ── Bindings ─────────────────────────────────────────────────────────────────

struct GlobalState {
    view_proj:       mat4x4<f32>,
    inv_view:        mat4x4<f32>,
    inv_proj:        mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    light_dir:       vec4<f32>,
    cam_pos:         vec4<f32>,
    brick_origin:    vec4<f32>,
    brick_dims:      vec4<f32>,
    /// x = viewport_width, y = viewport_height, z = 1/w, w = 1/h
    screen:          vec4<f32>,
    params:          vec4<f32>,
    light_params:    vec4<f32>,
    sun_color:       vec4<f32>,
    bg_color:        vec4<f32>,
}

@group(0) @binding(0) var<storage, read> g:       GlobalState;
@group(0) @binding(1) var t_depth:                texture_depth_2d;
@group(0) @binding(2) var t_color:                texture_2d<f32>;
@group(0) @binding(3) var samp_linear:            sampler;
@group(0) @binding(4) var samp_depth:             sampler;

struct SsrOpts {
    strength:  f32,
    max_steps: f32,
    thickness: f32,
    enabled:   f32,
}
@group(0) @binding(5) var<uniform> ssr: SsrOpts;
@group(0) @binding(6) var t_normal: texture_2d<f32>;

// ── Sky env probe ────────────────────────────────────────────────────────────
//
// Mirrors the analytic sky in sky.wgsl (kept in sync: any change there should
// land here too). Metals reflect the actual sky colors — bg_color, sun
// position, day/night, atmosphere — so the env fallback must follow suit
// rather than use a static hemisphere. When SSR doesn't hit an on-screen
// surface, it samples this in the reflection direction.
//
// The functions below are lifted from sky.wgsl. Keeping post_ssr as a single
// self-contained module (no shader-module imports in wgpu) costs duplication
// but guarantees any sky-color change the user makes shows up in reflections.

fn sun_air_mass(sun_y: f32) -> f32 { return 1.0 / max(sun_y, 0.032); }

fn extinction_rgb(air_mass: f32) -> vec3<f32> {
    return vec3<f32>(
        exp(-0.052 * air_mass),
        exp(-0.082 * air_mass),
        exp(-0.128 * air_mass)
    );
}

fn rayleigh_phase(cos_theta: f32) -> f32 {
    return 0.65 * (1.0 + cos_theta * cos_theta);
}

fn day_sky_blend_weight(amb: f32, sun_lvl: f32) -> f32 {
    let k = clamp(amb + sun_lvl, 0.0, 8.0);
    return smoothstep(0.1, 0.82, pow(k, 1.42));
}

fn day_sky_brightness(amb: f32, sun_lvl: f32) -> f32 {
    let e = amb * 0.48 + sun_lvl * 0.52;
    return clamp(0.12 + 0.88 * smoothstep(0.04, 1.35, e), 0.08, 1.35);
}

fn night_sky_radiance(
    dir: vec3<f32>,
    sun_dir: vec3<f32>,
    sun_color: vec3<f32>,
    amb: f32,
    sun_lvl: f32,
) -> vec3<f32> {
    let vy = dir.y;
    let up = clamp(vy * 0.5 + 0.5, 0.0, 1.0);
    let amb_floor = 0.035 + amb * 0.55;
    let zen = vec3<f32>(0.006, 0.008, 0.022) * amb_floor;
    let hor = vec3<f32>(0.02, 0.024, 0.045) * amb_floor;
    var rgb = mix(hor, zen, pow(up, 0.65));
    let mu = max(dot(dir, sun_dir), 0.0);
    let disk = pow(mu, 2800.0) * sun_lvl * 6.5;
    rgb = rgb + sun_color * disk;
    let glow = sun_color * pow(mu, 12.0) * sun_lvl * 0.045;
    return rgb + glow;
}

fn sky_atmosphere(dir: vec3<f32>, sun_dir: vec3<f32>, sun_color: vec3<f32>) -> vec3<f32> {
    let sy = clamp(sun_dir.y, 0.0, 1.0);
    let cos_theta = clamp(dot(dir, sun_dir), -1.0, 1.0);
    let mu = max(cos_theta, 0.0);
    let am_sun = sun_air_mass(sy);
    let ext = extinction_rgb(am_sun);
    let sunlight = sun_color * ext;
    let vy = dir.y;
    let horiz_path = 1.0 / max(abs(vy) + 0.1, 0.06);
    let haze_w = clamp(horiz_path * 0.038, 0.0, 1.6);
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
    let pr = rayleigh_phase(cos_theta);
    let rayleigh_rgb = vec3<f32>(0.2, 0.42, 0.92) * sunlight * pr * (0.1 + 0.26 * sy);
    let haze_rgb = vec3<f32>(0.82, 0.76, 0.68) * sunlight * haze_w * (0.28 + 0.72 * (1.0 - sy));
    var rgb = base * (0.42 + 0.58 * sy) + rayleigh_rgb + haze_rgb * 0.62;
    let mie_exp = mix(5.5, 90.0, sy);
    let mie = sunlight * pow(mu, mie_exp) * (0.28 + 0.55 * (1.0 - sy));
    rgb = rgb + mie;
    let disk = smoothstep(0.99925, 0.99982, cos_theta);
    rgb = rgb + sunlight * disk * (3.2 + 2.8 * sy);
    return min(rgb, vec3<f32>(14.0));
}

/// Sample the sky in a given direction using the same model the sky pass uses.
/// `g.light_params.w` gates analytic-sky vs solid bg_color (matches sky shader).
fn sample_sky_env(dir: vec3<f32>) -> vec3<f32> {
    if (g.light_params.w < 0.5) {
        return g.bg_color.xyz;
    }
    let sun_dir = normalize(g.light_dir.xyz);
    let sc = max(g.sun_color.xyz, vec3<f32>(1e-4));
    let amb = max(g.light_params.x, 0.0);
    let sun_lvl = max(g.light_params.y, 0.0);
    let w = day_sky_blend_weight(amb, sun_lvl);
    let day = sky_atmosphere(dir, sun_dir, sc) * day_sky_brightness(amb, sun_lvl);
    let night = night_sky_radiance(dir, sun_dir, sc, amb, sun_lvl);
    return mix(night, day, w);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Reconstruct world position from depth at `uv`.
/// Returns w=1 on valid geometry, w=0 for sky.
fn reconstruct_world_pos(uv: vec2<f32>) -> vec4<f32> {
    let depth = textureSample(t_depth, samp_depth, uv);
    if depth >= 0.9999 { return vec4<f32>(0.0); }
    let ndc    = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let view_h = g.inv_proj * ndc;
    let view   = view_h.xyz / view_h.w;
    let world  = (g.inv_view * vec4<f32>(view, 1.0)).xyz;
    return vec4<f32>(world, 1.0);
}

// ── Main fragment ─────────────────────────────────────────────────────────────

@fragment
fn fs_ssr(i: FullscreenOut) -> @location(0) vec4<f32> {
    // Read GBuffer: rgb = world normal * 0.5 + 0.5, a = metalness.
    // Non-metal and sky pixels contribute no env reflection.
    let gbuf = textureSample(t_normal, samp_linear, i.uv);
    let metalness = gbuf.a;
    if metalness < 0.5 { return vec4<f32>(0.0); }

    let depth = textureSample(t_depth, samp_depth, i.uv);
    if depth >= 0.9999 { return vec4<f32>(0.0); }

    let p0 = reconstruct_world_pos(i.uv);
    if p0.w < 0.5 { return vec4<f32>(0.0); }

    let n = normalize(gbuf.rgb * 2.0 - 1.0);

    let v       = normalize(g.cam_pos.xyz - p0.xyz);
    let ray_dir = reflect(-v, n);

    // Metallic Fresnel: metals tint reflections with their base color.
    // Read base color from the opaque HDR buffer for the F0 term.
    let base = textureSample(t_color, samp_linear, i.uv).rgb;
    let ndv = max(dot(n, v), 0.0);
    let f0 = base * 0.96 + vec3<f32>(0.04);
    let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - ndv, 5.0);

    // Env fallback: environment colour in the reflection direction. This is
    // the *sole* source of environment-reflection energy for metals — the
    // scene shader used to bake a copy as well, but that double-counted
    // whenever SSR found a real hit.
    //
    // When the analytic sky is enabled the env colour tracks day/night, sun
    // position, and atmospheric haze — all of which can change live. When the
    // sky is disabled the scene has a solid-colour background, so reflections
    // should just sample that colour (no artificial sky/ground split).
    var env_fallback: vec3<f32>;
    if (g.light_params.w < 0.5) {
        // Solid-background mode: the environment is one colour in all
        // directions, so the reflection is that colour.
        env_fallback = g.bg_color.xyz;
    } else {
        let sky_env = sample_sky_env(ray_dir);
        // Dim ambient-tinted ground for rays pointing downward.
        let amb = max(g.light_params.x, 0.0);
        let ground_env = g.bg_color.xyz * 0.12 + vec3<f32>(0.04, 0.045, 0.05) * amb;
        // Smooth blend across the horizon so downward rays sample the ground.
        let horizon_t = smoothstep(-0.12, 0.12, ray_dir.y);
        env_fallback = mix(ground_env, sky_env, horizon_t);
    }

    // Trace the reflection ray in screen space when SSR is enabled and the ray
    // points toward the camera (back-facing rays can't produce valid hits).
    var hit_color  = vec3<f32>(0.0);
    var confidence = 0.0;
    let can_trace = ssr.enabled >= 0.5 && dot(ray_dir, v) >= 0.0;

    if can_trace {
        let max_steps = i32(clamp(ssr.max_steps, 8.0, 64.0));
        let max_dist  = 48.0;
        let step_size = max_dist / f32(max_steps);

        for (var s = 1; s <= max_steps; s++) {
            let t_val      = f32(s) * step_size;
            let ray_world  = p0.xyz + ray_dir * t_val;

            // Project ray position to screen.
            let clip = g.view_proj * vec4<f32>(ray_world, 1.0);
            if clip.w <= 0.001 { break; }
            let ndc_xyz = clip.xyz / clip.w;
            let ray_uv  = vec2<f32>(ndc_xyz.x * 0.5 + 0.5, -ndc_xyz.y * 0.5 + 0.5);
            let ray_d   = ndc_xyz.z;

            // Stop when marching outside the viewport.
            if any(ray_uv < vec2<f32>(0.005)) || any(ray_uv > vec2<f32>(0.995)) { break; }

            let scene_d = textureSample(t_depth, samp_depth, ray_uv);

            // Hit: ray has gone behind the geometry.
            if ray_d > scene_d + 0.0001 {
                // Thickness test: skip hits where the geometry is much further back
                // than our ray (avoids false hits through thin floors/walls).
                let scene_world = reconstruct_world_pos(ray_uv);
                if scene_world.w > 0.5 {
                    let gap = length(scene_world.xyz - ray_world);
                    if gap < ssr.thickness {
                        hit_color = textureSample(t_color, samp_linear, ray_uv).rgb;

                        // Fade near screen edges.
                        let edge = min(
                            min(ray_uv.x, 1.0 - ray_uv.x),
                            min(ray_uv.y, 1.0 - ray_uv.y),
                        );
                        let edge_fade = clamp(edge * 8.0, 0.0, 1.0);

                        // Fade reflections that require many steps (long-range misses
                        // tend to be noisier / hit the wrong surface).
                        let step_fade = 1.0 - f32(s) / f32(max_steps);

                        confidence = edge_fade * (0.4 + 0.6 * step_fade) * ssr.strength;
                    }
                }
                break;
            }
        }
    }

    // Confidence blends the real SSR hit with the env fallback. At c=0 (miss
    // or SSR disabled) we get pure env; at c=1 we get the traced reflection.
    let reflection = mix(env_fallback, hit_color, confidence);

    // Output the Fresnel-weighted reflection as an additive term. The OIT
    // composite adds this to the opaque buffer; it is the only place where
    // env reflection energy enters the metal shading path.
    let reflected = reflection * fresnel;
    return vec4<f32>(reflected, 1.0);
}
