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
    tone_mode: u32,
    grain_strength: f32,
    vignette_strength: f32,
    distance_tint_strength: f32,
}

@group(0) @binding(3)
var<uniform> post_opts: PostCompositeOpts;

/// ACES (Narkowicz fit): better shadow/mid contrast than Reinhard; still maps HDR highlights.
fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp(
        (color * (a * color + b)) / (color * (c * color + d) + e),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
}

fn reinhard_v3(c: vec3<f32>) -> vec3<f32> {
    return c / (c + vec3<f32>(1.0));
}

/// Softer mids than raw ACES — rough analogue to Three.js Neutral tone mapping.
fn neutral_tonemap(c: vec3<f32>) -> vec3<f32> {
    return aces_tonemap(c * 0.94);
}

fn linear_to_display_srgb(c: vec3<f32>) -> vec3<f32> {
    return pow(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
}

/// Simple filmic shoulder (not full AgX).
fn agx_like_tonemap(c: vec3<f32>) -> vec3<f32> {
    let x = max(c, vec3<f32>(1e-4));
    let t = x / (x + vec3<f32>(0.155));
    return pow(clamp(t, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.4));
}

fn apply_tone(mode: u32, rgb: vec3<f32>) -> vec3<f32> {
    switch mode {
        case 0u: {
            return neutral_tonemap(rgb);
        }
        case 1u: {
            return aces_tonemap(rgb);
        }
        case 2u: {
            return linear_to_display_srgb(rgb);
        }
        case 3u: {
            return clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
        }
        case 4u: {
            return agx_like_tonemap(rgb);
        }
        case 5u: {
            return linear_to_display_srgb(reinhard_v3(rgb));
        }
        default: {
            return aces_tonemap(rgb);
        }
    }
}

fn hash12(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

@fragment
fn fs_composite(i: FullscreenOut) -> @location(0) vec4<f32> {
    let hdr = textureSample(t_hdr, samp_linear, i.uv).rgb;
    let blo = textureSample(t_bloom, samp_linear, i.uv).rgb;
    // Bloom is thresholded in extract; keep strength modest so mids stay punchy.
    let rgb0 = hdr + blo * 0.42;
    // Scene + sky are linear scene-referred (Rgba16Float). Display-referred → sRGB swapchain.
    var mapped = apply_tone(post_opts.tone_mode, rgb0);
    let g = post_opts.grain_strength;
    if g > 0.001 {
        let n = hash12(i.uv * vec2<f32>(1920.0, 1080.0) + mapped.xy);
        mapped = mapped + (n - 0.5) * g * 0.35;
    }
    let vig = post_opts.vignette_strength;
    if vig > 0.001 {
        let d = distance(i.uv, vec2<f32>(0.5, 0.5)) * 1.414;
        let factor = 1.0 - smoothstep(0.2, 0.95, d) * vig;
        mapped = mapped * factor;
    }
    let dt = post_opts.distance_tint_strength;
    if dt > 0.001 {
        let radial = distance(i.uv, vec2<f32>(0.5, 0.5)) * 1.414;
        let fog_amt = smoothstep(0.15, 1.0, radial) * dt;
        let horizon = vec3<f32>(0.52, 0.58, 0.66);
        mapped = mix(mapped, horizon, fog_amt);
    }
    return vec4<f32>(mapped, 1.0);
}
