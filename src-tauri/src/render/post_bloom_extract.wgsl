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
var samp_linear: sampler;

struct BloomExtractU {
    /// Current exposure in EV stops (matches post_composite_opts.exposure_ev).
    exposure_ev: f32,
    // three scalar pads to match Rust [f32; 3] — vec3<f32> would cause 32-byte layout
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(2)
var<uniform> ex: BloomExtractU;

/// Bloom source: only energy above `threshold` (scene-linear HDR). Soft knee avoids a hard cutoff.
fn bloom_threshold_rgb(c: vec3<f32>, threshold: f32, knee: f32) -> vec3<f32> {
    let lum = dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    if (lum <= threshold) {
        return vec3<f32>(0.0);
    }
    let over = lum - threshold;
    let soft = min(over, 2.0 * knee);
    let soft_q = soft * soft / (4.0 * knee + 1e-5);
    let contrib_lum = max(soft_q, over - knee);
    let scale = contrib_lum / max(lum, 1e-5);
    return c.rgb * scale;
}

@fragment
fn fs_bloom_extract(i: FullscreenOut) -> @location(0) vec4<f32> {
    let c = textureSample(t_hdr, samp_linear, i.uv);
    let glow = c.a;
    // Emissive voxels: bloom proportional to luminance starting from 0 (no hard bypass).
    // A dim glow voxel blooms softly; a bright one blooms strongly. More physical than
    // passing full color unconditionally.
    if (glow > 0.5) {
        let rgb = bloom_threshold_rgb(c.rgb, 0.0, 0.2);
        return vec4<f32>(rgb, 1.0);
    }
    // Scale threshold by exposure: at higher EV the sensor saturates at lower scene values,
    // so dimmer surfaces bloom. Clamp to [0.05, 2.0] to avoid degenerate extremes.
    let effective_threshold = clamp(0.74 * exp2(-ex.exposure_ev), 0.05, 2.0);
    let rgb = bloom_threshold_rgb(c.rgb, effective_threshold, 0.14);
    return vec4<f32>(rgb, 1.0);
}
