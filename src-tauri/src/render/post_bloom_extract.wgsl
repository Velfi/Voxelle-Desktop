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

/// Bloom source: only energy above `threshold` (scene-linear HDR). Soft knee avoids a hard cutoff.
/// Emissive voxels (`glow` from MRT) still pass full color so glow materials bloom reliably.
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
    if (glow > 0.5) {
        return vec4<f32>(c.rgb, 1.0);
    }
    // ~sRGB-linear white is 1.0; only clearly HDR or sun-lit surfaces bloom.
    let rgb = bloom_threshold_rgb(c.rgb, 0.74, 0.14);
    return vec4<f32>(rgb, 1.0);
}
