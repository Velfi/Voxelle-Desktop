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
var t_blur_src: texture_2d<f32>;

@group(0) @binding(1)
var samp_linear: sampler;

struct PostU {
    blur_dir: vec4<f32>,
}

@group(0) @binding(2)
var<uniform> post: PostU;

@fragment
fn fs_blur(i: FullscreenOut) -> @location(0) vec4<f32> {
    let dir = post.blur_dir.xy;
    // blur_dir.z carries a per-iteration step multiplier (1, 2, 4 …) so successive
    // passes cover exponentially wider radii without extra samples.
    let step = max(post.blur_dir.z, 1.0);
    let dims = textureDimensions(t_blur_src);
    let texel = vec2<f32>(1.0 / max(f32(dims.x), 1.0), 1.0 / max(f32(dims.y), 1.0));
    var sum = vec3<f32>(0.0);
    let w0 = 0.227027;
    let w1 = 0.1945946;
    let w2 = 0.1216216;
    let w3 = 0.054054;
    let w4 = 0.016216;
    sum += textureSample(t_blur_src, samp_linear, i.uv).rgb * w0;
    sum += textureSample(t_blur_src, samp_linear, i.uv + dir * texel * step * 1.0).rgb * w1;
    sum += textureSample(t_blur_src, samp_linear, i.uv + dir * texel * step * 2.0).rgb * w2;
    sum += textureSample(t_blur_src, samp_linear, i.uv + dir * texel * step * 3.0).rgb * w3;
    sum += textureSample(t_blur_src, samp_linear, i.uv + dir * texel * step * 4.0).rgb * w4;
    sum += textureSample(t_blur_src, samp_linear, i.uv - dir * texel * step * 1.0).rgb * w1;
    sum += textureSample(t_blur_src, samp_linear, i.uv - dir * texel * step * 2.0).rgb * w2;
    sum += textureSample(t_blur_src, samp_linear, i.uv - dir * texel * step * 3.0).rgb * w3;
    sum += textureSample(t_blur_src, samp_linear, i.uv - dir * texel * step * 4.0).rgb * w4;
    return vec4<f32>(sum, 1.0);
}
