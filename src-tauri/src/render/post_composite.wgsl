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
var t_ao: texture_2d<f32>;

@group(0) @binding(2)
var t_bloom: texture_2d<f32>;

@group(0) @binding(3)
var samp_linear: sampler;

@fragment
fn fs_composite(i: FullscreenOut) -> @location(0) vec4<f32> {
    let hdr = textureSample(t_hdr, samp_linear, i.uv).rgb;
    let ao = textureSample(t_ao, samp_linear, i.uv).r;
    let blo = textureSample(t_bloom, samp_linear, i.uv).rgb;
    let rgb = hdr * ao + blo * 0.88;
    let mapped = rgb / (rgb + vec3<f32>(1.0));
    let gamma = pow(mapped, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(gamma, 1.0);
}
