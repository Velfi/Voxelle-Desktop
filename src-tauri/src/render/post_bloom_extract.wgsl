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

@fragment
fn fs_bloom_extract(i: FullscreenOut) -> @location(0) vec4<f32> {
    let c = textureSample(t_hdr, samp_linear, i.uv);
    let lum = dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let glow = c.a;
    let m = select(select(0.0, 1.0, lum > 0.15), 1.0, glow > 0.5);
    return vec4<f32>(c.rgb * m, 1.0);
}
