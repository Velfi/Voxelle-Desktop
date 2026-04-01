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

@group(0) @binding(0) var t_accum: texture_2d<f32>;
@group(0) @binding(1) var t_revealage: texture_2d<f32>;
@group(0) @binding(2) var t_opaque: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

@fragment
fn fs_oit_composite(in: FullscreenOut) -> @location(0) vec4<f32> {
    let accum = textureSample(t_accum, samp, in.uv);
    let revealage = textureSample(t_revealage, samp, in.uv).r;
    let opaque = textureSample(t_opaque, samp, in.uv).rgb;

    // No transparent fragments accumulated at this pixel.
    if (accum.a < 1e-4) {
        return vec4<f32>(opaque, 1.0);
    }

    // Weighted average of transparent colors.
    let avg_color = accum.rgb / max(accum.a, 1e-5);

    // Composite: transparent contribution + transmitted opaque background.
    let result = avg_color * (1.0 - revealage) + opaque * revealage;

    return vec4<f32>(result, 1.0);
}
