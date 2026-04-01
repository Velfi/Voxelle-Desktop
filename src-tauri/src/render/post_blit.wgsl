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
var t_src: texture_2d<f32>;

@group(0) @binding(1)
var samp: sampler;

@fragment
fn fs_blit(i: FullscreenOut) -> @location(0) vec4<f32> {
    return textureSample(t_src, samp, i.uv);
}

// Weighted blit: multiplies the sample by a scalar uniform before additive blending.
// Used in the bloom upsample pyramid so coarser levels contribute less than finer ones.
// three scalar pads to match Rust [f32; 3] — vec3<f32> would cause 32-byte layout vs 16-byte buffer
struct BlitWeightU { weight: f32, _pad0: f32, _pad1: f32, _pad2: f32 }
@group(0) @binding(2) var<uniform> bu: BlitWeightU;

@fragment
fn fs_blit_weighted(i: FullscreenOut) -> @location(0) vec4<f32> {
    return textureSample(t_src, samp, i.uv) * bu.weight;
}
