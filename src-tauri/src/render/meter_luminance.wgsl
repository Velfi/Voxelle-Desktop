/// Average scene linear luminance (before exposure/tone map) for auto exposure.
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
fn fs_meter(i: FullscreenOut) -> @location(0) vec4<f32> {
    var sum = 0.0;
    let n = 8.0;
    for (var j = 0; j < 8; j++) {
        for (var k = 0; k < 8; k++) {
            let uv = (vec2<f32>(f32(k), f32(j)) + vec2<f32>(0.5)) / n;
            let c = textureSample(t_hdr, samp_linear, uv).rgb;
            sum += dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
        }
    }
    let lum = sum / 64.0;
    // R32Float attachment: only .r is written; vec4 required by WGSL color output rules.
    return vec4<f32>(lum, 0.0, 0.0, 1.0);
}
