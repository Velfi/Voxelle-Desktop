struct GlobalState {
    view_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    light_dir: vec4<f32>,
    cam_pos: vec4<f32>,
    brick_origin: vec4<f32>,
    brick_dims: vec4<f32>,
    screen: vec4<f32>,
    params: vec4<f32>,
}

@group(0) @binding(0)
var<storage, read> g: GlobalState;

@group(0) @binding(1)
var t_ao: texture_2d<f32>;

@group(0) @binding(2)
var t_depth: texture_depth_2d;

@group(0) @binding(3)
var samp_linear: sampler;

struct PostU {
    blur_dir: vec4<f32>,
}

@group(0) @binding(4)
var<uniform> post: PostU;

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

fn linearize_depth(d: f32) -> f32 {
    let n = g.params.w;
    let f = 5000.0;
    return (2.0 * n) / (f + n - d * (f - n));
}

@fragment
fn fs_ao_blur(i: FullscreenOut) -> @location(0) f32 {
    let dir = post.blur_dir.xy;
    let dims = textureDimensions(t_ao);
    let ddim = textureDimensions(t_depth);
    let texel = vec2<f32>(1.0 / max(f32(dims.x), 1.0), 1.0 / max(f32(dims.y), 1.0));
    let px = vec2<i32>(
        i32(clamp(i.uv.x * f32(ddim.x), 0.0, f32(ddim.x) - 1.0)),
        i32(clamp(i.uv.y * f32(ddim.y), 0.0, f32(ddim.y) - 1.0)),
    );
    let d_c = textureLoad(t_depth, px, 0);
    let lin_c = linearize_depth(d_c);
    let k_depth = 1850.0;

    let w0 = 0.227027;
    let w1 = 0.1945946;
    let w2 = 0.1216216;
    let w3 = 0.054054;
    let w4 = 0.016216;

    var acc = 0.0;
    var wsum = 0.0;

    // center
    acc += textureSample(t_ao, samp_linear, i.uv).r * w0;
    wsum += w0;

    // +1 / -1
    var uv_p = i.uv + dir * texel * 1.0;
    var uv_m = i.uv - dir * texel * 1.0;
    var p_p = vec2<i32>(
        i32(clamp(uv_p.x * f32(ddim.x), 0.0, f32(ddim.x) - 1.0)),
        i32(clamp(uv_p.y * f32(ddim.y), 0.0, f32(ddim.y) - 1.0)),
    );
    var p_m = vec2<i32>(
        i32(clamp(uv_m.x * f32(ddim.x), 0.0, f32(ddim.x) - 1.0)),
        i32(clamp(uv_m.y * f32(ddim.y), 0.0, f32(ddim.y) - 1.0)),
    );
    var wp = w1 * exp(-abs(linearize_depth(textureLoad(t_depth, p_p, 0)) - lin_c) * k_depth);
    var wm = w1 * exp(-abs(linearize_depth(textureLoad(t_depth, p_m, 0)) - lin_c) * k_depth);
    acc += textureSample(t_ao, samp_linear, uv_p).r * wp;
    acc += textureSample(t_ao, samp_linear, uv_m).r * wm;
    wsum += wp + wm;

    // ±2
    uv_p = i.uv + dir * texel * 2.0;
    uv_m = i.uv - dir * texel * 2.0;
    p_p = vec2<i32>(
        i32(clamp(uv_p.x * f32(ddim.x), 0.0, f32(ddim.x) - 1.0)),
        i32(clamp(uv_p.y * f32(ddim.y), 0.0, f32(ddim.y) - 1.0)),
    );
    p_m = vec2<i32>(
        i32(clamp(uv_m.x * f32(ddim.x), 0.0, f32(ddim.x) - 1.0)),
        i32(clamp(uv_m.y * f32(ddim.y), 0.0, f32(ddim.y) - 1.0)),
    );
    wp = w2 * exp(-abs(linearize_depth(textureLoad(t_depth, p_p, 0)) - lin_c) * k_depth);
    wm = w2 * exp(-abs(linearize_depth(textureLoad(t_depth, p_m, 0)) - lin_c) * k_depth);
    acc += textureSample(t_ao, samp_linear, uv_p).r * wp;
    acc += textureSample(t_ao, samp_linear, uv_m).r * wm;
    wsum += wp + wm;

    // ±3
    uv_p = i.uv + dir * texel * 3.0;
    uv_m = i.uv - dir * texel * 3.0;
    p_p = vec2<i32>(
        i32(clamp(uv_p.x * f32(ddim.x), 0.0, f32(ddim.x) - 1.0)),
        i32(clamp(uv_p.y * f32(ddim.y), 0.0, f32(ddim.y) - 1.0)),
    );
    p_m = vec2<i32>(
        i32(clamp(uv_m.x * f32(ddim.x), 0.0, f32(ddim.x) - 1.0)),
        i32(clamp(uv_m.y * f32(ddim.y), 0.0, f32(ddim.y) - 1.0)),
    );
    wp = w3 * exp(-abs(linearize_depth(textureLoad(t_depth, p_p, 0)) - lin_c) * k_depth);
    wm = w3 * exp(-abs(linearize_depth(textureLoad(t_depth, p_m, 0)) - lin_c) * k_depth);
    acc += textureSample(t_ao, samp_linear, uv_p).r * wp;
    acc += textureSample(t_ao, samp_linear, uv_m).r * wm;
    wsum += wp + wm;

    // ±4
    uv_p = i.uv + dir * texel * 4.0;
    uv_m = i.uv - dir * texel * 4.0;
    p_p = vec2<i32>(
        i32(clamp(uv_p.x * f32(ddim.x), 0.0, f32(ddim.x) - 1.0)),
        i32(clamp(uv_p.y * f32(ddim.y), 0.0, f32(ddim.y) - 1.0)),
    );
    p_m = vec2<i32>(
        i32(clamp(uv_m.x * f32(ddim.x), 0.0, f32(ddim.x) - 1.0)),
        i32(clamp(uv_m.y * f32(ddim.y), 0.0, f32(ddim.y) - 1.0)),
    );
    wp = w4 * exp(-abs(linearize_depth(textureLoad(t_depth, p_p, 0)) - lin_c) * k_depth);
    wm = w4 * exp(-abs(linearize_depth(textureLoad(t_depth, p_m, 0)) - lin_c) * k_depth);
    acc += textureSample(t_ao, samp_linear, uv_p).r * wp;
    acc += textureSample(t_ao, samp_linear, uv_m).r * wm;
    wsum += wp + wm;

    return acc / max(wsum, 1e-5);
}
