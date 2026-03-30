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
var t_depth: texture_depth_2d;

@group(0) @binding(2)
var t_normal: texture_2d<f32>;

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
fn fs_ssao(i: FullscreenOut) -> @location(0) f32 {
    let dims = textureDimensions(t_depth);
    let uv = i.uv;
    let coord = vec2<i32>(i32(uv.x * f32(dims.x)), i32(uv.y * f32(dims.y)));
    let c0 = vec2<i32>(clamp(coord, vec2<i32>(0), vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1)));
    let d = textureLoad(t_depth, c0, 0);
    let lin = linearize_depth(d);
    let n_enc = textureLoad(t_normal, c0, 0).xyz;
    let n = normalize(n_enc * 2.0 - 1.0);
    var occ = 0.0;
    let k = 8;
    for (var j = 0; j < k; j++) {
        let ang = f32(j) * 0.785398 + uv.x * 12.0;
        let off = vec2<i32>(i32(cos(ang) * 14.0), i32(sin(ang) * 14.0));
        let c2 = vec2<i32>(clamp(c0 + off, vec2<i32>(0), vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1)));
        let d2 = textureLoad(t_depth, c2, 0);
        let lin2 = linearize_depth(d2);
        occ += select(0.0, 1.0, (lin2 - lin) > 0.00012);
    }
    let ao = 1.0 - (occ / f32(k)) * 0.85;
    return clamp(ao, 0.25, 1.0);
}
