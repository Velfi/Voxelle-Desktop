/// Cold-start background: matches former CSS
/// `radial-gradient(ellipse 120% 80% at 50% 20%, rgb(28,32,42), rgb(10,11,14))`.
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
    light_params: vec4<f32>,
    sun_color: vec4<f32>,
    bg_color: vec4<f32>,
}

@group(0) @binding(0)
var<storage, read> g: GlobalState;

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

fn srgb_channel_to_linear(c: f32) -> f32 {
    if (c <= 0.04045) {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

fn srgb_to_linear3(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        srgb_channel_to_linear(rgb.r),
        srgb_channel_to_linear(rgb.g),
        srgb_channel_to_linear(rgb.b),
    );
}

struct MrtOut {
    @location(0) color: vec4<f32>,
    @location(1) gbuf_n: vec4<f32>,
}

@fragment
fn fs_start_screen_mrt(in: FullscreenOut) -> MrtOut {
    let center = vec2<f32>(0.5, 0.2);
    let d = in.uv - center;
    let r = length(vec2<f32>(d.x / 0.52, d.y / 0.42));
    let t = clamp(r * 1.05, 0.0, 1.0);
    // `g.params.x`: 0 = dark radial, 1 = light paper tones (from UI appearance).
    let k = clamp(g.params.x, 0.0, 1.0);
    let dark_c0 = srgb_to_linear3(vec3<f32>(28.0 / 255.0, 32.0 / 255.0, 42.0 / 255.0));
    let dark_c1 = srgb_to_linear3(vec3<f32>(10.0 / 255.0, 11.0 / 255.0, 14.0 / 255.0));
    let light_c0 = srgb_to_linear3(vec3<f32>(250.0 / 255.0, 246.0 / 255.0, 239.0 / 255.0));
    let light_c1 = srgb_to_linear3(vec3<f32>(232.0 / 255.0, 224.0 / 255.0, 212.0 / 255.0));
    let c0 = mix(dark_c0, light_c0, k);
    let c1 = mix(dark_c1, light_c1, k);
    let rgb = mix(c0, c1, t);
    var out: MrtOut;
    out.color = vec4<f32>(rgb, 0.0);
    let vn = normalize((g.inv_view * vec4<f32>(0.0, 1.0, 0.0, 0.0)).xyz);
    out.gbuf_n = vec4<f32>(vn * 0.5 + 0.5, 1.0);
    return out;
}
