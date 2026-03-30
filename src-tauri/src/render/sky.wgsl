/// Fullscreen sky fill: matches web `HemisphereLight` colors + soft horizon (sky sphere ~0x9ec8f0).
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

const SKY_ZENITH: vec3<f32> = vec3<f32>(0.722, 0.831, 0.910);
const SKY_GROUND: vec3<f32> = vec3<f32>(0.290, 0.333, 0.408);
const SKY_HORIZON: vec3<f32> = vec3<f32>(0.620, 0.784, 0.941);

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

fn world_ray_dir(uv: vec2<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let clip = vec4<f32>(ndc, 1.0, 1.0);
    var view = g.inv_proj * clip;
    view = vec4<f32>(view.xyz / max(view.w, 1e-6), 1.0);
    let world = (g.inv_view * vec4<f32>(view.xyz, 0.0)).xyz;
    return normalize(world);
}

struct SkyOut {
    @location(0) color: vec4<f32>,
    @location(1) gbuf_n: vec4<f32>,
}

@fragment
fn fs_sky_mrt(in: FullscreenOut) -> SkyOut {
    let dir = world_ray_dir(in.uv);
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    var rgb = mix(SKY_GROUND, SKY_ZENITH, t);
    let h = 1.0 - abs(dir.y);
    rgb = mix(rgb, SKY_HORIZON, h * h * 0.35);
    var out: SkyOut;
    out.color = vec4<f32>(rgb, 0.0);
    let vn = normalize((g.inv_view * vec4<f32>(0.0, 1.0, 0.0, 0.0)).xyz);
    out.gbuf_n = vec4<f32>(vn * 0.5 + 0.5, 1.0);
    return out;
}
