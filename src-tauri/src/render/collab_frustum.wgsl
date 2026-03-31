// Peer camera frustum wireframe. Uses same @group(0) @binding(0) as scene.
// Extended vertex format: pos + color + alpha for near→far fade.
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

struct VertexIn {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) alpha: f32,
}

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) alpha: f32,
}

struct OpaqueOut {
    @location(0) color: vec4<f32>,
    @location(1) gbuf_n: vec4<f32>,
}

fn preview_tonemap(rgb: vec3<f32>) -> vec3<f32> {
    return rgb / (rgb * vec3<f32>(0.42) + vec3<f32>(1.0));
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip_pos = g.view_proj * vec4<f32>(in.pos, 1.0);
    out.color = in.color;
    out.world_pos = in.pos;
    out.alpha = in.alpha;
    return out;
}

@fragment
fn fs_frustum_front(in: VertexOut) -> OpaqueOut {
    var out: OpaqueOut;
    let c = clamp(in.color, vec3<f32>(0.0), vec3<f32>(1.0));
    let rgb = preview_tonemap(c * 1.15);
    out.color = vec4<f32>(rgb, 0.9 * in.alpha);
    out.gbuf_n = vec4<f32>(0.0);
    return out;
}

@fragment
fn fs_frustum_occluded(in: VertexOut) -> OpaqueOut {
    var out: OpaqueOut;
    let c = clamp(in.color, vec3<f32>(0.0), vec3<f32>(1.0));
    let rgb = preview_tonemap(c * 0.72);
    out.color = vec4<f32>(rgb, 0.82 * in.alpha);
    out.gbuf_n = vec4<f32>(0.0);
    return out;
}
