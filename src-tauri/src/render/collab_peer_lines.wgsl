// Peer look direction: line from eye → orbit target. Uses same @group(0) @binding(0) as scene.
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
}

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
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
    return out;
}

@fragment
fn fs_collab_line_front(in: VertexOut) -> OpaqueOut {
    var out: OpaqueOut;
    let c = clamp(in.color, vec3<f32>(0.0), vec3<f32>(1.0));
    let rgb = preview_tonemap(c * 1.15);
    out.color = vec4<f32>(rgb, 0.9);
    out.gbuf_n = vec4<f32>(0.0);
    return out;
}

@fragment
fn fs_collab_line_occluded(in: VertexOut) -> OpaqueOut {
    var out: OpaqueOut;
    let c = clamp(in.color, vec3<f32>(0.0), vec3<f32>(1.0));
    let rgb = preview_tonemap(c * 0.72);
    out.color = vec4<f32>(rgb, 0.82);
    out.gbuf_n = vec4<f32>(0.0);
    return out;
}

/// Selection transform gizmo — front pass: full-brightness axis colors.
@fragment
fn fs_gizmo_front(in: VertexOut) -> OpaqueOut {
    var out: OpaqueOut;
    let c = clamp(in.color, vec3<f32>(0.0), vec3<f32>(1.0));
    out.color = vec4<f32>(c, 1.0);
    out.gbuf_n = vec4<f32>(0.0);
    return out;
}

/// Selection transform gizmo — occluded pass: dimmed, semi-transparent ghost.
@fragment
fn fs_gizmo_occluded(in: VertexOut) -> OpaqueOut {
    var out: OpaqueOut;
    let c = clamp(in.color, vec3<f32>(0.0), vec3<f32>(1.0));
    out.color = vec4<f32>(c * 0.6, 0.35);
    out.gbuf_n = vec4<f32>(0.0);
    return out;
}

/// Voxel grid borders: darkening stroke; fades for moiré (screen), grazing view, and **camera distance**
/// (far + zoomed out → coplanar z-fight with opaque mesh; fade before it reads as shimmer).
@fragment
fn fs_grid_border_line(in: VertexOut) -> OpaqueOut {
    var out: OpaqueOut;
    let dark = vec3<f32>(0.0, 0.0, 0.0);
    let rgb = preview_tonemap(dark);

    let inv_w = 1.0 / max(abs(in.clip_pos.w), 1e-5);
    let ndc = vec2<f32>(in.clip_pos.x * inv_w, in.clip_pos.y * inv_w);
    let ndc_grad = max(fwidth(ndc.x), fwidth(ndc.y));
    let screen_fade = smoothstep(0.0001, 0.005, ndc_grad);

    let dist = length(g.cam_pos.xyz - in.world_pos);
    // Earlier / tighter than before: far camera → thin lines fight the depth buffer.
    let dist_fade = 1.0 - smoothstep(48.0, 150.0, dist);

    let a = 0.72 * screen_fade * dist_fade;
    out.color = vec4<f32>(rgb, a);
    out.gbuf_n = vec4<f32>(0.0);
    return out;
}
