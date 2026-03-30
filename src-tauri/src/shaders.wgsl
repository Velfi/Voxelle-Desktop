struct Uniforms {
    view_proj: mat4x4<f32>,
    light_dir: vec4<f32>,
    cam_pos: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) mat_kind: f32,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) mat_kind: f32,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world = in.position;
    out.clip_pos = u.view_proj * vec4<f32>(world, 1.0);
    out.world_pos = world;
    out.normal = in.normal;
    out.color = in.color;
    out.mat_kind = in.mat_kind;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let l = normalize(-u.light_dir.xyz);
    let v = normalize(u.cam_pos.xyz - in.world_pos);
    let h = normalize(l + v);
    var base = in.color;
    var glow = 0.0;
    var spec = 0.0;
    if in.mat_kind > 0.5 && in.mat_kind < 1.5 {
        glow = 0.6;
    } else if in.mat_kind > 1.5 {
        spec = 0.35;
        base = mix(base, vec3<f32>(1.0), 0.15);
    } else {
        spec = 0.12;
    }
    let ndl = max(dot(n, l), 0.0);
    let amb = 0.28;
    let diff = ndl * 0.72;
    let spec_term = pow(max(dot(n, h), 0.0), 32.0) * spec;
    var rgb = base * (amb + diff) + vec3<f32>(spec_term) + base * glow;
    rgb = rgb / (rgb + vec3<f32>(1.0));
    return vec4<f32>(pow(rgb, vec3<f32>(1.0 / 2.2)), 1.0);
}
