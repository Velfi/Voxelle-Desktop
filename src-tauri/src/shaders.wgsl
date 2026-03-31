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
    let base = in.color;
    let ndl = max(dot(n, l), 0.0);
    let ndh = max(dot(n, h), 0.0);
    let ndv = max(dot(n, v), 0.0);
    let amb = 0.28;

    let is_metal = in.mat_kind > 0.25 && in.mat_kind < 0.75;
    let is_glow  = in.mat_kind > 0.75 && in.mat_kind < 1.25;

    var rgb: vec3<f32>;
    if (is_metal) {
        let f0 = base * 0.96 + vec3<f32>(0.04);
        let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - ndv, 5.0);
        let spec = fresnel * pow(ndh, 96.0) * 1.8;
        let ambient_refl = base * 0.72 * amb;
        rgb = ambient_refl + base * 0.15 * ndl + spec;
    } else if (is_glow) {
        let emissive = base * 2.8;
        let shape = base * (amb * 0.12 + 0.18 * ndl);
        rgb = emissive + shape;
    } else if (in.mat_kind > 1.5) {
        let spec = pow(ndh, 32.0) * 0.35;
        let tinted = mix(base, vec3<f32>(1.0), 0.15);
        rgb = tinted * (amb + ndl * 0.72) + vec3<f32>(spec);
    } else {
        let spec = pow(ndh, 32.0) * 0.12;
        rgb = base * (amb + ndl * 0.72) + vec3<f32>(spec);
    }
    rgb = rgb / (rgb + vec3<f32>(1.0));
    return vec4<f32>(pow(rgb, vec3<f32>(1.0 / 2.2)), 1.0);
}
