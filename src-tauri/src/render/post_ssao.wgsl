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

fn hash2(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

/// View-space position from UV and hardware depth [0,1] (matches scene projection + inv_proj).
fn view_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc_x = uv.x * 2.0 - 1.0;
    let ndc_y = 1.0 - uv.y * 2.0;
    let ndc_z = depth * 2.0 - 1.0;
    let clip = vec4<f32>(ndc_x, ndc_y, ndc_z, 1.0);
    let inv = g.inv_proj;
    var v = inv * clip;
    return v.xyz / max(v.w, 1e-6);
}

@fragment
fn fs_ssao(i: FullscreenOut) -> @location(0) f32 {
    let dims = textureDimensions(t_depth);
    let uv = i.uv;
    let px = vec2<f32>(uv.x * f32(dims.x) - 0.5, uv.y * f32(dims.y) - 0.5);
    let c0 = vec2<i32>(
        clamp(i32(floor(px.x + 0.5)), 0, i32(dims.x) - 1),
        clamp(i32(floor(px.y + 0.5)), 0, i32(dims.y) - 1),
    );
    let d_c = textureLoad(t_depth, c0, 0);
    let pos_c = view_position(uv, d_c);
    let n_enc = textureLoad(t_normal, c0, 0).xyz;
    let n_c = normalize(n_enc * 2.0 - 1.0);

    let rnd = hash2(px + vec2<f32>(g.screen.x, g.screen.y)) * 6.2831853;
    let texel = vec2<f32>(1.0 / max(f32(dims.x), 1.0), 1.0 / max(f32(dims.y), 1.0));

    let radius_px = 18.0;
    let rings = 3u;
    let dirs = 8u;
    var occ = 0.0;
    var wsum = 0.0;

    for (var ring = 1u; ring <= rings; ring++) {
        let r = f32(ring) / f32(rings) * radius_px;
        for (var s = 0u; s < dirs; s++) {
            let ang = rnd + f32(s) * 0.78539816;
            let off_f = vec2<f32>(cos(ang), sin(ang)) * r;
            let c1 = vec2<i32>(
                clamp(c0.x + i32(round(off_f.x)), 0, i32(dims.x) - 1),
                clamp(c0.y + i32(round(off_f.y)), 0, i32(dims.y) - 1),
            );
            let d_s = textureLoad(t_depth, c1, 0);
            let uv_s = (vec2<f32>(f32(c1.x) + 0.5, f32(c1.y) + 0.5)) * texel;
            let pos_s = view_position(uv_s, d_s);

            let delta = pos_s - pos_c;
            let dist = length(delta);
            if (dist < 1e-4) {
                continue;
            }
            let d_hat = delta / dist;
            let plane_align = abs(dot(d_hat, n_c));

            // Suppress self-occlusion along nearly tangent directions (flat expanses, coplanar neighbors).
            let tangent_gate = smoothstep(0.08, 0.22, plane_align);

            let lin_c = linearize_depth(d_c);
            let lin_s = linearize_depth(d_s);
            let dz = lin_s - lin_c;
            let range_f = smoothstep(0.0, 0.012, abs(dz));

            let bias = 0.00035 + 0.00008 * f32(ring);
            let depth_occ = smoothstep(bias, bias + 0.0045, dz);

            let w = (1.0 / f32(ring)) * tangent_gate * range_f;
            occ += depth_occ * w;
            wsum += w;
        }
    }

    var ao = 1.0;
    if (wsum > 1e-5) {
        ao = 1.0 - clamp((occ / wsum) * 0.82, 0.0, 0.75);
    }
    return clamp(ao, 0.28, 1.0);
}
