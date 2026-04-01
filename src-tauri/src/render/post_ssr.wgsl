/// Screen-Space Reflections pass.
///
/// Runs after the opaque pass and texture copy, before the transmission pass.
/// Outputs (rgb = reflected colour, a = confidence 0..1) into `ssr_texture`.
/// The transmission pass (`fs_trans`) reads this and blends it via Fresnel.
///
/// Algorithm:
///   1. Reconstruct world position from depth buffer.
///   2. Reconstruct world normal via cross-product of adjacent depth samples
///      (finite-differences). Works well for axis-aligned voxel faces.
///   3. Reflect the view ray around the surface normal.
///   4. March the reflected ray in world space, projecting each step back to
///      screen-space UV to compare against the depth buffer.
///   5. On hit: sample `hdr_opaque`, fade confidence near screen edges and at
///      large march distances to hide the miss boundary.

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
    o.uv  = vec2<f32>(x, y);
    return o;
}

// ── Bindings ─────────────────────────────────────────────────────────────────

struct GlobalState {
    view_proj:       mat4x4<f32>,
    inv_view:        mat4x4<f32>,
    inv_proj:        mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    light_dir:       vec4<f32>,
    cam_pos:         vec4<f32>,
    brick_origin:    vec4<f32>,
    brick_dims:      vec4<f32>,
    /// x = viewport_width, y = viewport_height, z = 1/w, w = 1/h
    screen:          vec4<f32>,
    params:          vec4<f32>,
    light_params:    vec4<f32>,
    sun_color:       vec4<f32>,
    bg_color:        vec4<f32>,
}

@group(0) @binding(0) var<storage, read> g:       GlobalState;
@group(0) @binding(1) var t_depth:                texture_depth_2d;
@group(0) @binding(2) var t_color:                texture_2d<f32>;
@group(0) @binding(3) var samp_linear:            sampler;
@group(0) @binding(4) var samp_depth:             sampler;

struct SsrOpts {
    strength:  f32,
    max_steps: f32,
    thickness: f32,
    enabled:   f32,
}
@group(0) @binding(5) var<uniform> ssr: SsrOpts;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Reconstruct world position from depth at `uv`.
/// Returns w=1 on valid geometry, w=0 for sky.
fn reconstruct_world_pos(uv: vec2<f32>) -> vec4<f32> {
    let depth = textureSample(t_depth, samp_depth, uv);
    if depth >= 0.9999 { return vec4<f32>(0.0); }
    let ndc    = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let view_h = g.inv_proj * ndc;
    let view   = view_h.xyz / view_h.w;
    let world  = (g.inv_view * vec4<f32>(view, 1.0)).xyz;
    return vec4<f32>(world, 1.0);
}

/// Reconstruct world-space surface normal at `uv` by finite-differencing the
/// depth buffer.  Ensures the result faces the camera.
fn reconstruct_normal(uv: vec2<f32>) -> vec3<f32> {
    let dx = vec2<f32>(g.screen.z, 0.0);
    let dy = vec2<f32>(0.0, g.screen.w);

    let p0 = reconstruct_world_pos(uv);
    let pr = reconstruct_world_pos(uv + dx);
    let pd = reconstruct_world_pos(uv + dy);

    if p0.w < 0.5 || pr.w < 0.5 || pd.w < 0.5 { return vec3<f32>(0.0); }

    let t = pr.xyz - p0.xyz;
    let b = pd.xyz - p0.xyz;
    var n = normalize(cross(t, b));

    // Flip if pointing away from the camera.
    let v = normalize(g.cam_pos.xyz - p0.xyz);
    if dot(n, v) < 0.0 { n = -n; }
    return n;
}

// ── Main fragment ─────────────────────────────────────────────────────────────

@fragment
fn fs_ssr(i: FullscreenOut) -> @location(0) vec4<f32> {
    if ssr.enabled < 0.5 { return vec4<f32>(0.0); }

    // Skip sky.
    let depth = textureSample(t_depth, samp_depth, i.uv);
    if depth >= 0.9999 { return vec4<f32>(0.0); }

    let p0 = reconstruct_world_pos(i.uv);
    if p0.w < 0.5 { return vec4<f32>(0.0); }

    let n = reconstruct_normal(i.uv);
    if dot(n, n) < 0.25 { return vec4<f32>(0.0); }

    let v       = normalize(g.cam_pos.xyz - p0.xyz);
    let ray_dir = reflect(-v, n);

    // Don't trace rays pointing away from camera (back-facing).
    if dot(ray_dir, v) < 0.0 { return vec4<f32>(0.0); }

    let max_steps = i32(clamp(ssr.max_steps, 8.0, 64.0));
    // Step size in world units; 0.5 gives sub-voxel precision.
    let step_size = 0.5;

    var hit_color  = vec3<f32>(0.0);
    var confidence = 0.0;

    for (var s = 1; s <= max_steps; s++) {
        let t_val      = f32(s) * step_size;
        let ray_world  = p0.xyz + ray_dir * t_val;

        // Project ray position to screen.
        let clip = g.view_proj * vec4<f32>(ray_world, 1.0);
        if clip.w <= 0.001 { break; }
        let ndc_xyz = clip.xyz / clip.w;
        let ray_uv  = vec2<f32>(ndc_xyz.x * 0.5 + 0.5, -ndc_xyz.y * 0.5 + 0.5);
        let ray_d   = ndc_xyz.z;

        // Stop when marching outside the viewport.
        if any(ray_uv < vec2<f32>(0.005)) || any(ray_uv > vec2<f32>(0.995)) { break; }

        let scene_d = textureSample(t_depth, samp_depth, ray_uv);

        // Hit: ray has gone behind the geometry.
        if ray_d > scene_d + 0.0001 {
            // Thickness test: skip hits where the geometry is much further back
            // than our ray (avoids false hits through thin floors/walls).
            let scene_world = reconstruct_world_pos(ray_uv);
            if scene_world.w > 0.5 {
                let gap = length(scene_world.xyz - ray_world);
                if gap < ssr.thickness {
                    hit_color = textureSample(t_color, samp_linear, ray_uv).rgb;

                    // Fade near screen edges.
                    let edge = min(
                        min(ray_uv.x, 1.0 - ray_uv.x),
                        min(ray_uv.y, 1.0 - ray_uv.y),
                    );
                    let edge_fade = clamp(edge * 8.0, 0.0, 1.0);

                    // Fade reflections that require many steps (long-range misses
                    // tend to be noisier / hit the wrong surface).
                    let step_fade = 1.0 - f32(s) / f32(max_steps);

                    confidence = edge_fade * (0.4 + 0.6 * step_fade);
                }
            }
            break;
        }
    }

    return vec4<f32>(hit_color, confidence * ssr.strength);
}
