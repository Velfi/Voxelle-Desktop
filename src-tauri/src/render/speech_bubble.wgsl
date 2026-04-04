// Screen-space speech bubble / floating note overlay.
//
// The vertex stage emits a fullscreen triangle; the SDF fragment stage
// shapes the output into a rounded-rectangle body with a triangular tail.
// Rendered on the swapchain surface with alpha blending (no depth test).
//
// Uniform coordinates are in swapchain pixels, Y-down, origin top-left.

struct BubbleUniforms {
    /// Bubble body rect: x, y (top-left), w, h — swapchain pixels.
    rect: vec4<f32>,
    /// Tail tip in swapchain pixels (absolute); points toward the anchor.
    tail_tip: vec2<f32>,
    /// Horizontal shake translation applied to the body (pixels).
    shake_x: f32,
    /// Body corner radius (pixels).
    corner_r: f32,
    /// Background fill colour (linear RGBA).
    bg_color: vec4<f32>,
    /// Border stroke colour (linear RGBA).
    border_color: vec4<f32>,
    /// Inner border half-width (pixels).
    border_w: f32,
    // Three f32 pads keep alignment at 4 bytes each (vec3 would jump to align-16, bloating to 96 B).
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0)
var<uniform> u: BubbleUniforms;

struct VOut {
    @builtin(position) pos: vec4<f32>,
}

// ── Vertex ────────────────────────────────────────────────────────────────────

/// Fullscreen triangle — scissor rect is set by the CPU to the bubble AABB
/// so only the relevant region is shaded.
@vertex
fn vs_bubble(@builtin(vertex_index) vi: u32) -> VOut {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return VOut(vec4<f32>(corners[vi], 0.0, 1.0));
}

// ── SDF helpers ───────────────────────────────────────────────────────────────

fn sdf_round_rect(p: vec2<f32>, half_ext: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - half_ext + vec2<f32>(r, r);
    return length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - r;
}

fn sdf_triangle(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
    let e0 = b - a;
    let e1 = c - b;
    let e2 = a - c;
    let v0 = p - a;
    let v1 = p - b;
    let v2 = p - c;
    let pq0 = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
    let pq1 = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
    let pq2 = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);
    let s = sign(e0.x * e2.y - e0.y * e2.x);
    let d2 = min(
        min(
            vec2<f32>(dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x)),
            vec2<f32>(dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x)),
        ),
        vec2<f32>(dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x)),
    );
    return -sqrt(d2.x) * sign(d2.y);
}

// ── Fragment ──────────────────────────────────────────────────────────────────

@fragment
fn fs_bubble(in: VOut) -> @location(0) vec4<f32> {
    // Fragment centre in swapchain pixels (Y-down, origin top-left).
    let px = in.pos.xy;

    // Shake displaces the body only; the tail tip stays anchored.
    let bx = u.rect.x + u.shake_x;
    let by = u.rect.y;
    let bw = u.rect.z;
    let bh = u.rect.w;

    let center = vec2<f32>(bx + bw * 0.5, by + bh * 0.5);
    let half   = vec2<f32>(bw * 0.5, bh * 0.5);

    // Body SDF.
    let body_d = sdf_round_rect(px - center, half, u.corner_r);

    // Tail: wedge from the bubble edge toward the anchor point.
    // The CPU already clamps the tip to MAX_TAIL_LEN beyond the body edge,
    // so we just use the uniform value directly.
    let tip = u.tail_tip;

    // Signed distances from tip to each edge (positive = outside).
    let d_left   = bx - tip.x;
    let d_right  = tip.x - (bx + bw);
    let d_top    = by - tip.y;
    let d_bottom = tip.y - (by + bh);

    // Pick the edge closest to the tail tip.
    let max_h = max(d_left, d_right);
    let max_v = max(d_top, d_bottom);

    var t_a: vec2<f32>;
    var t_b: vec2<f32>;

    // Unshaken body x for stable tail length (so base width doesn't jitter with sway/shake).
    let rest_bx = bx - u.shake_x;

    if max_h > max_v {
        // Tail exits left or right — base runs vertically.
        let edge_y = clamp(tip.y, by + u.corner_r, by + bh - u.corner_r);
        let is_left = f32(d_left > d_right);
        let edge_x = bx * is_left + (bx + bw) * (1.0 - is_left);
        let rest_edge_x = rest_bx * is_left + (rest_bx + bw) * (1.0 - is_left);
        let tail_len = abs(tip.x - rest_edge_x);
        let base_half = clamp(tail_len * 0.35, 8.0, 28.0);
        t_a = vec2<f32>(edge_x, edge_y - base_half);
        t_b = vec2<f32>(edge_x, edge_y + base_half);
    } else {
        // Tail exits top or bottom — base runs horizontally.
        let edge_x = clamp(tip.x, bx + u.corner_r, bx + bw - u.corner_r);
        let is_top = f32(d_top > d_bottom);
        let edge_y = by * is_top + (by + bh) * (1.0 - is_top);
        let tail_len = abs(tip.y - edge_y);
        let base_half = clamp(tail_len * 0.35, 8.0, 28.0);
        t_a = vec2<f32>(edge_x - base_half, edge_y);
        t_b = vec2<f32>(edge_x + base_half, edge_y);
    }

    let tail_d = sdf_triangle(px, t_a, t_b, tip);

    // Union of body and tail.
    let d = min(body_d, tail_d);

    // Early-out for pixels clearly outside (saves bandwidth).
    if d > 2.0 {
        discard;
    }

    // Anti-aliased fill.
    let aa: f32 = 1.0;
    let fill_alpha = 1.0 - smoothstep(-aa, aa, d);

    // Inner border ring of width `border_w`.
    // is_border → 0 deep inside, 1 in border zone (d near 0 from inside).
    let is_border = smoothstep(-aa, aa, d + u.border_w);

    var col = mix(u.bg_color, u.border_color, is_border);
    col.a = col.a * fill_alpha;

    if col.a < 0.004 {
        discard;
    }
    return col;
}
