// Per-slice greedy mesh (bitmap ≤ 64×64). No cross-slice atomics — disjoint output ranges via slice_vtx_base / slice_idx_base.

struct MeshParams {
    max_vertices: u32,
    max_indices: u32,
    slice_count: u32,
    _pad: u32,
}

struct SliceHeader {
    axis: u32,
    sign_i: i32,
    depth: i32,
    color: u32,
    mat_kind: f32,
    u0: i32,
    v0: i32,
    width: u32,
    height: u32,
    bit_start: u32,
    bit_word_count: u32,
}

struct GpuVertex {
    pos: vec3<f32>,
    n: vec3<f32>,
    col: vec3<f32>,
    mk: f32,
}

@group(0) @binding(0) var<uniform> mesh_params: MeshParams;
@group(0) @binding(1) var<storage, read> slice_headers: array<SliceHeader>;
@group(0) @binding(2) var<storage, read> slice_bits: array<u32>;
@group(0) @binding(3) var<storage, read> slice_vtx_base: array<u32>;
@group(0) @binding(4) var<storage, read> slice_idx_base: array<u32>;
@group(0) @binding(5) var<storage, read_write> vtx_out: array<GpuVertex>;
@group(0) @binding(6) var<storage, read_write> idx_out: array<u32>;
@group(0) @binding(7) var<storage, read_write> total_counts: array<atomic<u32>>;

fn face_normal(axis: u32, sign: i32) -> vec3<f32> {
    if axis == 0u {
        return vec3<f32>(f32(sign), 0.0, 0.0);
    }
    if axis == 1u {
        return vec3<f32>(0.0, f32(sign), 0.0);
    }
    return vec3<f32>(0.0, 0.0, f32(sign));
}

fn quad_corner(axis: u32, sign: i32, depth: i32, u: i32, v: i32) -> vec3<f32> {
    let fo = 0.5 * f32(sign);
    if axis == 0u {
        return vec3<f32>(f32(depth) + fo, f32(u) - 0.5, f32(v) - 0.5);
    }
    if axis == 1u {
        return vec3<f32>(f32(u) - 0.5, f32(depth) + fo, f32(v) - 0.5);
    }
    return vec3<f32>(f32(u) - 0.5, f32(v) - 0.5, f32(depth) + fo);
}

@compute @workgroup_size(1, 1, 1)
fn greedy_slice(@builtin(global_invocation_id) gid: vec3<u32>) {
    let si = gid.x;
    if si >= mesh_params.slice_count {
        return;
    }
    let h = slice_headers[si];
    let w = h.width;
    let hgt = h.height;
    if w == 0u || hgt == 0u || w > 64u || hgt > 64u {
        return;
    }

    var bits: array<u32, 128>;
    var consumed: array<u32, 128>;
    let nw = h.bit_word_count;
    let bs = h.bit_start;
    for (var i = 0u; i < 128u; i++) {
        bits[i] = 0u;
        consumed[i] = 0u;
    }
    for (var i = 0u; i < nw && i < 128u; i++) {
        bits[i] = slice_bits[bs + i];
    }

    let col = vec3<f32>(
        f32((h.color >> 16u) & 0xffu) / 255.0,
        f32((h.color >> 8u) & 0xffu) / 255.0,
        f32(h.color & 0xffu) / 255.0,
    );
    let n = face_normal(h.axis, h.sign_i);
    let ccw = select(
        select(n.z > 0.0, n.y < 0.0, n.y != 0.0),
        n.x > 0.0,
        n.x != 0.0,
    );

    let g_v_base = slice_vtx_base[si];
    let g_i_base = slice_idx_base[si];
    var v_off = 0u;
    var i_off = 0u;

    for (var v = 0u; v < hgt; v++) {
        for (var u = 0u; u < w; u++) {
            let cidx = u + v * w;
            let wi = cidx / 32u;
            let bi = cidx % 32u;
            if ((bits[wi] >> bi) & 1u) == 0u {
                continue;
            }
            if ((consumed[wi] >> bi) & 1u) != 0u {
                continue;
            }

            var rw = 1u;
            loop {
                if u + rw >= w {
                    break;
                }
                let ni = (u + rw) + v * w;
                let nwi = ni / 32u;
                let nbi = ni % 32u;
                if ((bits[nwi] >> nbi) & 1u) == 0u {
                    break;
                }
                if ((consumed[nwi] >> nbi) & 1u) != 0u {
                    break;
                }
                rw = rw + 1u;
            }

            var rh = 1u;
            loop {
                if v + rh >= hgt {
                    break;
                }
                var row_ok = true;
                for (var du = 0u; du < rw; du++) {
                    let ni = (u + du) + (v + rh) * w;
                    let nwi = ni / 32u;
                    let nbi = ni % 32u;
                    if ((bits[nwi] >> nbi) & 1u) == 0u {
                        row_ok = false;
                        break;
                    }
                    if ((consumed[nwi] >> nbi) & 1u) != 0u {
                        row_ok = false;
                        break;
                    }
                }
                if !row_ok {
                    break;
                }
                rh = rh + 1u;
            }

            for (var dv = 0u; dv < rh; dv++) {
                for (var du = 0u; du < rw; du++) {
                    let ni = (u + du) + (v + dv) * w;
                    let nwi = ni / 32u;
                    let nbi = ni % 32u;
                    consumed[nwi] = consumed[nwi] | (1u << nbi);
                }
            }

            let u0 = i32(u) + h.u0;
            let v0 = i32(v) + h.v0;
            let p00 = quad_corner(h.axis, h.sign_i, h.depth, u0, v0);
            let p10 = quad_corner(h.axis, h.sign_i, h.depth, u0 + i32(rw), v0);
            let p11 = quad_corner(h.axis, h.sign_i, h.depth, u0 + i32(rw), v0 + i32(rh));
            let p01 = quad_corner(h.axis, h.sign_i, h.depth, u0, v0 + i32(rh));

            let vbase = g_v_base + v_off;
            let ibase = g_i_base + i_off;
            if vbase + 4u > mesh_params.max_vertices || ibase + 6u > mesh_params.max_indices {
                return;
            }

            vtx_out[vbase + 0u] = GpuVertex(p00, n, col, h.mat_kind);
            vtx_out[vbase + 1u] = GpuVertex(p10, n, col, h.mat_kind);
            vtx_out[vbase + 2u] = GpuVertex(p11, n, col, h.mat_kind);
            vtx_out[vbase + 3u] = GpuVertex(p01, n, col, h.mat_kind);

            let g0 = vbase + 0u;
            let g1 = vbase + 1u;
            let g2 = vbase + 2u;
            let g3 = vbase + 3u;

            if ccw {
                idx_out[ibase + 0u] = g0;
                idx_out[ibase + 1u] = g1;
                idx_out[ibase + 2u] = g2;
                idx_out[ibase + 3u] = g0;
                idx_out[ibase + 4u] = g2;
                idx_out[ibase + 5u] = g3;
            } else {
                idx_out[ibase + 0u] = g0;
                idx_out[ibase + 1u] = g2;
                idx_out[ibase + 2u] = g1;
                idx_out[ibase + 3u] = g0;
                idx_out[ibase + 4u] = g3;
                idx_out[ibase + 5u] = g2;
            }

            v_off = v_off + 4u;
            i_off = i_off + 6u;
        }
    }

    atomicAdd(&total_counts[0], v_off);
    atomicAdd(&total_counts[1], i_off);
}
