// Per-slice greedy mesh (bitmap ≤ 64×64). Outputs are packed densely via atomics
// so CPU copy [0..v_total) / [0..i_total) matches the mesh (no holes from per-slice reservation).

struct MeshParams {
    max_vertices: u32,
    max_indices: u32,
    slice_count: u32,
    _pad0: u32,
    brick_ox: i32,
    brick_oy: i32,
    brick_oz: i32,
    _pad1: i32,
    brick_dx: u32,
    brick_dy: u32,
    brick_dz: u32,
    _pad2: u32,
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
    ao: f32,
}

@group(0) @binding(0) var<uniform> mesh_params: MeshParams;
@group(0) @binding(1) var<storage, read> slice_headers: array<SliceHeader>;
@group(0) @binding(2) var<storage, read> slice_bits: array<u32>;
@group(0) @binding(3) var<storage, read_write> vtx_out: array<GpuVertex>;
@group(0) @binding(4) var<storage, read_write> idx_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> alloc: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> brick_cells: array<u32>;

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

fn brick_occupied_at(ix: vec3<i32>) -> bool {
    let rel = ix - vec3<i32>(mesh_params.brick_ox, mesh_params.brick_oy, mesh_params.brick_oz);
    let dx = i32(mesh_params.brick_dx);
    let dy = i32(mesh_params.brick_dy);
    let dz = i32(mesh_params.brick_dz);
    if (rel.x < 0 || rel.y < 0 || rel.z < 0) {
        return false;
    }
    if (rel.x >= dx || rel.y >= dy || rel.z >= dz) {
        return false;
    }
    let idx = u32(rel.x) + u32(rel.y) * u32(dx) + u32(rel.z) * u32(dx) * u32(dy);
    let cell = brick_cells[idx];
    return (cell & (1u << 31u)) != 0u;
}

/// Minecraft-style corner AO (parity with `greedy_mesh::corner_ao_factor`).
fn corner_ao_brick(axis: u32, depth: i32, cu: i32, cv: i32) -> f32 {
    var occ = 0u;
    if axis == 0u {
        if brick_occupied_at(vec3<i32>(depth, cu - 1, cv)) {
            occ = occ + 1u;
        }
        if brick_occupied_at(vec3<i32>(depth, cu, cv - 1)) {
            occ = occ + 1u;
        }
        if brick_occupied_at(vec3<i32>(depth, cu - 1, cv - 1)) {
            occ = occ + 1u;
        }
    } else if axis == 1u {
        if brick_occupied_at(vec3<i32>(cu - 1, depth, cv)) {
            occ = occ + 1u;
        }
        if brick_occupied_at(vec3<i32>(cu, depth, cv - 1)) {
            occ = occ + 1u;
        }
        if brick_occupied_at(vec3<i32>(cu - 1, depth, cv - 1)) {
            occ = occ + 1u;
        }
    } else {
        if brick_occupied_at(vec3<i32>(cu - 1, cv, depth)) {
            occ = occ + 1u;
        }
        if brick_occupied_at(vec3<i32>(cu, cv - 1, depth)) {
            occ = occ + 1u;
        }
        if brick_occupied_at(vec3<i32>(cu - 1, cv - 1, depth)) {
            occ = occ + 1u;
        }
    }
    let f = 1.0 - 0.2 * f32(occ);
    return clamp(f, 0.4, 1.0);
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

    for (var vv = 0u; vv < hgt; vv++) {
        for (var u = 0u; u < w; u++) {
            let cidx = u + vv * w;
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
                let ni = (u + rw) + vv * w;
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
                if vv + rh >= hgt {
                    break;
                }
                var row_ok = true;
                for (var du = 0u; du < rw; du++) {
                    let ni = (u + du) + (vv + rh) * w;
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
                    let ni = (u + du) + (vv + dv) * w;
                    let nwi = ni / 32u;
                    let nbi = ni % 32u;
                    consumed[nwi] = consumed[nwi] | (1u << nbi);
                }
            }

            let u0 = i32(u) + h.u0;
            let v0 = i32(vv) + h.v0;
            let p00 = quad_corner(h.axis, h.sign_i, h.depth, u0, v0);
            let p10 = quad_corner(h.axis, h.sign_i, h.depth, u0 + i32(rw), v0);
            let p11 = quad_corner(h.axis, h.sign_i, h.depth, u0 + i32(rw), v0 + i32(rh));
            let p01 = quad_corner(h.axis, h.sign_i, h.depth, u0, v0 + i32(rh));

            let ao00 = corner_ao_brick(h.axis, h.depth, u0, v0);
            let ao10 = corner_ao_brick(h.axis, h.depth, u0 + i32(rw) - 1, v0);
            let ao11 = corner_ao_brick(h.axis, h.depth, u0 + i32(rw) - 1, v0 + i32(rh) - 1);
            let ao01 = corner_ao_brick(h.axis, h.depth, u0, v0 + i32(rh) - 1);
            let ao_face = (ao00 + ao10 + ao11 + ao01) * 0.25;

            let vbase = atomicAdd(&alloc[0], 4u);
            if vbase + 4u > mesh_params.max_vertices {
                atomicSub(&alloc[0], 4u);
                return;
            }
            let ibase = atomicAdd(&alloc[1], 6u);
            if ibase + 6u > mesh_params.max_indices {
                atomicSub(&alloc[1], 6u);
                atomicSub(&alloc[0], 4u);
                return;
            }

            vtx_out[vbase + 0u] = GpuVertex(p00, n, col, h.mat_kind, ao_face);
            vtx_out[vbase + 1u] = GpuVertex(p10, n, col, h.mat_kind, ao_face);
            vtx_out[vbase + 2u] = GpuVertex(p11, n, col, h.mat_kind, ao_face);
            vtx_out[vbase + 3u] = GpuVertex(p01, n, col, h.mat_kind, ao_face);

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
        }
    }
}
