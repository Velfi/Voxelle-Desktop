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

// Must match [`OPAQUE_VERTEX_STRIDE`] (56) / CPU interleaved layout: 3+3+3+1+1+3 floats.
// WGSL `vec3` aligns to 16B in structs; do not use vec3 fields here or storage stride diverges from Rust.
struct GpuVertex {
    data: array<f32, 14>,
}

@group(0) @binding(0) var<uniform> mesh_params: MeshParams;
@group(0) @binding(1) var<storage, read> slice_headers: array<SliceHeader>;
@group(0) @binding(2) var<storage, read> slice_bits: array<u32>;
@group(0) @binding(3) var<storage, read_write> vtx_out: array<GpuVertex>;
@group(0) @binding(4) var<storage, read_write> idx_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> alloc: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> brick_cells: array<u32>;

fn write_opaque_vertex(slot: u32, pos: vec3<f32>, n: vec3<f32>, col: vec3<f32>, mk: f32, ao: f32) {
    vtx_out[slot].data[0] = pos.x;
    vtx_out[slot].data[1] = pos.y;
    vtx_out[slot].data[2] = pos.z;
    vtx_out[slot].data[3] = n.x;
    vtx_out[slot].data[4] = n.y;
    vtx_out[slot].data[5] = n.z;
    vtx_out[slot].data[6] = col.x;
    vtx_out[slot].data[7] = col.y;
    vtx_out[slot].data[8] = col.z;
    vtx_out[slot].data[9] = mk;
    vtx_out[slot].data[10] = ao;
    // emission_tint: GPU mesh path doesn't compute glow irradiance; zero-fill to match CPU layout.
    vtx_out[slot].data[11] = 0.0;
    vtx_out[slot].data[12] = 0.0;
    vtx_out[slot].data[13] = 0.0;
}

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

fn brick_cell_at(ix: vec3<i32>) -> u32 {
    let rel = ix - vec3<i32>(mesh_params.brick_ox, mesh_params.brick_oy, mesh_params.brick_oz);
    let dx = i32(mesh_params.brick_dx);
    let dy = i32(mesh_params.brick_dy);
    let dz = i32(mesh_params.brick_dz);
    if (rel.x < 0 || rel.y < 0 || rel.z < 0) {
        return 0u;
    }
    if (rel.x >= dx || rel.y >= dy || rel.z >= dz) {
        return 0u;
    }
    let idx = u32(rel.x) + u32(rel.y) * u32(dx) + u32(rel.z) * u32(dx) * u32(dy);
    return brick_cells[idx];
}

fn unpack_mat(packed: u32) -> u32 {
    return (packed >> 24u) & 0xFu;
}

/// Matches [`greedy_mesh::srgb_to_linear`] / CPU `color_rgb` so GPU greedy chunks match CPU mesh shading.
fn srgb_channel_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        return c / 12.92;
    }
    let t = (c + 0.055) / 1.055;
    return pow(t, 2.4);
}

/// Non-transmissive solid occludes AO. CPU also requires same `object_id`; the brick buffer has no per-voxel object id, so GPU only excludes glass/water (parity for materials, not object seams).
fn ao_cell_occludes_brick(ix: vec3<i32>) -> bool {
    let cell = brick_cell_at(ix);
    if ((cell & (1u << 31u)) == 0u) {
        return false;
    }
    let m = unpack_mat(cell);
    return m != 3u && m != 4u;
}

fn ao_du_dv(ci: u32, k: u32) -> vec2<i32> {
    switch ci {
        case 0u: {
            switch k {
                case 0u: {
                    return vec2<i32>(-1, 0);
                }
                case 1u: {
                    return vec2<i32>(0, -1);
                }
                default: {
                    return vec2<i32>(-1, -1);
                }
            }
        }
        case 1u: {
            switch k {
                case 0u: {
                    return vec2<i32>(1, 0);
                }
                case 1u: {
                    return vec2<i32>(0, -1);
                }
                default: {
                    return vec2<i32>(1, -1);
                }
            }
        }
        case 2u: {
            switch k {
                case 0u: {
                    return vec2<i32>(1, 0);
                }
                case 1u: {
                    return vec2<i32>(0, 1);
                }
                default: {
                    return vec2<i32>(1, 1);
                }
            }
        }
        default: {
            switch k {
                case 0u: {
                    return vec2<i32>(-1, 0);
                }
                case 1u: {
                    return vec2<i32>(0, 1);
                }
                default: {
                    return vec2<i32>(-1, 1);
                }
            }
        }
    }
}

fn ao_get_state(s1: u32, s2: u32, c: u32) -> u32 {
    if (s1 != 0u && s2 != 0u) {
        return 0u;
    }
    return 3u - (s1 + s2 + c);
}

fn ao_preset_strong(idx: u32) -> f32 {
    switch idx {
        case 0u: {
            return 0.55;
        }
        case 1u: {
            return 0.72;
        }
        case 2u: {
            return 0.88;
        }
        default: {
            return 1.0;
        }
    }
}

/// Per-corner AO for a single voxel face (kept for reference / tests).
fn corner_ao_brick(axis: u32, sign: i32, depth: i32, cu: i32, cv: i32, corner_idx: u32) -> f32 {
    var s1 = 0u;
    var s2 = 0u;
    var sc = 0u;
    for (var k = 0u; k < 3u; k++) {
        let duv = ao_du_dv(corner_idx, k);
        var ix = vec3<i32>(0, 0, 0);
        if axis == 0u {
            ix = vec3<i32>(depth + sign, cu + duv.x, cv + duv.y);
        } else if axis == 1u {
            ix = vec3<i32>(cu + duv.x, depth + sign, cv + duv.y);
        } else {
            ix = vec3<i32>(cu + duv.x, cv + duv.y, depth + sign);
        }
        let b = select(0u, 1u, ao_cell_occludes_brick(ix));
        if k == 0u {
            s1 = b;
        } else if k == 1u {
            s2 = b;
        } else {
            sc = b;
        }
    }
    let st = ao_get_state(s1, s2, sc);
    return ao_preset_strong(st);
}

/// Canonical vertex AO — averages all 4 L-shaped neighborhood evaluations
/// around a vertex so that shared edges between merged quads never produce
/// seams.  `vu`/`vv` are *vertex* grid positions (quad boundary), not cell
/// centres.  Matches `greedy_mesh::canonical_vertex_ao` on the CPU.
fn canonical_vertex_ao_brick(axis: u32, sign: i32, depth: i32, vu: i32, vv: i32) -> f32 {
    // Check all 4 voxels one layer above the face surrounding this vertex.
    var pa = vec3<i32>(0);
    var pb = vec3<i32>(0);
    var pc = vec3<i32>(0);
    var pd = vec3<i32>(0);
    if axis == 0u {
        pa = vec3<i32>(depth + sign, vu - 1, vv - 1);
        pb = vec3<i32>(depth + sign, vu,     vv - 1);
        pc = vec3<i32>(depth + sign, vu - 1, vv    );
        pd = vec3<i32>(depth + sign, vu,     vv    );
    } else if axis == 1u {
        pa = vec3<i32>(vu - 1, depth + sign, vv - 1);
        pb = vec3<i32>(vu,     depth + sign, vv - 1);
        pc = vec3<i32>(vu - 1, depth + sign, vv    );
        pd = vec3<i32>(vu,     depth + sign, vv    );
    } else {
        pa = vec3<i32>(vu - 1, vv - 1, depth + sign);
        pb = vec3<i32>(vu,     vv - 1, depth + sign);
        pc = vec3<i32>(vu - 1, vv,     depth + sign);
        pd = vec3<i32>(vu,     vv,     depth + sign);
    }
    let a = select(0u, 1u, ao_cell_occludes_brick(pa));
    let b = select(0u, 1u, ao_cell_occludes_brick(pb));
    let c = select(0u, 1u, ao_cell_occludes_brick(pc));
    let d = select(0u, 1u, ao_cell_occludes_brick(pd));

    // L0: side1=C, side2=B, corner=A
    // L1: side1=D, side2=A, corner=B
    // L2: side1=B, side2=C, corner=D
    // L3: side1=A, side2=D, corner=C
    let m0 = ao_preset_strong(ao_get_state(c, b, a));
    let m1 = ao_preset_strong(ao_get_state(d, a, b));
    let m2 = ao_preset_strong(ao_get_state(b, c, d));
    let m3 = ao_preset_strong(ao_get_state(a, d, c));
    return (m0 + m1 + m2 + m3) * 0.25;
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

    let cr = f32((h.color >> 16u) & 0xffu) / 255.0;
    let cg = f32((h.color >> 8u) & 0xffu) / 255.0;
    let cb = f32(h.color & 0xffu) / 255.0;
    let col = vec3<f32>(
        srgb_channel_to_linear(cr),
        srgb_channel_to_linear(cg),
        srgb_channel_to_linear(cb),
    );
    let n = face_normal(h.axis, h.sign_i);
    let ccw = select(
        select(n.z > 0.0, n.y < 0.0, n.y != 0.0),
        n.x > 0.0,
        n.x != 0.0,
    );

    // ── AO-aware merge: pre-compute which cells have fully-lit AO ──────
    // Only cells with all 4 vertex AO values = 1.0 are eligible for
    // greedy merging.  The rest are emitted as 1×1 quads with accurate
    // per-vertex AO, eliminating gradient banding across large quads.
    var ao_ok: array<u32, 128>;
    for (var i2 = 0u; i2 < 128u; i2++) {
        ao_ok[i2] = 0u;
    }
    for (var vv2 = 0u; vv2 < hgt; vv2++) {
        for (var u2 = 0u; u2 < w; u2++) {
            let cidx2 = u2 + vv2 * w;
            let wi2 = cidx2 / 32u;
            let bi2 = cidx2 % 32u;
            if ((bits[wi2] >> bi2) & 1u) == 0u {
                continue;
            }
            let cu2 = i32(u2) + h.u0;
            let cv2 = i32(vv2) + h.v0;
            let a0 = canonical_vertex_ao_brick(h.axis, h.sign_i, h.depth, cu2, cv2);
            let a1 = canonical_vertex_ao_brick(h.axis, h.sign_i, h.depth, cu2 + 1, cv2);
            let a2 = canonical_vertex_ao_brick(h.axis, h.sign_i, h.depth, cu2 + 1, cv2 + 1);
            let a3 = canonical_vertex_ao_brick(h.axis, h.sign_i, h.depth, cu2, cv2 + 1);
            if a0 >= 1.0 && a1 >= 1.0 && a2 >= 1.0 && a3 >= 1.0 {
                ao_ok[wi2] = ao_ok[wi2] | (1u << bi2);
            }
        }
    }

    // ── Pass 1: greedy merge only fully-lit cells ──────────────────────
    for (var vv = 0u; vv < hgt; vv++) {
        for (var u = 0u; u < w; u++) {
            let cidx = u + vv * w;
            let wi = cidx / 32u;
            let bi = cidx % 32u;
            if ((ao_ok[wi] >> bi) & 1u) == 0u {
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
                if ((ao_ok[nwi] >> nbi) & 1u) == 0u {
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
                    if ((ao_ok[nwi] >> nbi) & 1u) == 0u {
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

            // All vertices are fully lit — hardcode AO = 1.0.
            let ao00 = 1.0;
            let ao10 = 1.0;
            let ao11 = 1.0;
            let ao01 = 1.0;

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

            write_opaque_vertex(vbase + 0u, p00, n, col, h.mat_kind, ao00);
            write_opaque_vertex(vbase + 1u, p10, n, col, h.mat_kind, ao10);
            write_opaque_vertex(vbase + 2u, p11, n, col, h.mat_kind, ao11);
            write_opaque_vertex(vbase + 3u, p01, n, col, h.mat_kind, ao01);

            let g0 = vbase + 0u;
            let g1 = vbase + 1u;
            let g2 = vbase + 2u;
            let g3 = vbase + 3u;

            // Quad diagonal flip: pair the two brighter corners to
            // minimise the visible crease from the triangle split.
            let flip = (ao00 + ao11) < (ao10 + ao01);

            if ccw {
                if flip {
                    idx_out[ibase + 0u] = g0;
                    idx_out[ibase + 1u] = g1;
                    idx_out[ibase + 2u] = g3;
                    idx_out[ibase + 3u] = g1;
                    idx_out[ibase + 4u] = g2;
                    idx_out[ibase + 5u] = g3;
                } else {
                    idx_out[ibase + 0u] = g0;
                    idx_out[ibase + 1u] = g1;
                    idx_out[ibase + 2u] = g2;
                    idx_out[ibase + 3u] = g0;
                    idx_out[ibase + 4u] = g2;
                    idx_out[ibase + 5u] = g3;
                }
            } else {
                if flip {
                    idx_out[ibase + 0u] = g0;
                    idx_out[ibase + 1u] = g3;
                    idx_out[ibase + 2u] = g1;
                    idx_out[ibase + 3u] = g1;
                    idx_out[ibase + 4u] = g3;
                    idx_out[ibase + 5u] = g2;
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

    // ── Pass 2: emit non-merged cells as 1×1 quads with per-vertex AO ─
    for (var vv3 = 0u; vv3 < hgt; vv3++) {
        for (var u3 = 0u; u3 < w; u3++) {
            let cidx3 = u3 + vv3 * w;
            let wi3 = cidx3 / 32u;
            let bi3 = cidx3 % 32u;
            if ((bits[wi3] >> bi3) & 1u) == 0u {
                continue;
            }
            if ((consumed[wi3] >> bi3) & 1u) != 0u {
                continue;
            }

            let cu3 = i32(u3) + h.u0;
            let cv3 = i32(vv3) + h.v0;
            let sp00 = quad_corner(h.axis, h.sign_i, h.depth, cu3, cv3);
            let sp10 = quad_corner(h.axis, h.sign_i, h.depth, cu3 + 1, cv3);
            let sp11 = quad_corner(h.axis, h.sign_i, h.depth, cu3 + 1, cv3 + 1);
            let sp01 = quad_corner(h.axis, h.sign_i, h.depth, cu3, cv3 + 1);

            let sao00 = canonical_vertex_ao_brick(h.axis, h.sign_i, h.depth, cu3, cv3);
            let sao10 = canonical_vertex_ao_brick(h.axis, h.sign_i, h.depth, cu3 + 1, cv3);
            let sao11 = canonical_vertex_ao_brick(h.axis, h.sign_i, h.depth, cu3 + 1, cv3 + 1);
            let sao01 = canonical_vertex_ao_brick(h.axis, h.sign_i, h.depth, cu3, cv3 + 1);

            let sv = atomicAdd(&alloc[0], 4u);
            if sv + 4u > mesh_params.max_vertices {
                atomicSub(&alloc[0], 4u);
                return;
            }
            let si3 = atomicAdd(&alloc[1], 6u);
            if si3 + 6u > mesh_params.max_indices {
                atomicSub(&alloc[1], 6u);
                atomicSub(&alloc[0], 4u);
                return;
            }

            write_opaque_vertex(sv + 0u, sp00, n, col, h.mat_kind, sao00);
            write_opaque_vertex(sv + 1u, sp10, n, col, h.mat_kind, sao10);
            write_opaque_vertex(sv + 2u, sp11, n, col, h.mat_kind, sao11);
            write_opaque_vertex(sv + 3u, sp01, n, col, h.mat_kind, sao01);

            let sg0 = sv + 0u;
            let sg1 = sv + 1u;
            let sg2 = sv + 2u;
            let sg3 = sv + 3u;

            let sflip = (sao00 + sao11) < (sao10 + sao01);
            if ccw {
                if sflip {
                    idx_out[si3 + 0u] = sg0; idx_out[si3 + 1u] = sg1; idx_out[si3 + 2u] = sg3;
                    idx_out[si3 + 3u] = sg1; idx_out[si3 + 4u] = sg2; idx_out[si3 + 5u] = sg3;
                } else {
                    idx_out[si3 + 0u] = sg0; idx_out[si3 + 1u] = sg1; idx_out[si3 + 2u] = sg2;
                    idx_out[si3 + 3u] = sg0; idx_out[si3 + 4u] = sg2; idx_out[si3 + 5u] = sg3;
                }
            } else {
                if sflip {
                    idx_out[si3 + 0u] = sg0; idx_out[si3 + 1u] = sg3; idx_out[si3 + 2u] = sg1;
                    idx_out[si3 + 3u] = sg1; idx_out[si3 + 4u] = sg3; idx_out[si3 + 5u] = sg2;
                } else {
                    idx_out[si3 + 0u] = sg0; idx_out[si3 + 1u] = sg2; idx_out[si3 + 2u] = sg1;
                    idx_out[si3 + 3u] = sg0; idx_out[si3 + 4u] = sg3; idx_out[si3 + 5u] = sg2;
                }
            }
        }
    }
}
