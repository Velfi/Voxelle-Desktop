//! Smooth surface extraction (marching cubes + dual contouring), aligned with web `marchingCubesCore.ts` / `dualContourCore.ts`.

use crate::greedy_mesh::{MeshBuffers, VoxelCoord};
use crate::marching_tables::{EDGE_TABLE, TRI_TABLE};
use crate::voxelle::{MaterialId, Voxel};
use ahash::AHashMap;
use std::time::Instant;

const ISO: f32 = 0.5;
const EPS: f32 = 1e-6;

struct LatticeField {
    node_min_x: i32,
    node_min_y: i32,
    node_min_z: i32,
    nx: usize,
    ny: usize,
    nz: usize,
    values: Vec<f32>,
    grad_x: Vec<f32>,
    grad_y: Vec<f32>,
    grad_z: Vec<f32>,
    col_r: Vec<f32>,
    col_g: Vec<f32>,
    col_b: Vec<f32>,
    bucket_material: MaterialId,
}

fn coord_key(x: i32, y: i32, z: i32) -> VoxelCoord {
    (x, y, z)
}

#[inline]
fn idx(x: usize, y: usize, z: usize, nx: usize, ny: usize) -> usize {
    (z * ny + y) * nx + x
}

fn sample_field(values: &[f32], x: i32, y: i32, z: i32, nx: usize, ny: usize, nz: usize) -> f32 {
    if x < 0 || y < 0 || z < 0 {
        return 0.0;
    }
    let (ux, uy, uz) = (x as usize, y as usize, z as usize);
    if ux >= nx || uy >= ny || uz >= nz {
        return 0.0;
    }
    values[idx(ux, uy, uz, nx, ny)]
}

/// Web `buildMarchingLatticeField`: `occupancy` bounds the lattice; colors prefer `color` map, else occupancy RGB.
fn build_lattice_field(
    color_voxels: &AHashMap<VoxelCoord, Voxel>,
    occupancy_voxels: &AHashMap<VoxelCoord, Voxel>,
) -> Option<LatticeField> {
    if occupancy_voxels.is_empty() || color_voxels.is_empty() {
        return None;
    }
    let first = color_voxels.values().next()?;
    let bucket_material = first.material;

    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut min_z = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;
    let mut max_z = i32::MIN;
    for (x, y, z) in occupancy_voxels.keys().copied() {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        min_z = min_z.min(z);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
        max_z = max_z.max(z);
    }

    let node_min_x = min_x - 1;
    let node_min_y = min_y - 1;
    let node_min_z = min_z - 1;
    let node_max_x = max_x + 2;
    let node_max_y = max_y + 2;
    let node_max_z = max_z + 2;

    let nx = (node_max_x - node_min_x + 1) as usize;
    let ny = (node_max_y - node_min_y + 1) as usize;
    let nz = (node_max_z - node_min_z + 1) as usize;
    let n = nx * ny * nz;

    let mut values = vec![0.0f32; n];
    let mut col_r = vec![0.0f32; n];
    let mut col_g = vec![0.0f32; n];
    let mut col_b = vec![0.0f32; n];

    for z in 0..nz {
        let gz = node_min_z + z as i32;
        for y in 0..ny {
            let gy = node_min_y + y as i32;
            for x in 0..nx {
                let gx = node_min_x + x as i32;
                let i = idx(x, y, z, nx, ny);

                let mut count_occ = 0i32;
                let mut sr_occ = 0i32;
                let mut sg_occ = 0i32;
                let mut sb_occ = 0i32;
                let mut count_b = 0i32;
                let mut sr_b = 0i32;
                let mut sg_b = 0i32;
                let mut sb_b = 0i32;

                for dz in -1..=0 {
                    for dy in -1..=0 {
                        for dx in -1..=0 {
                            let k = coord_key(gx + dx, gy + dy, gz + dz);
                            if let Some(vo) = occupancy_voxels.get(&k) {
                                count_occ += 1;
                                let rgb = vo.color;
                                sr_occ += ((rgb >> 16) & 0xff) as i32;
                                sg_occ += ((rgb >> 8) & 0xff) as i32;
                                sb_occ += (rgb & 0xff) as i32;
                            }
                            if let Some(vb) = color_voxels.get(&k) {
                                count_b += 1;
                                let rgb = vb.color;
                                sr_b += ((rgb >> 16) & 0xff) as i32;
                                sg_b += ((rgb >> 8) & 0xff) as i32;
                                sb_b += (rgb & 0xff) as i32;
                            }
                        }
                    }
                }

                values[i] = count_occ as f32;
                if count_b > 0 {
                    let cb = count_b as f32;
                    col_r[i] = sr_b as f32 / cb / 255.0;
                    col_g[i] = sg_b as f32 / cb / 255.0;
                    col_b[i] = sb_b as f32 / cb / 255.0;
                } else if count_occ > 0 {
                    let c = count_occ as f32;
                    col_r[i] = sr_occ as f32 / c / 255.0;
                    col_g[i] = sg_occ as f32 / c / 255.0;
                    col_b[i] = sb_occ as f32 / c / 255.0;
                }
            }
        }
    }

    let mut grad_x = vec![0.0f32; n];
    let mut grad_y = vec![0.0f32; n];
    let mut grad_z = vec![0.0f32; n];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = idx(x, y, z, nx, ny);
                let xi = x as i32;
                let yi = y as i32;
                let zi = z as i32;
                let gx = sample_field(&values, xi - 1, yi, zi, nx, ny, nz)
                    - sample_field(&values, xi + 1, yi, zi, nx, ny, nz);
                let gy = sample_field(&values, xi, yi - 1, zi, nx, ny, nz)
                    - sample_field(&values, xi, yi + 1, zi, nx, ny, nz);
                let gz = sample_field(&values, xi, yi, zi - 1, nx, ny, nz)
                    - sample_field(&values, xi, yi, zi + 1, nx, ny, nz);
                let gl = (gx * gx + gy * gy + gz * gz).sqrt();
                if gl > EPS {
                    grad_x[i] = gx / gl;
                    grad_y[i] = gy / gl;
                    grad_z[i] = gz / gl;
                }
            }
        }
    }

    Some(LatticeField {
        node_min_x,
        node_min_y,
        node_min_z,
        nx,
        ny,
        nz,
        values,
        grad_x,
        grad_y,
        grad_z,
        col_r,
        col_g,
        col_b,
        bucket_material,
    })
}

const CORNER_OFF: [[i32; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

const EDGE_PAIRS: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

fn mat_kind_for_material(m: MaterialId) -> f32 {
    match m {
        MaterialId::Wax => 0.15,
        MaterialId::Metal => 0.5,
        MaterialId::Glow => 1.0,
        MaterialId::Velvet => 1.4,
        MaterialId::Holographic => 1.75,
        MaterialId::Glass => 2.0,
        MaterialId::Water => 2.5,
        _ => 0.0,
    }
}

fn marching_cubes_bucket(
    color_map: &AHashMap<VoxelCoord, Voxel>,
    progress: Option<(usize, usize)>,
) -> Option<MeshBuffers> {
    let tag = progress
        .map(|(i, n)| format!("[bucket {i}/{n}] "))
        .unwrap_or_default();
    let field = build_lattice_field(color_map, color_map)?;
    let LatticeField {
        node_min_x,
        node_min_y,
        node_min_z,
        nx,
        ny,
        nz,
        values,
        grad_x,
        grad_y,
        grad_z,
        col_r,
        col_g,
        col_b,
        bucket_material,
    } = field;

    let mat_k = mat_kind_for_material(bucket_material);

    let cells = nx.saturating_mul(ny).saturating_mul(nz);
    let nv = color_map.len();
    if cells > 1_000_000 {
        log::warn!(
            target: "voxelle_load",
            "{}marching_cubes_bucket: large lattice {}×{}×{} ({} cells) for {} voxels — expect long CPU time",
            tag,
            nx,
            ny,
            nz,
            cells,
            nv
        );
    } else {
        log::info!(
            target: "voxelle_load",
            "{}marching_cubes_bucket: lattice {}×{}×{} ({} cells), {} voxels",
            tag,
            nx,
            ny,
            nz,
            cells,
            nv
        );
    }

    let t_mc = Instant::now();

    let mut raw_pos: Vec<f32> = Vec::new();
    let mut raw_norm: Vec<f32> = Vec::new();
    let mut raw_col: Vec<f32> = Vec::new();

    let mut edge_pos = [0.0f32; 12 * 3];
    let mut edge_norm = [0.0f32; 12 * 3];
    let mut edge_col = [0.0f32; 12 * 3];

    for z in 0..nz.saturating_sub(1) {
        for y in 0..ny.saturating_sub(1) {
            for x in 0..nx.saturating_sub(1) {
                let mut cube_index = 0u32;
                let mut corner_v = [0.0f32; 8];
                let mut corner_x = [0.0f32; 8];
                let mut corner_y = [0.0f32; 8];
                let mut corner_z = [0.0f32; 8];
                let mut corner_nx = [0.0f32; 8];
                let mut corner_ny = [0.0f32; 8];
                let mut corner_nz = [0.0f32; 8];
                let mut corner_r = [0.0f32; 8];
                let mut corner_g = [0.0f32; 8];
                let mut corner_b = [0.0f32; 8];

                for c in 0..8 {
                    let ox = CORNER_OFF[c][0];
                    let oy = CORNER_OFF[c][1];
                    let oz = CORNER_OFF[c][2];
                    let sx = x as i32 + ox;
                    let sy = y as i32 + oy;
                    let sz = z as i32 + oz;
                    let i = idx(sx as usize, sy as usize, sz as usize, nx, ny);
                    let v = values[i];
                    corner_v[c] = v;
                    if v < ISO {
                        cube_index |= 1 << c;
                    }
                    corner_x[c] = node_min_x as f32 + sx as f32 + 0.5;
                    corner_y[c] = node_min_y as f32 + sy as f32 + 0.5;
                    corner_z[c] = node_min_z as f32 + sz as f32 + 0.5;
                    corner_nx[c] = grad_x[i];
                    corner_ny[c] = grad_y[i];
                    corner_nz[c] = grad_z[i];
                    corner_r[c] = col_r[i];
                    corner_g[c] = col_g[i];
                    corner_b[c] = col_b[i];
                }

                let edge_mask = EDGE_TABLE[cube_index as usize];
                if edge_mask == 0 {
                    continue;
                }

                for e in 0..12 {
                    if (edge_mask & (1 << e)) == 0 {
                        continue;
                    }
                    let a = EDGE_PAIRS[e][0];
                    let b = EDGE_PAIRS[e][1];
                    let va = corner_v[a];
                    let vb = corner_v[b];
                    let denom = vb - va;
                    let mu = if denom.abs() < EPS {
                        0.5
                    } else {
                        (ISO - va) / denom
                    };
                    let t = mu.clamp(0.0, 1.0);
                    let base = e * 3;

                    edge_pos[base] = corner_x[a] + (corner_x[b] - corner_x[a]) * t;
                    edge_pos[base + 1] = corner_y[a] + (corner_y[b] - corner_y[a]) * t;
                    edge_pos[base + 2] = corner_z[a] + (corner_z[b] - corner_z[a]) * t;

                    let nx_l = corner_nx[a] + (corner_nx[b] - corner_nx[a]) * t;
                    let ny_l = corner_ny[a] + (corner_ny[b] - corner_ny[a]) * t;
                    let nz_l = corner_nz[a] + (corner_nz[b] - corner_nz[a]) * t;
                    let nl = (nx_l * nx_l + ny_l * ny_l + nz_l * nz_l).sqrt();
                    if nl > EPS {
                        edge_norm[base] = nx_l / nl;
                        edge_norm[base + 1] = ny_l / nl;
                        edge_norm[base + 2] = nz_l / nl;
                    } else {
                        edge_norm[base] = 0.0;
                        edge_norm[base + 1] = 1.0;
                        edge_norm[base + 2] = 0.0;
                    }

                    let use_a = va >= vb;
                    edge_col[base] = if use_a { corner_r[a] } else { corner_r[b] };
                    edge_col[base + 1] = if use_a { corner_g[a] } else { corner_g[b] };
                    edge_col[base + 2] = if use_a { corner_b[a] } else { corner_b[b] };
                }

                let tri_offset = cube_index as usize * 16;
                let mut ti = 0;
                loop {
                    let t0 = TRI_TABLE[tri_offset + ti];
                    if t0 < 0 {
                        break;
                    }
                    let e0 = t0 as usize;
                    let e1 = TRI_TABLE[tri_offset + ti + 1] as usize;
                    let e2 = TRI_TABLE[tri_offset + ti + 2] as usize;
                    let b0 = e0 * 3;
                    let b1 = e1 * 3;
                    let b2 = e2 * 3;
                    for &bb in &[b0, b1, b2] {
                        raw_pos.push(edge_pos[bb]);
                        raw_pos.push(edge_pos[bb + 1]);
                        raw_pos.push(edge_pos[bb + 2]);
                        raw_norm.push(edge_norm[bb]);
                        raw_norm.push(edge_norm[bb + 1]);
                        raw_norm.push(edge_norm[bb + 2]);
                        raw_col.push(edge_col[bb]);
                        raw_col.push(edge_col[bb + 1]);
                        raw_col.push(edge_col[bb + 2]);
                    }
                    ti += 3;
                }
            }
        }
    }

    if raw_pos.is_empty() {
        return None;
    }

    log::info!(
        target: "voxelle_load",
        "{}marching_cubes_bucket: iso-surface pass {:?}",
        tag,
        t_mc.elapsed()
    );

    let t_weld = Instant::now();
    let out = weld_vertices_to_mesh(&raw_pos, &raw_norm, &raw_col, mat_k);
    log::info!(
        target: "voxelle_load",
        "{}marching_cubes_bucket: weld {:?}",
        tag,
        t_weld.elapsed()
    );
    out
}

fn weld_vertices_to_mesh(
    raw_pos: &[f32],
    raw_norm: &[f32],
    raw_col: &[f32],
    mat_k: f32,
) -> Option<MeshBuffers> {
    let n_tri_v = raw_pos.len() / 3;
    let mut vertex_map: AHashMap<(i32, i32, i32), u32> = AHashMap::new();
    let mut positions: Vec<f32> = Vec::new();
    let mut acc_nx: Vec<f32> = Vec::new();
    let mut acc_ny: Vec<f32> = Vec::new();
    let mut acc_nz: Vec<f32> = Vec::new();
    let mut acc_cr: Vec<f32> = Vec::new();
    let mut acc_cg: Vec<f32> = Vec::new();
    let mut acc_cb: Vec<f32> = Vec::new();
    let mut counts: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    fn q(x: f32) -> i32 {
        (x * 1_000_000.0).round() as i32
    }

    for i in 0..n_tri_v {
        let px = raw_pos[i * 3];
        let py = raw_pos[i * 3 + 1];
        let pz = raw_pos[i * 3 + 2];
        let key = (q(px), q(py), q(pz));
        let vi = *vertex_map.entry(key).or_insert_with(|| {
            let vi = (positions.len() / 3) as u32;
            positions.extend_from_slice(&[px, py, pz]);
            acc_nx.push(0.0);
            acc_ny.push(0.0);
            acc_nz.push(0.0);
            acc_cr.push(0.0);
            acc_cg.push(0.0);
            acc_cb.push(0.0);
            counts.push(0.0);
            vi
        });
        let vi = vi as usize;
        acc_nx[vi] += raw_norm[i * 3];
        acc_ny[vi] += raw_norm[i * 3 + 1];
        acc_nz[vi] += raw_norm[i * 3 + 2];
        acc_cr[vi] += raw_col[i * 3];
        acc_cg[vi] += raw_col[i * 3 + 1];
        acc_cb[vi] += raw_col[i * 3 + 2];
        counts[vi] += 1.0;
        indices.push(vi as u32);
    }

    let nv = positions.len() / 3;
    let mut normals = vec![0.0f32; nv * 3];
    let mut colors = vec![0.0f32; nv * 3];
    let mat_kind = vec![mat_k; nv];
    let ao = vec![1.0f32; nv];

    for i in 0..nv {
        let c = counts[i].max(1.0);
        let nxv = acc_nx[i];
        let nyv = acc_ny[i];
        let nzv = acc_nz[i];
        let nl = (nxv * nxv + nyv * nyv + nzv * nzv).sqrt();
        if nl > EPS {
            normals[i * 3] = nxv / nl;
            normals[i * 3 + 1] = nyv / nl;
            normals[i * 3 + 2] = nzv / nl;
        } else {
            normals[i * 3] = 0.0;
            normals[i * 3 + 1] = 1.0;
            normals[i * 3 + 2] = 0.0;
        }
        colors[i * 3] = (acc_cr[i] / c).clamp(0.0, 1.0);
        colors[i * 3 + 1] = (acc_cg[i] / c).clamp(0.0, 1.0);
        colors[i * 3 + 2] = (acc_cb[i] / c).clamp(0.0, 1.0);
    }

    let nv = positions.len() / 3;
    Some(MeshBuffers {
        positions,
        normals,
        colors,
        mat_kind,
        ao,
        emission_tint: vec![0.0f32; nv * 3],
        indices,
    })
}

fn sample_grad(
    values: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    x: i32,
    y: i32,
    z: i32,
    out: &mut [f32; 3],
) {
    let gx = sample_field(values, x - 1, y, z, nx, ny, nz)
        - sample_field(values, x + 1, y, z, nx, ny, nz);
    let gy = sample_field(values, x, y - 1, z, nx, ny, nz)
        - sample_field(values, x, y + 1, z, nx, ny, nz);
    let gz = sample_field(values, x, y, z - 1, nx, ny, nz)
        - sample_field(values, x, y, z + 1, nx, ny, nz);
    let gl = (gx * gx + gy * gy + gz * gz).sqrt();
    if gl > EPS {
        out[0] = gx / gl;
        out[1] = gy / gl;
        out[2] = gz / gl;
    } else {
        out[0] = 0.0;
        out[1] = 1.0;
        out[2] = 0.0;
    }
}

fn solve3x3(
    a00: f64,
    a01: f64,
    a02: f64,
    a11: f64,
    a12: f64,
    a22: f64,
    b0: f64,
    b1: f64,
    b2: f64,
) -> Option<[f64; 3]> {
    let mut m = [
        [a00, a01, a02, b0],
        [a01, a11, a12, b1],
        [a02, a12, a22, b2],
    ];
    for col in 0..3 {
        let mut pivot = col;
        let mut max_abs = m[col][col].abs();
        for r in (col + 1)..3 {
            let v = m[r][col].abs();
            if v > max_abs {
                max_abs = v;
                pivot = r;
            }
        }
        if max_abs < 1e-12 {
            return None;
        }
        if pivot != col {
            m.swap(col, pivot);
        }
        let div = m[col][col];
        for c in col..4 {
            m[col][c] /= div;
        }
        for r in 0..3 {
            if r == col {
                continue;
            }
            let f = m[r][col];
            if f.abs() < 1e-15 {
                continue;
            }
            for c in col..4 {
                m[r][c] -= f * m[col][c];
            }
        }
    }
    Some([m[0][3], m[1][3], m[2][3]])
}

fn dual_contour_bucket(
    color_map: &AHashMap<VoxelCoord, Voxel>,
    full_map: &AHashMap<VoxelCoord, Voxel>,
) -> Option<MeshBuffers> {
    let field = build_lattice_field(color_map, full_map)?;
    let LatticeField {
        node_min_x,
        node_min_y,
        node_min_z,
        nx,
        ny,
        nz,
        values,
        grad_x: _,
        grad_y: _,
        grad_z: _,
        col_r,
        col_g,
        col_b,
        bucket_material,
    } = field;

    let mat_k = mat_kind_for_material(bucket_material);

    let mut cell_vertex: AHashMap<(i32, i32, i32), u32> = AHashMap::new();
    let mut positions: Vec<f32> = Vec::new();
    let mut normals: Vec<f32> = Vec::new();
    let mut colors: Vec<f32> = Vec::new();

    let mut corner_v = [0.0f32; 8];
    let mut corner_x = [0.0f32; 8];
    let mut corner_y = [0.0f32; 8];
    let mut corner_z = [0.0f32; 8];
    let mut corner_nx = [0.0f32; 8];
    let mut corner_ny = [0.0f32; 8];
    let mut corner_nz = [0.0f32; 8];
    let mut corner_r = [0.0f32; 8];
    let mut corner_g = [0.0f32; 8];
    let mut corner_b = [0.0f32; 8];
    let mut gtmp = [0.0f32; 3];

    for cz in -1i32..(nz as i32) {
        for cy in -1i32..(ny as i32) {
            for cx in -1i32..(nx as i32) {
                let mut cube_index = 0u32;
                for c in 0..8 {
                    let ox = CORNER_OFF[c][0];
                    let oy = CORNER_OFF[c][1];
                    let oz = CORNER_OFF[c][2];
                    let sx = cx + ox;
                    let sy = cy + oy;
                    let sz = cz + oz;
                    if sx < 0 || sy < 0 || sz < 0 {
                        corner_v[c] = 0.0;
                    } else {
                        let ux = sx as usize;
                        let uy = sy as usize;
                        let uz = sz as usize;
                        if ux >= nx || uy >= ny || uz >= nz {
                            corner_v[c] = 0.0;
                        } else {
                            let i = idx(ux, uy, uz, nx, ny);
                            corner_v[c] = values[i];
                        }
                    }
                    if corner_v[c] < ISO {
                        cube_index |= 1 << c;
                    }
                    corner_x[c] = node_min_x as f32 + sx as f32 + 0.5;
                    corner_y[c] = node_min_y as f32 + sy as f32 + 0.5;
                    corner_z[c] = node_min_z as f32 + sz as f32 + 0.5;
                    sample_grad(&values, nx, ny, nz, sx, sy, sz, &mut gtmp);
                    corner_nx[c] = gtmp[0];
                    corner_ny[c] = gtmp[1];
                    corner_nz[c] = gtmp[2];
                    if sx >= 0 && sy >= 0 && sz >= 0 {
                        let ux = sx as usize;
                        let uy = sy as usize;
                        let uz = sz as usize;
                        if ux < nx && uy < ny && uz < nz {
                            let i = idx(ux, uy, uz, nx, ny);
                            corner_r[c] = col_r[i];
                            corner_g[c] = col_g[i];
                            corner_b[c] = col_b[i];
                        } else {
                            corner_r[c] = 0.0;
                            corner_g[c] = 0.0;
                            corner_b[c] = 0.0;
                        }
                    } else {
                        corner_r[c] = 0.0;
                        corner_g[c] = 0.0;
                        corner_b[c] = 0.0;
                    }
                }

                if cube_index == 0 || cube_index == 255 {
                    continue;
                }

                let mut a00 = 0.0f64;
                let mut a01 = 0.0f64;
                let mut a02 = 0.0f64;
                let mut a11 = 0.0f64;
                let mut a12 = 0.0f64;
                let mut a22 = 0.0f64;
                let mut bb0 = 0.0f64;
                let mut bb1 = 0.0f64;
                let mut bb2 = 0.0f64;
                let mut cx_sum = 0.0f64;
                let mut cy_sum = 0.0f64;
                let mut cz_sum = 0.0f64;
                let mut nx_sum = 0.0f64;
                let mut ny_sum = 0.0f64;
                let mut nz_sum = 0.0f64;
                let mut n_herm = 0i32;
                let mut col_sr = 0.0f64;
                let mut col_sg = 0.0f64;
                let mut col_sb = 0.0f64;

                for e in 0..12 {
                    let a = EDGE_PAIRS[e][0];
                    let b = EDGE_PAIRS[e][1];
                    let va = corner_v[a];
                    let vb = corner_v[b];
                    let inside_a = va >= ISO;
                    let inside_b = vb >= ISO;
                    if inside_a == inside_b {
                        continue;
                    }
                    let denom = vb - va;
                    let mu = if denom.abs() < EPS {
                        0.5
                    } else {
                        (ISO - va) / denom
                    };
                    let t = mu.clamp(0.0, 1.0);
                    let px = corner_x[a] + (corner_x[b] - corner_x[a]) * t;
                    let py = corner_y[a] + (corner_y[b] - corner_y[a]) * t;
                    let pz = corner_z[a] + (corner_z[b] - corner_z[a]) * t;

                    let mut nx_l = corner_nx[a] + (corner_nx[b] - corner_nx[a]) * t;
                    let mut ny_l = corner_ny[a] + (corner_ny[b] - corner_ny[a]) * t;
                    let mut nz_l = corner_nz[a] + (corner_nz[b] - corner_nz[a]) * t;
                    let nl = (nx_l * nx_l + ny_l * ny_l + nz_l * nz_l).sqrt();
                    if nl > EPS {
                        nx_l /= nl;
                        ny_l /= nl;
                        nz_l /= nl;
                    } else {
                        nx_l = 0.0;
                        ny_l = 1.0;
                        nz_l = 0.0;
                    }

                    let d = nx_l * px + ny_l * py + nz_l * pz;
                    a00 += (nx_l * nx_l) as f64;
                    a01 += (nx_l * ny_l) as f64;
                    a02 += (nx_l * nz_l) as f64;
                    a11 += (ny_l * ny_l) as f64;
                    a12 += (ny_l * nz_l) as f64;
                    a22 += (nz_l * nz_l) as f64;
                    bb0 += (nx_l * d) as f64;
                    bb1 += (ny_l * d) as f64;
                    bb2 += (nz_l * d) as f64;

                    cx_sum += px as f64;
                    cy_sum += py as f64;
                    cz_sum += pz as f64;
                    nx_sum += nx_l as f64;
                    ny_sum += ny_l as f64;
                    nz_sum += nz_l as f64;
                    n_herm += 1;

                    let use_a = va >= vb;
                    col_sr += if use_a {
                        corner_r[a] as f64
                    } else {
                        corner_r[b] as f64
                    };
                    col_sg += if use_a {
                        corner_g[a] as f64
                    } else {
                        corner_g[b] as f64
                    };
                    col_sb += if use_a {
                        corner_b[a] as f64
                    } else {
                        corner_b[b] as f64
                    };
                }

                if n_herm == 0 {
                    continue;
                }

                let min_wx = node_min_x as f32 + cx as f32 + 0.5;
                let max_wx = node_min_x as f32 + cx as f32 + 1.5;
                let min_wy = node_min_y as f32 + cy as f32 + 0.5;
                let max_wy = node_min_y as f32 + cy as f32 + 1.5;
                let min_wz = node_min_z as f32 + cz as f32 + 0.5;
                let max_wz = node_min_z as f32 + cz as f32 + 1.5;

                let nh = n_herm as f64;
                let mx = cx_sum / nh;
                let my = cy_sum / nh;
                let mz = cz_sum / nh;

                // Tikhonov regularization: add λI to pull the QEF solution toward the
                // mass point when normals are nearly coplanar (rank-deficient system).
                // λ scales with 1/nh so poorly-constrained cells (few crossings, prone to
                // spikes) get strong regularization while well-constrained cells (many
                // diverse normals, real sharp corners) get weak relative pull and stay sharp.
                let lambda = 0.1 / nh;
                let ra00 = a00 + lambda;
                let ra11 = a11 + lambda;
                let ra22 = a22 + lambda;
                let rb0 = bb0 + lambda * mx;
                let rb1 = bb1 + lambda * my;
                let rb2 = bb2 + lambda * mz;

                let (px, py, pz) =
                    if let Some(s) = solve3x3(ra00, a01, a02, ra11, a12, ra22, rb0, rb1, rb2) {
                        (
                            s[0].clamp(min_wx as f64, max_wx as f64) as f32,
                            s[1].clamp(min_wy as f64, max_wy as f64) as f32,
                            s[2].clamp(min_wz as f64, max_wz as f64) as f32,
                        )
                    } else {
                        (
                            mx.clamp(min_wx as f64, max_wx as f64) as f32,
                            my.clamp(min_wy as f64, max_wy as f64) as f32,
                            mz.clamp(min_wz as f64, max_wz as f64) as f32,
                        )
                    };

                let vi = (positions.len() / 3) as u32;
                cell_vertex.insert((cx, cy, cz), vi);
                positions.extend_from_slice(&[px, py, pz]);

                // Use the average of the Hermite normals accumulated during QEF building.
                // These are the actual surface normals at each edge crossing — much more
                // accurate at corners than a nearest-neighbour lattice gradient lookup.
                let mut nnx = (nx_sum / nh) as f32;
                let mut nny = (ny_sum / nh) as f32;
                let mut nnz = (nz_sum / nh) as f32;
                let gn = (nnx * nnx + nny * nny + nnz * nnz).sqrt();
                if gn > EPS {
                    nnx /= gn;
                    nny /= gn;
                    nnz /= gn;
                } else {
                    nnx = 0.0;
                    nny = 1.0;
                    nnz = 0.0;
                }
                normals.extend_from_slice(&[nnx, nny, nnz]);

                let nh_f = n_herm as f32;
                colors.extend_from_slice(&[
                    (col_sr as f32 / nh_f).clamp(0.0, 1.0),
                    (col_sg as f32 / nh_f).clamp(0.0, 1.0),
                    (col_sb as f32 / nh_f).clamp(0.0, 1.0),
                ]);
            }
        }
    }

    if positions.is_empty() {
        return None;
    }

    let mut raw_indices: Vec<u32> = Vec::new();

    for ix in 0..nx - 1 {
        for iy in 0..ny {
            for iz in 0..nz {
                let va = sample_field(&values, ix as i32, iy as i32, iz as i32, nx, ny, nz);
                let vb = sample_field(&values, ix as i32 + 1, iy as i32, iz as i32, nx, ny, nz);
                if (va >= ISO) == (vb >= ISO) {
                    continue;
                }
                if let (Some(&c0), Some(&c1), Some(&c2), Some(&c3)) = (
                    cell_vertex.get(&(ix as i32, iy as i32 - 1, iz as i32 - 1)),
                    cell_vertex.get(&(ix as i32, iy as i32, iz as i32 - 1)),
                    cell_vertex.get(&(ix as i32, iy as i32, iz as i32)),
                    cell_vertex.get(&(ix as i32, iy as i32 - 1, iz as i32)),
                ) {
                    if va < ISO {
                        raw_indices.extend([c0, c3, c2, c0, c2, c1]);
                    } else {
                        raw_indices.extend([c0, c1, c2, c0, c2, c3]);
                    }
                }
            }
        }
    }

    for iy in 0..ny - 1 {
        for ix in 0..nx {
            for iz in 0..nz {
                let va = sample_field(&values, ix as i32, iy as i32, iz as i32, nx, ny, nz);
                let vb = sample_field(&values, ix as i32, iy as i32 + 1, iz as i32, nx, ny, nz);
                if (va >= ISO) == (vb >= ISO) {
                    continue;
                }
                if let (Some(&c0), Some(&c1), Some(&c2), Some(&c3)) = (
                    cell_vertex.get(&(ix as i32 - 1, iy as i32, iz as i32 - 1)),
                    cell_vertex.get(&(ix as i32 - 1, iy as i32, iz as i32)),
                    cell_vertex.get(&(ix as i32, iy as i32, iz as i32)),
                    cell_vertex.get(&(ix as i32, iy as i32, iz as i32 - 1)),
                ) {
                    if va < ISO {
                        raw_indices.extend([c0, c3, c2, c0, c2, c1]);
                    } else {
                        raw_indices.extend([c0, c1, c2, c0, c2, c3]);
                    }
                }
            }
        }
    }

    for iz in 0..nz - 1 {
        for ix in 0..nx {
            for iy in 0..ny {
                let va = sample_field(&values, ix as i32, iy as i32, iz as i32, nx, ny, nz);
                let vb = sample_field(&values, ix as i32, iy as i32, iz as i32 + 1, nx, ny, nz);
                if (va >= ISO) == (vb >= ISO) {
                    continue;
                }
                if let (Some(&c0), Some(&c1), Some(&c2), Some(&c3)) = (
                    cell_vertex.get(&(ix as i32 - 1, iy as i32 - 1, iz as i32)),
                    cell_vertex.get(&(ix as i32, iy as i32 - 1, iz as i32)),
                    cell_vertex.get(&(ix as i32, iy as i32, iz as i32)),
                    cell_vertex.get(&(ix as i32 - 1, iy as i32, iz as i32)),
                ) {
                    if va < ISO {
                        raw_indices.extend([c0, c3, c2, c0, c2, c1]);
                    } else {
                        raw_indices.extend([c0, c1, c2, c0, c2, c3]);
                    }
                }
            }
        }
    }

    if raw_indices.is_empty() {
        return None;
    }

    let nv = positions.len() / 3;
    Some(MeshBuffers {
        positions,
        normals,
        colors,
        mat_kind: vec![mat_k; nv],
        ao: vec![1.0f32; nv],
        emission_tint: vec![0.0f32; nv * 3],
        indices: raw_indices,
    })
}

fn bucket_key_parts(v: &Voxel) -> (u32, u8) {
    let mat_tag = match v.material {
        MaterialId::Plastic => 0u8,
        MaterialId::Metal => 1,
        MaterialId::Rubber => 2,
        MaterialId::Glass => 3,
        MaterialId::Water => 4,
        MaterialId::Glow => 5,
        MaterialId::Velvet => 6,
        MaterialId::Wax => 7,
        MaterialId::Holographic => 8,
    };
    (v.color, mat_tag)
}

fn merge_mesh_buffers(parts: Vec<MeshBuffers>) -> MeshBuffers {
    let mut out = MeshBuffers::default();
    for p in parts {
        let base = (out.positions.len() / 3) as u32;
        out.positions.extend_from_slice(&p.positions);
        out.normals.extend_from_slice(&p.normals);
        out.colors.extend_from_slice(&p.colors);
        out.mat_kind.extend_from_slice(&p.mat_kind);
        out.ao.extend_from_slice(&p.ao);
        out.emission_tint.extend_from_slice(&p.emission_tint);
        for &i in &p.indices {
            out.indices.push(base + i);
        }
    }
    out
}

/// Marching cubes per color|material bucket (matches web `computeMarchingCubes`).
pub fn build_marching_cubes_merged(voxels: &[Voxel]) -> MeshBuffers {
    build_marching_cubes_merged_with_progress(voxels, |_, _, _| {})
}

/// Like [`build_marching_cubes_merged`] but calls `on_progress(fraction, done, total)` after each bucket.
pub fn build_marching_cubes_merged_with_progress<F>(voxels: &[Voxel], on_progress: F) -> MeshBuffers
where
    F: Fn(f32, usize, usize),
{
    build_marching_cubes_merged_cancellable(voxels, on_progress, || false)
}

/// Like [`build_marching_cubes_merged_with_progress`] but checks `should_cancel()` after each bucket
/// and returns an empty [`MeshBuffers`] immediately when it returns `true`.
pub fn build_marching_cubes_merged_cancellable<F, C>(
    voxels: &[Voxel],
    on_progress: F,
    should_cancel: C,
) -> MeshBuffers
where
    F: Fn(f32, usize, usize),
    C: Fn() -> bool,
{
    let t0 = Instant::now();
    let mut buckets: AHashMap<(u32, u8), AHashMap<VoxelCoord, Voxel>> = AHashMap::new();
    for v in voxels {
        let k = bucket_key_parts(v);
        buckets
            .entry(k)
            .or_default()
            .insert(coord_key(v.x, v.y, v.z), *v);
    }
    let n_buckets = buckets.len();
    log::info!(
        target: "voxelle_load",
        "marching_cubes_merged: {} input voxels → {} color|material buckets (bucketing {:?})",
        voxels.len(),
        n_buckets,
        t0.elapsed()
    );
    let mut parts = Vec::new();
    for (i, b) in buckets.values().enumerate() {
        if should_cancel() {
            log::info!(target: "voxelle_load", "marching_cubes_merged: cancelled after {i}/{n_buckets} buckets");
            return MeshBuffers::default();
        }
        let t_bucket = Instant::now();
        let idx = i + 1;
        match marching_cubes_bucket(b, Some((idx, n_buckets))) {
            Some(m) => {
                log::info!(
                    target: "voxelle_load",
                    "marching_cubes_merged: finished bucket {idx}/{n_buckets} in {:?}",
                    t_bucket.elapsed()
                );
                parts.push(m);
            }
            None => {
                log::info!(
                    target: "voxelle_load",
                    "marching_cubes_merged: bucket {idx}/{n_buckets} empty mesh in {:?}",
                    t_bucket.elapsed()
                );
            }
        }
        on_progress(idx as f32 / n_buckets as f32, idx, n_buckets);
    }
    log::info!(
        target: "voxelle_load",
        "marching_cubes_merged: done {:?} total",
        t0.elapsed()
    );
    merge_mesh_buffers(parts)
}

/// Dual contouring per bucket with full-scene occupancy (matches web `computeDualContour`).
pub fn build_dual_contour_merged(voxels: &[Voxel]) -> MeshBuffers {
    build_dual_contour_merged_with_progress(voxels, |_, _, _| {})
}

/// Like [`build_dual_contour_merged`] but calls `on_progress(fraction, done, total)` after each bucket.
pub fn build_dual_contour_merged_with_progress<F>(voxels: &[Voxel], on_progress: F) -> MeshBuffers
where
    F: Fn(f32, usize, usize),
{
    build_dual_contour_merged_cancellable(voxels, on_progress, || false)
}

/// Like [`build_dual_contour_merged_with_progress`] but checks `should_cancel()` after each bucket
/// and returns an empty [`MeshBuffers`] immediately when it returns `true`.
pub fn build_dual_contour_merged_cancellable<F, C>(
    voxels: &[Voxel],
    on_progress: F,
    should_cancel: C,
) -> MeshBuffers
where
    F: Fn(f32, usize, usize),
    C: Fn() -> bool,
{
    let t0 = Instant::now();
    let full = crate::greedy_mesh::voxel_map(voxels);
    let mut buckets: AHashMap<(u32, u8), AHashMap<VoxelCoord, Voxel>> = AHashMap::new();
    for v in voxels {
        let k = bucket_key_parts(v);
        buckets
            .entry(k)
            .or_default()
            .insert(coord_key(v.x, v.y, v.z), *v);
    }
    let n_buckets = buckets.len();
    log::info!(
        target: "voxelle_load",
        "dual_contour_merged: {} input voxels → {} color|material buckets (bucketing {:?})",
        voxels.len(),
        n_buckets,
        t0.elapsed()
    );
    let mut parts = Vec::new();
    for (i, b) in buckets.values().enumerate() {
        if should_cancel() {
            log::info!(target: "voxelle_load", "dual_contour_merged: cancelled after {i}/{n_buckets} buckets");
            return MeshBuffers::default();
        }
        let t_bucket = Instant::now();
        let idx = i + 1;
        match dual_contour_bucket(b, &full) {
            Some(m) => {
                log::info!(
                    target: "voxelle_load",
                    "dual_contour_merged: finished bucket {idx}/{n_buckets} in {:?}",
                    t_bucket.elapsed()
                );
                parts.push(m);
            }
            None => {
                log::info!(
                    target: "voxelle_load",
                    "dual_contour_merged: bucket {idx}/{n_buckets} empty mesh in {:?}",
                    t_bucket.elapsed()
                );
            }
        }
        on_progress(idx as f32 / n_buckets as f32, idx, n_buckets);
    }
    log::info!(
        target: "voxelle_load",
        "dual_contour_merged: done {:?} total",
        t0.elapsed()
    );
    merge_mesh_buffers(parts)
}
