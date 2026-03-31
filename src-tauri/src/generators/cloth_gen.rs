//! Cloth patch generator (web `getClothPatchFromPinsVoxels` parity): PBD + gravity on a UV grid,
//! then voxelize + brush thicken like rope.

use crate::greedy_mesh::VoxelCoord;
use crate::voxel_edit::{ensure_grid_fits_coord, voxel_line_dda, BrushShape, VoxelEditDelta};
use crate::voxelle::{MaterialId, Voxel, VoxelleFile};
use ahash::AHashMap;
use std::collections::HashSet;

use super::rope_gen::thicken_centerline_voxels;

const CLOTH_PATCH_MAX_CELLS: usize = 2200;
const CLOTH_PATCH_MAX_DIM: i32 = 46;

/// Web `ClothSimOptions` (percent scales from UI are divided by 100 before passing).
pub struct ClothSimOptions {
    pub gravity_scale: f64,
    pub stiffness_scale: f64,
    /// `None` = auto from tension (`round(28 + 22 * t)`).
    pub iterations: Option<u32>,
    pub constraint_passes: u32,
}

impl Default for ClothSimOptions {
    fn default() -> Self {
        Self {
            gravity_scale: 1.0,
            stiffness_scale: 1.0,
            iterations: None,
            constraint_passes: 2,
        }
    }
}

fn cloth_patch_point_in_polygon_2d(px: f64, py: f64, poly: &[(f64, f64)]) -> bool {
    let mut inside = false;
    let n = poly.len();
    if n < 3 {
        return false;
    }
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if (yi > py) != (yj > py) {
            let x_int = ((xj - xi) * (py - yi)) / (yj - yi) + xi;
            if px < x_int {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

fn vec_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec_add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn vec_scale(p: [f64; 3], s: f64) -> [f64; 3] {
    [p[0] * s, p[1] * s, p[2] * s]
}

fn vec_len(p: [f64; 3]) -> f64 {
    (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
}

fn vec_norm(p: [f64; 3]) -> [f64; 3] {
    let l = vec_len(p);
    if l < 1e-12 {
        return [0.0, 0.0, 0.0];
    }
    [p[0] / l, p[1] / l, p[2] / l]
}

fn vec_cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn vec_dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Web `ropeGravityVector` / `RopeGravityDirection`.
pub fn rope_gravity_unit(gravity_direction: &str) -> [f64; 3] {
    vec_norm(match gravity_direction {
        "down" => [0.0, -1.0, 0.0],
        "up" => [0.0, 1.0, 0.0],
        "left" => [-1.0, 0.0, 0.0],
        "right" => [1.0, 0.0, 0.0],
        "forward" => [0.0, 0.0, -1.0],
        "back" => [0.0, 0.0, 1.0],
        _ => [0.0, -1.0, 0.0],
    })
}

struct Node {
    iu: i32,
    iv: i32,
    pos: [f64; 3],
    init: [f64; 3],
    plane_init: [f64; 3],
    pinned: bool,
}

struct Edge {
    a: usize,
    b: usize,
    rest: f64,
}

/// Closed polygon of 3+ pin voxels → draped voxel path (before brush thicken).
pub fn cloth_patch_centerline_from_pins(
    pins: &[[i32; 3]],
    tension: f64,
    gravity_direction: &str,
    sim: &ClothSimOptions,
) -> Vec<VoxelCoord> {
    let raw: Vec<[i32; 3]> = pins
        .iter()
        .enumerate()
        .filter(|(i, p)| {
            *i == 0
                || pins[i - 1][0] != p[0]
                || pins[i - 1][1] != p[1]
                || pins[i - 1][2] != p[2]
        })
        .map(|(_, p)| *p)
        .collect();

    if raw.len() < 3 {
        return Vec::new();
    }

    let o = [
        raw[0][0] as f64,
        raw[0][1] as f64,
        raw[0][2] as f64,
    ];
    let e1 = vec_sub(
        [raw[1][0] as f64, raw[1][1] as f64, raw[1][2] as f64],
        o,
    );
    let mut nvec = vec_cross(
        e1,
        vec_sub(
            [raw[2][0] as f64, raw[2][1] as f64, raw[2][2] as f64],
            o,
        ),
    );
    let mut k = 3usize;
    while vec_len(nvec) < 1e-9 && k < raw.len() {
        nvec = vec_cross(
            e1,
            vec_sub(
                [raw[k][0] as f64, raw[k][1] as f64, raw[k][2] as f64],
                o,
            ),
        );
        k += 1;
    }

    if vec_len(nvec) < 1e-9 {
        let mut out: Vec<VoxelCoord> = Vec::new();
        let mut seen = HashSet::new();
        let n = raw.len();
        for i in 0..n {
            let seg = voxel_line_dda(
                (raw[i][0], raw[i][1], raw[i][2]),
                (raw[(i + 1) % n][0], raw[(i + 1) % n][1], raw[(i + 1) % n][2]),
            );
            for p in seg {
                if seen.insert(p) {
                    out.push(p);
                }
            }
        }
        return out;
    }

    let nunit = vec_norm(nvec);
    let mut uaxis = vec_norm(e1);
    if vec_dot3(uaxis, nunit).abs() > 0.99 {
        uaxis = vec_norm(vec_cross(nunit, [1.0, 0.0, 0.0]));
        if vec_len(uaxis) < 1e-6 {
            uaxis = vec_norm(vec_cross(nunit, [0.0, 1.0, 0.0]));
        }
    }
    let vaxis = vec_norm(vec_cross(nunit, uaxis));

    let mut uv_poly: Vec<(f64, f64)> = Vec::with_capacity(raw.len());
    for p in &raw {
        let d = vec_sub(
            [p[0] as f64, p[1] as f64, p[2] as f64],
            o,
        );
        uv_poly.push((vec_dot3(d, uaxis), vec_dot3(d, vaxis)));
    }

    let mut area2 = 0.0;
    for i in 0..uv_poly.len() {
        let p = uv_poly[i];
        let q = uv_poly[(i + 1) % uv_poly.len()];
        area2 += p.0 * q.1 - q.0 * p.1;
    }
    if area2 < 0.0 {
        uv_poly.reverse();
    }

    let mut umin = uv_poly[0].0;
    let mut umax = uv_poly[0].0;
    let mut vmin = uv_poly[0].1;
    let mut vmax = uv_poly[0].1;
    for q in &uv_poly {
        umin = umin.min(q.0);
        umax = umax.max(q.0);
        vmin = vmin.min(q.1);
        vmax = vmax.max(q.1);
    }

    let ur = (umax - umin).max(1.0);
    let vr = (vmax - vmin).max(1.0);
    let dim_slack = CLOTH_PATCH_MAX_DIM - 3;
    let mut step_u = (ur / dim_slack as f64).ceil().max(1.0) as i32;
    let mut step_v = (vr / dim_slack as f64).ceil().max(1.0) as i32;

    for _guard in 0..512 {
        let nu0 = (ur / step_u as f64).ceil() as i32 + 3;
        let nv0 = (vr / step_v as f64).ceil() as i32 + 3;
        if (nu0 as usize * nv0 as usize) <= CLOTH_PATCH_MAX_CELLS
            && nu0.max(nv0) <= CLOTH_PATCH_MAX_DIM
        {
            break;
        }
        if ur / step_u as f64 >= vr / step_v as f64 {
            step_u += 1;
        } else {
            step_v += 1;
        }
    }

    {
        let mut nu0 = (ur / step_u as f64).ceil() as i32 + 3;
        let mut nv0 = (vr / step_v as f64).ceil() as i32 + 3;
        for _safety in 0..10000 {
            if (nu0 as usize * nv0 as usize) <= CLOTH_PATCH_MAX_CELLS
                && nu0.max(nv0) <= CLOTH_PATCH_MAX_DIM
            {
                break;
            }
            if ur / step_u as f64 >= vr / step_v as f64 {
                step_u += 1;
            } else {
                step_v += 1;
            }
            nu0 = (ur / step_u as f64).ceil() as i32 + 3;
            nv0 = (vr / step_v as f64).ceil() as i32 + 3;
        }
    }

    let mut pinned_grid_keys: ahash::AHashMap<String, usize> = ahash::AHashMap::new();
    for (j, uvp) in uv_poly.iter().enumerate() {
        let base_iu = ((uvp.0 / step_u as f64).round() as i32) * step_u;
        let base_iv = ((uvp.1 / step_v as f64).round() as i32) * step_v;
        let mut best_key: Option<String> = None;
        let mut best_d2 = f64::INFINITY;
        let mut diu = -step_u;
        while diu <= step_u {
            let mut div = -step_v;
            while div <= step_v {
                let ciu = base_iu + diu;
                let civ = base_iv + div;
                if cloth_patch_point_in_polygon_2d(ciu as f64, civ as f64, &uv_poly) {
                    let du = (ciu as f64) - uvp.0;
                    let dv = (civ as f64) - uvp.1;
                    let d2 = du * du + dv * dv;
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best_key = Some(format!("{ciu},{civ}"));
                    }
                }
                div += step_v;
            }
            diu += step_u;
        }
        if let Some(key) = best_key {
            pinned_grid_keys.insert(key, j);
        }
    }

    fn interpolate_from_pins(
        iu: i32,
        iv: i32,
        uv_poly: &[(f64, f64)],
        raw: &[[i32; 3]],
    ) -> [f64; 3] {
        let iuf = iu as f64;
        let ivf = iv as f64;
        let mut w_sum = 0.0;
        let mut wx = 0.0;
        let mut wy = 0.0;
        let mut wz = 0.0;
        for (j, uvp) in uv_poly.iter().enumerate() {
            let du = iuf - uvp.0;
            let dv = ivf - uvp.1;
            let d2 = du * du + dv * dv;
            if d2 < 0.01 {
                return [
                    raw[j][0] as f64,
                    raw[j][1] as f64,
                    raw[j][2] as f64,
                ];
            }
            let w = 1.0 / d2;
            w_sum += w;
            wx += w * raw[j][0] as f64;
            wy += w * raw[j][1] as f64;
            wz += w * raw[j][2] as f64;
        }
        [wx / w_sum, wy / w_sum, wz / w_sum]
    }

    let u_start = (umin / step_u as f64).floor() as i32 * step_u - step_u;
    let u_end = (umax / step_u as f64).ceil() as i32 * step_u + step_u;
    let v_start = (vmin / step_v as f64).floor() as i32 * step_v - step_v;
    let v_end = (vmax / step_v as f64).ceil() as i32 * step_v + step_v;

    let mut node_index_by_key: ahash::AHashMap<String, usize> = ahash::AHashMap::new();
    let mut nodes: Vec<Node> = Vec::new();

    let mut iu = u_start;
    while iu <= u_end {
        let mut iv = v_start;
        while iv <= v_end {
            if cloth_patch_point_in_polygon_2d(iu as f64, iv as f64, &uv_poly) {
                let plane_init = [
                    o[0] + iu as f64 * uaxis[0] + iv as f64 * vaxis[0],
                    o[1] + iu as f64 * uaxis[1] + iv as f64 * vaxis[1],
                    o[2] + iu as f64 * uaxis[2] + iv as f64 * vaxis[2],
                ];
                let gk = format!("{iu},{iv}");
                let pin_idx = pinned_grid_keys.get(&gk).copied();
                let pinned = pin_idx.is_some();
                let pos = if let Some(pi) = pin_idx {
                    [
                        raw[pi][0] as f64,
                        raw[pi][1] as f64,
                        raw[pi][2] as f64,
                    ]
                } else {
                    interpolate_from_pins(iu, iv, &uv_poly, &raw)
                };
                let ni = nodes.len();
                nodes.push(Node {
                    iu,
                    iv,
                    pos,
                    init: pos,
                    plane_init,
                    pinned,
                });
                node_index_by_key.insert(gk, ni);
            }
            iv += step_v;
        }
        iu += step_u;
    }

    if nodes.is_empty() {
        return Vec::new();
    }

    let mut edges: Vec<Edge> = Vec::new();
    let neigh: [(i32, i32); 4] = [(step_u, 0), (-step_u, 0), (0, step_v), (0, -step_v)];
    for ni in 0..nodes.len() {
        let a = &nodes[ni];
        for (du, dv) in neigh {
            let key = format!("{},{}", a.iu + du, a.iv + dv);
            let Some(&bj) = node_index_by_key.get(&key) else {
                continue;
            };
            if bj <= ni {
                continue;
            }
            let b = &nodes[bj];
            let rest = vec_len(vec_sub(b.plane_init, a.plane_init));
            if rest < 1e-9 {
                continue;
            }
            edges.push(Edge { a: ni, b: bj, rest });
        }
    }

    let mut pos: Vec<[f64; 3]> = nodes.iter().map(|n| n.pos).collect();
    let pinned_arr: Vec<bool> = nodes.iter().map(|n| n.pinned).collect();
    let init: Vec<[f64; 3]> = nodes.iter().map(|n| n.init).collect();

    let t0 = tension.clamp(0.0, 1.0);
    let gravity_scale = sim.gravity_scale.max(0.0);
    let stiffness_scale = sim.stiffness_scale.clamp(0.05, 2.0);
    let iterations = if let Some(it) = sim.iterations {
        (it as i32).clamp(4, 96) as usize
    } else {
        (28.0 + 22.0 * t0).round() as usize
    };
    let relax = ((0.35 + 0.6 * t0) * stiffness_scale).min(0.99);
    let cell = step_u.max(step_v) as f64;
    let down = rope_gravity_unit(gravity_direction);
    let grav_step = cell * (0.02 + 0.18 * (1.0 - t0)) * gravity_scale;
    let constraint_passes = sim.constraint_passes.clamp(1, 6) as usize;

    for _it in 0..iterations {
        for p in 0..pos.len() {
            if !pinned_arr[p] {
                pos[p] = vec_add(pos[p], vec_scale(down, grav_step));
            }
        }
        for _pass in 0..constraint_passes {
            for e in &edges {
                let ia = e.a;
                let ib = e.b;
                if pinned_arr[ia] && pinned_arr[ib] {
                    continue;
                }
                let pa = pos[ia];
                let pb = pos[ib];
                let d = vec_sub(pb, pa);
                let dist = vec_len(d);
                if dist < 1e-12 {
                    continue;
                }
                let diff = (dist - e.rest) / dist;
                let correction = vec_scale(d, diff * 0.5 * relax);
                if !pinned_arr[ia] && !pinned_arr[ib] {
                    pos[ia] = vec_add(pa, correction);
                    pos[ib] = vec_sub(pb, correction);
                } else if pinned_arr[ia] {
                    pos[ib] = vec_sub(pb, vec_scale(correction, 2.0));
                } else {
                    pos[ia] = vec_add(pa, vec_scale(correction, 2.0));
                }
            }
        }
        for p in 0..pos.len() {
            if pinned_arr[p] {
                pos[p] = init[p];
            }
        }
    }

    let mut path: Vec<VoxelCoord> = Vec::new();
    let mut path_seen: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut push_path = |q: VoxelCoord| {
        if path_seen.insert(q) {
            path.push(q);
        }
    };

    for p in &pos {
        push_path((
            p[0].round() as i32,
            p[1].round() as i32,
            p[2].round() as i32,
        ));
    }

    let g_dir = down;
    const MAX_BRIDGE_CARDINAL: i32 = 36;
    const MAX_BRIDGE_DIAG: i32 = 24;
    const MAX_VERTICAL_ALIGN: f64 = 0.92;

    let bridge_chord = |pa: VoxelCoord, pb: VoxelCoord, max_cheb: i32, path_seen: &mut HashSet<(i32, i32, i32)>, path: &mut Vec<VoxelCoord>| {
        let dx = pb.0 - pa.0;
        let dy = pb.1 - pa.1;
        let dz = pb.2 - pa.2;
        let cheb = dx.abs().max(dy.abs()).max(dz.abs());
        if cheb <= 1 {
            return;
        }
        if cheb > max_cheb {
            return;
        }
        let dist = ((dx * dx + dy * dy + dz * dz) as f64).sqrt();
        if dist < 1e-9 {
            return;
        }
        let align = ((dx as f64) * g_dir[0] + (dy as f64) * g_dir[1] + (dz as f64) * g_dir[2]).abs() / dist;
        if align > MAX_VERTICAL_ALIGN {
            return;
        }
        for s in voxel_line_dda(pa, pb) {
            if path_seen.insert(s) {
                path.push(s);
            }
        }
    };

    for e in &edges {
        let pa = (
            pos[e.a][0].round() as i32,
            pos[e.a][1].round() as i32,
            pos[e.a][2].round() as i32,
        );
        let pb = (
            pos[e.b][0].round() as i32,
            pos[e.b][1].round() as i32,
            pos[e.b][2].round() as i32,
        );
        bridge_chord(pa, pb, MAX_BRIDGE_CARDINAL, &mut path_seen, &mut path);
    }

    let path_diag_neigh: [(i32, i32); 4] = [
        (step_u, step_v),
        (step_u, -step_v),
        (-step_u, step_v),
        (-step_u, -step_v),
    ];
    for ni in 0..nodes.len() {
        for (du, dv) in path_diag_neigh {
            let key = format!("{},{}", nodes[ni].iu + du, nodes[ni].iv + dv);
            let Some(&bj) = node_index_by_key.get(&key) else {
                continue;
            };
            if bj <= ni {
                continue;
            }
            let pa = (
                pos[ni][0].round() as i32,
                pos[ni][1].round() as i32,
                pos[ni][2].round() as i32,
            );
            let pb = (
                pos[bj][0].round() as i32,
                pos[bj][1].round() as i32,
                pos[bj][2].round() as i32,
            );
            bridge_chord(pa, pb, MAX_BRIDGE_DIAG, &mut path_seen, &mut path);
        }
    }

    path
}

/// Cloth voxel footprint for hover preview (no file mutation).
pub fn preview_cloth_voxels(
    pins: &[[i32; 3]],
    tension: f32,
    gravity_direction: &str,
    brush_radius_index: u32,
    brush_shape: BrushShape,
    sim: &ClothSimOptions,
) -> Vec<VoxelCoord> {
    let path = cloth_patch_centerline_from_pins(
        pins,
        tension as f64,
        gravity_direction,
        sim,
    );
    if path.is_empty() {
        return Vec::new();
    }
    thicken_centerline_voxels(&path, brush_radius_index, brush_shape)
}

pub fn generator_cloth_from_pins(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    pins: &[[i32; 3]],
    tension: f32,
    gravity_direction: &str,
    brush_radius_index: u32,
    brush_shape: BrushShape,
    color: u32,
    material: MaterialId,
    sim: ClothSimOptions,
) -> Result<Vec<VoxelEditDelta>, String> {
    let path = cloth_patch_centerline_from_pins(
        pins,
        tension as f64,
        gravity_direction,
        &sim,
    );
    if path.is_empty() {
        return Ok(Vec::new());
    }
    let cells = thicken_centerline_voxels(&path, brush_radius_index, brush_shape);
    let mut out = Vec::new();
    let mut seen: HashSet<VoxelCoord> = HashSet::new();
    for (x, y, z) in cells {
        ensure_grid_fits_coord(file, x, y, z);
        if !seen.insert((x, y, z)) {
            continue;
        }
        if voxel_map.contains_key(&(x, y, z)) {
            continue;
        }
        let nv = Voxel {
            x,
            y,
            z,
            color,
            material,
            object_id: file.active_object_id,
        };
        let idx = file.voxels.len();
        file.voxels.push(nv);
        voxel_map.insert((x, y, z), idx);
        out.push(VoxelEditDelta::Added(nv));
    }
    Ok(out)
}
