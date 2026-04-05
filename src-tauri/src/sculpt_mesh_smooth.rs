//! Web parity for sculpt **smooth**: voxel majority (`sculptOps.ts`) and mesh Taubin + revoxelize (`sculptMeshLaplacian.ts`).

use crate::greedy_mesh::{self, VoxelCoord};
use crate::voxel_edit::VoxelEditDelta;
use crate::voxelle::{MaterialId, Voxel, VoxelleFile};
use ahash::{AHashMap, AHashSet};
use glam::Vec3;

/// Skip mesh Laplacian when ROI cell count exceeds this (web `MAX_LAPLACIAN_ROI_CELLS`).
pub const MAX_LAPLACIAN_ROI_CELLS: i64 = 70_000;

const SMOOTH_NEIGHBOR_RADIUS_MAX: i32 = 6;
const PROXY_COLOR: u32 = 0x888888;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SculptSmoothVariant {
    #[default]
    Majority,
    MeshLaplacian,
}

#[inline]
fn smooth_thresholds(neighbor_count: i32, aggressiveness: u32) -> (i32, i32) {
    let t = (aggressiveness.min(100) as f32) / 100.0;
    let fill_ratio = (5.0 / 6.0) * (1.0 - t) + (4.0 / 6.0) * t;
    let remove_ratio = (1.0 / 6.0) * (1.0 - t) + (2.0 / 6.0) * t;
    let min_filled_to_add = (fill_ratio * neighbor_count as f32).ceil() as i32;
    let max_filled_to_remove = (remove_ratio * neighbor_count as f32).floor() as i32;
    (min_filled_to_add, max_filled_to_remove)
}

fn blend_voxels_for_smooth(neighbors: &[Voxel]) -> (u32, MaterialId) {
    if neighbors.is_empty() {
        return (PROXY_COLOR, MaterialId::Plastic);
    }
    let n = neighbors.len() as i32;
    let mut sr = 0i32;
    let mut sg = 0i32;
    let mut sb = 0i32;
    for v in neighbors {
        let c = v.color;
        sr += ((c >> 16) & 0xff) as i32;
        sg += ((c >> 8) & 0xff) as i32;
        sb += (c & 0xff) as i32;
    }
    let color = (((sr / n).clamp(0, 255) as u32) << 16)
        | (((sg / n).clamp(0, 255) as u32) << 8)
        | ((sb / n).clamp(0, 255) as u32);

    let mut counts = [0u32; 6];
    for v in neighbors {
        let idx = v.material.material_index() as usize;
        if idx < 6 {
            counts[idx] += 1;
        }
    }
    let mut best_m = MaterialId::Plastic;
    let mut best_c = 0u32;
    for (i, &c) in counts.iter().enumerate() {
        if c > best_c {
            best_c = c;
            best_m = MaterialId::from_index(i as u8);
        }
    }
    (color, best_m)
}

#[inline]
fn in_grid_bounds(x: i32, y: i32, z: i32, grid_size: i32) -> bool {
    crate::voxel_edit::in_grid(x, y, z, grid_size)
}

/// Web `applySmooth` for footprint cells (unique); one logical pass.
pub fn apply_sculpt_smooth_majority_pass(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    footprint: &[VoxelCoord],
    grid_size: i32,
    neighbor_radius: u32,
    aggressiveness: u32,
    _stroke_color: u32,
    _stroke_material: MaterialId,
) -> Vec<VoxelEditDelta> {
    let r = (neighbor_radius.min(SMOOTH_NEIGHBOR_RADIUS_MAX.max(0) as u32) as i32)
        .min(SMOOTH_NEIGHBOR_RADIUS_MAX);
    let mut seen: AHashSet<VoxelCoord> = AHashSet::with_capacity(footprint.len());
    let mut unique: Vec<VoxelCoord> = Vec::new();
    for &p in footprint {
        if seen.insert(p) {
            unique.push(p);
        }
    }

    let mut out: Vec<VoxelEditDelta> = Vec::new();

    for (x, y, z) in unique {
        if !in_grid_bounds(x, y, z, grid_size) {
            continue;
        }
        let filled = voxel_map.contains_key(&(x, y, z));
        let mut filled_count = 0i32;
        let mut neighbor_voxels: Vec<Voxel> = Vec::new();
        let mut neighbor_slot_count = 0i32;

        if r <= 0 {
            for (nx, ny, nz) in crate::voxel_edit::neighbors_6((x, y, z)) {
                if !in_grid_bounds(nx, ny, nz, grid_size) {
                    continue;
                }
                neighbor_slot_count += 1;
                if let Some(&idx) = voxel_map.get(&(nx, ny, nz)) {
                    filled_count += 1;
                    neighbor_voxels.push(file.voxels[idx]);
                }
            }
        } else {
            for dz in -r..=r {
                for dy in -r..=r {
                    for dx in -r..=r {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        let nx = x + dx;
                        let ny = y + dy;
                        let nz = z + dz;
                        if !in_grid_bounds(nx, ny, nz, grid_size) {
                            continue;
                        }
                        neighbor_slot_count += 1;
                        if let Some(&idx) = voxel_map.get(&(nx, ny, nz)) {
                            filled_count += 1;
                            neighbor_voxels.push(file.voxels[idx]);
                        }
                    }
                }
            }
        }

        if neighbor_slot_count == 0 {
            continue;
        }
        let (min_filled_to_add, max_filled_to_remove) =
            smooth_thresholds(neighbor_slot_count, aggressiveness);

        if !filled && filled_count >= min_filled_to_add {
            let (nc, nm) = blend_voxels_for_smooth(&neighbor_voxels);
            let nv = Voxel {
                x,
                y,
                z,
                color: nc,
                material: nm,
                object_id: file.active_object_id,
            };
            let idx = file.voxels.len();
            file.voxels.push(nv);
            voxel_map.insert((x, y, z), idx);
            out.push(VoxelEditDelta::Added(nv));
        } else if filled && filled_count <= max_filled_to_remove {
            let Some(&remove_idx) = voxel_map.get(&(x, y, z)) else {
                continue;
            };
            let removed_voxel = file.voxels[remove_idx];
            let last = file.voxels.len() - 1;
            if remove_idx != last {
                file.voxels.swap(remove_idx, last);
                let moved = file.voxels[remove_idx];
                voxel_map.insert((moved.x, moved.y, moved.z), remove_idx);
            }
            file.voxels.pop();
            voxel_map.remove(&(x, y, z));
            out.push(VoxelEditDelta::Removed {
                voxel: removed_voxel,
            });
        }
    }

    out
}

fn roi_from_footprint_margin(
    footprint: &[VoxelCoord],
    margin: i32,
    grid_size: i32,
) -> Option<(i32, i32, i32, i32, i32, i32)> {
    if footprint.is_empty() {
        return None;
    }
    let (lo, hi) = crate::voxel_edit::grid_valid_range(grid_size);
    let mut min_x = i32::MAX;
    let mut max_x = i32::MIN;
    let mut min_y = i32::MAX;
    let mut max_y = i32::MIN;
    let mut min_z = i32::MAX;
    let mut max_z = i32::MIN;
    for &(x, y, z) in footprint {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
        min_z = min_z.min(z);
        max_z = max_z.max(z);
    }
    let m = margin.max(0);
    Some((
        (min_x - m).max(lo),
        (min_y - m).max(lo),
        (min_z - m).max(lo),
        (max_x + m).min(hi),
        (max_y + m).min(hi),
        (max_z + m).min(hi),
    ))
}

fn roi_cell_count(min_x: i32, min_y: i32, min_z: i32, max_x: i32, max_y: i32, max_z: i32) -> i64 {
    if min_x > max_x || min_y > max_y || min_z > max_z {
        return 0;
    }
    let w = (max_x - min_x + 1) as i64;
    let h = (max_y - min_y + 1) as i64;
    let d = (max_z - min_z + 1) as i64;
    w * h * d
}

fn build_full_voxel_map(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
) -> AHashMap<VoxelCoord, Voxel> {
    let mut m = AHashMap::with_capacity(voxel_map.len());
    for (&coord, &idx) in voxel_map {
        m.insert(coord, file.voxels[idx]);
    }
    m
}

fn build_adjacency(vertex_count: usize, indices: &[u32]) -> Vec<Vec<usize>> {
    let mut adj: Vec<Vec<usize>> = (0..vertex_count).map(|_| Vec::new()).collect();
    for t in (0..indices.len()).step_by(3) {
        if t + 2 >= indices.len() {
            break;
        }
        let a = indices[t] as usize;
        let b = indices[t + 1] as usize;
        let c = indices[t + 2] as usize;
        if a >= vertex_count || b >= vertex_count || c >= vertex_count {
            continue;
        }
        let push_unique = |adj: &mut [Vec<usize>], i: usize, j: usize| {
            if !adj[i].contains(&j) {
                adj[i].push(j);
            }
        };
        push_unique(&mut adj, a, b);
        push_unique(&mut adj, a, c);
        push_unique(&mut adj, b, a);
        push_unique(&mut adj, b, c);
        push_unique(&mut adj, c, a);
        push_unique(&mut adj, c, b);
    }
    adj
}

fn is_pinned_vertex(
    px: f64,
    py: f64,
    pz: f64,
    min_x: f64,
    min_y: f64,
    min_z: f64,
    max_x: f64,
    max_y: f64,
    max_z: f64,
) -> bool {
    const EPS: f64 = 1e-5;
    let on_x = (px - min_x).abs() < EPS || (max_x + 1.0 - px).abs() < EPS;
    let on_y = (py - min_y).abs() < EPS || (max_y + 1.0 - py).abs() < EPS;
    let on_z = (pz - min_z).abs() < EPS || (max_z + 1.0 - pz).abs() < EPS;
    on_x || on_y || on_z
}

fn umbrella_laplacian_step(pos: &mut [f64], adj: &[Vec<usize>], pinned: &[bool], lambda: f64) {
    let n = pos.len() / 3;
    let mut nx = vec![0.0f64; pos.len()];
    nx.copy_from_slice(pos);
    for i in 0..n {
        if pinned[i] {
            continue;
        }
        let nb = &adj[i];
        if nb.is_empty() {
            continue;
        }
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut sz = 0.0;
        for &j in nb {
            sx += pos[j * 3];
            sy += pos[j * 3 + 1];
            sz += pos[j * 3 + 2];
        }
        let k = nb.len() as f64;
        let lx = sx / k - pos[i * 3];
        let ly = sy / k - pos[i * 3 + 1];
        let lz = sz / k - pos[i * 3 + 2];
        nx[i * 3] = pos[i * 3] + lambda * lx;
        nx[i * 3 + 1] = pos[i * 3 + 1] + lambda * ly;
        nx[i * 3 + 2] = pos[i * 3 + 2] + lambda * lz;
    }
    pos.copy_from_slice(&nx);
}

fn taubin_smooth_mesh(
    positions: &[f32],
    indices: &[u32],
    roi_min_x: f64,
    roi_min_y: f64,
    roi_min_z: f64,
    roi_max_x: f64,
    roi_max_y: f64,
    roi_max_z: f64,
    iterations: u32,
    relax_pct: u32,
) -> Vec<f64> {
    let n = positions.len() / 3;
    let mut pos = vec![0.0f64; positions.len()];
    for i in 0..positions.len() {
        pos[i] = positions[i] as f64;
    }
    let adj = build_adjacency(n, indices);
    let mut pinned = vec![false; n];
    for i in 0..n {
        pinned[i] = is_pinned_vertex(
            pos[i * 3],
            pos[i * 3 + 1],
            pos[i * 3 + 2],
            roi_min_x,
            roi_min_y,
            roi_min_z,
            roi_max_x,
            roi_max_y,
            roi_max_z,
        );
    }
    let s = (relax_pct.min(100) as f64) / 100.0;
    let lambda = 0.33 * s;
    let mu = -0.34 * s;
    let iters = iterations.clamp(1, 20);
    for _ in 0..iters {
        if lambda > 1e-8 {
            umbrella_laplacian_step(&mut pos, &adj, &pinned, lambda);
        }
        if mu < -1e-8 {
            umbrella_laplacian_step(&mut pos, &adj, &pinned, mu);
        }
    }
    pos
}

#[inline]
fn ray_triangle_moller(orig: Vec3, dir: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<f32> {
    const EPS: f32 = 1e-7;
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let p = dir.cross(e2);
    let det = e1.dot(p);
    if det.abs() < EPS {
        return None;
    }
    let inv_det = 1.0 / det;
    let t = orig - v0;
    let u = t.dot(p) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = t.cross(e1);
    let v = dir.dot(q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let dist = e2.dot(q) * inv_det;
    if dist > EPS {
        Some(dist)
    } else {
        None
    }
}

fn count_ray_hits_plus_x(ox: f64, oy: f64, oz: f64, pos: &[f64], indices: &[u32]) -> i32 {
    const T_MERGE: f64 = 1e-4;
    let orig = Vec3::new(ox as f32, oy as f32, oz as f32);
    let dir = Vec3::X;
    let mut ts: Vec<f32> = Vec::new();
    for t in (0..indices.len()).step_by(3) {
        if t + 2 >= indices.len() {
            break;
        }
        let i0 = indices[t] as usize * 3;
        let i1 = indices[t + 1] as usize * 3;
        let i2 = indices[t + 2] as usize * 3;
        if i2 + 2 >= pos.len() {
            continue;
        }
        let v0 = Vec3::new(pos[i0] as f32, pos[i0 + 1] as f32, pos[i0 + 2] as f32);
        let v1 = Vec3::new(pos[i1] as f32, pos[i1 + 1] as f32, pos[i1 + 2] as f32);
        let v2 = Vec3::new(pos[i2] as f32, pos[i2 + 1] as f32, pos[i2 + 2] as f32);
        if let Some(dist) = ray_triangle_moller(orig, dir, v0, v1, v2) {
            ts.push(dist);
        }
    }
    if ts.is_empty() {
        return 0;
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut crossings = 0;
    let mut last = f32::NEG_INFINITY;
    for dist in ts {
        if (dist - last) > T_MERGE as f32 {
            crossings += 1;
            last = dist;
        }
    }
    crossings
}

fn vertex_axis_bounds(pos: &[f64]) -> (f64, f64, f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut min_z = f64::INFINITY;
    let mut max_z = f64::NEG_INFINITY;
    for i in (0..pos.len()).step_by(3) {
        if i + 2 >= pos.len() {
            break;
        }
        min_x = min_x.min(pos[i]);
        max_x = max_x.max(pos[i]);
        min_y = min_y.min(pos[i + 1]);
        max_y = max_y.max(pos[i + 1]);
        min_z = min_z.min(pos[i + 2]);
        max_z = max_z.max(pos[i + 2]);
    }
    (min_x, max_x, min_y, max_y, min_z, max_z)
}

fn voxel_cell_inside_mesh(
    x: i32,
    y: i32,
    z: i32,
    pos: &[f64],
    indices: &[u32],
    was_solid: bool,
    vb: Option<(f64, f64, f64, f64, f64, f64)>,
) -> bool {
    let ox = x as f64 + 0.5 + 1e-4;
    let oy = y as f64 + 0.5 + 2e-4;
    let oz = z as f64 + 0.5 + 3e-4;
    if count_ray_hits_plus_x(ox, oy, oz, pos, indices) % 2 == 1 {
        return true;
    }
    if was_solid {
        if let Some((min_x, max_x, min_y, max_y, min_z, max_z)) = vb {
            let cx = x as f64 + 0.5;
            let cy = y as f64 + 0.5;
            let cz = z as f64 + 0.5;
            const PAD: f64 = 0.51;
            if cx >= min_x - PAD
                && cx <= max_x + PAD
                && cy >= min_y - PAD
                && cy <= max_y + PAD
                && cz >= min_z - PAD
                && cz <= max_z + PAD
            {
                return true;
            }
        }
    }
    false
}

fn nearest_original_voxel(
    cx: f64,
    cy: f64,
    cz: f64,
    originals: &[(i32, i32, i32, Voxel)],
) -> Option<Voxel> {
    let mut best_d = f64::INFINITY;
    let mut best: Option<Voxel> = None;
    for &(ox, oy, oz, v) in originals {
        let dx = cx - (ox as f64 + 0.5);
        let dy = cy - (oy as f64 + 0.5);
        let dz = cz - (oz as f64 + 0.5);
        let d = dx * dx + dy * dy + dz * dz;
        if d < best_d {
            best_d = d;
            best = Some(v);
        }
    }
    best
}

/// Mesh Taubin smooth + revoxelize (web `applyMeshLaplacianSmooth`). Falls back to majority on large ROI / empty mesh.
pub fn apply_sculpt_smooth_mesh_laplacian(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    footprint: &[VoxelCoord],
    grid_size: i32,
    neighbor_margin: i32,
    laplacian_iterations: u32,
    laplacian_relax_pct: u32,
    majority_neighbor_radius: u32,
    majority_aggressiveness: u32,
    stroke_color: u32,
    stroke_material: MaterialId,
) -> Vec<VoxelEditDelta> {
    let margin = neighbor_margin.max(0);
    let Some((rmin_x, rmin_y, rmin_z, rmax_x, rmax_y, rmax_z)) =
        roi_from_footprint_margin(footprint, margin, grid_size)
    else {
        return Vec::new();
    };
    let ncells = roi_cell_count(rmin_x, rmin_y, rmin_z, rmax_x, rmax_y, rmax_z);
    if ncells > MAX_LAPLACIAN_ROI_CELLS {
        return apply_sculpt_smooth_majority_pass(
            file,
            voxel_map,
            footprint,
            grid_size,
            majority_neighbor_radius,
            majority_aggressiveness,
            stroke_color,
            stroke_material,
        );
    }

    let full_map = build_full_voxel_map(file, voxel_map);
    let mut proxy_keys: AHashSet<VoxelCoord> = AHashSet::new();
    let mut emit: Vec<Voxel> = Vec::new();
    for x in rmin_x..=rmax_x {
        for y in rmin_y..=rmax_y {
            for z in rmin_z..=rmax_z {
                if !in_grid_bounds(x, y, z, grid_size) {
                    continue;
                }
                if full_map.contains_key(&(x, y, z)) {
                    proxy_keys.insert((x, y, z));
                    emit.push(Voxel {
                        x,
                        y,
                        z,
                        color: PROXY_COLOR,
                        material: MaterialId::Plastic,
                        object_id: file.active_object_id,
                    });
                }
            }
        }
    }

    if emit.is_empty() {
        return Vec::new();
    }

    let mut combined = full_map.clone();
    for v in &emit {
        combined.insert((v.x, v.y, v.z), *v);
    }

    let mesh = greedy_mesh::build_greedy_mesh_mapped(&emit, &combined);
    if mesh.indices.len() < 3 {
        return apply_sculpt_smooth_majority_pass(
            file,
            voxel_map,
            footprint,
            grid_size,
            majority_neighbor_radius,
            majority_aggressiveness,
            stroke_color,
            stroke_material,
        );
    }

    let roi_min_x = rmin_x as f64;
    let roi_min_y = rmin_y as f64;
    let roi_min_z = rmin_z as f64;
    let roi_max_x = rmax_x as f64;
    let roi_max_y = rmax_y as f64;
    let roi_max_z = rmax_z as f64;

    let smoothed = taubin_smooth_mesh(
        &mesh.positions,
        &mesh.indices,
        roi_min_x,
        roi_min_y,
        roi_min_z,
        roi_max_x,
        roi_max_y,
        roi_max_z,
        laplacian_iterations,
        laplacian_relax_pct,
    );

    let vb = vertex_axis_bounds(&smoothed);
    let mut originals: Vec<(i32, i32, i32, Voxel)> = Vec::new();
    for x in rmin_x..=rmax_x {
        for y in rmin_y..=rmax_y {
            for z in rmin_z..=rmax_z {
                if let Some(&v) = full_map.get(&(x, y, z)) {
                    originals.push((x, y, z, v));
                }
            }
        }
    }

    let vb_opt = Some(vb);
    let mut out: Vec<VoxelEditDelta> = Vec::new();

    for x in rmin_x..=rmax_x {
        for y in rmin_y..=rmax_y {
            for z in rmin_z..=rmax_z {
                if !in_grid_bounds(x, y, z, grid_size) {
                    continue;
                }
                let k = (x, y, z);
                if let Some(&idx) = voxel_map.get(&k) {
                    let removed = file.voxels[idx];
                    let last = file.voxels.len() - 1;
                    if idx != last {
                        file.voxels.swap(idx, last);
                        let moved = file.voxels[idx];
                        voxel_map.insert((moved.x, moved.y, moved.z), idx);
                    }
                    file.voxels.pop();
                    voxel_map.remove(&k);
                    out.push(VoxelEditDelta::Removed { voxel: removed });
                }
            }
        }
    }

    for x in rmin_x..=rmax_x {
        for y in rmin_y..=rmax_y {
            for z in rmin_z..=rmax_z {
                if !in_grid_bounds(x, y, z, grid_size) {
                    continue;
                }
                let was_solid = proxy_keys.contains(&(x, y, z));
                if !voxel_cell_inside_mesh(x, y, z, &smoothed, &mesh.indices, was_solid, vb_opt) {
                    continue;
                }
                let near = nearest_original_voxel(
                    x as f64 + 0.5,
                    y as f64 + 0.5,
                    z as f64 + 0.5,
                    &originals,
                );
                let template = near.unwrap_or(Voxel {
                    x,
                    y,
                    z,
                    color: stroke_color,
                    material: stroke_material,
                    object_id: file.active_object_id,
                });
                let nv = Voxel {
                    x,
                    y,
                    z,
                    color: template.color,
                    material: template.material,
                    object_id: file.active_object_id,
                };
                let idx = file.voxels.len();
                file.voxels.push(nv);
                voxel_map.insert((x, y, z), idx);
                out.push(VoxelEditDelta::Added(nv));
            }
        }
    }

    out
}
