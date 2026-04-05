use super::common::{Rng, GOLDEN_ANGLE_RAD};
use crate::camera::OrbitCamera;
use crate::greedy_mesh::VoxelCoord;
use crate::voxel_edit::{
    effective_ray_grid_size, ensure_grid_fits_coord, ray_first_solid, screen_to_world_ray,
    VoxelEditDelta,
};
use crate::voxelle::{MaterialId, Scene, Voxel, VoxelleFile};
use ahash::AHashMap;
use std::collections::HashSet;

const VOXEL_CAP: usize = 150_000;

// ---------------------------------------------------------------------------
// Vector helpers (f64 triples) — kept local because this generator uses f64
// precision throughout its braid / wobble paths.
// ---------------------------------------------------------------------------

type V3 = (f64, f64, f64);

fn v3_add(a: V3, b: V3) -> V3 {
    (a.0 + b.0, a.1 + b.1, a.2 + b.2)
}

fn v3_scale(a: V3, s: f64) -> V3 {
    (a.0 * s, a.1 * s, a.2 * s)
}

fn v3_len(a: V3) -> f64 {
    (a.0 * a.0 + a.1 * a.1 + a.2 * a.2).sqrt()
}

fn v3_normalize(a: V3) -> V3 {
    let l = v3_len(a);
    if l < 1e-12 {
        (0.0, 1.0, 0.0)
    } else {
        (a.0 / l, a.1 / l, a.2 / l)
    }
}

fn v3_cross(a: V3, b: V3) -> V3 {
    (
        a.1 * b.2 - a.2 * b.1,
        a.2 * b.0 - a.0 * b.2,
        a.0 * b.1 - a.1 * b.0,
    )
}

fn v3_round(a: V3) -> (i32, i32, i32) {
    (a.0.round() as i32, a.1.round() as i32, a.2.round() as i32)
}

// ---------------------------------------------------------------------------
// Tangent frame from face normal
// ---------------------------------------------------------------------------

fn tangent_vectors(nx: i32, ny: i32, nz: i32) -> (V3, V3) {
    let n = (nx as f64, ny as f64, nz as f64);
    let arbitrary = if nx.abs() > nz.abs() {
        (0.0, 0.0, 1.0)
    } else {
        (1.0, 0.0, 0.0)
    };
    let t1 = v3_normalize(v3_cross(n, arbitrary));
    let t2 = v3_normalize(v3_cross(n, t1));
    (t1, t2)
}

// ---------------------------------------------------------------------------
// Mean backbone with wobble
// ---------------------------------------------------------------------------

fn build_mean_backbone(
    base: V3,
    normal: V3,
    t1: V3,
    t2: V3,
    height: i32,
    wobble: f64,
    rng: &mut Rng,
) -> Vec<V3> {
    let mut spine = Vec::with_capacity(height as usize);
    let mut drift_u = 0.0_f64;
    let mut drift_v = 0.0_f64;
    for k in 0..height {
        let frac = k as f64;
        // Accumulate lateral wobble
        drift_u += rng.next_signed_f64() * wobble * 0.6;
        drift_v += rng.next_signed_f64() * wobble * 0.6;
        let along = v3_scale(normal, frac);
        let lateral = v3_add(v3_scale(t1, drift_u), v3_scale(t2, drift_v));
        spine.push(v3_add(base, v3_add(along, lateral)));
    }
    spine
}

// ---------------------------------------------------------------------------
// Braid offsets for multi-strand stems
// ---------------------------------------------------------------------------

fn braid_offsets(
    strand_index: usize,
    strand_count: usize,
    step: usize,
    twist: f64,
    t1: V3,
    t2: V3,
) -> V3 {
    if strand_count <= 1 {
        return (0.0, 0.0, 0.0);
    }
    let r = 1.0 + (twist * 2.2).floor();
    let base_angle = (strand_index as f64 / strand_count as f64) * std::f64::consts::TAU;
    let angle = base_angle + step as f64 * (0.35 + twist * 0.95);
    let ou = r * angle.cos();
    let ov = r * angle.sin();
    v3_add(v3_scale(t1, ou), v3_scale(t2, ov))
}

// ---------------------------------------------------------------------------
// Effective girth at a given step (with taper)
// ---------------------------------------------------------------------------

fn effective_girth_at(girth: f64, taper: f64, step: usize, total_height: usize) -> f64 {
    if total_height <= 1 {
        return girth;
    }
    girth * (1.0 - taper * (step as f64 / (total_height - 1) as f64))
}

// ---------------------------------------------------------------------------
// Euclidean disk placement
// ---------------------------------------------------------------------------

fn euclidean_radius_steps(r: f64) -> i32 {
    (r * 1.85).round() as i32
}

fn place_disk(center: V3, radius: f64, t1: V3, t2: V3, coords: &mut HashSet<VoxelCoord>) {
    if radius < 0.01 {
        let c = v3_round(center);
        coords.insert(c);
        return;
    }
    let steps = euclidean_radius_steps(radius);
    let r2 = radius * radius;
    for u in -steps..=steps {
        for v in -steps..=steps {
            let uf = u as f64 / 1.85;
            let vf = v as f64 / 1.85;
            if uf * uf + vf * vf > r2 {
                continue;
            }
            let pos = v3_add(center, v3_add(v3_scale(t1, uf), v3_scale(t2, vf)));
            coords.insert(v3_round(pos));
        }
    }
}

// ---------------------------------------------------------------------------
// Bresenham-style 3D line for branch paths
// ---------------------------------------------------------------------------

fn bresenham_3d(a: (i32, i32, i32), b: (i32, i32, i32)) -> Vec<(i32, i32, i32)> {
    let mut out = Vec::new();
    let dx = (b.0 - a.0).abs();
    let dy = (b.1 - a.1).abs();
    let dz = (b.2 - a.2).abs();
    let sx = if a.0 < b.0 { 1 } else { -1 };
    let sy = if a.1 < b.1 { 1 } else { -1 };
    let sz = if a.2 < b.2 { 1 } else { -1 };
    let dm = dx.max(dy).max(dz);
    let mut x = a.0;
    let mut y = a.1;
    let mut z = a.2;
    let mut ex = dm / 2;
    let mut ey = dm / 2;
    let mut ez = dm / 2;
    for _ in 0..=dm {
        out.push((x, y, z));
        ex -= dx;
        if ex < 0 {
            ex += dm;
            x += sx;
        }
        ey -= dy;
        if ey < 0 {
            ey += dm;
            y += sy;
        }
        ez -= dz;
        if ez < 0 {
            ez += dm;
            z += sz;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Branch generation
// ---------------------------------------------------------------------------

fn generate_branches(
    spine: &[V3],
    normal: V3,
    t1: V3,
    t2: V3,
    branch_count: i32,
    branch_depth: i32,
    branch_start: f64,
    branch_spread: f64,
    girth: f64,
    _taper: f64,
    canopy: f64,
    rng: &mut Rng,
    coords: &mut HashSet<VoxelCoord>,
) {
    if branch_count <= 0 || spine.len() < 2 {
        return;
    }
    let total = spine.len();
    let start_idx = ((branch_start * total as f64) as usize).max(1);
    let usable = total - start_idx;
    if usable < 1 {
        return;
    }

    for bi in 0..branch_count {
        if coords.len() >= VOXEL_CAP {
            break;
        }
        // Evenly spaced fork points along the usable range
        let frac = if branch_count == 1 {
            0.5
        } else {
            bi as f64 / (branch_count - 1) as f64
        };
        let spine_idx = start_idx + (frac * (usable - 1) as f64).round() as usize;
        let spine_idx = spine_idx.min(total - 1);

        // Direction: outward in tangent plane using golden angle spiral
        let angle = bi as f64 * (GOLDEN_ANGLE_RAD as f64);
        let out_dir = v3_normalize(v3_add(v3_scale(t1, angle.cos()), v3_scale(t2, angle.sin())));
        // Mix in some upward (normal) bias
        let branch_dir = v3_normalize(v3_add(
            v3_scale(out_dir, branch_spread.max(0.3)),
            v3_scale(normal, 0.5),
        ));

        let branch_len = ((total as f64 * 0.4 * branch_spread).max(2.0)) as i32;
        let origin = spine[spine_idx];
        let tip = v3_add(origin, v3_scale(branch_dir, branch_len as f64));

        let origin_i = v3_round(origin);
        let tip_i = v3_round(tip);
        let path = bresenham_3d(origin_i, tip_i);

        // Walk the branch path with tapering disk
        let path_len = path.len();
        for (pi, &(px, py, pz)) in path.iter().enumerate() {
            if coords.len() >= VOXEL_CAP {
                break;
            }
            let branch_frac = pi as f64 / path_len.max(1) as f64;
            let r = effective_girth_at(girth * 0.5, 0.8, pi, path_len).max(0.0);
            let center = (px as f64, py as f64, pz as f64);
            place_disk(center, r, t1, t2, coords);

            // Canopy at branch tip
            if canopy > 0.01 && branch_frac > 0.7 {
                let canopy_r = girth * canopy * (1.0 - (branch_frac - 0.7) / 0.3).max(0.0) * 1.5;
                place_canopy_at(center, canopy_r, t1, t2, normal, rng, coords);
            }
        }

        // Recursive sub-branches (depth 2)
        if branch_depth >= 2 && path.len() > 2 {
            let sub_count = (branch_count / 2).clamp(1, 3);
            let sub_spine: Vec<V3> = path
                .iter()
                .map(|&(x, y, z)| (x as f64, y as f64, z as f64))
                .collect();
            generate_branches(
                &sub_spine,
                normal,
                t1,
                t2,
                sub_count,
                1, // no further recursion
                0.3,
                branch_spread * 0.6,
                girth * 0.4,
                _taper,
                canopy * 0.5,
                rng,
                coords,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Canopy placement
// ---------------------------------------------------------------------------

fn place_canopy_at(
    center: V3,
    radius: f64,
    t1: V3,
    t2: V3,
    normal: V3,
    rng: &mut Rng,
    coords: &mut HashSet<VoxelCoord>,
) {
    if radius < 0.5 {
        return;
    }
    // Multi-layer canopy: a few disks stacked along normal
    let layers = (radius * 0.8).ceil() as i32;
    for layer in -layers..=layers {
        if coords.len() >= VOXEL_CAP {
            break;
        }
        let layer_frac = (layer.abs() as f64) / layers.max(1) as f64;
        let layer_r = radius * (1.0 - layer_frac * layer_frac).max(0.0); // spherical falloff
        let layer_center = v3_add(center, v3_scale(normal, layer as f64));

        let steps = euclidean_radius_steps(layer_r);
        let r2 = layer_r * layer_r;
        for u in -steps..=steps {
            for v in -steps..=steps {
                let uf = u as f64 / 1.85;
                let vf = v as f64 / 1.85;
                if uf * uf + vf * vf > r2 {
                    continue;
                }
                // Random dropout for organic look
                if rng.next_f64() < 0.2 {
                    continue;
                }
                let pos = v3_add(layer_center, v3_add(v3_scale(t1, uf), v3_scale(t2, vf)));
                coords.insert(v3_round(pos));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Core flora generation
// ---------------------------------------------------------------------------

pub fn generate_flora_deltas(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    face_empty: VoxelCoord,
    solid: VoxelCoord,
    seed: i32,
    height: i32,
    girth: i32,
    wobble: f32,
    taper: f32,
    stem_count: i32,
    cluster_radius: i32,
    branch_count: i32,
    branch_depth: i32,
    branch_start: f32,
    branch_spread: f32,
    braid_strands: i32,
    braid_twist: f32,
    canopy: f32,
    color: u32,
    material: MaterialId,
) -> Vec<VoxelEditDelta> {
    // Clamp parameters
    let height = height.clamp(1, 96);
    let girth = girth.clamp(0, 20);
    let wobble = (wobble as f64).clamp(0.0, 1.0);
    let taper = (taper as f64).clamp(0.0, 1.0);
    let stem_count = stem_count.clamp(1, 8);
    let cluster_radius = cluster_radius.clamp(0, 4);
    let branch_count = branch_count.clamp(0, 6);
    let branch_depth = branch_depth.clamp(1, 2);
    let branch_start = (branch_start as f64).clamp(0.0, 0.9);
    let branch_spread = (branch_spread as f64).clamp(0.0, 3.0);
    let braid_strands = braid_strands.clamp(1, 5);
    let braid_twist = (braid_twist as f64).clamp(0.0, 1.0);
    let canopy = (canopy as f64).clamp(0.0, 1.0);
    let girth_f = girth as f64;

    // Face normal
    let nx = face_empty.0 - solid.0;
    let ny = face_empty.1 - solid.1;
    let nz = face_empty.2 - solid.2;
    if nx.abs() + ny.abs() + nz.abs() != 1 {
        return Vec::new();
    }
    let normal = v3_normalize((nx as f64, ny as f64, nz as f64));
    let (t1, t2) = tangent_vectors(nx, ny, nz);
    let base = (
        face_empty.0 as f64,
        face_empty.1 as f64,
        face_empty.2 as f64,
    );

    let mut rng = Rng::new(seed as u32);
    let mut coords: HashSet<VoxelCoord> = HashSet::new();

    // Generate each stem in the cluster
    for si in 0..stem_count {
        if coords.len() >= VOXEL_CAP {
            break;
        }

        // Cluster offset: spread stems around the base
        let cluster_off = if stem_count > 1 && cluster_radius > 0 {
            let angle =
                (si as f64 / stem_count as f64) * std::f64::consts::TAU + rng.next_f64() * 0.5;
            let r = rng.next_f64() * cluster_radius as f64;
            v3_add(v3_scale(t1, r * angle.cos()), v3_scale(t2, r * angle.sin()))
        } else {
            (0.0, 0.0, 0.0)
        };
        let stem_base = v3_add(base, cluster_off);

        // Build the mean backbone for this stem
        let spine = build_mean_backbone(stem_base, normal, t1, t2, height, wobble, &mut rng);

        // Place disks along each braid strand
        for strand in 0..braid_strands {
            if coords.len() >= VOXEL_CAP {
                break;
            }
            for (k, center) in spine.iter().enumerate() {
                if coords.len() >= VOXEL_CAP {
                    break;
                }
                let braid_off = braid_offsets(
                    strand as usize,
                    braid_strands as usize,
                    k,
                    braid_twist,
                    t1,
                    t2,
                );
                let disk_center = v3_add(*center, braid_off);
                let r = effective_girth_at(girth_f, taper, k, height as usize);
                place_disk(disk_center, r, t1, t2, &mut coords);
            }
        }

        // Canopy at the tip
        if canopy > 0.01 && !spine.is_empty() {
            let tip = *spine.last().unwrap();
            let canopy_r = girth_f * canopy * 2.0 + 1.0;
            place_canopy_at(tip, canopy_r, t1, t2, normal, &mut rng, &mut coords);
        }

        // Branches
        if branch_count > 0 {
            generate_branches(
                &spine,
                normal,
                t1,
                t2,
                branch_count,
                branch_depth,
                branch_start,
                branch_spread,
                girth_f,
                taper,
                canopy,
                &mut rng,
                &mut coords,
            );
        }
    }

    // Convert collected coords to voxel edit deltas
    let mut out = Vec::with_capacity(coords.len().min(VOXEL_CAP));
    for (x, y, z) in coords {
        if out.len() >= VOXEL_CAP {
            break;
        }
        if voxel_map.contains_key(&(x, y, z)) {
            continue;
        }
        ensure_grid_fits_coord(file, x, y, z);
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
    out
}

/// Face-click flora generator (web parity).
pub fn generator_flora_at_screen(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    seed: i32,
    flora_height: i32,
    girth: i32,
    wobble: f32,
    taper: f32,
    stem_count: i32,
    cluster_radius: i32,
    branch_count: i32,
    branch_depth: i32,
    branch_start: f32,
    branch_spread: f32,
    braid_strands: i32,
    braid_twist: f32,
    canopy: f32,
    color: u32,
    material: MaterialId,
) -> Result<Vec<VoxelEditDelta>, String> {
    let grid_size = effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((solid, prev)) = ray_first_solid(origin, dir, voxel_map, grid_size) else {
        return Ok(Vec::new());
    };
    let Some(face_empty) = prev else {
        return Ok(Vec::new());
    };
    Ok(generate_flora_deltas(
        file,
        voxel_map,
        face_empty,
        solid,
        seed,
        flora_height,
        girth,
        wobble,
        taper,
        stem_count,
        cluster_radius,
        branch_count,
        branch_depth,
        branch_start,
        branch_spread,
        braid_strands,
        braid_twist,
        canopy,
        color,
        material,
    ))
}

/// Preview-only: compute the set of voxel coords flora would occupy,
/// without mutating the real file. Used for hover preview.
#[allow(clippy::too_many_arguments)]
pub fn preview_flora_at_screen(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    seed: i32,
    flora_height: i32,
    girth: i32,
    wobble: f32,
    taper: f32,
    stem_count: i32,
    cluster_radius: i32,
    branch_count: i32,
    branch_depth: i32,
    branch_start: f32,
    branch_spread: f32,
    braid_strands: i32,
    braid_twist: f32,
    canopy: f32,
    color: u32,
    material: MaterialId,
) -> Vec<(VoxelCoord, u32)> {
    let grid_size = effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((solid, prev)) = ray_first_solid(origin, dir, voxel_map, grid_size) else {
        return Vec::new();
    };
    let Some(face_empty) = prev else {
        return Vec::new();
    };
    let mut stub_file = VoxelleFile {
        version: 0,
        grid_size: file.grid_size,
        scene: Scene::default(),
        scene_extra: None,
        mood: None,
        lighting: None,
        voxels: Vec::new(),
        objects: Vec::new(),
        active_object_id: 0,
    };
    let mut stub_map: AHashMap<VoxelCoord, usize> = AHashMap::new();
    generate_flora_deltas(
        &mut stub_file,
        &mut stub_map,
        face_empty,
        solid,
        seed,
        flora_height,
        girth,
        wobble,
        taper,
        stem_count,
        cluster_radius,
        branch_count,
        branch_depth,
        branch_start,
        branch_spread,
        braid_strands,
        braid_twist,
        canopy,
        color,
        material,
    )
    .into_iter()
    .filter_map(|d| {
        if let VoxelEditDelta::Added(v) = d {
            if !voxel_map.contains_key(&(v.x, v.y, v.z)) {
                return Some(((v.x, v.y, v.z), v.color));
            }
        }
        None
    })
    .collect()
}
