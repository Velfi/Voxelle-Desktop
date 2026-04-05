//! Flood fill algorithms, connected component logic, and fill boundary detection.

use crate::camera::OrbitCamera;
use crate::greedy_mesh::VoxelCoord;
use crate::stroke_modes::PlaneAxis;
use crate::voxelle::{MaterialId, Voxel, VoxelleFile};
use ahash::{AHashMap, AHashSet};
use glam::Vec3;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};

use super::raycasting::{ray_first_solid_scene, screen_to_world_ray};
/// Web parity with `MAX_GRID_SIZE` in `store/core.ts`: symmetric grid about origin, capped for safety.
/// (Re-exported from parent; available here for fill BFS grid checks.)
use super::{effective_ray_grid_size, in_grid};

/// Web parity with `MAX_GRID_SIZE` in `store/core.ts`: symmetric grid about origin, capped for safety.
/// Unconstrained fills that would exceed this many cells show a "large fill" path.
pub const FILL_UNCONSTRAINED_LARGE_THRESHOLD: usize = 256;
/// Cooperative cancel / progress checks during BFS (dequeue steps).
pub const FILL_BFS_PROGRESS_INTERVAL: usize = 2048;
/// Cheap cancel poll every N dequeues. Duplicates in the queue can inflate dequeues vs. `out.len()`,
/// so this must be much smaller than [`FILL_BFS_PROGRESS_INTERVAL`] or Escape/Cancel lags badly.
pub const FILL_BFS_CANCEL_CHECK_INTERVAL: usize = 32;
/// Cancel/yield cadence during the fast "would this unconstrained fill be large?" probe (separate BFS).
pub const FILL_THRESHOLD_PROBE_CANCEL_INTERVAL: usize = 32;
/// Hard safety cap — refuse to allocate or apply beyond this (matches web "don't freeze" intent).
pub const FILL_ABSOLUTE_MAX_CELLS: usize = 50_000_000;

/// Result of a cancellable flood over solid selection coords.
#[derive(Debug)]
pub struct FillCoordOutcome {
    pub coords: Vec<VoxelCoord>,
    pub cancelled: bool,
    pub hit_absolute_cap: bool,
}

/// Result of flood fill edits (remove / paint / empty-add).
#[derive(Debug)]
pub struct FloodFillEditOutcome {
    pub deltas: Vec<super::VoxelEditDelta>,
    pub cancelled: bool,
    pub hit_absolute_cap: bool,
}

pub fn neighbors_6(c: VoxelCoord) -> [VoxelCoord; 6] {
    let (x, y, z) = c;
    [
        (x + 1, y, z),
        (x - 1, y, z),
        (x, y + 1, z),
        (x, y - 1, z),
        (x, y, z + 1),
        (x, y, z - 1),
    ]
}

fn neighbors_26(c: VoxelCoord) -> Vec<VoxelCoord> {
    let (x, y, z) = c;
    let mut v = Vec::with_capacity(26);
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                v.push((x + dx, y + dy, z + dz));
            }
        }
    }
    v
}

fn camera_view_forward(cam: &OrbitCamera) -> Vec3 {
    let eye = cam.smooth_eye();
    let t = cam.smooth_target;
    (t - eye).normalize()
}

/// Face entry axis 0|1|2 from air cell `prev` to solid `hit` (6-connected step).
fn face_axis_from_prev_hit(prev: VoxelCoord, hit: VoxelCoord) -> Option<usize> {
    let dx = hit.0 - prev.0;
    let dy = hit.1 - prev.1;
    let dz = hit.2 - prev.2;
    if dx.abs() + dy.abs() + dz.abs() != 1 {
        return None;
    }
    if dx != 0 {
        Some(0)
    } else if dy != 0 {
        Some(1)
    } else {
        Some(2)
    }
}

/// Web `selection.ts` `voxelInConstrainPlane` — `seed` is the fill origin (solid hit or empty seed).
fn voxel_in_fill_plane(
    cell: VoxelCoord,
    seed: VoxelCoord,
    plane_axis: PlaneAxis,
    face_axis: Option<usize>,
    cam_forward: Vec3,
) -> bool {
    let (nx, ny, nz) = cell;
    let (sx, sy, sz) = seed;
    match plane_axis {
        PlaneAxis::X => nx == sx,
        PlaneAxis::Y => ny == sy,
        PlaneAxis::Z => nz == sz,
        PlaneAxis::Auto => {
            let ax = face_axis.unwrap_or(1);
            match ax {
                0 => nx == sx,
                1 => ny == sy,
                _ => nz == sz,
            }
        }
        PlaneAxis::Camera => {
            let dx = (nx - sx) as f32;
            let dy = (ny - sy) as f32;
            let dz = (nz - sz) as f32;
            let dot = dx * cam_forward.x + dy * cam_forward.y + dz * cam_forward.z;
            dot.abs() < 0.5
        }
    }
}

pub(super) fn plane_axis_fixed(prev: VoxelCoord, hit: VoxelCoord) -> Option<(usize, i32)> {
    let dx = hit.0 - prev.0;
    let dy = hit.1 - prev.1;
    let dz = hit.2 - prev.2;
    if dx != 0 && dy == 0 && dz == 0 {
        Some((0, hit.0))
    } else if dy != 0 && dx == 0 && dz == 0 {
        Some((1, hit.1))
    } else if dz != 0 && dx == 0 && dy == 0 {
        Some((2, hit.2))
    } else {
        None
    }
}

#[inline]
pub(super) fn voxel_on_plane(c: VoxelCoord, axis: usize, fixed: i32) -> bool {
    match axis {
        0 => c.0 == fixed,
        1 => c.1 == fixed,
        2 => c.2 == fixed,
        _ => false,
    }
}

pub(super) fn neighbors_on_face_plane(axis: usize, c: VoxelCoord) -> [(i32, i32, i32); 4] {
    let (x, y, z) = c;
    match axis {
        0 => [(x, y + 1, z), (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)],
        1 => [(x + 1, y, z), (x - 1, y, z), (x, y, z + 1), (x, y, z - 1)],
        2 => [(x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z)],
        _ => [(0, 0, 0); 4],
    }
}

/// Remove a single solid voxel (swap-remove); same semantics as [`apply_edit`] remove.
pub fn remove_voxel_at_coord(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    coord: VoxelCoord,
) -> Option<super::VoxelEditDelta> {
    let remove_idx = *voxel_map.get(&coord)?;
    let removed_voxel = file.voxels[remove_idx];
    let last = file.voxels.len() - 1;
    if remove_idx != last {
        file.voxels.swap(remove_idx, last);
        let moved = file.voxels[remove_idx];
        voxel_map.insert((moved.x, moved.y, moved.z), remove_idx);
    }
    file.voxels.pop();
    voxel_map.remove(&coord);
    Some(super::VoxelEditDelta::Removed {
        voxel: removed_voxel,
    })
}

/// Flood-fill remove: connected solid region from screen pick (same region as selection fill).
pub fn flood_fill_remove_at_screen(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    fill_diagonals: bool,
    fill_respects_color: bool,
    match_material: bool,
    fill_constrain_plane: bool,
    plane_axis: PlaneAxis,
    cancel: Option<&AtomicBool>,
    mut on_progress: impl FnMut(usize),
) -> Result<FloodFillEditOutcome, String> {
    let o = flood_fill_selection_coords_with_control(
        file,
        voxel_map,
        camera,
        width,
        height,
        sx,
        sy,
        fill_diagonals,
        fill_respects_color,
        match_material,
        fill_constrain_plane,
        plane_axis,
        cancel,
        &mut on_progress,
    );
    if o.cancelled {
        return Ok(FloodFillEditOutcome {
            deltas: Vec::new(),
            cancelled: true,
            hit_absolute_cap: false,
        });
    }
    if o.hit_absolute_cap {
        return Ok(FloodFillEditOutcome {
            deltas: Vec::new(),
            cancelled: false,
            hit_absolute_cap: true,
        });
    }
    let coords = o.coords;
    let mut out = Vec::with_capacity(coords.len());
    for c in coords {
        if let Some(d) = remove_voxel_at_coord(file, voxel_map, c) {
            out.push(d);
        }
    }
    Ok(FloodFillEditOutcome {
        deltas: out,
        cancelled: false,
        hit_absolute_cap: false,
    })
}

/// Flood-fill add: connected **empty** cells from add-placement seed (air in front of first solid).
pub fn flood_fill_empty_at_screen(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    fill_diagonals: bool,
    color_resolver: impl Fn(i32, i32, i32) -> u32,
    material: MaterialId,
    fill_constrain_plane: bool,
    plane_axis: PlaneAxis,
    cancel: Option<&AtomicBool>,
    mut on_progress: impl FnMut(usize),
) -> Result<FloodFillEditOutcome, String> {
    // Fixed bound for BFS: do not grow `file.grid_size` during the walk (that made `in_grid` unbounded).
    let grid_limit = effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((hit, prev, _oid)) = ray_first_solid_scene(origin, dir, file, voxel_map, grid_limit)
    else {
        return Ok(FloodFillEditOutcome {
            deltas: Vec::new(),
            cancelled: false,
            hit_absolute_cap: false,
        });
    };
    let Some(seed) = prev else {
        return Ok(FloodFillEditOutcome {
            deltas: Vec::new(),
            cancelled: false,
            hit_absolute_cap: false,
        });
    };
    if voxel_map.contains_key(&seed) {
        return Ok(FloodFillEditOutcome {
            deltas: Vec::new(),
            cancelled: false,
            hit_absolute_cap: false,
        });
    }
    if !in_grid(seed.0, seed.1, seed.2, grid_limit) {
        return Ok(FloodFillEditOutcome {
            deltas: Vec::new(),
            cancelled: false,
            hit_absolute_cap: false,
        });
    }
    let face_axis = face_axis_from_prev_hit(seed, hit);
    let cam_forward = camera_view_forward(camera);
    super::ensure_grid_fits_coord(file, seed.0, seed.1, seed.2);
    let mut visited: AHashSet<VoxelCoord> = AHashSet::new();
    let mut queue: VecDeque<VoxelCoord> = VecDeque::new();
    queue.push_back(seed);
    let mut out: Vec<super::VoxelEditDelta> = Vec::new();
    let mut steps: usize = 0;

    while let Some(c) = queue.pop_front() {
        steps += 1;
        if steps.is_multiple_of(FILL_BFS_CANCEL_CHECK_INTERVAL)
            && cancel.map(|c| c.load(Ordering::Relaxed)).unwrap_or(false)
        {
            return Ok(FloodFillEditOutcome {
                deltas: out,
                cancelled: true,
                hit_absolute_cap: false,
            });
        }
        if steps.is_multiple_of(FILL_BFS_PROGRESS_INTERVAL) {
            on_progress(out.len());
        }
        if visited.contains(&c) {
            continue;
        }
        visited.insert(c);
        if voxel_map.contains_key(&c) {
            continue;
        }
        if out.len() >= FILL_ABSOLUTE_MAX_CELLS {
            return Ok(FloodFillEditOutcome {
                deltas: out,
                cancelled: false,
                hit_absolute_cap: true,
            });
        }
        // `out.len()` is strictly below cap here
        let nv = Voxel {
            x: c.0,
            y: c.1,
            z: c.2,
            color: color_resolver(c.0, c.1, c.2),
            material,
            object_id: file.active_object_id,
        };
        let idx = file.voxels.len();
        file.voxels.push(nv);
        voxel_map.insert(c, idx);
        out.push(super::VoxelEditDelta::Added(nv));

        let neigh: Vec<VoxelCoord> = if fill_diagonals {
            neighbors_26(c)
        } else {
            neighbors_6(c).to_vec()
        };
        for n in neigh {
            if !in_grid(n.0, n.1, n.2, grid_limit) {
                continue;
            }
            if fill_constrain_plane
                && !voxel_in_fill_plane(n, seed, plane_axis, face_axis, cam_forward)
            {
                continue;
            }
            if !visited.contains(&n) {
                queue.push_back(n);
            }
        }
    }
    Ok(FloodFillEditOutcome {
        deltas: out,
        cancelled: false,
        hit_absolute_cap: false,
    })
}

/// Flood-fill paint: same connected region as selection fill, then recolor.
pub fn flood_fill_paint_at_screen(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    color_resolver: impl Fn(i32, i32, i32) -> u32,
    new_material: MaterialId,
    match_material: bool,
    fill_diagonals: bool,
    fill_respects_color: bool,
    fill_constrain_plane: bool,
    plane_axis: PlaneAxis,
    cancel: Option<&AtomicBool>,
    mut on_progress: impl FnMut(usize),
) -> Result<FloodFillEditOutcome, String> {
    let o = flood_fill_selection_coords_with_control(
        file,
        voxel_map,
        camera,
        width,
        height,
        sx,
        sy,
        fill_diagonals,
        fill_respects_color,
        match_material,
        fill_constrain_plane,
        plane_axis,
        cancel,
        &mut on_progress,
    );
    if o.cancelled {
        return Ok(FloodFillEditOutcome {
            deltas: Vec::new(),
            cancelled: true,
            hit_absolute_cap: false,
        });
    }
    if o.hit_absolute_cap {
        return Ok(FloodFillEditOutcome {
            deltas: Vec::new(),
            cancelled: false,
            hit_absolute_cap: true,
        });
    }
    let coords = o.coords;
    let mut out = Vec::with_capacity(coords.len());
    for c in coords {
        let Some(&idx) = voxel_map.get(&c) else {
            continue;
        };
        let before = file.voxels[idx];
        let resolved_color = color_resolver(c.0, c.1, c.2);
        if before.color == resolved_color && before.material == new_material {
            continue;
        }
        let after = Voxel {
            color: resolved_color,
            material: new_material,
            ..before
        };
        file.voxels[idx] = after;
        out.push(super::VoxelEditDelta::Painted { before, after });
    }
    Ok(FloodFillEditOutcome {
        deltas: out,
        cancelled: false,
        hit_absolute_cap: false,
    })
}

/// 6-connected solid voxels matching the seed hit's color (and optionally material) — selection / flood without edits.
pub fn connected_solid_same_color_from_screen(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    match_material: bool,
) -> Option<Vec<VoxelCoord>> {
    let grid_size = effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let (hit, _, _oid) = ray_first_solid_scene(origin, dir, file, voxel_map, grid_size)?;
    let seed_idx = *voxel_map.get(&hit)?;
    let seed = file.voxels[seed_idx];
    let tc = seed.color;
    let tm = seed.material;

    let mut out: Vec<VoxelCoord> = Vec::new();
    let mut visited: AHashSet<VoxelCoord> = AHashSet::new();
    let mut queue: VecDeque<VoxelCoord> = VecDeque::new();
    visited.insert(hit);
    queue.push_back(hit);

    while let Some(c) = queue.pop_front() {
        let Some(&idx) = voxel_map.get(&c) else {
            continue;
        };
        let v = file.voxels[idx];
        if v.color != tc || (match_material && v.material != tm) {
            continue;
        }
        out.push(c);
        for n in neighbors_6(c) {
            if !in_grid(n.0, n.1, n.2, grid_size) {
                continue;
            }
            if visited.insert(n) {
                queue.push_back(n);
            }
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Face-connected empty cells on the same plane as the add-cell in front of the ray hit.
pub fn coplanar_empty_connected_from_screen(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> Option<Vec<VoxelCoord>> {
    let grid_size = effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let (hit, prev, _oid) = ray_first_solid_scene(origin, dir, file, voxel_map, grid_size)?;
    let prev = prev?;
    let (axis, fixed) = plane_axis_fixed(prev, hit)?;
    if voxel_map.contains_key(&prev) {
        return None;
    }
    if !voxel_on_plane(prev, axis, fixed) {
        return None;
    }

    let mut visited: AHashSet<VoxelCoord> = AHashSet::new();
    let mut queue: VecDeque<VoxelCoord> = VecDeque::new();
    let mut out: Vec<VoxelCoord> = Vec::new();
    visited.insert(prev);
    queue.push_back(prev);

    while let Some(c) = queue.pop_front() {
        if voxel_map.contains_key(&c) {
            continue;
        }
        if !voxel_on_plane(c, axis, fixed) {
            continue;
        }
        out.push(c);
        for n in neighbors_on_face_plane(axis, c) {
            if !in_grid(n.0, n.1, n.2, grid_size) {
                continue;
            }
            if !voxel_on_plane(n, axis, fixed) {
                continue;
            }
            if voxel_map.contains_key(&n) {
                continue;
            }
            if visited.insert(n) {
                queue.push_back(n);
            }
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Face-connected voxels on the axis-aligned plane through the hit face (from ray `prev` → `hit`).
pub fn coplanar_connected_from_screen(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> Option<Vec<VoxelCoord>> {
    if file.voxels.is_empty() {
        return None;
    }
    let grid_size = effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let (hit, prev, _oid) = ray_first_solid_scene(origin, dir, file, voxel_map, grid_size)?;
    let prev = prev?;
    let (axis, fixed) = plane_axis_fixed(prev, hit)?;
    if !voxel_map.contains_key(&hit) {
        return None;
    }

    let mut visited: AHashSet<VoxelCoord> = AHashSet::new();
    let mut queue: VecDeque<VoxelCoord> = VecDeque::new();
    let mut out: Vec<VoxelCoord> = Vec::new();
    visited.insert(hit);
    queue.push_back(hit);

    while let Some(c) = queue.pop_front() {
        if !voxel_map.contains_key(&c) {
            continue;
        }
        if !voxel_on_plane(c, axis, fixed) {
            continue;
        }
        out.push(c);
        for n in neighbors_on_face_plane(axis, c) {
            if !in_grid(n.0, n.1, n.2, grid_size) {
                continue;
            }
            if !voxel_on_plane(n, axis, fixed) {
                continue;
            }
            if !voxel_map.contains_key(&n) {
                continue;
            }
            if visited.insert(n) {
                queue.push_back(n);
            }
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Keep only coords whose voxels match `seed` color (and optionally material).
pub fn filter_coords_by_seed_color(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    coords: &[VoxelCoord],
    seed: Voxel,
    match_material: bool,
) -> Vec<VoxelCoord> {
    coords
        .iter()
        .copied()
        .filter(|c| {
            voxel_map.get(c).is_some_and(|&i| {
                let v = file.voxels[i];
                v.color == seed.color && (!match_material || v.material == seed.material)
            })
        })
        .collect()
}

/// Keep coords that lie on the same face plane as the coplanar pick from screen (solid hit).
pub fn filter_coords_coplanar_solid_from_screen(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    coords: &[VoxelCoord],
) -> Vec<VoxelCoord> {
    let grid_size = effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((hit, prev, _oid)) = ray_first_solid_scene(origin, dir, file, voxel_map, grid_size)
    else {
        return Vec::new();
    };
    let Some(prev) = prev else {
        return Vec::new();
    };
    let Some((axis, fixed)) = plane_axis_fixed(prev, hit) else {
        return Vec::new();
    };
    coords
        .iter()
        .copied()
        .filter(|c| voxel_map.contains_key(c) && voxel_on_plane(*c, axis, fixed))
        .collect()
}

/// Keep empty coords that lie on the coplanar-void plane from screen (same plane as `coplanar_empty_connected_from_screen`).
pub fn filter_coords_coplanar_empty_from_screen(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    coords: &[VoxelCoord],
) -> Vec<VoxelCoord> {
    let grid_size = effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((hit, prev, _oid)) = ray_first_solid_scene(origin, dir, file, voxel_map, grid_size)
    else {
        return Vec::new();
    };
    let Some(prev) = prev else {
        return Vec::new();
    };
    let Some((axis, fixed)) = plane_axis_fixed(prev, hit) else {
        return Vec::new();
    };
    coords
        .iter()
        .copied()
        .filter(|c| !voxel_map.contains_key(c) && voxel_on_plane(*c, axis, fixed))
        .collect()
}

/// Flood BFS over solid voxels from screen pick (selection fill). `respect_color`: only like-colored
/// to seed; if false, include any solid connected in the chosen adjacency.
///
/// Cooperative cancel (Escape) and progress; stops at [`FILL_ABSOLUTE_MAX_CELLS`] with `hit_absolute_cap`.
pub fn flood_fill_selection_coords_with_control(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    fill_diagonals: bool,
    respect_color: bool,
    match_material: bool,
    fill_constrain_plane: bool,
    plane_axis: PlaneAxis,
    cancel: Option<&AtomicBool>,
    on_progress: &mut impl FnMut(usize),
) -> FillCoordOutcome {
    let grid_size = effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((hit, prev, _oid)) = ray_first_solid_scene(origin, dir, file, voxel_map, grid_size)
    else {
        return FillCoordOutcome {
            coords: Vec::new(),
            cancelled: false,
            hit_absolute_cap: false,
        };
    };
    let Some(&seed_idx) = voxel_map.get(&hit) else {
        return FillCoordOutcome {
            coords: Vec::new(),
            cancelled: false,
            hit_absolute_cap: false,
        };
    };
    let seed_v = file.voxels[seed_idx];
    let tc = seed_v.color;
    let tm = seed_v.material;

    let face_axis = prev.and_then(|p| face_axis_from_prev_hit(p, hit));
    let cam_forward = camera_view_forward(camera);

    let mut out: Vec<VoxelCoord> = Vec::new();
    let mut visited: AHashSet<VoxelCoord> = AHashSet::new();
    let mut queue: VecDeque<VoxelCoord> = VecDeque::new();
    let mut steps: usize = 0;
    queue.push_back(hit);

    while let Some(c) = queue.pop_front() {
        steps += 1;
        if steps.is_multiple_of(FILL_BFS_CANCEL_CHECK_INTERVAL)
            && cancel.map(|c| c.load(Ordering::Relaxed)).unwrap_or(false)
        {
            return FillCoordOutcome {
                coords: out,
                cancelled: true,
                hit_absolute_cap: false,
            };
        }
        if steps.is_multiple_of(FILL_BFS_PROGRESS_INTERVAL) {
            on_progress(out.len());
        }
        if visited.contains(&c) {
            continue;
        }
        visited.insert(c);
        let Some(&idx) = voxel_map.get(&c) else {
            continue;
        };
        let v = file.voxels[idx];
        if respect_color && (v.color != tc || (match_material && v.material != tm)) {
            continue;
        }
        if out.len() >= FILL_ABSOLUTE_MAX_CELLS {
            return FillCoordOutcome {
                coords: out,
                cancelled: false,
                hit_absolute_cap: true,
            };
        }
        out.push(c);

        let neigh: Vec<VoxelCoord> = if fill_diagonals {
            neighbors_26(c)
        } else {
            neighbors_6(c).to_vec()
        };
        for n in neigh {
            if !in_grid(n.0, n.1, n.2, grid_size) {
                continue;
            }
            if fill_constrain_plane
                && !voxel_in_fill_plane(n, hit, plane_axis, face_axis, cam_forward)
            {
                continue;
            }
            if !visited.contains(&n) {
                queue.push_back(n);
            }
        }
    }
    FillCoordOutcome {
        coords: out,
        cancelled: false,
        hit_absolute_cap: false,
    }
}

/// Fast check: would an unconstrained solid flood from this pick exceed `threshold` cells?
/// [`Err(())`] means the caller's [`AtomicBool`] cancel flag was set (Escape / Cancel).
#[allow(clippy::result_unit_err)] // Cancel uses `Err(())` sentinel; no further error detail needed.
pub fn flood_fill_selection_region_exceeds_threshold(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    fill_diagonals: bool,
    respect_color: bool,
    match_material: bool,
    fill_constrain_plane: bool,
    plane_axis: PlaneAxis,
    threshold: usize,
    cancel: Option<&AtomicBool>,
) -> Result<bool, ()> {
    let grid_size = effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((hit, prev, _oid)) = ray_first_solid_scene(origin, dir, file, voxel_map, grid_size)
    else {
        return Ok(false);
    };
    let Some(&seed_idx) = voxel_map.get(&hit) else {
        return Ok(false);
    };
    let seed_v = file.voxels[seed_idx];
    let tc = seed_v.color;
    let tm = seed_v.material;

    let face_axis = prev.and_then(|p| face_axis_from_prev_hit(p, hit));
    let cam_forward = camera_view_forward(camera);

    let mut matched: usize = 0;
    let mut visited: AHashSet<VoxelCoord> = AHashSet::new();
    let mut queue: VecDeque<VoxelCoord> = VecDeque::new();
    let mut steps: usize = 0;
    queue.push_back(hit);

    while let Some(c) = queue.pop_front() {
        steps += 1;
        if steps.is_multiple_of(FILL_THRESHOLD_PROBE_CANCEL_INTERVAL) {
            if cancel.map(|c| c.load(Ordering::Relaxed)).unwrap_or(false) {
                return Err(());
            }
            std::thread::yield_now();
        }
        if visited.contains(&c) {
            continue;
        }
        visited.insert(c);
        let Some(&idx) = voxel_map.get(&c) else {
            continue;
        };
        let v = file.voxels[idx];
        if respect_color && (v.color != tc || (match_material && v.material != tm)) {
            continue;
        }
        matched += 1;
        if matched > threshold {
            return Ok(true);
        }

        let neigh: Vec<VoxelCoord> = if fill_diagonals {
            neighbors_26(c)
        } else {
            neighbors_6(c).to_vec()
        };
        for n in neigh {
            if !in_grid(n.0, n.1, n.2, grid_size) {
                continue;
            }
            if fill_constrain_plane
                && !voxel_in_fill_plane(n, hit, plane_axis, face_axis, cam_forward)
            {
                continue;
            }
            if !visited.contains(&n) {
                queue.push_back(n);
            }
        }
    }
    Ok(false)
}

/// Fast check for empty-cell flood (add fill): would region exceed `threshold` empty cells?
/// [`Err(())`] means cancel.
#[allow(clippy::result_unit_err)]
pub fn flood_fill_empty_region_exceeds_threshold(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    fill_diagonals: bool,
    fill_constrain_plane: bool,
    plane_axis: PlaneAxis,
    threshold: usize,
    cancel: Option<&AtomicBool>,
) -> Result<bool, ()> {
    let grid_limit = effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((hit, prev, _oid)) = ray_first_solid_scene(origin, dir, file, voxel_map, grid_limit)
    else {
        return Ok(false);
    };
    let Some(seed) = prev else {
        return Ok(false);
    };
    if voxel_map.contains_key(&seed) {
        return Ok(false);
    }
    if !in_grid(seed.0, seed.1, seed.2, grid_limit) {
        return Ok(false);
    }
    let face_axis = face_axis_from_prev_hit(seed, hit);
    let cam_forward = camera_view_forward(camera);

    let mut matched: usize = 0;
    let mut visited: AHashSet<VoxelCoord> = AHashSet::new();
    let mut queue: VecDeque<VoxelCoord> = VecDeque::new();
    let mut steps: usize = 0;
    queue.push_back(seed);

    while let Some(c) = queue.pop_front() {
        steps += 1;
        if steps.is_multiple_of(FILL_THRESHOLD_PROBE_CANCEL_INTERVAL) {
            if cancel.map(|c| c.load(Ordering::Relaxed)).unwrap_or(false) {
                return Err(());
            }
            std::thread::yield_now();
        }
        if visited.contains(&c) {
            continue;
        }
        visited.insert(c);
        if voxel_map.contains_key(&c) {
            continue;
        }
        matched += 1;
        if matched > threshold {
            return Ok(true);
        }

        let neigh: Vec<VoxelCoord> = if fill_diagonals {
            neighbors_26(c)
        } else {
            neighbors_6(c).to_vec()
        };
        for n in neigh {
            if !in_grid(n.0, n.1, n.2, grid_limit) {
                continue;
            }
            if fill_constrain_plane
                && !voxel_in_fill_plane(n, seed, plane_axis, face_axis, cam_forward)
            {
                continue;
            }
            if !visited.contains(&n) {
                queue.push_back(n);
            }
        }
    }
    Ok(false)
}

/// Flood BFS over solid voxels from screen pick (selection fill). `respect_color`: only like-colored
/// to seed; if false, include any solid connected in the chosen adjacency.
#[allow(dead_code)] // Used by tests; convenience wrapper around [`flood_fill_selection_coords_with_control`].
pub fn flood_fill_selection_coords(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    fill_diagonals: bool,
    respect_color: bool,
    match_material: bool,
    fill_constrain_plane: bool,
    plane_axis: PlaneAxis,
) -> Vec<VoxelCoord> {
    flood_fill_selection_coords_with_control(
        file,
        voxel_map,
        camera,
        width,
        height,
        sx,
        sy,
        fill_diagonals,
        respect_color,
        match_material,
        fill_constrain_plane,
        plane_axis,
        None,
        &mut |_| {},
    )
    .coords
}
