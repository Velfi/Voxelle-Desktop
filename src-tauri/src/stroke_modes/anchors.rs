//! Per-stroke-mode anchor center computation and axis-aligned geometry helpers.

use crate::greedy_mesh::VoxelCoord;
use crate::voxel_edit::{anchor_for_stroke_edit, voxel_line_dda, world_to_voxel, EditTool};
use crate::voxelle::VoxelleFile;
use ahash::AHashMap;
use glam::Vec3;
use std::collections::HashSet;

use super::{
    DrawStrokeMode, PlaneAxis, StrokeAux,
    polygon::{
        convex_hull_2d, extrude_base_positions,
        fill_polygon_2d, fill_polygon_axis_aligned, fill_polygon_hull_axis_aligned,
        fill_solid_polygon_hull_projected, fill_solid_polygon_simple_projected,
        lift_plane_2d_to_voxels, project_vertices_to_plane_2d, solid_polygon_fixed_plane,
        stroke_aux_is_solid_family,
    },
    symmetry::{axis_align_line_endpoints, axis_from_plane_axis, face_normal_axis},
};

pub(super) fn disk_in_axis_plane(center: VoxelCoord, plane_axis: usize, radius: i32) -> Vec<VoxelCoord> {
    let r = radius.max(0);
    let mut out = Vec::new();
    let (cx, cy, cz) = center;
    let r2 = r * r;
    match plane_axis {
        0 => {
            for dy in -r..=r {
                for dz in -r..=r {
                    if dy * dy + dz * dz <= r2 {
                        out.push((cx, cy + dy, cz + dz));
                    }
                }
            }
        }
        1 => {
            for dx in -r..=r {
                for dz in -r..=r {
                    if dx * dx + dz * dz <= r2 {
                        out.push((cx + dx, cy, cz + dz));
                    }
                }
            }
        }
        2 => {
            for dx in -r..=r {
                for dy in -r..=r {
                    if dx * dx + dy * dy <= r2 {
                        out.push((cx + dx, cy + dy, cz));
                    }
                }
            }
        }
        _ => {}
    }
    out
}

/// Disk with the interior removed: voxels with `inner_r^2 < dist^2 <= outer_r^2` in the plane (Euclidean).
pub(super) fn annulus_in_axis_plane(center: VoxelCoord, plane_axis: usize, outer_r: i32) -> Vec<VoxelCoord> {
    let outer_r = outer_r.max(0);
    if outer_r == 0 {
        return Vec::new();
    }
    let inner_r = outer_r.saturating_sub(1);
    let o2 = outer_r * outer_r;
    let i2 = inner_r * inner_r;
    let mut out = Vec::new();
    let (cx, cy, cz) = center;
    match plane_axis {
        0 => {
            for dy in -outer_r..=outer_r {
                for dz in -outer_r..=outer_r {
                    let d2 = dy * dy + dz * dz;
                    if d2 <= o2 && d2 > i2 {
                        out.push((cx, cy + dy, cz + dz));
                    }
                }
            }
        }
        1 => {
            for dx in -outer_r..=outer_r {
                for dz in -outer_r..=outer_r {
                    let d2 = dx * dx + dz * dz;
                    if d2 <= o2 && d2 > i2 {
                        out.push((cx + dx, cy, cz + dz));
                    }
                }
            }
        }
        2 => {
            for dx in -outer_r..=outer_r {
                for dy in -outer_r..=outer_r {
                    let d2 = dx * dx + dy * dy;
                    if d2 <= o2 && d2 > i2 {
                        out.push((cx + dx, cy + dy, cz));
                    }
                }
            }
        }
        _ => {}
    }
    out
}

pub(super) fn circle_radius_in_plane(center: [i32; 3], edge: [i32; 3], plane_axis: usize) -> i32 {
    let r = match plane_axis {
        0 => {
            let dy = edge[1] - center[1];
            let dz = edge[2] - center[2];
            ((dy * dy + dz * dz) as f64).sqrt().round() as i32
        }
        1 => {
            let dx = edge[0] - center[0];
            let dz = edge[2] - center[2];
            ((dx * dx + dz * dz) as f64).sqrt().round() as i32
        }
        _ => {
            let dx = edge[0] - center[0];
            let dy = edge[1] - center[1];
            ((dx * dx + dy * dy) as f64).sqrt().round() as i32
        }
    };
    r.max(0)
}

/// Thick segment: voxels within `radius` of the polyline A–B in 3D (Manhattan tube, axis-aligned only).
pub(super) fn cylinder_axis_aligned_caps(a: VoxelCoord, b: VoxelCoord, radius: i32) -> Vec<VoxelCoord> {
    let line = voxel_line_dda(a, b);
    let mut seen: HashSet<VoxelCoord> = HashSet::new();
    let r = radius.max(0);
    for c in line {
        for dx in -r..=r {
            for dy in -r..=r {
                for dz in -r..=r {
                    if dx * dx + dy * dy + dz * dz <= r * r {
                        let p = (c.0 + dx, c.1 + dy, c.2 + dz);
                        seen.insert(p);
                    }
                }
            }
        }
    }
    seen.into_iter().collect()
}

fn intersect_ray_axis_aligned_plane(
    origin: Vec3,
    dir: Vec3,
    plane_axis: usize,
    plane_coord: f32,
) -> Option<Vec3> {
    let d = match plane_axis {
        0 => dir.x,
        1 => dir.y,
        2 => dir.z,
        _ => return None,
    };
    if d.abs() < 1e-8 {
        return None;
    }
    let o = match plane_axis {
        0 => origin.x,
        1 => origin.y,
        2 => origin.z,
        _ => return None,
    };
    let t = (plane_coord - o) / d;
    if t < 0.0 {
        return None;
    }
    Some(origin + dir * t)
}

/// Compute the end-point voxel `b` for a drag-plane operation.
///
/// When the start and end screen coordinates are the same (a click with no
/// drag), we return `a` directly instead of re-deriving via ray–plane
/// intersection + rounding — the two paths use the same `world_to_voxel`
/// formula but feed it different inputs (DDA step vs plane intersection),
/// which can disagree by one voxel on boundaries.
fn drag_plane_end_voxel(
    a: VoxelCoord,
    plane_ax: usize,
    camera: &crate::camera::OrbitCamera,
    width: f32,
    height: f32,
    lsx: f32,
    lsy: f32,
    sx: f32,
    sy: f32,
) -> Option<VoxelCoord> {
    if (sx - lsx).abs() < 1e-6 && (sy - lsy).abs() < 1e-6 {
        return Some(a);
    }
    let (origin1, dir1) = crate::voxel_edit::screen_to_world_ray(camera, width, height, sx, sy);
    let plane_coord = match plane_ax {
        0 => a.0 as f32 + 0.5,
        1 => a.1 as f32 + 0.5,
        _ => a.2 as f32 + 0.5,
    };
    let p = intersect_ray_axis_aligned_plane(origin1, dir1, plane_ax, plane_coord)?;
    let mut b = world_to_voxel(p);
    match plane_ax {
        0 => b.0 = a.0,
        1 => b.1 = a.1,
        _ => b.2 = a.2,
    }
    Some(b)
}

/// Web `getAxisAlignedPlaneFromNormal`: filled rectangle in the axis-aligned face plane through `a`, spanning to `b`.
pub(super) fn fill_axis_aligned_plane_rectangle(
    a: VoxelCoord,
    b: VoxelCoord,
    fixed_axis: usize,
) -> Vec<VoxelCoord> {
    let mut out = Vec::new();
    match fixed_axis {
        0 => {
            let x = a.0;
            let y0 = a.1.min(b.1);
            let y1 = a.1.max(b.1);
            let z0 = a.2.min(b.2);
            let z1 = a.2.max(b.2);
            for py in y0..=y1 {
                for pz in z0..=z1 {
                    out.push((x, py, pz));
                }
            }
        }
        1 => {
            let y = a.1;
            let x0 = a.0.min(b.0);
            let x1 = a.0.max(b.0);
            let z0 = a.2.min(b.2);
            let z1 = a.2.max(b.2);
            for px in x0..=x1 {
                for pz in z0..=z1 {
                    out.push((px, y, pz));
                }
            }
        }
        _ => {
            let z = a.2;
            let x0 = a.0.min(b.0);
            let x1 = a.0.max(b.0);
            let y0 = a.1.min(b.1);
            let y1 = a.1.max(b.1);
            for px in x0..=x1 {
                for py in y0..=y1 {
                    out.push((px, py, z));
                }
            }
        }
    }
    out
}

/// Outer rectangle minus inner rectangle eroded by `wall` in the two free axes (hollow plane shell).
fn hollow_plane_rectangle_frame(
    a: VoxelCoord,
    b: VoxelCoord,
    fixed_axis: usize,
    wall: i32,
) -> Vec<VoxelCoord> {
    let w = wall.max(1);
    let full = fill_axis_aligned_plane_rectangle(a, b, fixed_axis);
    let coord = |c: VoxelCoord, ax: usize| -> i32 {
        match ax {
            0 => c.0,
            1 => c.1,
            _ => c.2,
        }
    };
    let (u_axis, v_axis) = match fixed_axis {
        0 => (1usize, 2usize),
        1 => (0usize, 2usize),
        _ => (0usize, 1usize),
    };
    let u0 = coord(a, u_axis).min(coord(b, u_axis));
    let u1 = coord(a, u_axis).max(coord(b, u_axis));
    let v0 = coord(a, v_axis).min(coord(b, v_axis));
    let v1 = coord(a, v_axis).max(coord(b, v_axis));
    let iu0 = u0 + w;
    let iu1 = u1 - w;
    let iv0 = v0 + w;
    let iv1 = v1 - w;
    if iu0 > iu1 || iv0 > iv1 {
        return full;
    }
    let inner_a = match fixed_axis {
        0 => (a.0, iu0, iv0),
        1 => (iu0, a.1, iv0),
        _ => (iu0, iv0, a.2),
    };
    let inner_b = match fixed_axis {
        0 => (a.0, iu1, iv1),
        1 => (iu1, a.1, iv1),
        _ => (iu1, iv1, a.2),
    };
    let inner: HashSet<VoxelCoord> =
        fill_axis_aligned_plane_rectangle(inner_a, inner_b, fixed_axis)
            .into_iter()
            .collect();
    full.into_iter().filter(|c| !inner.contains(c)).collect()
}

pub(super) const NEIGHBORS6: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

pub(super) fn neighbors_in_fixed_plane(fixed_axis: usize) -> &'static [(i32, i32, i32)] {
    match fixed_axis {
        0 => &[(0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
        1 => &[(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)],
        _ => &[(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)],
    }
}

/// Web `hollowSolidToShell` — erodes `thickness` layers from the interior using `neighbors`.
pub(super) fn hollow_solid_to_shell(
    solid: &[VoxelCoord],
    thickness: i32,
    neighbors: &[(i32, i32, i32)],
) -> Vec<VoxelCoord> {
    let t = thickness.max(1);
    if solid.is_empty() {
        return Vec::new();
    }
    let mut r: HashSet<VoxelCoord> = solid.iter().copied().collect();
    for _ in 0..t {
        if r.is_empty() {
            break;
        }
        let mut next: HashSet<VoxelCoord> = HashSet::new();
        for &(x, y, z) in &r {
            let mut all_inside = true;
            for &(dx, dy, dz) in neighbors {
                let p = (x + dx, y + dy, z + dz);
                if !r.contains(&p) {
                    all_inside = false;
                    break;
                }
            }
            if all_inside {
                next.insert((x, y, z));
            }
        }
        r = next;
    }
    let core = r;
    solid
        .iter()
        .copied()
        .filter(|p| !core.contains(p))
        .collect()
}

/// Web `getAxisAlignedCuboid`. `face_n*` is outward from solid into air (e.g. `prev - hit` from [`ray_first_solid`]).
pub fn axis_aligned_cuboid_from_plane(
    a: VoxelCoord,
    b: VoxelCoord,
    face_nx: i32,
    face_ny: i32,
    face_nz: i32,
    depth: i32,
    plane_hollow: bool,
    hollow_wall_thickness: i32,
    plane_ax: usize,
) -> Vec<VoxelCoord> {
    let ax = face_nx.abs();
    let ay = face_ny.abs();
    let az = face_nz.abs();
    let fixed_axis = if ax >= ay && ax >= az {
        0usize
    } else if ay >= az {
        1usize
    } else {
        2usize
    };
    let wall = hollow_wall_thickness.max(1);
    let plane_positions = fill_axis_aligned_plane_rectangle(a, b, plane_ax);
    if depth == 0 {
        if !plane_hollow {
            return plane_positions;
        }
        return hollow_solid_to_shell(&plane_positions, wall, neighbors_in_fixed_plane(fixed_axis));
    }
    let mut positions: Vec<VoxelCoord> = plane_positions.clone();
    let comp = match fixed_axis {
        0 => face_nx,
        1 => face_ny,
        _ => face_nz,
    };
    let step = if comp > 0 {
        1
    } else if comp < 0 {
        -1
    } else {
        0
    };
    let layers = depth.abs();
    let dir = if depth > 0 { step } else { -step };
    if dir != 0 {
        for k in 1..=layers {
            let dk = dir * k;
            for &(px, py, pz) in &plane_positions {
                let p = match fixed_axis {
                    0 => (px + dk, py, pz),
                    1 => (px, py + dk, pz),
                    _ => (px, py, pz + dk),
                };
                positions.push(p);
            }
        }
    }
    if !plane_hollow {
        return positions;
    }
    hollow_solid_to_shell(&positions, wall, &NEIGHBORS6)
}

/// Web `getAxisAlignedCircleFromNormal`.
fn axis_aligned_circle_from_normal(
    center: VoxelCoord,
    edge: VoxelCoord,
    face_nx: i32,
    face_ny: i32,
    face_nz: i32,
    hollow: bool,
    hollow_wall_thickness: i32,
) -> Vec<VoxelCoord> {
    let ax = face_nx.abs();
    let ay = face_ny.abs();
    let az = face_nz.abs();
    let fixed_axis = if ax >= ay && ax >= az {
        0usize
    } else if ay >= az {
        1usize
    } else {
        2usize
    };
    let wall = hollow_wall_thickness.max(1);

    let (cu, cv, eu, ev) = match fixed_axis {
        0 => (center.1, center.2, edge.1, edge.2),
        1 => (center.0, center.2, edge.0, edge.2),
        _ => (center.0, center.1, edge.0, edge.1),
    };
    let du = eu - cu;
    let dv = ev - cv;
    let r_sq = du * du + dv * dv;
    if r_sq == 0 {
        return vec![center];
    }
    let ru = ((r_sq as f64).sqrt().ceil() as i32).max(0);
    let mut filled: Vec<VoxelCoord> = Vec::new();
    for u in (cu - ru)..=(cu + ru) {
        for v in (cv - ru)..=(cv + ru) {
            let ddu = u - cu;
            let ddv = v - cv;
            if ddu * ddu + ddv * ddv <= r_sq {
                let p = match fixed_axis {
                    0 => (center.0, u, v),
                    1 => (u, center.1, v),
                    _ => (u, v, center.2),
                };
                filled.push(p);
            }
        }
    }
    if !hollow {
        return filled;
    }
    hollow_solid_to_shell(&filled, wall, neighbors_in_fixed_plane(fixed_axis))
}

/// Web `getAxisAlignedCylinder` — `face_n*` outward air→solid (`prev - hit`).
pub fn axis_aligned_cylinder_from_plane(
    center: VoxelCoord,
    edge: VoxelCoord,
    face_nx: i32,
    face_ny: i32,
    face_nz: i32,
    depth: i32,
    taper_pct: i32,
    plane_hollow: bool,
    hollow_wall_thickness: i32,
) -> Vec<VoxelCoord> {
    if depth == 0 {
        return axis_aligned_circle_from_normal(
            center,
            edge,
            face_nx,
            face_ny,
            face_nz,
            plane_hollow,
            hollow_wall_thickness,
        );
    }
    let ax = face_nx.abs();
    let ay = face_ny.abs();
    let az = face_nz.abs();
    let fixed_axis = if ax >= ay && ax >= az {
        0usize
    } else if ay >= az {
        1usize
    } else {
        2usize
    };
    let wall = hollow_wall_thickness.max(1);

    let (cu, cv, eu, ev) = match fixed_axis {
        0 => (center.1, center.2, edge.1, edge.2),
        1 => (center.0, center.2, edge.0, edge.2),
        _ => (center.0, center.1, edge.0, edge.1),
    };
    let base_w = match fixed_axis {
        0 => center.0,
        1 => center.1,
        _ => center.2,
    };
    let du = eu - cu;
    let dv = ev - cv;
    let base_r_sq = du * du + dv * dv;

    let comp = match fixed_axis {
        0 => face_nx,
        1 => face_ny,
        _ => face_nz,
    };
    let step = if comp > 0 {
        1
    } else if comp < 0 {
        -1
    } else {
        0
    };
    let layers = depth.abs();
    let dir = if depth > 0 { step } else { -step };
    let taper = taper_pct.clamp(0, 100);

    let mut seen: HashSet<VoxelCoord> = HashSet::new();
    let mut positions: Vec<VoxelCoord> = Vec::new();

    for k in 0..=layers {
        let w = base_w + dir * k;
        let r_sq_i64: i64 = if taper > 0 && layers > 0 {
            let scale = 1.0 - (taper as f64 / 100.0) * (k as f64 / layers as f64);
            let s2 = scale * scale;
            ((base_r_sq as f64) * s2).round() as i64
        } else {
            base_r_sq as i64
        };

        if r_sq_i64 <= 0 {
            let p = match fixed_axis {
                0 => (w, cu, cv),
                1 => (cu, w, cv),
                _ => (cu, cv, w),
            };
            if seen.insert(p) {
                positions.push(p);
            }
            continue;
        }
        let ru = ((r_sq_i64 as f64).sqrt().ceil() as i32).max(0);
        for u in (cu - ru)..=(cu + ru) {
            for v in (cv - ru)..=(cv + ru) {
                let ddu = (u - cu) as i64;
                let ddv = (v - cv) as i64;
                if ddu * ddu + ddv * ddv <= r_sq_i64 {
                    let p = match fixed_axis {
                        0 => (w, u, v),
                        1 => (u, w, v),
                        _ => (u, v, w),
                    };
                    if seen.insert(p) {
                        positions.push(p);
                    }
                }
            }
        }
    }

    if !plane_hollow {
        return positions;
    }
    hollow_solid_to_shell(&positions, wall, &NEIGHBORS6)
}

/// Returns the frozen depth-phase geometry from `StrokeAux` when all five fields are present,
/// bypassing the camera-dependent raycast so camera movement during depth phase cannot alter the
/// extrusion direction.
fn frozen_cuboid_geo(
    aux: &StrokeAux,
) -> Option<(
    crate::greedy_mesh::VoxelCoord,
    crate::greedy_mesh::VoxelCoord,
    usize,
    crate::greedy_mesh::VoxelCoord,
    crate::greedy_mesh::VoxelCoord,
)> {
    let [ax, ay, az] = aux.cuboid_frozen_a?;
    let [bx, by, bz] = aux.cuboid_frozen_b?;
    let plane_ax = aux.cuboid_frozen_plane_ax? as usize;
    let [hx, hy, hz] = aux.cuboid_frozen_hit?;
    let [px, py, pz] = aux.cuboid_frozen_prev?;
    Some((
        (ax, ay, az),
        (bx, by, bz),
        plane_ax,
        (hx, hy, hz),
        (px, py, pz),
    ))
}

/// Public re-export of [`cuboid_drag_plane_geometry`] for use by `lib.rs`'s
/// `query_cuboid_plane_geometry` command.
pub fn cuboid_drag_plane_geometry_pub(
    tool: EditTool,
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &crate::camera::OrbitCamera,
    width: f32,
    height: f32,
    lsx: f32,
    lsy: f32,
    sx: f32,
    sy: f32,
    plane_axis: PlaneAxis,
    snap_to_surface: bool,
) -> Option<(VoxelCoord, VoxelCoord, usize, VoxelCoord, VoxelCoord)> {
    cuboid_drag_plane_geometry(
        tool,
        file,
        voxel_map,
        camera,
        width,
        height,
        lsx,
        lsy,
        sx,
        sy,
        plane_axis,
        snap_to_surface,
    )
}

/// `a`, `b`, `plane_ax`, solid hit, air cell before hit (for outward normal `prev - hit`).
fn cuboid_drag_plane_geometry(
    tool: EditTool,
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &crate::camera::OrbitCamera,
    width: f32,
    height: f32,
    lsx: f32,
    lsy: f32,
    sx: f32,
    sy: f32,
    plane_axis: PlaneAxis,
    snap_to_surface: bool,
) -> Option<(VoxelCoord, VoxelCoord, usize, VoxelCoord, VoxelCoord)> {
    let grid_size = crate::voxel_edit::effective_ray_grid_size(file);
    let (origin0, dir0) = crate::voxel_edit::screen_to_world_ray(camera, width, height, lsx, lsy);
    let (hit0, prev0, _oid0) =
        crate::voxel_edit::ray_first_solid_scene(origin0, dir0, file, voxel_map, grid_size)?;
    let prev0 = prev0?;
    let face_ax = face_normal_axis(prev0, hit0);
    let plane_ax = axis_from_plane_axis(plane_axis, face_ax).unwrap_or(2);
    let a = anchor_for_stroke_edit(
        tool,
        snap_to_surface,
        file,
        voxel_map,
        camera,
        width,
        height,
        lsx,
        lsy,
    )?;
    let b = drag_plane_end_voxel(a, plane_ax, camera, width, height, lsx, lsy, sx, sy)?;
    // When plane_axis is fixed (X/Y/Z), override the extrusion direction to be
    // perpendicular to the chosen plane, not along the clicked face's normal.
    let (hit_out, prev_out) = match plane_axis {
        PlaneAxis::Auto | PlaneAxis::Camera => (hit0, prev0),
        _ => {
            // Determine sign along plane_ax: prefer original face projection,
            // fall back to camera-to-target direction.
            let orig_delta = [prev0.0 - hit0.0, prev0.1 - hit0.1, prev0.2 - hit0.2];
            let sign = {
                let proj = orig_delta[plane_ax];
                if proj != 0 {
                    proj.signum()
                } else {
                    let eye = camera.eye();
                    let cam = [
                        camera.target.x - eye.x,
                        camera.target.y - eye.y,
                        camera.target.z - eye.z,
                    ];
                    if cam[plane_ax] < 0.0 {
                        1
                    } else {
                        -1
                    }
                }
            };
            let syn_prev = match plane_ax {
                0 => (hit0.0 + sign, hit0.1, hit0.2),
                1 => (hit0.0, hit0.1 + sign, hit0.2),
                _ => (hit0.0, hit0.1, hit0.2 + sign),
            };
            (hit0, syn_prev)
        }
    };
    Some((a, b, plane_ax, hit_out, prev_out))
}

/// Web Surface plane / Solid cuboid plane phase: rectangle (or hollow shell) from drag start to current in the face plane.
fn axis_aligned_drag_plane_rectangle(
    tool: EditTool,
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &crate::camera::OrbitCamera,
    width: f32,
    height: f32,
    lsx: f32,
    lsy: f32,
    sx: f32,
    sy: f32,
    plane_axis: PlaneAxis,
    plane_hollow: bool,
    snap_to_surface: bool,
) -> Option<Vec<VoxelCoord>> {
    let grid_size = crate::voxel_edit::effective_ray_grid_size(file);
    let (origin0, dir0) = crate::voxel_edit::screen_to_world_ray(camera, width, height, lsx, lsy);
    let (_hit0, prev0, _oid0) =
        crate::voxel_edit::ray_first_solid_scene(origin0, dir0, file, voxel_map, grid_size)?;
    let prev0 = prev0?;
    let face_ax = face_normal_axis(prev0, _hit0);
    let plane_ax = axis_from_plane_axis(plane_axis, face_ax).unwrap_or(2);
    let a = anchor_for_stroke_edit(
        tool,
        snap_to_surface,
        file,
        voxel_map,
        camera,
        width,
        height,
        lsx,
        lsy,
    )?;
    let b = drag_plane_end_voxel(a, plane_ax, camera, width, height, lsx, lsy, sx, sy)?;
    Some(if plane_hollow {
        hollow_plane_rectangle_frame(a, b, plane_ax, 1)
    } else {
        fill_axis_aligned_plane_rectangle(a, b, plane_ax)
    })
}

/// Web circle/cylinder base drag: disk in the face plane from start anchor to current (cone intersection point).
fn axis_aligned_drag_plane_circle(
    tool: EditTool,
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &crate::camera::OrbitCamera,
    width: f32,
    height: f32,
    lsx: f32,
    lsy: f32,
    sx: f32,
    sy: f32,
    plane_axis: PlaneAxis,
    plane_hollow: bool,
    snap_to_surface: bool,
) -> Option<Vec<VoxelCoord>> {
    let grid_size = crate::voxel_edit::effective_ray_grid_size(file);
    let (origin0, dir0) = crate::voxel_edit::screen_to_world_ray(camera, width, height, lsx, lsy);
    let (_hit0, prev0, _oid0) =
        crate::voxel_edit::ray_first_solid_scene(origin0, dir0, file, voxel_map, grid_size)?;
    let prev0 = prev0?;
    let face_ax = face_normal_axis(prev0, _hit0);
    let plane_ax = axis_from_plane_axis(plane_axis, face_ax).unwrap_or(2);
    let a = anchor_for_stroke_edit(
        tool,
        snap_to_surface,
        file,
        voxel_map,
        camera,
        width,
        height,
        lsx,
        lsy,
    )?;
    let b = drag_plane_end_voxel(a, plane_ax, camera, width, height, lsx, lsy, sx, sy)?;
    let cc = [a.0, a.1, a.2];
    let ce = [b.0, b.1, b.2];
    let r = circle_radius_in_plane(cc, ce, plane_ax);
    let center = (a.0, a.1, a.2);
    Some(if plane_hollow && r > 0 {
        annulus_in_axis_plane(center, plane_ax, r)
    } else {
        disk_in_axis_plane(center, plane_ax, r)
    })
}

/// Two-click circle: `plane_ax` must match the face plane (same as [`axis_aligned_drag_plane_circle`]),
/// not "first equal coordinate" between center and edge — that mis-picks when two coords match (e.g. top‑face drag along X).
fn circle_plane_axis_two_click(
    plane_axis: PlaneAxis,
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &crate::camera::OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    cc: [i32; 3],
    ce: [i32; 3],
) -> usize {
    let grid_size = crate::voxel_edit::effective_ray_grid_size(file);
    let (origin, dir) = crate::voxel_edit::screen_to_world_ray(camera, width, height, sx, sy);
    if let Some((hit, Some(prev), _oid)) =
        crate::voxel_edit::ray_first_solid_scene(origin, dir, file, voxel_map, grid_size)
    {
        if let Some(face_ax) = face_normal_axis(prev, hit) {
            if let Some(ax) = axis_from_plane_axis(plane_axis, Some(face_ax)) {
                return ax;
            }
        }
    }
    // Ray miss / degenerate hit: prefer Z then Y then X as the constant coordinate (common horizontal‑first scenes).
    if cc[2] == ce[2] {
        2
    } else if cc[1] == ce[1] {
        1
    } else if cc[0] == ce[0] {
        0
    } else {
        2
    }
}

/// When depth is negative and the tool is Add, flip the anchor and face normal
/// so the cuboid/cylinder extrudes through the clicked surface into the empty
/// space on the far side.  For Remove (and positive depth) the original
/// geometry is returned unchanged.
pub(super) fn flip_depth_anchor_if_needed(
    tool: EditTool,
    depth: i32,
    a: VoxelCoord,
    b: VoxelCoord,
    hit: VoxelCoord,
    prev: VoxelCoord,
) -> (VoxelCoord, VoxelCoord, i32, i32, i32, i32) {
    if depth < 0 && matches!(tool, EditTool::Add) {
        // Shift anchor from `prev` (empty space) to `hit` (surface),
        // reverse the face normal, and use |depth|.
        let dx = hit.0 - prev.0;
        let dy = hit.1 - prev.1;
        let dz = hit.2 - prev.2;
        (
            (a.0 + dx, a.1 + dy, a.2 + dz),
            (b.0 + dx, b.1 + dy, b.2 + dz),
            dx,
            dy,
            dz,
            depth.abs(),
        )
    } else {
        (a, b, prev.0 - hit.0, prev.1 - hit.1, prev.2 - hit.2, depth)
    }
}

/// Stroke anchor cells for draw/remove/paint (brush applied per center afterward).
///
/// `spray_constraint_plane`: when `Some((point, normal))`, spray mode raycasts against the invisible
/// plane instead of voxels (web constrain-to-plane parity).
#[allow(clippy::too_many_arguments)]
pub fn compute_anchors(
    mode: DrawStrokeMode,
    plane_axis: PlaneAxis,
    aux: &StrokeAux,
    tool: EditTool,
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &crate::camera::OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    brush_radius: u32,
    stroke_line_start: Option<(f32, f32)>,
    stroke_segment_prev: Option<(f32, f32)>,
    spray_constraint_plane: Option<(Vec3, Vec3)>,
) -> Vec<VoxelCoord> {
    let snap = aux.stroke_snap_to_surface;
    match mode {
        DrawStrokeMode::Fill => Vec::new(),
        DrawStrokeMode::Precise => {
            anchor_for_stroke_edit(tool, snap, file, voxel_map, camera, width, height, sx, sy)
                .into_iter()
                .collect()
        }
        DrawStrokeMode::Spray => {
            // When constrain-to-plane is active, raycast against the invisible plane.
            let anchor_fn = |sx_: f32, sy_: f32| -> Option<VoxelCoord> {
                if let Some((pp, pn)) = spray_constraint_plane {
                    crate::voxel_edit::anchor_on_plane(camera, width, height, sx_, sy_, pp, pn)
                } else {
                    anchor_for_stroke_edit(
                        tool, snap, file, voxel_map, camera, width, height, sx_, sy_,
                    )
                }
            };
            if let Some((px, py)) = stroke_segment_prev {
                match (anchor_fn(px, py), anchor_fn(sx, sy)) {
                    (Some(a), Some(b)) => voxel_line_dda(a, b),
                    _ => anchor_fn(sx, sy).into_iter().collect(),
                }
            } else {
                anchor_fn(sx, sy).into_iter().collect()
            }
        }
        DrawStrokeMode::Line => {
            let align = aux.stroke_axis_align;
            let line_pts = |a: VoxelCoord, b: VoxelCoord| {
                let (a, b) = if align {
                    axis_align_line_endpoints(a, b)
                } else {
                    (a, b)
                };
                voxel_line_dda(a, b)
            };
            if let Some((lsx, lsy)) = stroke_line_start {
                match (
                    anchor_for_stroke_edit(
                        tool, snap, file, voxel_map, camera, width, height, lsx, lsy,
                    ),
                    anchor_for_stroke_edit(
                        tool, snap, file, voxel_map, camera, width, height, sx, sy,
                    ),
                ) {
                    (Some(a), Some(b)) => line_pts(a, b),
                    _ => anchor_for_stroke_edit(
                        tool, snap, file, voxel_map, camera, width, height, sx, sy,
                    )
                    .into_iter()
                    .collect(),
                }
            } else if let Some((px, py)) = stroke_segment_prev {
                match (
                    anchor_for_stroke_edit(
                        tool, snap, file, voxel_map, camera, width, height, px, py,
                    ),
                    anchor_for_stroke_edit(
                        tool, snap, file, voxel_map, camera, width, height, sx, sy,
                    ),
                ) {
                    (Some(a), Some(b)) => line_pts(a, b),
                    _ => anchor_for_stroke_edit(
                        tool, snap, file, voxel_map, camera, width, height, sx, sy,
                    )
                    .into_iter()
                    .collect(),
                }
            } else {
                anchor_for_stroke_edit(tool, snap, file, voxel_map, camera, width, height, sx, sy)
                    .into_iter()
                    .collect()
            }
        }
        DrawStrokeMode::Plane => {
            if let Some((lsx, lsy)) = stroke_line_start {
                if let Some(cells) = axis_aligned_drag_plane_rectangle(
                    tool,
                    file,
                    voxel_map,
                    camera,
                    width,
                    height,
                    lsx,
                    lsy,
                    sx,
                    sy,
                    plane_axis,
                    aux.plane_hollow,
                    snap,
                ) {
                    return cells;
                }
            }
            let grid_size = crate::voxel_edit::effective_ray_grid_size(file);
            let (origin, dir) =
                crate::voxel_edit::screen_to_world_ray(camera, width, height, sx, sy);
            let Some((hit, prev, _oid)) =
                crate::voxel_edit::ray_first_solid_scene(origin, dir, file, voxel_map, grid_size)
            else {
                return Vec::new();
            };
            let Some(prev) = prev else {
                return Vec::new();
            };
            let face_ax = face_normal_axis(prev, hit);
            let plane_ax = axis_from_plane_axis(plane_axis, face_ax).unwrap_or(2);
            let anchor =
                anchor_for_stroke_edit(tool, snap, file, voxel_map, camera, width, height, sx, sy);
            let Some(c) = anchor else {
                return Vec::new();
            };
            if aux.plane_hollow {
                annulus_in_axis_plane(c, plane_ax, brush_radius as i32)
            } else {
                disk_in_axis_plane(c, plane_ax, brush_radius as i32)
            }
        }
        DrawStrokeMode::Circle => {
            if let (Some(&cc), Some(&ce)) = (aux.circle_center.as_ref(), aux.circle_edge.as_ref()) {
                let center = (cc[0], cc[1], cc[2]);
                let plane_ax = circle_plane_axis_two_click(
                    plane_axis, file, voxel_map, camera, width, height, sx, sy, cc, ce,
                );
                let r = circle_radius_in_plane(cc, ce, plane_ax);
                disk_in_axis_plane(center, plane_ax, r)
            } else if let Some((lsx, lsy)) = stroke_line_start {
                if let Some(v) = axis_aligned_drag_plane_circle(
                    tool,
                    file,
                    voxel_map,
                    camera,
                    width,
                    height,
                    lsx,
                    lsy,
                    sx,
                    sy,
                    plane_axis,
                    aux.plane_hollow,
                    snap,
                ) {
                    v
                } else {
                    anchor_for_stroke_edit(
                        tool, snap, file, voxel_map, camera, width, height, sx, sy,
                    )
                    .into_iter()
                    .collect()
                }
            } else {
                anchor_for_stroke_edit(tool, snap, file, voxel_map, camera, width, height, sx, sy)
                    .into_iter()
                    .collect()
            }
        }
        DrawStrokeMode::Cuboid => {
            if let Some((lsx, lsy)) = stroke_line_start {
                let wall = aux.cuboid_hollow_wall_thickness.unwrap_or(1);
                if let Some(depth) = aux.cuboid_depth {
                    let geo = frozen_cuboid_geo(aux).or_else(|| {
                        cuboid_drag_plane_geometry(
                            tool, file, voxel_map, camera, width, height, lsx, lsy, sx, sy,
                            plane_axis, snap,
                        )
                    });
                    if let Some((a, b, plane_ax, hit, prev)) = geo {
                        // Negative depth + Add: flip anchor & normal so the
                        // cuboid extrudes through the surface into empty space
                        // on the far side, letting users "grow down" as well
                        // as up.
                        let (fa, fb, fnx, fny, fnz, fd) =
                            flip_depth_anchor_if_needed(tool, depth, a, b, hit, prev);
                        return axis_aligned_cuboid_from_plane(
                            fa,
                            fb,
                            fnx,
                            fny,
                            fnz,
                            fd,
                            aux.plane_hollow,
                            wall,
                            plane_ax,
                        );
                    }
                    return Vec::new();
                }
                if let Some(v) = axis_aligned_drag_plane_rectangle(
                    tool,
                    file,
                    voxel_map,
                    camera,
                    width,
                    height,
                    lsx,
                    lsy,
                    sx,
                    sy,
                    plane_axis,
                    aux.plane_hollow,
                    snap,
                ) {
                    v
                } else {
                    anchor_for_stroke_edit(
                        tool, snap, file, voxel_map, camera, width, height, sx, sy,
                    )
                    .into_iter()
                    .collect()
                }
            } else {
                anchor_for_stroke_edit(tool, snap, file, voxel_map, camera, width, height, sx, sy)
                    .into_iter()
                    .collect()
            }
        }
        DrawStrokeMode::Cylinder => {
            if let Some((lsx, lsy)) = stroke_line_start {
                let wall = aux.cuboid_hollow_wall_thickness.unwrap_or(1);
                if let Some(depth) = aux.cylinder_depth {
                    let geo = frozen_cuboid_geo(aux).or_else(|| {
                        cuboid_drag_plane_geometry(
                            tool, file, voxel_map, camera, width, height, lsx, lsy, sx, sy,
                            plane_axis, snap,
                        )
                    });
                    if let Some((a, b, _plane_ax, hit, prev)) = geo {
                        let taper = aux.cylinder_taper_pct.unwrap_or(0).clamp(0, 100);
                        let (fa, fb, fnx, fny, fnz, fd) =
                            flip_depth_anchor_if_needed(tool, depth, a, b, hit, prev);
                        return axis_aligned_cylinder_from_plane(
                            fa,
                            fb,
                            fnx,
                            fny,
                            fnz,
                            fd,
                            taper,
                            aux.plane_hollow,
                            wall,
                        );
                    }
                    return Vec::new();
                }
                if let Some(v) = axis_aligned_drag_plane_circle(
                    tool,
                    file,
                    voxel_map,
                    camera,
                    width,
                    height,
                    lsx,
                    lsy,
                    sx,
                    sy,
                    plane_axis,
                    aux.plane_hollow,
                    snap,
                ) {
                    v
                } else {
                    anchor_for_stroke_edit(
                        tool, snap, file, voxel_map, camera, width, height, sx, sy,
                    )
                    .into_iter()
                    .collect()
                }
            } else if let (Some(a), Some(b)) = (aux.cylinder_a, aux.cylinder_b) {
                cylinder_axis_aligned_caps(
                    (a[0], a[1], a[2]),
                    (b[0], b[1], b[2]),
                    brush_radius as i32,
                )
            } else {
                anchor_for_stroke_edit(tool, snap, file, voxel_map, camera, width, height, sx, sy)
                    .into_iter()
                    .collect()
            }
        }
        DrawStrokeMode::Polygon => {
            if aux.polygon_vertices.len() >= 3 {
                if stroke_aux_is_solid_family(aux) {
                    if let Some(depth) = aux.polygon_depth {
                        if let Some((fixed_axis, fixed_coord)) =
                            solid_polygon_fixed_plane(&aux.polygon_vertices, plane_axis)
                        {
                            let poly2d =
                                project_vertices_to_plane_2d(&aux.polygon_vertices, fixed_axis);
                            let base = lift_plane_2d_to_voxels(
                                fixed_axis,
                                fixed_coord,
                                &fill_polygon_2d(&poly2d),
                            );
                            return extrude_base_positions(base, fixed_axis, depth);
                        }
                        return fill_polygon_axis_aligned(&aux.polygon_vertices);
                    }
                    if let Some(v) =
                        fill_solid_polygon_simple_projected(&aux.polygon_vertices, plane_axis)
                    {
                        v
                    } else {
                        fill_polygon_axis_aligned(&aux.polygon_vertices)
                    }
                } else {
                    fill_polygon_axis_aligned(&aux.polygon_vertices)
                }
            } else {
                anchor_for_stroke_edit(tool, snap, file, voxel_map, camera, width, height, sx, sy)
                    .into_iter()
                    .collect()
            }
        }
        DrawStrokeMode::PolygonHull => {
            if aux.polygon_vertices.len() >= 3 {
                if stroke_aux_is_solid_family(aux) {
                    if let Some(depth) = aux.polygon_depth {
                        if let Some((fixed_axis, fixed_coord)) =
                            solid_polygon_fixed_plane(&aux.polygon_vertices, plane_axis)
                        {
                            let pts =
                                project_vertices_to_plane_2d(&aux.polygon_vertices, fixed_axis);
                            let hull = convex_hull_2d(pts);
                            if hull.len() >= 3 {
                                let base = lift_plane_2d_to_voxels(
                                    fixed_axis,
                                    fixed_coord,
                                    &fill_polygon_2d(&hull),
                                );
                                return extrude_base_positions(base, fixed_axis, depth);
                            }
                        }
                        return fill_polygon_hull_axis_aligned(&aux.polygon_vertices);
                    }
                    if let Some(v) =
                        fill_solid_polygon_hull_projected(&aux.polygon_vertices, plane_axis)
                    {
                        v
                    } else {
                        fill_polygon_hull_axis_aligned(&aux.polygon_vertices)
                    }
                } else {
                    fill_polygon_hull_axis_aligned(&aux.polygon_vertices)
                }
            } else {
                anchor_for_stroke_edit(tool, snap, file, voxel_map, camera, width, height, sx, sy)
                    .into_iter()
                    .collect()
            }
        }
    }
}
