//! Brush shape generation (sphere, cube, cylinder, pyramid), spray constraints, and brush mask types.

use crate::camera::OrbitCamera;
use crate::greedy_mesh::VoxelCoord;
use crate::stroke_modes::{stroke_anchor_centers_with_mode, DrawStrokeMode, PlaneAxis, StrokeAux};
use crate::voxelle::VoxelleFile;
use ahash::AHashMap;
use glam::Vec3;
use std::collections::HashSet;

/// Polygon / polygonHull area uses exact lattice fill (web parity). Brush radius must not thicken
/// the filled region — otherwise each interior cell is expanded into a thick brush footprint.
#[inline]
pub(super) fn brush_radius_for_area_polygon_stroke(
    stroke_mode: DrawStrokeMode,
    brush_radius: u32,
) -> u32 {
    match stroke_mode {
        DrawStrokeMode::Polygon | DrawStrokeMode::PolygonHull => 0,
        _ => brush_radius,
    }
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BrushShape {
    #[default]
    Sphere,
    Cube,
    Pyramid,
    /// 2D flat rectangle in the face tangent plane (single layer, locked to one world axis).
    Square,
    /// 2D flat disk in the face tangent plane (single layer, locked to one world axis).
    Circle,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ExtrudeProfile {
    #[default]
    Cube,
    Cylinder,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ExtrudeEndCap {
    #[default]
    Flat,
    Rounded,
    Pointed,
}

/// Direction reference for straight-line extrude (matches web `branchExtrudeRef`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ExtrudeDirectionRef {
    /// View plane: drag maps through camera right/up.
    #[default]
    Camera,
    /// Dominant axis of the start face normal (falls back to camera if no face).
    Auto,
    /// World ±X, sign from drag vs view plane.
    X,
    /// World ±Y, sign from drag vs view plane.
    Y,
    /// World ±Z, sign from drag vs view plane.
    Z,
}

/// Resolve the world-space extrusion direction from screen drag, camera, and direction reference.
/// Matches web `resolveBranchExtrudeDirection`.
pub fn resolve_extrude_direction(
    dir_ref: ExtrudeDirectionRef,
    camera: &OrbitCamera,
    screen_dx: f32,
    screen_dy: f32,
    face_normal: Option<(i32, i32, i32)>,
) -> Vec3 {
    let eye = camera.smooth_eye();
    let target = camera.smooth_target;
    let view_dir = (target - eye).normalize_or_zero();
    let world_up = Vec3::Y;
    let right = view_dir.cross(world_up).normalize_or_zero();
    let up = right.cross(view_dir).normalize_or_zero();
    // Map screen drag to camera-relative world direction
    let raw = right * screen_dx + up * screen_dy;

    let axis_sign_from_drag = |axis: Vec3| -> f32 {
        let d = raw.dot(axis);
        if d.abs() < 1e-9 || d > 0.0 {
            1.0
        } else {
            -1.0
        }
    };

    let snap_normal_to_axis = |n: (i32, i32, i32)| -> Vec3 {
        let ax = n.0.abs();
        let ay = n.1.abs();
        let az = n.2.abs();
        if ax >= ay && ax >= az {
            Vec3::new(n.0.signum() as f32, 0.0, 0.0)
        } else if ay >= ax && ay >= az {
            Vec3::new(0.0, n.1.signum() as f32, 0.0)
        } else {
            Vec3::new(0.0, 0.0, n.2.signum() as f32)
        }
    };

    match dir_ref {
        ExtrudeDirectionRef::Camera => {
            let len = raw.length();
            if len > 1e-6 {
                raw / len
            } else {
                up.normalize_or_zero()
            }
        }
        ExtrudeDirectionRef::Auto => {
            if let Some(n) = face_normal {
                let axis = snap_normal_to_axis(n);
                let sign = axis_sign_from_drag(axis);
                axis * sign
            } else {
                // Fallback to camera mode
                let len = raw.length();
                if len > 1e-6 {
                    raw / len
                } else {
                    up.normalize_or_zero()
                }
            }
        }
        ExtrudeDirectionRef::X => {
            let axis = Vec3::X;
            let sign = axis_sign_from_drag(axis);
            axis * sign
        }
        ExtrudeDirectionRef::Y => {
            let axis = Vec3::Y;
            let sign = axis_sign_from_drag(axis);
            axis * sign
        }
        ExtrudeDirectionRef::Z => {
            let axis = Vec3::Z;
            let sign = axis_sign_from_drag(axis);
            axis * sign
        }
    }
}

/// Generate a straight-line path of voxel coordinates from `origin` along `direction`.
/// Matches web `getRayDirectionPath`.
pub fn get_ray_direction_path(origin: VoxelCoord, direction: Vec3, length: u32) -> Vec<VoxelCoord> {
    if length == 0 {
        return vec![origin];
    }
    let len = direction.length();
    if len < 1e-9 {
        return vec![origin];
    }
    let nd = direction / len;
    let mut positions = Vec::with_capacity(length as usize + 1);
    let mut seen = ahash::AHashSet::with_capacity(length as usize + 1);
    for i in 0..=length {
        let x = (origin.0 as f32 + i as f32 * nd.x).round() as i32;
        let y = (origin.1 as f32 + i as f32 * nd.y).round() as i32;
        let z = (origin.2 as f32 + i as f32 * nd.z).round() as i32;
        let c = (x, y, z);
        if seen.insert(c) {
            positions.push(c);
        }
    }
    positions
}

/// Generate a straight voxel path between two coordinates, including both endpoints.
pub fn get_line_path_inclusive(origin: VoxelCoord, end: VoxelCoord) -> Vec<VoxelCoord> {
    let dx = end.0 - origin.0;
    let dy = end.1 - origin.1;
    let dz = end.2 - origin.2;
    let steps = dx.abs().max(dy.abs()).max(dz.abs()) as u32;
    if steps == 0 {
        return vec![origin];
    }
    get_ray_direction_path(origin, Vec3::new(dx as f32, dy as f32, dz as f32), steps)
}

/// Compute the extrude footprint for a selection: copies all selected coords forward
/// by `direction` (unit vector) for each depth step 1..=length.
pub fn extrude_selection_footprint(
    selection: &ahash::AHashSet<VoxelCoord>,
    direction: Vec3,
    length: u32,
) -> Vec<VoxelCoord> {
    let mut result = Vec::new();
    for depth in 1..=length {
        let offset = direction * depth as f32;
        for &(x, y, z) in selection {
            result.push((
                x + offset.x.round() as i32,
                y + offset.y.round() as i32,
                z + offset.z.round() as i32,
            ));
        }
    }
    result
}

/// Compute the extrude footprint for a straight-line ray spine.
/// This handles both cube and cylinder profiles, matching the web version's behavior.
pub fn extrude_ray_footprint(
    spine: &[VoxelCoord],
    brush_radius: u32,
    brush_shape: BrushShape,
    brush_strength: u32,
    brush_falloff: u32,
    stroke_seed: u32,
    extrude_profile: ExtrudeProfile,
    extrude_end_cap: ExtrudeEndCap,
    extrude_taper: bool,
    extrude_taper_start: f32,
    extrude_taper_end: f32,
) -> Vec<VoxelCoord> {
    if spine.is_empty() {
        return Vec::new();
    }
    if extrude_profile == ExtrudeProfile::Cylinder {
        let r = (brush_radius + 1) as f32 / 2.0;
        let footprint = if extrude_taper {
            let start_r = extrude_taper_start.max(0.0);
            let end_r = extrude_taper_end.max(0.0);
            extrude_tapered_cylinder_footprint(spine, start_r, end_r, extrude_end_cap)
        } else {
            extrude_uniform_cylinder_footprint(spine, r, extrude_end_cap)
        };
        filter_sculpt_footprint_stochastic(
            footprint,
            spine,
            brush_radius,
            brush_falloff,
            brush_strength,
            stroke_seed,
        )
    } else {
        let out = extrude_shaped_brush_footprint(
            spine,
            brush_shape,
            if extrude_taper {
                extrude_taper_start.max(0.0)
            } else {
                brush_radius as f32
            },
            if extrude_taper {
                extrude_taper_end.max(0.0)
            } else {
                brush_radius as f32
            },
            extrude_end_cap,
        );
        filter_sculpt_footprint_stochastic(
            out,
            spine,
            brush_radius,
            brush_falloff,
            brush_strength,
            stroke_seed,
        )
    }
}

fn add_brush_shape_cap(
    seen: &mut HashSet<VoxelCoord>,
    out: &mut Vec<VoxelCoord>,
    origin: VoxelCoord,
    dir: [f32; 3],
    base_radius: f32,
    brush_shape: BrushShape,
    rounded: bool,
) {
    if base_radius <= 0.0 {
        return;
    }
    let Some(t) = normalize3_opt(dir) else {
        return;
    };
    let k_max = base_radius.ceil().max(1.0) as i32;
    let mut cached_radius: Option<u32> = None;
    let mut cached_offsets: Vec<VoxelCoord> = Vec::new();
    for k in 1..=k_max {
        let frac = k as f32 / (k_max as f32 + 1.0);
        let radius = if rounded {
            base_radius * (1.0 - frac * frac).sqrt()
        } else {
            base_radius * (1.0 - frac)
        };
        let radius = radius.round().max(0.0) as u32;
        let center = (
            origin.0 + (k as f32 * t[0]).round() as i32,
            origin.1 + (k as f32 * t[1]).round() as i32,
            origin.2 + (k as f32 * t[2]).round() as i32,
        );
        if cached_radius != Some(radius) {
            cached_offsets = brush_offset_cells(brush_shape, radius, None, None);
            cached_radius = Some(radius);
        }
        for &(ox, oy, oz) in &cached_offsets {
            let c = (center.0 + ox, center.1 + oy, center.2 + oz);
            if seen.insert(c) {
                out.push(c);
            }
        }
    }
}

/// Brush-shaped extrude footprint with optional taper and end caps.
pub fn extrude_shaped_brush_footprint(
    spine: &[VoxelCoord],
    brush_shape: BrushShape,
    start_radius: f32,
    end_radius: f32,
    cap: ExtrudeEndCap,
) -> Vec<VoxelCoord> {
    if spine.is_empty() {
        return Vec::new();
    }
    let n = spine.len();
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut cached_radius: Option<u32> = None;
    let mut cached_offsets: Vec<VoxelCoord> = Vec::new();

    for (i, &(cx, cy, cz)) in spine.iter().enumerate() {
        let t = if n == 1 {
            0.0
        } else {
            i as f32 / (n as f32 - 1.0)
        };
        let radius = (start_radius + t * (end_radius - start_radius))
            .round()
            .max(0.0) as u32;
        if cached_radius != Some(radius) {
            cached_offsets = brush_offset_cells(brush_shape, radius, None, None);
            cached_radius = Some(radius);
        }
        for &(ox, oy, oz) in &cached_offsets {
            let c = (cx + ox, cy + oy, cz + oz);
            if seen.insert(c) {
                out.push(c);
            }
        }
    }

    if cap == ExtrudeEndCap::Rounded {
        if let Some(t0) = extrude_tangent_at(spine, 0) {
            add_brush_shape_cap(
                &mut seen,
                &mut out,
                spine[0],
                [-t0[0], -t0[1], -t0[2]],
                start_radius,
                brush_shape,
                true,
            );
        }
        if let Some(t1) = extrude_tangent_at(spine, n - 1) {
            add_brush_shape_cap(
                &mut seen,
                &mut out,
                spine[n - 1],
                t1,
                end_radius,
                brush_shape,
                true,
            );
        }
    }

    if cap == ExtrudeEndCap::Pointed {
        if let Some(t0) = extrude_tangent_at(spine, 0) {
            add_brush_shape_cap(
                &mut seen,
                &mut out,
                spine[0],
                [-t0[0], -t0[1], -t0[2]],
                start_radius,
                brush_shape,
                false,
            );
        }
        if let Some(t1) = extrude_tangent_at(spine, n - 1) {
            add_brush_shape_cap(
                &mut seen,
                &mut out,
                spine[n - 1],
                t1,
                end_radius,
                brush_shape,
                false,
            );
        }
    }

    out
}

#[inline]
pub(super) fn spray_passes(cell: (i32, i32, i32), spray: f32) -> bool {
    if spray <= 0.0 {
        return true;
    }
    let h = cell.0.wrapping_mul(73856093)
        ^ cell.1.wrapping_mul(19349663)
        ^ cell.2.wrapping_mul(83492791);
    let u = (h as u32 as f64 / u32::MAX as f64) as f32;
    u < spray.clamp(0.0, 1.0)
}

/// Deterministic scatter offset for a spray stamp center (web `expandPathWithBrushStamps` scatter).
/// Returns a random offset in `[-scatter, scatter]` for the given axis (0/1/2).
#[inline]
pub(super) fn spray_scatter_offset(center: (i32, i32, i32), scatter: u32, axis: u32) -> i32 {
    if scatter == 0 {
        return 0;
    }
    let h = center
        .0
        .wrapping_mul(73856093_i32.wrapping_add(axis as i32 * 17))
        ^ center
            .1
            .wrapping_mul(19349663_i32.wrapping_add(axis as i32 * 31))
        ^ center
            .2
            .wrapping_mul(83492791_i32.wrapping_add(axis as i32 * 47));
    let u = h as u32 as f64 / u32::MAX as f64;
    ((u * 2.0 - 1.0) * scatter as f64).round() as i32
}

/// Deterministic random radius for a spray stamp (web `sprayRadiusRange`).
/// Returns a radius in `[min, max]`.
#[inline]
pub(super) fn spray_random_radius(center: (i32, i32, i32), min: u32, max: u32) -> u32 {
    if min >= max {
        return min;
    }
    let h = center.0.wrapping_mul(73856093_i32.wrapping_add(7 * 17))
        ^ center.1.wrapping_mul(19349663_i32.wrapping_add(7 * 31))
        ^ center.2.wrapping_mul(83492791_i32.wrapping_add(7 * 47));
    let u = h as u32 as f64 / u32::MAX as f64;
    min + (u * (max - min + 1) as f64).floor().min((max - min) as f64) as u32
}

/// Dominant world axis of an axis-aligned face normal (0=X, 1=Y, 2=Z).
pub fn face_normal_to_axis(n: (i32, i32, i32)) -> u8 {
    if n.0.abs() >= n.1.abs() && n.0.abs() >= n.2.abs() {
        0
    } else if n.1.abs() >= n.2.abs() {
        1
    } else {
        2
    }
}

/// Map 2D tangent-plane offsets `(du, dv)` to 3D given the locked (face-normal) axis.
fn expand_2d_to_3d(locked_axis: u8, du: i32, dv: i32) -> (i32, i32, i32) {
    match locked_axis {
        0 => (0, du, dv),
        1 => (du, 0, dv),
        _ => (du, dv, 0),
    }
}

/// Build a brush offset list where `size` is the diameter in voxels (1 = single cell).
///
/// Odd sizes use a voxel-centered sphere/cube; even sizes shift the center to (0.5, 0.5, 0.5)
/// between voxels so the cross-section is exactly `size` voxels wide on every axis.
pub fn brush_offset_cells_for_size(
    shape: BrushShape,
    size: u32,
    clip_half_normal: Option<(i32, i32, i32)>,
    face_normal_axis: Option<u8>,
) -> Vec<(i32, i32, i32)> {
    if size <= 1 {
        return if let Some(n) = clip_half_normal {
            let v = (0, 0, 0);
            if v.0 * n.0 + v.1 * n.1 + v.2 * n.2 >= 0 {
                vec![v]
            } else {
                vec![]
            }
        } else {
            vec![(0, 0, 0)]
        };
    }
    let even = size.is_multiple_of(2);
    let half = size as i32 / 2;
    let (lo, hi) = if even {
        (-(half - 1), half)
    } else {
        (-half, half)
    };
    // For even sizes the sphere is centered between voxels; for odd it is on a voxel.
    let c = if even { 0.5_f32 } else { 0.0_f32 };
    let r2 = (size as f32 / 2.0).powi(2);
    let axis = face_normal_axis.unwrap_or(1);
    let mut out = Vec::new();
    match shape {
        BrushShape::Cube => {
            for dx in lo..=hi {
                for dy in lo..=hi {
                    for dz in lo..=hi {
                        out.push((dx, dy, dz));
                    }
                }
            }
        }
        BrushShape::Sphere => {
            for dx in lo..=hi {
                for dy in lo..=hi {
                    for dz in lo..=hi {
                        let fx = dx as f32 - c;
                        let fy = dy as f32 - c;
                        let fz = dz as f32 - c;
                        if fx * fx + fy * fy + fz * fz <= r2 + 1e-4 {
                            out.push((dx, dy, dz));
                        }
                    }
                }
            }
        }
        BrushShape::Pyramid => {
            // Octahedron: L1 norm <= half-size. Even sizes use fractional center.
            let thresh = size as f32 / 2.0;
            for dx in lo..=hi {
                for dy in lo..=hi {
                    for dz in lo..=hi {
                        let fx = (dx as f32 - c).abs();
                        let fy = (dy as f32 - c).abs();
                        let fz = (dz as f32 - c).abs();
                        if fx + fy + fz <= thresh + 1e-4 {
                            out.push((dx, dy, dz));
                        }
                    }
                }
            }
        }
        BrushShape::Square => {
            for du in lo..=hi {
                for dv in lo..=hi {
                    out.push(expand_2d_to_3d(axis, du, dv));
                }
            }
        }
        BrushShape::Circle => {
            let r2_2d = (size as f32 / 2.0).powi(2);
            for du in lo..=hi {
                for dv in lo..=hi {
                    let fu = du as f32 - c;
                    let fv = dv as f32 - c;
                    if fu * fu + fv * fv <= r2_2d + 1e-4 {
                        out.push(expand_2d_to_3d(axis, du, dv));
                    }
                }
            }
        }
    }
    if let Some(n) = clip_half_normal {
        out.retain(|o| o.0 * n.0 + o.1 * n.1 + o.2 * n.2 >= 0);
    }
    out.sort_by_key(|(a, b, c)| (a.abs() + b.abs() + c.abs(), *a, *b, *c));
    out
}

/// Brush offset list keyed by a display-size index (`radius` = display_value − 1, so 0 = 1-voxel).
/// Optional `clip_half_normal`: axis-aligned outward normal — keep offsets with `dx*nx+dy*ny+dz*nz >= 0`.
/// Optional `face_normal_axis`: for 2D shapes (Square/Circle), the world axis to lock (0=X, 1=Y, 2=Z).
pub fn brush_offset_cells(
    shape: BrushShape,
    radius: u32,
    clip_half_normal: Option<(i32, i32, i32)>,
    face_normal_axis: Option<u8>,
) -> Vec<(i32, i32, i32)> {
    brush_offset_cells_for_size(shape, radius + 1, clip_half_normal, face_normal_axis)
}

/// Voxel steps from brush center to the deepest part of the footprint toward the solid, along
/// `-outward_normal`. `outward_normal` is axis-aligned (from [`outward_face_normal_from_screen_ray`]).
pub(super) fn brush_footprint_extent_toward_solid(
    shape: BrushShape,
    radius: u32,
    outward_normal: (i32, i32, i32),
    clip_half_normal: Option<(i32, i32, i32)>,
) -> i32 {
    let offsets = brush_offset_cells(shape, radius, clip_half_normal, None);
    let mut min_dot = 0i32;
    for o in offsets {
        let d = o.0 * outward_normal.0 + o.1 * outward_normal.1 + o.2 * outward_normal.2;
        min_dot = min_dot.min(d);
    }
    -min_dot
}

/// Snap-to-surface uses the empty cell in front of the solid as the *contact* cell. Shift add
/// brush centers along the face outward normal so the footprint sits on that plane instead of
/// straddling it (orb half-embedded).
pub(super) fn adjust_add_centers_for_surface_snap_brush(
    centers: Vec<VoxelCoord>,
    tool: super::EditTool,
    brush_shape: BrushShape,
    brush_radius: u32,
    stroke_aux: &StrokeAux,
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> Vec<VoxelCoord> {
    if !matches!(tool, super::EditTool::Add)
        || !stroke_aux.stroke_snap_to_surface
        || brush_radius == 0
    {
        return centers;
    }
    let Some(n) = super::raycasting::outward_face_normal_from_screen_ray(
        file, voxel_map, camera, width, height, sx, sy,
    ) else {
        return centers;
    };
    let clip_half = if stroke_aux.brush_clip_bottom_half {
        Some(n)
    } else {
        None
    };
    let ext = brush_footprint_extent_toward_solid(brush_shape, brush_radius, n, clip_half);
    if ext == 0 {
        return centers;
    }
    centers
        .into_iter()
        .map(|c| (c.0 + n.0 * ext, c.1 + n.1 * ext, c.2 + n.2 * ext))
        .collect()
}

/// When `clip_half_normal` is `Some(n)` (axis-aligned), keep only offsets with `o·n >= 0` (outward from the hit face).
pub(super) fn brush_clip_half_normal_from_screen(
    clip: bool,
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> Option<(i32, i32, i32)> {
    if !clip {
        return None;
    }
    Some(
        super::raycasting::outward_face_normal_from_screen_ray(
            file, voxel_map, camera, width, height, sx, sy,
        )
        .unwrap_or((0, 1, 0)),
    )
}

pub fn snap_normal_to_axis(n: (i32, i32, i32)) -> (i32, i32, i32) {
    let ax = n.0.abs();
    let ay = n.1.abs();
    let az = n.2.abs();
    if ax >= ay && ax >= az {
        return (if n.0 >= 0 { 1 } else { -1 }, 0, 0);
    }
    if ay >= az {
        return (0, if n.1 >= 0 { 1 } else { -1 }, 0);
    }
    (0, 0, if n.2 >= 0 { 1 } else { -1 })
}

/// Web `SprayDirection` for wall extrusion.
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SprayDirection {
    #[default]
    Auto,
    None,
    Right,
    Left,
    Up,
    Down,
    Back,
    Forward,
}

pub fn spray_direction_vector(
    dir: SprayDirection,
    face_normal: Option<(i32, i32, i32)>,
) -> Option<(i32, i32, i32)> {
    match dir {
        SprayDirection::Auto => face_normal.map(snap_normal_to_axis),
        SprayDirection::None => None,
        SprayDirection::Down => Some((0, -1, 0)),
        SprayDirection::Up => Some((0, 1, 0)),
        SprayDirection::Forward => Some((0, 0, -1)),
        SprayDirection::Back => Some((0, 0, 1)),
        SprayDirection::Left => Some((-1, 0, 0)),
        SprayDirection::Right => Some((1, 0, 0)),
    }
}

pub(super) fn wall_lock_axis(
    dir: SprayDirection,
    face_n: Option<(i32, i32, i32)>,
) -> Option<usize> {
    match dir {
        SprayDirection::Auto => {
            let d = spray_direction_vector(SprayDirection::Auto, face_n)?;
            if d.0 != 0 {
                Some(0)
            } else if d.1 != 0 {
                Some(1)
            } else {
                Some(2)
            }
        }
        SprayDirection::Left | SprayDirection::Right => Some(0),
        SprayDirection::Down | SprayDirection::Up => Some(1),
        SprayDirection::Forward | SprayDirection::Back => Some(2),
        SprayDirection::None => None,
    }
}

pub(super) fn perpendicular_step_thick(dir: (i32, i32, i32)) -> (i32, i32, i32) {
    if dir.0 != 0 {
        (0, 1, 0)
    } else if dir.1 != 0 {
        (1, 0, 0)
    } else {
        (0, 1, 0)
    }
}

pub(super) fn thicken_path_in_plane_wall(
    positions: &[(i32, i32, i32)],
    radius: f32,
    plane_normal_axis: usize,
) -> Vec<(i32, i32, i32)> {
    if radius <= 0.0 {
        return positions.to_vec();
    }
    let lo = -radius.ceil() as i32;
    let hi = radius.floor() as i32;
    let mut seen: HashSet<(i32, i32, i32)> = positions.iter().copied().collect();
    let mut result: Vec<(i32, i32, i32)> = positions.to_vec();
    for &(px, py, pz) in positions {
        match plane_normal_axis {
            0 => {
                for dy in lo..=hi {
                    for dz in lo..=hi {
                        let p = (px, py + dy, pz + dz);
                        if seen.insert(p) {
                            result.push(p);
                        }
                    }
                }
            }
            1 => {
                for dx in lo..=hi {
                    for dz in lo..=hi {
                        let p = (px + dx, py, pz + dz);
                        if seen.insert(p) {
                            result.push(p);
                        }
                    }
                }
            }
            _ => {
                for dx in lo..=hi {
                    for dy in lo..=hi {
                        let p = (px + dx, py + dy, pz);
                        if seen.insert(p) {
                            result.push(p);
                        }
                    }
                }
            }
        }
    }
    result
}

pub(super) fn directional_streak_wall(
    base: &[(i32, i32, i32)],
    direction: (i32, i32, i32),
    streak_len: i32,
) -> Vec<(i32, i32, i32)> {
    let len = streak_len.max(0);
    if len == 0 {
        return base.to_vec();
    }
    let (dx, dy, dz) = direction;
    let mut seen: HashSet<(i32, i32, i32)> = base.iter().copied().collect();
    let mut result = base.to_vec();
    for &(px, py, pz) in base {
        for k in 1..=len {
            let p = (px + k * dx, py + k * dy, pz + k * dz);
            if seen.insert(p) {
                result.push(p);
            }
        }
    }
    result
}

// ── Sculpt stroke anchor helpers ─────────────────────────────────────────────

pub(super) fn stroke_anchor_centers_sculpt(
    mode: super::SculptStrokeMode,
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    stroke_line_start: Option<(f32, f32)>,
    stroke_segment_prev: Option<(f32, f32)>,
) -> Vec<(i32, i32, i32)> {
    let tool = sculpt_edit_tool(mode);
    stroke_anchor_centers_with_mode(
        DrawStrokeMode::Line,
        PlaneAxis::Auto,
        &StrokeAux::default(),
        tool,
        file,
        voxel_map,
        camera,
        width,
        height,
        sx,
        sy,
        0,
        stroke_line_start,
        stroke_segment_prev,
        None,
    )
}

/// Edit tool for the spine anchor placement during sculpt. Draw uses `Remove` so the
/// spine tracks the solid surface (not the empty cell in front), preventing frame-by-frame
/// stacking along the view ray during replay. The brush offsets still expand into empty space.
#[inline]
pub(super) fn sculpt_edit_tool(mode: super::SculptStrokeMode) -> super::EditTool {
    match mode {
        super::SculptStrokeMode::Draw => super::EditTool::Remove,
        super::SculptStrokeMode::Extrude
        | super::SculptStrokeMode::Wall
        | super::SculptStrokeMode::Terrain => super::EditTool::Add,
        super::SculptStrokeMode::Smooth | super::SculptStrokeMode::Gouge => super::EditTool::Remove,
    }
}

/// Mulberry32 — matches web `createSeededRng` (`strokeGeometry.ts`).
fn mulberry32_next(state: &mut u32) -> f32 {
    *state = state.wrapping_add(0x6d2b79f5);
    let mut t = *state;
    t = (t ^ (t >> 15)).wrapping_mul(t | 1);
    t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
    ((t ^ (t >> 14)) as f32) / 4294967296.0
}

fn dist_sq_point_segment(
    px: f32,
    py: f32,
    pz: f32,
    ax: f32,
    ay: f32,
    az: f32,
    bx: f32,
    by: f32,
    bz: f32,
) -> f32 {
    let abx = bx - ax;
    let aby = by - ay;
    let abz = bz - az;
    let apx = px - ax;
    let apy = py - ay;
    let apz = pz - az;
    let ab_len_sq = abx * abx + aby * aby + abz * abz;
    if ab_len_sq < 1e-12 {
        let dx = px - ax;
        let dy = py - ay;
        let dz = pz - az;
        return dx * dx + dy * dy + dz * dz;
    }
    let mut t = (apx * abx + apy * aby + apz * abz) / ab_len_sq;
    t = t.clamp(0.0, 1.0);
    let qx = ax + t * abx;
    let qy = ay + t * aby;
    let qz = az + t * abz;
    let dx = px - qx;
    let dy = py - qy;
    let dz = pz - qz;
    dx * dx + dy * dy + dz * dz
}

fn min_dist_point_to_polyline(px: f32, py: f32, pz: f32, spine: &[(i32, i32, i32)]) -> f32 {
    if spine.is_empty() {
        return 0.0;
    }
    if spine.len() == 1 {
        let sx = spine[0].0 as f32 + 0.5;
        let sy = spine[0].1 as f32 + 0.5;
        let sz = spine[0].2 as f32 + 0.5;
        let dx = px - sx;
        let dy = py - sy;
        let dz = pz - sz;
        return (dx * dx + dy * dy + dz * dz).sqrt();
    }
    let mut min_d = f32::INFINITY;
    for i in 0..spine.len() - 1 {
        let ax = spine[i].0 as f32 + 0.5;
        let ay = spine[i].1 as f32 + 0.5;
        let az = spine[i].2 as f32 + 0.5;
        let bx = spine[i + 1].0 as f32 + 0.5;
        let by = spine[i + 1].1 as f32 + 0.5;
        let bz = spine[i + 1].2 as f32 + 0.5;
        let d = dist_sq_point_segment(px, py, pz, ax, ay, az, bx, by, bz).sqrt();
        if d < min_d {
            min_d = d;
        }
    }
    min_d
}

/// Web `computeSculptVoxelWeights` + `filterPositionsBySculptBrush` (sculptBrushWeights.ts).
pub(super) fn filter_sculpt_footprint_stochastic(
    footprint: Vec<VoxelCoord>,
    spine: &[(i32, i32, i32)],
    brush_radius: u32,
    falloff_100: u32,
    strength_100: u32,
    stroke_seed: u32,
) -> Vec<VoxelCoord> {
    let fall = (falloff_100.min(100) as f32) / 100.0;
    let str = (strength_100.clamp(1, 100) as f32) / 100.0;
    if fall <= 1e-9 && str >= 1.0 - 1e-9 {
        return footprint;
    }

    let r_vox = ((brush_radius + 1) as f32 / 2.0).max(1e-6);

    let mut spine_eff: Vec<(i32, i32, i32)> = spine.to_vec();
    if spine_eff.is_empty() && !footprint.is_empty() {
        spine_eff.push(footprint[0]);
    }

    let mut rng_state = stroke_seed;
    let mut out: Vec<VoxelCoord> = Vec::with_capacity(footprint.len());
    let mut seen: HashSet<(i32, i32, i32)> = HashSet::new();

    for (x, y, z) in footprint {
        if !seen.insert((x, y, z)) {
            continue;
        }
        let cx = x as f32 + 0.5;
        let cy = y as f32 + 0.5;
        let cz = z as f32 + 0.5;

        let mut w = 1.0f32;
        if fall > 1e-9 && !spine_eff.is_empty() {
            let d = min_dist_point_to_polyline(cx, cy, cz, &spine_eff);
            let t = (d / r_vox).min(1.0);
            let soft = (1.0 - t) * (1.0 - t);
            w = (1.0 - fall) + fall * soft;
            w = w.clamp(0.0, 1.0);
        }

        let p = w * str;
        if p >= 1.0 - 1e-9 {
            out.push((x, y, z));
            continue;
        }
        if p <= 1e-9 {
            continue;
        }
        let u = mulberry32_next(&mut rng_state);
        if u < p {
            out.push((x, y, z));
        }
    }
    out
}

// ── Extrude cylinder / capsule / taper geometry (web branch parity) ───────────

const BRANCH_R2_EPS: f32 = 1e-8;

fn normalize3_opt(v: [f32; 3]) -> Option<[f32; 3]> {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-9 {
        None
    } else {
        Some([v[0] / len, v[1] / len, v[2] / len])
    }
}

fn extrude_tangent_at(positions: &[VoxelCoord], i: usize) -> Option<[f32; 3]> {
    let n = positions.len();
    if n == 1 {
        return Some([0.0, 0.0, 1.0]);
    }
    if i == 0 {
        let (ax, ay, az) = positions[0];
        let (bx, by, bz) = positions[1];
        return normalize3_opt([(bx - ax) as f32, (by - ay) as f32, (bz - az) as f32]);
    }
    if i >= n - 1 {
        let (ax, ay, az) = positions[n - 2];
        let (bx, by, bz) = positions[n - 1];
        return normalize3_opt([(bx - ax) as f32, (by - ay) as f32, (bz - az) as f32]);
    }
    let (ax, ay, az) = positions[i - 1];
    let (bx, by, bz) = positions[i + 1];
    normalize3_opt([(bx - ax) as f32, (by - ay) as f32, (bz - az) as f32])
}

/// Flat-capped cylinder between two points: voxels within radius of the segment axis, clamped to [0, L].
fn add_flat_cylinder_segment(
    seen: &mut HashSet<VoxelCoord>,
    out: &mut Vec<VoxelCoord>,
    a: VoxelCoord,
    b: VoxelCoord,
    r: f32,
) {
    let (ax, ay, az) = (a.0 as f32, a.1 as f32, a.2 as f32);
    let (bx, by, bz) = (b.0 as f32, b.1 as f32, b.2 as f32);
    let abx = bx - ax;
    let aby = by - ay;
    let abz = bz - az;
    let len = (abx * abx + aby * aby + abz * abz).sqrt();
    if len < 1e-9 {
        return;
    }
    let tx = abx / len;
    let ty = aby / len;
    let tz = abz / len;
    let r2 = r * r + BRANCH_R2_EPS;
    let pad = r.ceil() as i32 + 2;
    let min_x = a.0.min(b.0) - pad;
    let max_x = a.0.max(b.0) + pad;
    let min_y = a.1.min(b.1) - pad;
    let max_y = a.1.max(b.1) + pad;
    let min_z = a.2.min(b.2) - pad;
    let max_z = a.2.max(b.2) + pad;
    for x in min_x..=max_x {
        for y in min_y..=max_y {
            for z in min_z..=max_z {
                let qx = x as f32 - ax;
                let qy = y as f32 - ay;
                let qz = z as f32 - az;
                let axial = qx * tx + qy * ty + qz * tz;
                if axial < 0.0 || axial > len {
                    continue;
                }
                let wx = qx - tx * axial;
                let wy = qy - ty * axial;
                let wz = qz - tz * axial;
                let perp2 = wx * wx + wy * wy + wz * wz;
                if perp2 <= r2 && seen.insert((x, y, z)) {
                    out.push((x, y, z));
                }
            }
        }
    }
}

/// Capsule between two points: voxels within radius of the closest point on the segment (rounded ends).
fn add_capsule_segment(
    seen: &mut HashSet<VoxelCoord>,
    out: &mut Vec<VoxelCoord>,
    a: VoxelCoord,
    b: VoxelCoord,
    r: f32,
) {
    let (ax, ay, az) = (a.0 as f32, a.1 as f32, a.2 as f32);
    let (bx, by, bz) = (b.0 as f32, b.1 as f32, b.2 as f32);
    let abx = bx - ax;
    let aby = by - ay;
    let abz = bz - az;
    let ab2 = abx * abx + aby * aby + abz * abz;
    if ab2 < 1e-18 {
        return;
    }
    let r2 = r * r + BRANCH_R2_EPS;
    let pad = r.ceil() as i32 + 2;
    let min_x = a.0.min(b.0) - pad;
    let max_x = a.0.max(b.0) + pad;
    let min_y = a.1.min(b.1) - pad;
    let max_y = a.1.max(b.1) + pad;
    let min_z = a.2.min(b.2) - pad;
    let max_z = a.2.max(b.2) + pad;
    for x in min_x..=max_x {
        for y in min_y..=max_y {
            for z in min_z..=max_z {
                let qx = x as f32 - ax;
                let qy = y as f32 - ay;
                let qz = z as f32 - az;
                let mut t = (qx * abx + qy * aby + qz * abz) / ab2;
                t = t.clamp(0.0, 1.0);
                let px = ax + t * abx;
                let py = ay + t * aby;
                let pz = az + t * abz;
                let dx = x as f32 - px;
                let dy = y as f32 - py;
                let dz = z as f32 - pz;
                if dx * dx + dy * dy + dz * dz <= r2 && seen.insert((x, y, z)) {
                    out.push((x, y, z));
                }
            }
        }
    }
}

/// Disk slab: single-voxel-thick disk perpendicular to tangent direction at center.
fn add_disk_slab(
    seen: &mut HashSet<VoxelCoord>,
    out: &mut Vec<VoxelCoord>,
    center: VoxelCoord,
    tangent: [f32; 3],
    r: f32,
) {
    if r <= 0.0 {
        if seen.insert(center) {
            out.push(center);
        }
        return;
    }
    let [tx, ty, tz] = tangent;
    let r2 = r * r + BRANCH_R2_EPS;
    let pad = r.ceil() as i32 + 2;
    let (cx, cy, cz) = center;
    for x in (cx - pad)..=(cx + pad) {
        for y in (cy - pad)..=(cy + pad) {
            for z in (cz - pad)..=(cz + pad) {
                let wx = (x - cx) as f32;
                let wy = (y - cy) as f32;
                let wz = (z - cz) as f32;
                let axial = wx * tx + wy * ty + wz * tz;
                if axial.abs() > 0.5001 {
                    continue;
                }
                let px = wx - tx * axial;
                let py = wy - ty * axial;
                let pz = wz - tz * axial;
                if px * px + py * py + pz * pz <= r2 && seen.insert((x, y, z)) {
                    out.push((x, y, z));
                }
            }
        }
    }
}

/// Hemisphere cap at a cylinder endpoint.
fn add_sphere_cap(
    seen: &mut HashSet<VoxelCoord>,
    out: &mut Vec<VoxelCoord>,
    center: VoxelCoord,
    r: f32,
    tangent: [f32; 3],
    outward_dot_positive: bool,
) {
    if r <= 0.0 {
        return;
    }
    let [tx, ty, tz] = tangent;
    let r2 = r * r + BRANCH_R2_EPS;
    let pad = r.ceil() as i32 + 2;
    let (cx, cy, cz) = center;
    for x in (cx - pad)..=(cx + pad) {
        for y in (cy - pad)..=(cy + pad) {
            for z in (cz - pad)..=(cz + pad) {
                let vx = (x - cx) as f32;
                let vy = (y - cy) as f32;
                let vz = (z - cz) as f32;
                let d2 = vx * vx + vy * vy + vz * vz;
                if d2 > r2 {
                    continue;
                }
                let dot = vx * tx + vy * ty + vz * tz;
                if outward_dot_positive {
                    if dot < -BRANCH_R2_EPS {
                        continue;
                    }
                } else if dot > BRANCH_R2_EPS {
                    continue;
                }
                if seen.insert((x, y, z)) {
                    out.push((x, y, z));
                }
            }
        }
    }
}

/// Pointed cone cap: tapered disk slabs extending from the endpoint along tangent.
fn add_pointed_cone_cap(
    seen: &mut HashSet<VoxelCoord>,
    out: &mut Vec<VoxelCoord>,
    origin: VoxelCoord,
    dir: [f32; 3],
    base_radius: f32,
) {
    if base_radius <= 0.0 {
        return;
    }
    let Some(t) = normalize3_opt(dir) else {
        return;
    };
    let k_max = base_radius.ceil().max(1.0) as i32;
    for k in 1..=k_max {
        let rk = base_radius * (1.0 - k as f32 / (k_max as f32 + 1.0));
        if rk <= 0.0 {
            continue;
        }
        let cx = origin.0 + (k as f32 * t[0]).round() as i32;
        let cy = origin.1 + (k as f32 * t[1]).round() as i32;
        let cz = origin.2 + (k as f32 * t[2]).round() as i32;
        add_disk_slab(seen, out, (cx, cy, cz), t, rk);
    }
}

/// Quantize continuous taper radius to discrete voxel sizes (web `taperRadiusToSize`).
fn taper_radius_to_size(c: f32) -> f32 {
    if c <= 0.0 || c < 0.25 {
        return 0.0;
    }
    if c < 0.75 {
        return 0.5;
    }
    if c < 1.25 {
        return 1.0;
    }
    if c < 1.75 {
        return 1.5;
    }
    if c <= 2.0 {
        return 2.0;
    }
    c
}

/// Compute extrude cylinder footprint from spine positions (web `thickenBranchUniformCylinder`).
pub(super) fn extrude_uniform_cylinder_footprint(
    spine: &[VoxelCoord],
    r: f32,
    cap: ExtrudeEndCap,
) -> Vec<VoxelCoord> {
    if spine.is_empty() {
        return Vec::new();
    }
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let n = spine.len();

    if n == 1 {
        // Single point: sphere + optional cones
        let c = spine[0];
        let ri = r.ceil() as i32;
        let r2 = r * r + BRANCH_R2_EPS;
        for dx in -ri..=ri {
            for dy in -ri..=ri {
                for dz in -ri..=ri {
                    if (dx * dx + dy * dy + dz * dz) as f32 <= r2 {
                        let p = (c.0 + dx, c.1 + dy, c.2 + dz);
                        if seen.insert(p) {
                            out.push(p);
                        }
                    }
                }
            }
        }
        if cap == ExtrudeEndCap::Pointed {
            add_pointed_cone_cap(&mut seen, &mut out, c, [0.0, 1.0, 0.0], r);
            add_pointed_cone_cap(&mut seen, &mut out, c, [0.0, -1.0, 0.0], r);
        }
        return out;
    }

    let use_capsule = cap == ExtrudeEndCap::Rounded;
    for i in 0..n - 1 {
        if use_capsule {
            add_capsule_segment(&mut seen, &mut out, spine[i], spine[i + 1], r);
        } else {
            add_flat_cylinder_segment(&mut seen, &mut out, spine[i], spine[i + 1], r);
        }
    }

    if cap == ExtrudeEndCap::Pointed {
        if let Some(t0) = extrude_tangent_at(spine, 0) {
            add_pointed_cone_cap(&mut seen, &mut out, spine[0], [-t0[0], -t0[1], -t0[2]], r);
        }
        if let Some(t1) = extrude_tangent_at(spine, n - 1) {
            add_pointed_cone_cap(&mut seen, &mut out, spine[n - 1], t1, r);
        }
    }

    out
}

/// Compute extrude tapered cylinder footprint (web `thickenBranchTaperedCylinder`).
pub(super) fn extrude_tapered_cylinder_footprint(
    spine: &[VoxelCoord],
    base_radius: f32,
    tip_radius: f32,
    cap: ExtrudeEndCap,
) -> Vec<VoxelCoord> {
    if spine.is_empty() {
        return Vec::new();
    }
    if base_radius <= 0.0 && tip_radius <= 0.0 {
        return spine.to_vec();
    }
    let n = spine.len();

    // Compute per-station radii
    let radii: Vec<f32> = (0..n)
        .map(|i| {
            let t = if n == 1 {
                0.0
            } else {
                i as f32 / (n as f32 - 1.0)
            };
            taper_radius_to_size((base_radius + t * (tip_radius - base_radius)).max(0.0))
        })
        .collect();

    if n == 1 {
        let r0 = radii[0];
        if r0 <= 0.0 {
            return vec![spine[0]];
        }
        return extrude_uniform_cylinder_footprint(spine, r0, cap);
    }

    let mut seen = HashSet::new();
    let mut out = Vec::new();

    // Disk slabs at each station with tapered radius
    for i in 0..n {
        let ri = radii[i];
        let p = spine[i];
        if ri <= 0.0 {
            if seen.insert(p) {
                out.push(p);
            }
            continue;
        }
        if let Some(t) = extrude_tangent_at(spine, i) {
            add_disk_slab(&mut seen, &mut out, p, t, ri);
        }
    }

    // Rounded end caps
    if cap == ExtrudeEndCap::Rounded {
        if let Some(t0) = extrude_tangent_at(spine, 0) {
            if radii[0] > 0.0 {
                add_sphere_cap(&mut seen, &mut out, spine[0], radii[0], t0, false);
            }
        }
        if let Some(t1) = extrude_tangent_at(spine, n - 1) {
            if radii[n - 1] > 0.0 {
                add_sphere_cap(&mut seen, &mut out, spine[n - 1], radii[n - 1], t1, true);
            }
        }
    }

    // Pointed cone caps
    if cap == ExtrudeEndCap::Pointed {
        if let Some(t0) = extrude_tangent_at(spine, 0) {
            if radii[0] > 0.0 {
                add_pointed_cone_cap(
                    &mut seen,
                    &mut out,
                    spine[0],
                    [-t0[0], -t0[1], -t0[2]],
                    radii[0],
                );
            }
        }
        if let Some(t1) = extrude_tangent_at(spine, n - 1) {
            if radii[n - 1] > 0.0 {
                add_pointed_cone_cap(&mut seen, &mut out, spine[n - 1], t1, radii[n - 1]);
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── face_normal_to_axis ────────────────────────────────────────────

    #[test]
    fn face_normal_to_axis_x_dominant() {
        assert_eq!(face_normal_to_axis((1, 0, 0)), 0);
        assert_eq!(face_normal_to_axis((-3, 0, 0)), 0);
    }

    #[test]
    fn face_normal_to_axis_y_dominant() {
        assert_eq!(face_normal_to_axis((0, 1, 0)), 1);
        assert_eq!(face_normal_to_axis((0, -5, 0)), 1);
    }

    #[test]
    fn face_normal_to_axis_z_dominant() {
        assert_eq!(face_normal_to_axis((0, 0, 1)), 2);
        assert_eq!(face_normal_to_axis((0, 0, -2)), 2);
    }

    #[test]
    fn face_normal_to_axis_mixed_picks_largest() {
        assert_eq!(face_normal_to_axis((3, 1, 2)), 0); // X largest
        assert_eq!(face_normal_to_axis((1, 5, 3)), 1); // Y largest
        assert_eq!(face_normal_to_axis((1, 2, 4)), 2); // Z largest
    }

    // ── snap_normal_to_axis ────────────────────────────────────────────

    #[test]
    fn snap_normal_positive_x() {
        assert_eq!(snap_normal_to_axis((1, 0, 0)), (1, 0, 0));
    }

    #[test]
    fn snap_normal_negative_x() {
        assert_eq!(snap_normal_to_axis((-3, 0, 0)), (-1, 0, 0));
    }

    #[test]
    fn snap_normal_y_dominant() {
        assert_eq!(snap_normal_to_axis((1, 5, 2)), (0, 1, 0));
        assert_eq!(snap_normal_to_axis((1, -5, 2)), (0, -1, 0));
    }

    #[test]
    fn snap_normal_z_dominant() {
        assert_eq!(snap_normal_to_axis((0, 1, 4)), (0, 0, 1));
    }

    // ── spray_passes ──────────────────────────────────────────────────

    #[test]
    fn spray_passes_zero_spray_always_true() {
        // spray <= 0.0 → early return true
        assert!(spray_passes((0, 0, 0), 0.0));
        assert!(spray_passes((100, -50, 200), 0.0));
    }

    #[test]
    fn spray_passes_is_deterministic() {
        let a = spray_passes((3, 7, -2), 0.5);
        let b = spray_passes((3, 7, -2), 0.5);
        assert_eq!(a, b);
    }

    #[test]
    fn spray_passes_different_cells_can_differ() {
        // With spray=0.5, some cells should pass and some shouldn't
        let mut pass_count = 0usize;
        let mut fail_count = 0usize;
        for i in 0..100_i32 {
            if spray_passes((i, i * 3, i * 7), 0.5) {
                pass_count += 1;
            } else {
                fail_count += 1;
            }
        }
        assert!(pass_count > 10, "too few passes: {pass_count}");
        assert!(fail_count > 10, "too few fails: {fail_count}");
    }

    // ── spray_scatter_offset ──────────────────────────────────────────

    #[test]
    fn spray_scatter_zero_scatter_returns_zero() {
        assert_eq!(spray_scatter_offset((0, 0, 0), 0, 0), 0);
        assert_eq!(spray_scatter_offset((5, 3, -1), 0, 2), 0);
    }

    #[test]
    fn spray_scatter_is_deterministic() {
        let a = spray_scatter_offset((1, 2, 3), 5, 0);
        let b = spray_scatter_offset((1, 2, 3), 5, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn spray_scatter_within_range() {
        let scatter = 4u32;
        for i in 0..50_i32 {
            let off = spray_scatter_offset((i, i + 1, i * 3), scatter, 1);
            assert!(
                off.abs() <= scatter as i32,
                "offset {off} out of range ±{scatter}"
            );
        }
    }

    // ── spray_random_radius ────────────────────────────────────────────

    #[test]
    fn spray_random_radius_equal_min_max() {
        assert_eq!(spray_random_radius((0, 0, 0), 5, 5), 5);
    }

    #[test]
    fn spray_random_radius_min_greater_than_max_returns_min() {
        assert_eq!(spray_random_radius((0, 0, 0), 7, 3), 7);
    }

    #[test]
    fn spray_random_radius_in_range() {
        let (min, max) = (2u32, 8u32);
        for i in 0..50_i32 {
            let r = spray_random_radius((i, i * 2, i * 3), min, max);
            assert!(r >= min && r <= max, "radius {r} out of [{min}, {max}]");
        }
    }

    #[test]
    fn spray_random_radius_is_deterministic() {
        assert_eq!(
            spray_random_radius((5, 10, 15), 0, 10),
            spray_random_radius((5, 10, 15), 0, 10)
        );
    }

    // ── brush_offset_cells_for_size ────────────────────────────────────

    #[test]
    fn brush_offset_cells_size_1_is_single_origin() {
        let cells = brush_offset_cells_for_size(BrushShape::Cube, 1, None, None);
        assert_eq!(cells, vec![(0, 0, 0)]);
    }

    #[test]
    fn brush_offset_cells_cube_size_3_has_27_cells() {
        let cells = brush_offset_cells_for_size(BrushShape::Cube, 3, None, None);
        assert_eq!(cells.len(), 27);
    }

    #[test]
    fn brush_offset_cells_cube_size_2_has_8_cells() {
        let cells = brush_offset_cells_for_size(BrushShape::Cube, 2, None, None);
        assert_eq!(cells.len(), 8);
    }

    #[test]
    fn brush_offset_cells_sphere_contains_origin() {
        let cells = brush_offset_cells_for_size(BrushShape::Sphere, 3, None, None);
        assert!(cells.contains(&(0, 0, 0)));
    }

    #[test]
    fn brush_offset_cells_sphere_smaller_than_cube() {
        let sphere = brush_offset_cells_for_size(BrushShape::Sphere, 5, None, None);
        let cube = brush_offset_cells_for_size(BrushShape::Cube, 5, None, None);
        assert!(sphere.len() < cube.len());
    }

    #[test]
    fn brush_offset_cells_clip_removes_negative_y() {
        // Clipping with normal (0,1,0) should keep only cells where dy >= 0
        let cells = brush_offset_cells_for_size(BrushShape::Cube, 3, Some((0, 1, 0)), None);
        for &(_x, y, _z) in &cells {
            assert!(y >= 0, "clip failed: found y={y}");
        }
        // Should have fewer cells than unclipped
        let unclipped = brush_offset_cells_for_size(BrushShape::Cube, 3, None, None);
        assert!(cells.len() < unclipped.len());
    }

    #[test]
    fn brush_offset_cells_wrapper_matches_size_plus_one() {
        let direct = brush_offset_cells_for_size(BrushShape::Sphere, 4, None, None);
        let via_wrapper = brush_offset_cells(BrushShape::Sphere, 3, None, None);
        assert_eq!(direct, via_wrapper);
    }
}
