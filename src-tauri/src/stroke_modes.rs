//! Draw stroke modes (parity with Voxelle web `StrokeMode` / `strokeGeometry.ts`).

use crate::greedy_mesh::VoxelCoord;
use crate::voxel_edit::{anchor_for_stroke_edit, voxel_line_dda, world_to_voxel, EditTool};
use crate::voxelle::VoxelleFile;
use ahash::AHashMap;
use glam::Vec3;
use std::collections::HashSet;

/// Matches web [`StrokeMode`](https://github.com/...) in `core.ts`.
#[derive(Clone, Copy, Debug, Default, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum DrawStrokeMode {
    #[default]
    Line,
    Plane,
    Circle,
    Precise,
    Cuboid,
    Cylinder,
    PolygonHull,
    Polygon,
    Fill,
    Spray,
}

#[derive(Clone, Copy, Debug, Default, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum PlaneAxis {
    #[default]
    Auto,
    X,
    Y,
    Z,
    /// View plane through seed (normal ≈ camera forward); web `constrainToPlaneRef === 'camera'`.
    #[serde(rename = "camera")]
    Camera,
}

/// Optional geometry from the UI for multi-point strokes (polygon, cuboid corners, etc.).
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StrokeAux {
    #[serde(default)]
    pub polygon_vertices: Vec<[i32; 3]>,
    #[serde(default)]
    pub circle_center: Option<[i32; 3]>,
    #[serde(default)]
    pub circle_edge: Option<[i32; 3]>,
    #[serde(default)]
    pub cuboid_min: Option<[i32; 3]>,
    #[serde(default)]
    pub cuboid_max: Option<[i32; 3]>,
    #[serde(default)]
    pub cylinder_a: Option<[i32; 3]>,
    #[serde(default)]
    pub cylinder_b: Option<[i32; 3]>,
    /// When true, plane stroke uses an annulus (outer brush radius, hollow center) instead of a filled disk.
    #[serde(default)]
    pub plane_hollow: bool,
    /// Solid cuboid: extrusion depth in voxel steps along the face normal (web `getAxisAlignedCuboid`).
    #[serde(default)]
    pub cuboid_depth: Option<i32>,
    /// Hollow shell thickness for cuboid/plane hollow (minimum 1). Web `clampPlaneCuboidHollowWallThickness`.
    #[serde(default)]
    pub cuboid_hollow_wall_thickness: Option<i32>,
    /// Solid cylinder: extrusion depth along face normal (web `getAxisAlignedCylinder`).
    #[serde(default)]
    pub cylinder_depth: Option<i32>,
    /// 0 = cylinder; 100 = cone; in-between = frustum (web `taperPct`).
    #[serde(default)]
    pub cylinder_taper_pct: Option<i32>,
    #[serde(default)]
    pub constrain_to_plane: bool,
    #[serde(default)]
    pub spray_size_range: bool,
    /// `"solid"` | `"stroke"` — web solid polygon uses projected plane fill (`getSolidPolygonBasePositions`).
    #[serde(default)]
    pub stroke_family_variant: Option<String>,
    /// When true (default), add anchors use surface-adjacent placement; when false, first empty cell along the ray.
    #[serde(default = "default_stroke_snap_to_surface")]
    pub stroke_snap_to_surface: bool,
    /// Line stroke: constrain endpoints so the segment is parallel to one world axis (dominant span).
    #[serde(default)]
    pub stroke_axis_align: bool,
    /// Sphere / cube / pyramid brush: keep only the half-space in the face **outward** direction (from ray hit).
    #[serde(default)]
    pub brush_clip_bottom_half: bool,
    /// Spray scatter: random offset of stamp centers (integer voxels, web `sprayScatter`).
    #[serde(default)]
    pub spray_scatter: u32,
    /// Spray radius min (used when `spray_size_range` is true).
    #[serde(default)]
    pub spray_radius_min: u32,
    /// Spray radius max (used when `spray_size_range` is true).
    #[serde(default)]
    pub spray_radius_max: u32,
    /// Separate brush shape for spray mode (overrides top-level `brush_shape` when present).
    #[serde(default)]
    pub spray_brush_shape: Option<crate::voxel_edit::BrushShape>,
    /// Plane reference for constrain-to-plane: `"auto"` | `"camera"` | `"x"` | `"y"` | `"z"`.
    /// Only meaningful when `constrain_to_plane` is true.
    #[serde(default)]
    pub constrain_to_plane_ref: Option<String>,
}

impl Default for StrokeAux {
    fn default() -> Self {
        Self {
            polygon_vertices: Vec::new(),
            circle_center: None,
            circle_edge: None,
            cuboid_min: None,
            cuboid_max: None,
            cylinder_a: None,
            cylinder_b: None,
            plane_hollow: false,
            cuboid_depth: None,
            cuboid_hollow_wall_thickness: None,
            cylinder_depth: None,
            cylinder_taper_pct: None,
            constrain_to_plane: false,
            spray_size_range: false,
            stroke_family_variant: None,
            stroke_snap_to_surface: true,
            stroke_axis_align: false,
            brush_clip_bottom_half: false,
            spray_scatter: 0,
            spray_radius_min: 0,
            spray_radius_max: 0,
            spray_brush_shape: None,
            constrain_to_plane_ref: None,
        }
    }
}

fn default_stroke_snap_to_surface() -> bool {
    true
}

/// Constrain `b` so the segment from `a` lies on a single axis (X, Y, or Z) through `a`.
fn axis_align_line_endpoints(a: VoxelCoord, b: VoxelCoord) -> (VoxelCoord, VoxelCoord) {
    let dx = (b.0 - a.0).abs();
    let dy = (b.1 - a.1).abs();
    let dz = (b.2 - a.2).abs();
    if dx >= dy && dx >= dz {
        (a, (b.0, a.1, a.2))
    } else if dy >= dz {
        (a, (a.0, b.1, a.2))
    } else {
        (a, (a.0, a.1, b.2))
    }
}

fn axis_from_plane_axis(pa: PlaneAxis, face_axis: Option<usize>) -> Option<usize> {
    match pa {
        // Camera uses the same axis-aligned plane as Auto (face from pick); view plane is fill-only.
        PlaneAxis::Auto | PlaneAxis::Camera => face_axis,
        PlaneAxis::X => Some(0),
        PlaneAxis::Y => Some(1),
        PlaneAxis::Z => Some(2),
    }
}

/// Face normal as axis index 0|1|2 from ray entry (air `prev` → solid `hit`).
fn face_normal_axis(prev: VoxelCoord, hit: VoxelCoord) -> Option<usize> {
    let dx = hit.0 - prev.0;
    let dy = hit.1 - prev.1;
    let dz = hit.2 - prev.2;
    let s = dx.abs() + dy.abs() + dz.abs();
    if s != 1 {
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

fn disk_in_axis_plane(center: VoxelCoord, plane_axis: usize, radius: i32) -> Vec<VoxelCoord> {
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
fn annulus_in_axis_plane(center: VoxelCoord, plane_axis: usize, outer_r: i32) -> Vec<VoxelCoord> {
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

fn point_in_polygon_2d(x: i32, y: i32, verts: &[(i32, i32)]) -> bool {
    let n = verts.len();
    if n < 3 {
        return false;
    }
    let mut c = false;
    for i in 0..n {
        let (x0, y0) = verts[i];
        let (x1, y1) = verts[(i + 1) % n];
        if (y0 > y) != (y1 > y) {
            let t = (y - y0) as f64 / ((y1 - y0) as f64).max(1e-9);
            let xi = x0 as f64 + t * (x1 - x0) as f64;
            if (x as f64) < xi {
                c = !c;
            }
        }
    }
    c
}

/// 2D polygon interior on the integer grid (ray-cast test per cell in bbox).
fn fill_polygon_2d(verts: &[(i32, i32)]) -> Vec<(i32, i32)> {
    if verts.len() < 3 {
        return verts.to_vec();
    }
    let minx = verts.iter().map(|p| p.0).min().unwrap();
    let maxx = verts.iter().map(|p| p.0).max().unwrap();
    let miny = verts.iter().map(|p| p.1).min().unwrap();
    let maxy = verts.iter().map(|p| p.1).max().unwrap();
    let mut out = Vec::new();
    for x in minx..=maxx {
        for y in miny..=maxy {
            if point_in_polygon_2d(x, y, verts) {
                out.push((x, y));
            }
        }
    }
    out
}

/// Web `COPLANAR_FILL_TOL` — plane distance threshold for integer voxel corners.
const COPLANAR_FILL_TOL: f64 = 0.08;

fn find_non_collinear_triple(vertices: &[[i32; 3]]) -> Option<(usize, usize, usize)> {
    let n = vertices.len();
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let a = vertices[i];
                let b = vertices[j];
                let c = vertices[k];
                let ab = (
                    b[0] - a[0],
                    b[1] - a[1],
                    b[2] - a[2],
                );
                let ac = (
                    c[0] - a[0],
                    c[1] - a[1],
                    c[2] - a[2],
                );
                let cx = ab.1 * ac.2 - ab.2 * ac.1;
                let cy = ab.2 * ac.0 - ab.0 * ac.2;
                let cz = ab.0 * ac.1 - ab.1 * ac.0;
                if (cx as i64) * (cx as i64) + (cy as i64) * (cy as i64) + (cz as i64) * (cz as i64)
                    >= 1
                {
                    return Some((i, j, k));
                }
            }
        }
    }
    None
}

fn are_coplanar(vertices: &[[i32; 3]], a: [i32; 3], b: [i32; 3], c: [i32; 3]) -> bool {
    let ab = (
        b[0] - a[0],
        b[1] - a[1],
        b[2] - a[2],
    );
    let ac = (
        c[0] - a[0],
        c[1] - a[1],
        c[2] - a[2],
    );
    let mut nx = (ab.1 * ac.2 - ab.2 * ac.1) as f64;
    let mut ny = (ab.2 * ac.0 - ab.0 * ac.2) as f64;
    let mut nz = (ab.0 * ac.1 - ab.1 * ac.0) as f64;
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len < 1e-9 {
        return true;
    }
    nx /= len;
    ny /= len;
    nz /= len;
    let d = -(nx * a[0] as f64 + ny * a[1] as f64 + nz * a[2] as f64);
    for p in vertices {
        let dist = (nx * p[0] as f64 + ny * p[1] as f64 + nz * p[2] as f64 + d).abs();
        if dist > COPLANAR_FILL_TOL {
            return false;
        }
    }
    true
}

/// Web `getCoplanarPolygonFillPositions` / triangle branch of `getPolygonVoxels`: arbitrary coplanar polygon.
fn fill_coplanar_polygon(vertices: &[[i32; 3]]) -> Option<Vec<VoxelCoord>> {
    if vertices.len() < 3 {
        return None;
    }
    let (i, j, k) = find_non_collinear_triple(vertices)?;
    let a = vertices[i];
    let b = vertices[j];
    let c = vertices[k];
    if !are_coplanar(vertices, a, b, c) {
        return None;
    }
    let ab = (
        b[0] - a[0],
        b[1] - a[1],
        b[2] - a[2],
    );
    let ac = (
        c[0] - a[0],
        c[1] - a[1],
        c[2] - a[2],
    );
    let mut nx = (ab.1 * ac.2 - ab.2 * ac.1) as f64;
    let mut ny = (ab.2 * ac.0 - ab.0 * ac.2) as f64;
    let mut nz = (ab.0 * ac.1 - ab.1 * ac.0) as f64;
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len < 1e-12 {
        return None;
    }
    nx /= len;
    ny /= len;
    nz /= len;
    let d = -(nx * a[0] as f64 + ny * a[1] as f64 + nz * a[2] as f64);

    let ax = nx.abs();
    let ay = ny.abs();
    let az = nz.abs();
    let drop_axis = if ax >= ay && ax >= az {
        0usize
    } else if ay >= az {
        1usize
    } else {
        2usize
    };
    let (u_axis, v_axis) = match drop_axis {
        0 => (1usize, 2usize),
        1 => (0usize, 2usize),
        _ => (0usize, 1usize),
    };
    let to_2d = |p: [i32; 3]| -> (i32, i32) { (p[u_axis], p[v_axis]) };

    let n = [nx, ny, nz];

    if vertices.len() == 3 {
        let a2 = to_2d(a);
        let b2 = to_2d(b);
        let c2 = to_2d(c);
        let v0x = (b2.0 - a2.0) as f64;
        let v0y = (b2.1 - a2.1) as f64;
        let v1x = (c2.0 - a2.0) as f64;
        let v1y = (c2.1 - a2.1) as f64;
        let denom = v0x * v1y - v0y * v1x;
        if denom.abs() < 1e-9 {
            return None;
        }
        let tri_tol = 1e-6_f64;
        let in_triangle = |pu: f64, pv: f64| {
            let px = pu - a2.0 as f64;
            let py = pv - a2.1 as f64;
            let s = (px * v1y - py * v1x) / denom;
            let t = (py * v0x - px * v0y) / denom;
            s >= -tri_tol && t >= -tri_tol && s + t <= 1.0 + tri_tol
        };
        let min_u = (a2.0.min(b2.0.min(c2.0))) as f64;
        let max_u = (a2.0.max(b2.0.max(c2.0))) as f64;
        let min_v = (a2.1.min(b2.1.min(c2.1))) as f64;
        let max_v = (a2.1.max(b2.1.max(c2.1))) as f64;
        let floor_u = min_u.floor() as i32;
        let ceil_u = max_u.ceil() as i32;
        let floor_v = min_v.floor() as i32;
        let ceil_v = max_v.ceil() as i32;
        let nd = n[drop_axis];
        if nd.abs() < 1e-9 {
            return None;
        }
        let mut out = Vec::new();
        let mut coord = [0i32; 3];
        for u in floor_u..=ceil_u {
            for v in floor_v..=ceil_v {
                let corners = [
                    (u as f64, v as f64),
                    ((u + 1) as f64, v as f64),
                    ((u + 1) as f64, (v + 1) as f64),
                    (u as f64, (v + 1) as f64),
                ];
                if !corners.iter().any(|&(pu, pv)| in_triangle(pu, pv)) {
                    continue;
                }
                let cu = u as f64 + 0.5;
                let cv = v as f64 + 0.5;
                let third = -(d + n[u_axis] * cu + n[v_axis] * cv) / nd;
                coord[u_axis] = u;
                coord[v_axis] = v;
                coord[drop_axis] = third.round() as i32;
                out.push((coord[0], coord[1], coord[2]));
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    } else {
        let polygon_2d: Vec<(i32, i32)> = vertices.iter().map(|p| to_2d(*p)).collect();
        let min_u = polygon_2d.iter().map(|p| p.0).min()? as f64;
        let max_u = polygon_2d.iter().map(|p| p.0).max()? as f64;
        let min_v = polygon_2d.iter().map(|p| p.1).min()? as f64;
        let max_v = polygon_2d.iter().map(|p| p.1).max()? as f64;
        let floor_u = min_u.floor() as i32;
        let ceil_u = max_u.ceil() as i32;
        let floor_v = min_v.floor() as i32;
        let ceil_v = max_v.ceil() as i32;
        let nd = n[drop_axis];
        if nd.abs() < 1e-9 {
            return None;
        }
        let verts_ref: &[(i32, i32)] = &polygon_2d;
        let mut out = Vec::new();
        let mut coord = [0i32; 3];
        for u in floor_u..=ceil_u {
            for v in floor_v..=ceil_v {
                let corners = [(u, v), (u + 1, v), (u + 1, v + 1), (u, v + 1)];
                if !corners
                    .iter()
                    .any(|&(cx, cy)| point_in_polygon_2d(cx, cy, verts_ref))
                {
                    continue;
                }
                let cu = u as f64 + 0.5;
                let cv = v as f64 + 0.5;
                let third = -(d + n[u_axis] * cu + n[v_axis] * cv) / nd;
                coord[u_axis] = u;
                coord[v_axis] = v;
                coord[drop_axis] = third.round() as i32;
                out.push((coord[0], coord[1], coord[2]));
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }
}

/// Convex hull in projected `(u,v)`, then fill — web `polygonHull` for coplanar non-axis-aligned rings.
fn fill_coplanar_hull(vertices: &[[i32; 3]]) -> Option<Vec<VoxelCoord>> {
    if vertices.len() < 3 {
        return None;
    }
    // Match web `getPolygonVoxels` / `getCoplanarPolygonFillPositions`: three corners use the
    // triangle/corner inclusion path, not `fill_polygon_2d` on a 2D hull loop.
    if vertices.len() == 3 {
        return fill_coplanar_polygon(vertices);
    }
    let (i, j, k) = find_non_collinear_triple(vertices)?;
    let a = vertices[i];
    let b = vertices[j];
    let c = vertices[k];
    if !are_coplanar(vertices, a, b, c) {
        return None;
    }
    let ab = (
        b[0] - a[0],
        b[1] - a[1],
        b[2] - a[2],
    );
    let ac = (
        c[0] - a[0],
        c[1] - a[1],
        c[2] - a[2],
    );
    let mut nx = (ab.1 * ac.2 - ab.2 * ac.1) as f64;
    let mut ny = (ab.2 * ac.0 - ab.0 * ac.2) as f64;
    let mut nz = (ab.0 * ac.1 - ab.1 * ac.0) as f64;
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len < 1e-12 {
        return None;
    }
    nx /= len;
    ny /= len;
    nz /= len;
    let d = -(nx * a[0] as f64 + ny * a[1] as f64 + nz * a[2] as f64);
    let n = [nx, ny, nz];

    let ax = nx.abs();
    let ay = ny.abs();
    let az = nz.abs();
    let drop_axis = if ax >= ay && ax >= az {
        0usize
    } else if ay >= az {
        1usize
    } else {
        2usize
    };
    let (u_axis, v_axis) = match drop_axis {
        0 => (1usize, 2usize),
        1 => (0usize, 2usize),
        _ => (0usize, 1usize),
    };
    let to_2d = |p: [i32; 3]| -> (i32, i32) { (p[u_axis], p[v_axis]) };

    let pts: Vec<(i32, i32)> = vertices.iter().map(|p| to_2d(*p)).collect();
    let hull = convex_hull_2d(pts);
    if hull.len() < 3 {
        return None;
    }
    let nd = n[drop_axis];
    if nd.abs() < 1e-9 {
        return None;
    }
    let filled_2d = fill_polygon_2d(&hull);
    let mut out = Vec::with_capacity(filled_2d.len());
    let mut coord = [0i32; 3];
    for (u, v) in filled_2d {
        let cu = u as f64 + 0.5;
        let cv = v as f64 + 0.5;
        let third = -(d + n[u_axis] * cu + n[v_axis] * cv) / nd;
        coord[u_axis] = u;
        coord[v_axis] = v;
        coord[drop_axis] = third.round() as i32;
        out.push((coord[0], coord[1], coord[2]));
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

// --- Web `getPolygonVoxels`: non-coplanar 4+ points → 3D convex hull, fill integer voxel centers inside.

const HULL_HALFSPACE_EPS: f64 = 1e-5;

#[inline]
fn v3_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn v3_cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn v3_dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn v3_len(a: [f64; 3]) -> f64 {
    v3_dot(a, a).sqrt()
}

#[inline]
fn v3_scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Supporting half-spaces `n · (p - a) <= 0` (inward) for each triangular face of the 3D convex hull.
/// Mirrors THREE `ConvexHull` + half-space containment used in web `getPolygonVoxels` for non-coplanar input.
fn convex_hull_3d_halfspaces(points: &[[f64; 3]]) -> Vec<([f64; 3], [f64; 3])> {
    let n = points.len();
    if n < 4 {
        return Vec::new();
    }
    let mut faces: Vec<([f64; 3], [f64; 3])> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let pi = points[i];
                let pj = points[j];
                let pk = points[k];
                let nvec = v3_cross(v3_sub(pj, pi), v3_sub(pk, pi));
                let l = v3_len(nvec);
                if l < 1e-12 {
                    continue;
                }
                let mut nu = v3_scale(nvec, 1.0 / l);
                let mut max_along = f64::NEG_INFINITY;
                let mut min_along = f64::INFINITY;
                for p in points {
                    let d = v3_dot(nu, v3_sub(*p, pi));
                    max_along = max_along.max(d);
                    min_along = min_along.min(d);
                }
                if max_along - min_along < 1e-9 {
                    // All input points lie in this plane — not a facet of a 3D hull.
                    continue;
                }
                let mut max_proj = f64::NEG_INFINITY;
                for p in points {
                    max_proj = max_proj.max(v3_dot(nu, v3_sub(*p, pi)));
                }
                if max_proj > HULL_HALFSPACE_EPS {
                    nu = v3_scale(nu, -1.0);
                    max_proj = f64::NEG_INFINITY;
                    for p in points {
                        max_proj = max_proj.max(v3_dot(nu, v3_sub(*p, pi)));
                    }
                }
                if max_proj > HULL_HALFSPACE_EPS {
                    continue;
                }
                faces.push((nu, pi));
            }
        }
    }
    faces
}

fn point_in_closed_halfspaces(q: [f64; 3], faces: &[([f64; 3], [f64; 3])]) -> bool {
    for &(n, a) in faces {
        if v3_dot(n, v3_sub(q, a)) > HULL_HALFSPACE_EPS {
            return false;
        }
    }
    true
}

/// Integer voxel centers inside the convex hull of `vertices` (web non-coplanar branch).
fn fill_non_coplanar_convex_hull_voxels(vertices: &[[i32; 3]]) -> Option<Vec<VoxelCoord>> {
    if vertices.len() < 4 {
        return None;
    }
    let pts: Vec<[f64; 3]> = vertices
        .iter()
        .map(|v| [v[0] as f64, v[1] as f64, v[2] as f64])
        .collect();
    let faces = convex_hull_3d_halfspaces(&pts);
    if faces.is_empty() {
        return None;
    }
    let mut min_x = i32::MAX;
    let mut max_x = i32::MIN;
    let mut min_y = i32::MAX;
    let mut max_y = i32::MIN;
    let mut min_z = i32::MAX;
    let mut max_z = i32::MIN;
    for v in vertices {
        min_x = min_x.min(v[0]);
        max_x = max_x.max(v[0]);
        min_y = min_y.min(v[1]);
        max_y = max_y.max(v[1]);
        min_z = min_z.min(v[2]);
        max_z = max_z.max(v[2]);
    }
    let mut out = Vec::new();
    for x in min_x..=max_x {
        for y in min_y..=max_y {
            for z in min_z..=max_z {
                let q = [x as f64, y as f64, z as f64];
                if point_in_closed_halfspaces(q, &faces) {
                    out.push((x, y, z));
                }
            }
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn fill_polygon_axis_aligned(vertices: &[[i32; 3]]) -> Vec<VoxelCoord> {
    if vertices.len() < 3 {
        return vertices.iter().map(|v| (v[0], v[1], v[2])).collect();
    }
    // Web `getPolygonVoxels`: exactly three vertices use coplanar triangle fill (barycentric +
    // per-voxel corner tests), not `fill_polygon_2d` on integer samples — otherwise the filled
    // region can disagree (wrong side of edges vs web).
    if vertices.len() == 3 {
        if let Some(v) = fill_coplanar_polygon(vertices) {
            return v;
        }
    }
    let xs: Vec<i32> = vertices.iter().map(|v| v[0]).collect();
    let ys: Vec<i32> = vertices.iter().map(|v| v[1]).collect();
    let zs: Vec<i32> = vertices.iter().map(|v| v[2]).collect();
    let axis_fill = if xs.iter().all(|&x| x == xs[0]) {
        let pts: Vec<(i32, i32)> = vertices.iter().map(|v| (v[1], v[2])).collect();
        fill_polygon_2d(&pts)
            .into_iter()
            .map(|(y, z)| (xs[0], y, z))
            .collect()
    } else if ys.iter().all(|&y| y == ys[0]) {
        let pts: Vec<(i32, i32)> = vertices.iter().map(|v| (v[0], v[2])).collect();
        fill_polygon_2d(&pts)
            .into_iter()
            .map(|(x, z)| (x, ys[0], z))
            .collect()
    } else if zs.iter().all(|&z| z == zs[0]) {
        let pts: Vec<(i32, i32)> = vertices.iter().map(|v| (v[0], v[1])).collect();
        fill_polygon_2d(&pts)
            .into_iter()
            .map(|(x, y)| (x, y, zs[0]))
            .collect()
    } else {
        Vec::new()
    };
    if !axis_fill.is_empty() {
        return axis_fill;
    }
    if let Some(coplanar) = fill_coplanar_polygon(vertices) {
        return coplanar;
    }
    if let Some(hull3) = fill_non_coplanar_convex_hull_voxels(vertices) {
        return hull3;
    }
    vertices.iter().map(|v| (v[0], v[1], v[2])).collect()
}

fn convex_hull_2d(mut pts: Vec<(i32, i32)>) -> Vec<(i32, i32)> {
    if pts.len() <= 3 {
        return pts;
    }
    pts.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let cross = |o: (i32, i32), a: (i32, i32), b: (i32, i32)| -> i64 {
        (a.0 as i64 - o.0 as i64) * (b.1 as i64 - o.1 as i64)
            - (a.1 as i64 - o.1 as i64) * (b.0 as i64 - o.0 as i64)
    };
    let mut lower: Vec<(i32, i32)> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<(i32, i32)> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn fill_polygon_hull_axis_aligned(vertices: &[[i32; 3]]) -> Vec<VoxelCoord> {
    if vertices.len() < 3 {
        return vertices.iter().map(|v| (v[0], v[1], v[2])).collect();
    }
    if vertices.len() == 3 {
        if let Some(v) = fill_coplanar_polygon(vertices) {
            return v;
        }
    }
    let xs: Vec<i32> = vertices.iter().map(|v| v[0]).collect();
    let ys: Vec<i32> = vertices.iter().map(|v| v[1]).collect();
    let zs: Vec<i32> = vertices.iter().map(|v| v[2]).collect();
    if xs.iter().all(|&x| x == xs[0]) {
        let pts: Vec<(i32, i32)> = vertices.iter().map(|v| (v[1], v[2])).collect();
        let hull = convex_hull_2d(pts);
        fill_polygon_2d(&hull)
            .into_iter()
            .map(|(y, z)| (xs[0], y, z))
            .collect()
    } else if ys.iter().all(|&y| y == ys[0]) {
        let pts: Vec<(i32, i32)> = vertices.iter().map(|v| (v[0], v[2])).collect();
        let hull = convex_hull_2d(pts);
        fill_polygon_2d(&hull)
            .into_iter()
            .map(|(x, z)| (x, ys[0], z))
            .collect()
    } else if zs.iter().all(|&z| z == zs[0]) {
        let pts: Vec<(i32, i32)> = vertices.iter().map(|v| (v[0], v[1])).collect();
        let hull = convex_hull_2d(pts);
        fill_polygon_2d(&hull)
            .into_iter()
            .map(|(x, y)| (x, y, zs[0]))
            .collect()
    } else if let Some(h) = fill_coplanar_hull(vertices) {
        h
    } else if let Some(h) = fill_non_coplanar_convex_hull_voxels(vertices) {
        h
    } else {
        vertices.iter().map(|v| (v[0], v[1], v[2])).collect()
    }
}

fn stroke_aux_is_solid_family(aux: &StrokeAux) -> bool {
    aux.stroke_family_variant.as_deref() == Some("solid")
}

/// Web `getSolidPolygonBasePositions`: corners projected onto plane through first vertex, orthogonal to `plane_axis` / auto-detected axis.
fn solid_polygon_fixed_plane(
    vertices: &[[i32; 3]],
    plane_axis: PlaneAxis,
) -> Option<(usize, i32)> {
    if vertices.is_empty() {
        return None;
    }
    let o = vertices[0];
    match plane_axis {
        PlaneAxis::Auto => {
            if vertices.iter().all(|v| v[0] == o[0]) {
                Some((0, o[0]))
            } else if vertices.iter().all(|v| v[1] == o[1]) {
                Some((1, o[1]))
            } else if vertices.iter().all(|v| v[2] == o[2]) {
                Some((2, o[2]))
            } else {
                None
            }
        }
        PlaneAxis::X => {
            if vertices.iter().all(|v| v[0] == o[0]) {
                Some((0, o[0]))
            } else {
                None
            }
        }
        PlaneAxis::Y => {
            if vertices.iter().all(|v| v[1] == o[1]) {
                Some((1, o[1]))
            } else {
                None
            }
        }
        PlaneAxis::Z => {
            if vertices.iter().all(|v| v[2] == o[2]) {
                Some((2, o[2]))
            } else {
                None
            }
        }
        PlaneAxis::Camera => None,
    }
}

fn project_vertices_to_plane_2d(vertices: &[[i32; 3]], fixed_axis: usize) -> Vec<(i32, i32)> {
    vertices
        .iter()
        .map(|v| match fixed_axis {
            0 => (v[1], v[2]),
            1 => (v[0], v[2]),
            _ => (v[0], v[1]),
        })
        .collect()
}

fn lift_plane_2d_to_voxels(
    fixed_axis: usize,
    fixed_coord: i32,
    cells: &[(i32, i32)],
) -> Vec<VoxelCoord> {
    cells
        .iter()
        .map(|&(a, b)| match fixed_axis {
            0 => (fixed_coord, a, b),
            1 => (a, fixed_coord, b),
            _ => (a, b, fixed_coord),
        })
        .collect()
}

/// Solid + polygon: filled **simple** polygon in the work plane (web `getSolidPolygonBasePositions`).
fn fill_solid_polygon_simple_projected(
    vertices: &[[i32; 3]],
    plane_axis: PlaneAxis,
) -> Option<Vec<VoxelCoord>> {
    let (fixed_axis, fixed_coord) = solid_polygon_fixed_plane(vertices, plane_axis)?;
    let poly2d = project_vertices_to_plane_2d(vertices, fixed_axis);
    let filled = fill_polygon_2d(&poly2d);
    Some(lift_plane_2d_to_voxels(fixed_axis, fixed_coord, &filled))
}

/// Solid + polygon **hull**: convex hull of projected corners, then fill (web Surface polygonHull, in work plane).
fn fill_solid_polygon_hull_projected(
    vertices: &[[i32; 3]],
    plane_axis: PlaneAxis,
) -> Option<Vec<VoxelCoord>> {
    let (fixed_axis, fixed_coord) = solid_polygon_fixed_plane(vertices, plane_axis)?;
    let pts = project_vertices_to_plane_2d(vertices, fixed_axis);
    let hull = convex_hull_2d(pts);
    if hull.len() < 3 {
        return None;
    }
    let filled = fill_polygon_2d(&hull);
    Some(lift_plane_2d_to_voxels(fixed_axis, fixed_coord, &filled))
}

fn circle_radius_in_plane(
    center: [i32; 3],
    edge: [i32; 3],
    plane_axis: usize,
) -> i32 {
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
fn cylinder_axis_aligned_caps(a: VoxelCoord, b: VoxelCoord, radius: i32) -> Vec<VoxelCoord> {
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

/// Web `getAxisAlignedPlaneFromNormal`: filled rectangle in the axis-aligned face plane through `a`, spanning to `b`.
fn fill_axis_aligned_plane_rectangle(a: VoxelCoord, b: VoxelCoord, fixed_axis: usize) -> Vec<VoxelCoord> {
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
fn hollow_plane_rectangle_frame(a: VoxelCoord, b: VoxelCoord, fixed_axis: usize, wall: i32) -> Vec<VoxelCoord> {
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
    let inner: HashSet<VoxelCoord> = fill_axis_aligned_plane_rectangle(inner_a, inner_b, fixed_axis)
        .into_iter()
        .collect();
    full.into_iter()
        .filter(|c| !inner.contains(c))
        .collect()
}

const NEIGHBORS6: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

fn neighbors_in_fixed_plane(fixed_axis: usize) -> &'static [(i32, i32, i32)] {
    match fixed_axis {
        0 => &[(0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
        1 => &[(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)],
        _ => &[(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)],
    }
}

/// Web `hollowSolidToShell` — erodes `thickness` layers from the interior using `neighbors`.
fn hollow_solid_to_shell(
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
        return hollow_solid_to_shell(
            &plane_positions,
            wall,
            neighbors_in_fixed_plane(fixed_axis),
        );
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
            let dk = dir * k as i32;
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
    hollow_solid_to_shell(
        &filled,
        wall,
        neighbors_in_fixed_plane(fixed_axis),
    )
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
        let w = base_w + dir * k as i32;
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
    Some((a, b, plane_ax, hit0, prev0))
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
/// not “first equal coordinate” between center and edge — that mis-picks when two coords match (e.g. top‑face drag along X).
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
    if let Some((hit, prev, _oid)) =
        crate::voxel_edit::ray_first_solid_scene(origin, dir, file, voxel_map, grid_size)
    {
        if let Some(prev) = prev {
            if let Some(face_ax) = face_normal_axis(prev, hit) {
                if let Some(ax) = axis_from_plane_axis(plane_axis, Some(face_ax)) {
                    return ax;
                }
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
fn flip_depth_anchor_if_needed(
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
pub fn stroke_anchor_centers_with_mode(
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
        DrawStrokeMode::Precise => anchor_for_stroke_edit(tool, snap, file, voxel_map, camera, width, height, sx, sy)
            .into_iter()
            .collect(),
        DrawStrokeMode::Spray => {
            // When constrain-to-plane is active, raycast against the invisible plane.
            let anchor_fn = |sx_: f32, sy_: f32| -> Option<VoxelCoord> {
                if let Some((pp, pn)) = spray_constraint_plane {
                    crate::voxel_edit::anchor_on_plane(camera, width, height, sx_, sy_, pp, pn)
                } else {
                    anchor_for_stroke_edit(tool, snap, file, voxel_map, camera, width, height, sx_, sy_)
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
                    anchor_for_stroke_edit(tool, snap, file, voxel_map, camera, width, height, lsx, lsy),
                    anchor_for_stroke_edit(tool, snap, file, voxel_map, camera, width, height, sx, sy),
                ) {
                    (Some(a), Some(b)) => line_pts(a, b),
                    _ => anchor_for_stroke_edit(tool, snap, file, voxel_map, camera, width, height, sx, sy)
                        .into_iter()
                        .collect(),
                }
            } else if let Some((px, py)) = stroke_segment_prev {
                match (
                    anchor_for_stroke_edit(tool, snap, file, voxel_map, camera, width, height, px, py),
                    anchor_for_stroke_edit(tool, snap, file, voxel_map, camera, width, height, sx, sy),
                ) {
                    (Some(a), Some(b)) => line_pts(a, b),
                    _ => anchor_for_stroke_edit(tool, snap, file, voxel_map, camera, width, height, sx, sy)
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
            let anchor = anchor_for_stroke_edit(tool, snap, file, voxel_map, camera, width, height, sx, sy);
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
                    plane_axis,
                    file,
                    voxel_map,
                    camera,
                    width,
                    height,
                    sx,
                    sy,
                    cc,
                    ce,
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
                    anchor_for_stroke_edit(tool, snap, file, voxel_map, camera, width, height, sx, sy)
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
                    if let Some((a, b, plane_ax, hit, prev)) = cuboid_drag_plane_geometry(
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
                        snap,
                    ) {
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
                    anchor_for_stroke_edit(tool, snap, file, voxel_map, camera, width, height, sx, sy)
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
                    if let Some((a, b, _plane_ax, hit, prev)) = cuboid_drag_plane_geometry(
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
                        snap,
                    ) {
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
                    anchor_for_stroke_edit(tool, snap, file, voxel_map, camera, width, height, sx, sy)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disk_radius_0_single() {
        let d = disk_in_axis_plane((0, 0, 0), 2, 0);
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn axis_aligned_cuboid_depth_zero_matches_plane() {
        let plane = fill_axis_aligned_plane_rectangle((0, 0, 0), (1, 1, 0), 2);
        let cuboid = axis_aligned_cuboid_from_plane(
            (0, 0, 0),
            (1, 1, 0),
            0,
            0,
            1,
            0,
            false,
            1,
            2,
        );
        assert_eq!(cuboid.len(), plane.len());
        assert_eq!(cuboid.len(), 4);
    }

    #[test]
    fn axis_aligned_cuboid_depth_one_extends_one_layer() {
        let result = axis_aligned_cuboid_from_plane(
            (0, 0, 0),
            (0, 0, 0),
            0,
            0,
            1,
            1,
            false,
            1,
            2,
        );
        assert_eq!(result.len(), 2);
        assert!(result.contains(&(0, 0, 0)));
        assert!(result.contains(&(0, 0, 1)));
    }

    #[test]
    fn axis_aligned_cylinder_depth_one_two_disks_along_normal() {
        // Face +Z: center (0,0,0), edge (2,0,0) → radius 2 in XY at z=0; depth +1 → z=0 and z=1.
        let cyl = axis_aligned_cylinder_from_plane(
            (0, 0, 0),
            (2, 0, 0),
            0,
            0,
            1,
            1,
            0,
            false,
            1,
        );
        let base = disk_in_axis_plane((0, 0, 0), 2, 2);
        let top = disk_in_axis_plane((0, 0, 1), 2, 2);
        assert_eq!(cyl.len(), base.len() + top.len());
        assert!(cyl.iter().all(|&p| p.2 == 0 || p.2 == 1));
    }

    #[test]
    fn axis_aligned_cuboid_negative_depth_extends_opposite() {
        // Face +Z: a single-voxel plane at z=0, depth -2 should extend to z=-1 and z=-2.
        let result = axis_aligned_cuboid_from_plane(
            (0, 0, 0),
            (0, 0, 0),
            0,
            0,
            -1, // face normal pointing -Z (flipped for negative depth Add)
            2,  // |depth|
            false,
            1,
            2,
        );
        assert_eq!(result.len(), 3);
        assert!(result.contains(&(0, 0, 0)));
        assert!(result.contains(&(0, 0, -1)));
        assert!(result.contains(&(0, 0, -2)));
    }

    #[test]
    fn flip_depth_anchor_add_negative_flips() {
        let a = (0, 1, 0); // prev (empty above surface)
        let b = (2, 1, 3); // corner in the drag plane
        let hit = (0, 0, 0);
        let prev = (0, 1, 0);
        let (fa, fb, fnx, fny, fnz, fd) =
            flip_depth_anchor_if_needed(EditTool::Add, -3, a, b, hit, prev);
        assert_eq!(fa, (0, 0, 0)); // shifted to hit
        assert_eq!(fb, (2, 0, 3)); // shifted same offset
        assert_eq!((fnx, fny, fnz), (0, -1, 0)); // normal reversed
        assert_eq!(fd, 3); // |depth|
    }

    #[test]
    fn flip_depth_anchor_remove_negative_unchanged() {
        let a = (0, 0, 0); // hit (surface) for Remove
        let b = (2, 0, 3);
        let hit = (0, 0, 0);
        let prev = (0, 1, 0);
        let (fa, fb, fnx, fny, fnz, fd) =
            flip_depth_anchor_if_needed(EditTool::Remove, -3, a, b, hit, prev);
        // Remove keeps original geometry
        assert_eq!(fa, (0, 0, 0));
        assert_eq!(fb, (2, 0, 3));
        assert_eq!((fnx, fny, fnz), (0, 1, 0));
        assert_eq!(fd, -3);
    }

    #[test]
    fn flip_depth_anchor_add_positive_unchanged() {
        let a = (0, 1, 0);
        let b = (2, 1, 3);
        let hit = (0, 0, 0);
        let prev = (0, 1, 0);
        let (fa, fb, fnx, fny, fnz, fd) =
            flip_depth_anchor_if_needed(EditTool::Add, 3, a, b, hit, prev);
        // Positive depth keeps original geometry
        assert_eq!(fa, (0, 1, 0));
        assert_eq!(fb, (2, 1, 3));
        assert_eq!((fnx, fny, fnz), (0, 1, 0));
        assert_eq!(fd, 3);
    }

    #[test]
    fn axis_aligned_cylinder_negative_depth_extends_opposite() {
        // Face +Z: center (0,0,0), edge (2,0,0), depth=-1 with flipped normal (0,0,-1)
        // should produce disks at z=0 and z=-1.
        let cyl = axis_aligned_cylinder_from_plane(
            (0, 0, 0),
            (2, 0, 0),
            0,
            0,
            -1, // flipped normal
            1,  // |depth|
            0,
            false,
            1,
        );
        assert!(cyl.iter().all(|&p| p.2 == 0 || p.2 == -1));
        let base = disk_in_axis_plane((0, 0, 0), 2, 2);
        let top = disk_in_axis_plane((0, 0, -1), 2, 2);
        assert_eq!(cyl.len(), base.len() + top.len());
    }

    #[test]
    fn plane_rect_matches_web_axis_aligned_plane() {
        let v = fill_axis_aligned_plane_rectangle((0, 0, 0), (2, 2, 0), 2);
        assert_eq!(v.len(), 9);
    }

    /// Axis-aligned z=plane triangle must use the same fill as web `getPolygonVoxels` (coplanar
    /// triangle / corner tests), not raw `fill_polygon_2d` on integer points.
    #[test]
    fn axis_aligned_triangle_includes_plausible_interior() {
        let v = [[0, 0, 0], [4, 0, 0], [0, 4, 0]];
        let filled = fill_polygon_axis_aligned(&v);
        assert!(
            filled.contains(&(1, 1, 0)),
            "interior lattice near centroid should be filled"
        );
    }

    /// Web `getPolygonVoxels` non-coplanar branch: 3D convex hull, integer voxel centers inside.
    #[test]
    fn non_coplanar_tetrahedron_hull_includes_interior_lattice() {
        let v = [
            [0, 0, 0],
            [4, 0, 0],
            [0, 4, 0],
            [0, 0, 4],
        ];
        let filled = fill_non_coplanar_convex_hull_voxels(&v).expect("expected 3D hull fill");
        assert!(filled.contains(&(1, 1, 1)), "interior lattice point should be inside hull");
        assert!(filled.len() > 8);
    }

    #[test]
    fn polygon_fill_non_coplanar_uses_hull() {
        let v = [
            [0, 0, 0],
            [4, 0, 0],
            [0, 4, 0],
            [0, 0, 4],
        ];
        let filled = fill_polygon_axis_aligned(&v);
        assert!(filled.contains(&(1, 1, 1)));
    }
}
