//! Polygon vertex placement, hull ordering, and polygon fill helpers.

use super::{PlaneAxis, StrokeAux};
use crate::greedy_mesh::VoxelCoord;

/// Web `COPLANAR_FILL_TOL` — plane distance threshold for integer voxel corners.
const COPLANAR_FILL_TOL: f64 = 0.08;

pub(super) fn point_in_polygon_2d(x: i32, y: i32, verts: &[(i32, i32)]) -> bool {
    let n = verts.len();
    if n < 3 {
        return false;
    }
    let mut c = false;
    for i in 0..n {
        let (x0, y0) = verts[i];
        let (x1, y1) = verts[(i + 1) % n];
        if (y0 > y) != (y1 > y) {
            let t = (y - y0) as f64 / (y1 - y0) as f64;
            let xi = x0 as f64 + t * (x1 - x0) as f64;
            if (x as f64) < xi {
                c = !c;
            }
        }
    }
    c
}

/// 2D polygon interior on the integer grid (ray-cast test per cell in bbox).
pub(super) fn fill_polygon_2d(verts: &[(i32, i32)]) -> Vec<(i32, i32)> {
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

fn find_non_collinear_triple(vertices: &[[i32; 3]]) -> Option<(usize, usize, usize)> {
    let n = vertices.len();
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let a = vertices[i];
                let b = vertices[j];
                let c = vertices[k];
                let ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2]);
                let ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2]);
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
    let ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    let ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2]);
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
pub(super) fn fill_coplanar_polygon(vertices: &[[i32; 3]]) -> Option<Vec<VoxelCoord>> {
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
    let ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    let ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2]);
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
pub(super) fn fill_coplanar_hull(vertices: &[[i32; 3]]) -> Option<Vec<VoxelCoord>> {
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
    let ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    let ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2]);
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
pub(super) fn fill_non_coplanar_convex_hull_voxels(
    vertices: &[[i32; 3]],
) -> Option<Vec<VoxelCoord>> {
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

pub(super) fn fill_polygon_axis_aligned(vertices: &[[i32; 3]]) -> Vec<VoxelCoord> {
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

pub(super) fn convex_hull_2d(mut pts: Vec<(i32, i32)>) -> Vec<(i32, i32)> {
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

pub(super) fn fill_polygon_hull_axis_aligned(vertices: &[[i32; 3]]) -> Vec<VoxelCoord> {
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

pub(super) fn stroke_aux_is_solid_family(aux: &StrokeAux) -> bool {
    aux.stroke_family_variant.as_deref() == Some("solid")
}

/// Web `getSolidPolygonBasePositions`: corners projected onto plane through first vertex, orthogonal to `plane_axis` / auto-detected axis.
pub(super) fn solid_polygon_fixed_plane(
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

pub(super) fn project_vertices_to_plane_2d(
    vertices: &[[i32; 3]],
    fixed_axis: usize,
) -> Vec<(i32, i32)> {
    vertices
        .iter()
        .map(|v| match fixed_axis {
            0 => (v[1], v[2]),
            1 => (v[0], v[2]),
            _ => (v[0], v[1]),
        })
        .collect()
}

pub(super) fn lift_plane_2d_to_voxels(
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
pub(super) fn fill_solid_polygon_simple_projected(
    vertices: &[[i32; 3]],
    plane_axis: PlaneAxis,
) -> Option<Vec<VoxelCoord>> {
    let (fixed_axis, fixed_coord) = solid_polygon_fixed_plane(vertices, plane_axis)?;
    let poly2d = project_vertices_to_plane_2d(vertices, fixed_axis);
    let filled = fill_polygon_2d(&poly2d);
    Some(lift_plane_2d_to_voxels(fixed_axis, fixed_coord, &filled))
}

/// Solid + polygon **hull**: convex hull of projected corners, then fill (web Surface polygonHull, in work plane).
pub(super) fn fill_solid_polygon_hull_projected(
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

/// Extrude a flat set of base voxel positions along `fixed_axis` by `depth` layers.
/// Positive depth extrudes in the +axis direction, negative in −axis. Depth 0 returns base unchanged.
pub(super) fn extrude_base_positions(
    base: Vec<VoxelCoord>,
    fixed_axis: usize,
    depth: i32,
) -> Vec<VoxelCoord> {
    if depth == 0 {
        return base;
    }
    let layers = depth.abs();
    let dir = if depth > 0 { 1i32 } else { -1i32 };
    let mut positions = base.clone();
    for k in 1..=layers {
        let dk = dir * k;
        for &(px, py, pz) in &base {
            let p = match fixed_axis {
                0 => (px + dk, py, pz),
                1 => (px, py + dk, pz),
                _ => (px, py, pz + dk),
            };
            positions.push(p);
        }
    }
    positions
}
