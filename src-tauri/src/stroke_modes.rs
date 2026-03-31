//! Draw stroke modes (parity with Voxelle web `StrokeMode` / `strokeGeometry.ts`).

use crate::greedy_mesh::VoxelCoord;
use crate::voxel_edit::{anchor_for_edit, voxel_line_dda, EditTool};
use crate::voxelle::VoxelleFile;
use ahash::AHashMap;
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
}

/// Optional geometry from the UI for multi-point strokes (polygon, cuboid corners, etc.).
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
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
    #[serde(default)]
    pub constrain_to_plane: bool,
    #[serde(default)]
    pub spray_size_range: bool,
}

fn axis_from_plane_axis(pa: PlaneAxis, face_axis: Option<usize>) -> Option<usize> {
    match pa {
        PlaneAxis::Auto => face_axis,
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

fn fill_aabb(min_c: VoxelCoord, max_c: VoxelCoord) -> Vec<VoxelCoord> {
    let x0 = min_c.0.min(max_c.0);
    let x1 = min_c.0.max(max_c.0);
    let y0 = min_c.1.min(max_c.1);
    let y1 = min_c.1.max(max_c.1);
    let z0 = min_c.2.min(max_c.2);
    let z1 = min_c.2.max(max_c.2);
    let mut out = Vec::new();
    for x in x0..=x1 {
        for y in y0..=y1 {
            for z in z0..=z1 {
                out.push((x, y, z));
            }
        }
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

fn fill_polygon_axis_aligned(vertices: &[[i32; 3]]) -> Vec<VoxelCoord> {
    if vertices.len() < 3 {
        return vertices.iter().map(|v| (v[0], v[1], v[2])).collect();
    }
    let xs: Vec<i32> = vertices.iter().map(|v| v[0]).collect();
    let ys: Vec<i32> = vertices.iter().map(|v| v[1]).collect();
    let zs: Vec<i32> = vertices.iter().map(|v| v[2]).collect();
    if xs.iter().all(|&x| x == xs[0]) {
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
        vertices.iter().map(|v| (v[0], v[1], v[2])).collect()
    }
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
    } else {
        fill_polygon_axis_aligned(vertices)
    }
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

/// Stroke anchor cells for draw/remove/paint (brush applied per center afterward).
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
) -> Vec<VoxelCoord> {
    match mode {
        DrawStrokeMode::Fill => Vec::new(),
        DrawStrokeMode::Precise => anchor_for_edit(tool, file, voxel_map, camera, width, height, sx, sy)
            .into_iter()
            .collect(),
        DrawStrokeMode::Spray => {
            if let Some((px, py)) = stroke_segment_prev {
                match (
                    anchor_for_edit(tool, file, voxel_map, camera, width, height, px, py),
                    anchor_for_edit(tool, file, voxel_map, camera, width, height, sx, sy),
                ) {
                    (Some(a), Some(b)) => voxel_line_dda(a, b),
                    _ => anchor_for_edit(tool, file, voxel_map, camera, width, height, sx, sy)
                        .into_iter()
                        .collect(),
                }
            } else {
                anchor_for_edit(tool, file, voxel_map, camera, width, height, sx, sy)
                    .into_iter()
                    .collect()
            }
        }
        DrawStrokeMode::Line => {
            if let Some((lsx, lsy)) = stroke_line_start {
                match (
                    anchor_for_edit(tool, file, voxel_map, camera, width, height, lsx, lsy),
                    anchor_for_edit(tool, file, voxel_map, camera, width, height, sx, sy),
                ) {
                    (Some(a), Some(b)) => voxel_line_dda(a, b),
                    _ => anchor_for_edit(tool, file, voxel_map, camera, width, height, sx, sy)
                        .into_iter()
                        .collect(),
                }
            } else if let Some((px, py)) = stroke_segment_prev {
                match (
                    anchor_for_edit(tool, file, voxel_map, camera, width, height, px, py),
                    anchor_for_edit(tool, file, voxel_map, camera, width, height, sx, sy),
                ) {
                    (Some(a), Some(b)) => voxel_line_dda(a, b),
                    _ => anchor_for_edit(tool, file, voxel_map, camera, width, height, sx, sy)
                        .into_iter()
                        .collect(),
                }
            } else {
                anchor_for_edit(tool, file, voxel_map, camera, width, height, sx, sy)
                    .into_iter()
                    .collect()
            }
        }
        DrawStrokeMode::Plane => {
            let grid_size = file.grid_size.max(1);
            let (origin, dir) =
                crate::voxel_edit::screen_to_world_ray(camera, width, height, sx, sy);
            let Some((hit, prev)) =
                crate::voxel_edit::ray_first_solid(origin, dir, voxel_map, grid_size)
            else {
                return Vec::new();
            };
            let Some(prev) = prev else {
                return Vec::new();
            };
            let face_ax = face_normal_axis(prev, hit);
            let plane_ax = axis_from_plane_axis(plane_axis, face_ax).unwrap_or(2);
            let anchor = anchor_for_edit(tool, file, voxel_map, camera, width, height, sx, sy);
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
                let plane_ax = if cc[0] == ce[0] {
                    0
                } else if cc[1] == ce[1] {
                    1
                } else if cc[2] == ce[2] {
                    2
                } else {
                    2
                };
                let r = circle_radius_in_plane(cc, ce, plane_ax);
                disk_in_axis_plane(center, plane_ax, r)
            } else {
                anchor_for_edit(tool, file, voxel_map, camera, width, height, sx, sy)
                    .into_iter()
                    .collect()
            }
        }
        DrawStrokeMode::Cuboid => {
            if let (Some(a), Some(b)) = (aux.cuboid_min, aux.cuboid_max) {
                fill_aabb((a[0], a[1], a[2]), (b[0], b[1], b[2]))
            } else {
                anchor_for_edit(tool, file, voxel_map, camera, width, height, sx, sy)
                    .into_iter()
                    .collect()
            }
        }
        DrawStrokeMode::Cylinder => {
            if let (Some(a), Some(b)) = (aux.cylinder_a, aux.cylinder_b) {
                cylinder_axis_aligned_caps(
                    (a[0], a[1], a[2]),
                    (b[0], b[1], b[2]),
                    brush_radius as i32,
                )
            } else {
                anchor_for_edit(tool, file, voxel_map, camera, width, height, sx, sy)
                    .into_iter()
                    .collect()
            }
        }
        DrawStrokeMode::Polygon => {
            if aux.polygon_vertices.len() >= 3 {
                fill_polygon_axis_aligned(&aux.polygon_vertices)
            } else {
                anchor_for_edit(tool, file, voxel_map, camera, width, height, sx, sy)
                    .into_iter()
                    .collect()
            }
        }
        DrawStrokeMode::PolygonHull => {
            if aux.polygon_vertices.len() >= 3 {
                fill_polygon_hull_axis_aligned(&aux.polygon_vertices)
            } else {
                anchor_for_edit(tool, file, voxel_map, camera, width, height, sx, sy)
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
    fn aabb_two_corners() {
        let v = fill_aabb((0, 0, 0), (1, 1, 1));
        assert_eq!(v.len(), 8);
    }
}
