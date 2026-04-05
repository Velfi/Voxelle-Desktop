//! Roof generator (web `getRoofFromPinsVoxels` parity): polygon footprint from pins,
//! extruded upward with configurable height profiles (gable, hip, mansard, etc.).

use crate::greedy_mesh::VoxelCoord;
use crate::voxel_edit::{ensure_grid_fits_coord, VoxelEditDelta};
use crate::voxelle::{MaterialId, Voxel, VoxelleFile};
use ahash::AHashMap;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Vector helpers (f64, matching cloth_gen patterns)
// ---------------------------------------------------------------------------

fn v3_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn v3_cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn v3_dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn v3_len(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn v3_norm(a: [f64; 3]) -> [f64; 3] {
    let l = v3_len(a);
    if l < 1e-12 {
        return [0.0, 0.0, 0.0];
    }
    [a[0] / l, a[1] / l, a[2] / l]
}

// ---------------------------------------------------------------------------
// 2D geometry helpers
// ---------------------------------------------------------------------------

/// Ray-casting point-in-polygon test (2D).
fn point_in_polygon_2d(px: f64, py: f64, poly: &[(f64, f64)]) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if (yi > py) != (yj > py) {
            let x_int = (xj - xi) * (py - yi) / (yj - yi) + xi;
            if px < x_int {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Signed distance from point to the nearest edge of a 2D polygon (positive = inside).
fn signed_dist_to_polygon_boundary(px: f64, py: f64, poly: &[(f64, f64)]) -> f64 {
    let n = poly.len();
    let mut min_d2 = f64::INFINITY;
    for i in 0..n {
        let (ax, ay) = poly[i];
        let (bx, by) = poly[(i + 1) % n];
        let dx = bx - ax;
        let dy = by - ay;
        let len2 = dx * dx + dy * dy;
        let t = if len2 < 1e-12 {
            0.0
        } else {
            ((px - ax) * dx + (py - ay) * dy) / len2
        }
        .clamp(0.0, 1.0);
        let cx = ax + t * dx - px;
        let cy = ay + t * dy - py;
        let d2 = cx * cx + cy * cy;
        if d2 < min_d2 {
            min_d2 = d2;
        }
    }
    let d = min_d2.sqrt();
    if point_in_polygon_2d(px, py, poly) {
        d
    } else {
        -d
    }
}

/// Convex hull via Andrew's monotone chain. Returns CCW-ordered hull.
fn convex_hull_2d(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut pts: Vec<(f64, f64)> = points.to_vec();
    pts.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap()
            .then(a.1.partial_cmp(&b.1).unwrap())
    });
    pts.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-12 && (a.1 - b.1).abs() < 1e-12);
    let n = pts.len();
    if n <= 1 {
        return pts;
    }
    if n == 2 {
        return pts;
    }

    fn cross2(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    }

    let mut lower: Vec<(f64, f64)> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross2(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<(f64, f64)> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross2(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Oriented bounding rectangle (minimum area) of 2D points. Returns (center, half_extents, axes).
/// axes[0] = longer axis direction, axes[1] = shorter axis direction.
/// half_extents = (half_length_along_axes0, half_length_along_axes1).
fn oriented_bounding_rect(hull: &[(f64, f64)]) -> ((f64, f64), (f64, f64), [(f64, f64); 2]) {
    if hull.len() < 2 {
        let c = hull.first().copied().unwrap_or((0.0, 0.0));
        return (c, (0.0, 0.0), [(1.0, 0.0), (0.0, 1.0)]);
    }
    let mut best_area = f64::INFINITY;
    let mut best_center = (0.0, 0.0);
    let mut best_half = (0.0, 0.0);
    let mut best_axes = [(1.0, 0.0), (0.0, 1.0)];

    let n = hull.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let dx = hull[j].0 - hull[i].0;
        let dy = hull[j].1 - hull[i].1;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-12 {
            continue;
        }
        let ux = dx / len;
        let uy = dy / len;
        let vx = -uy;
        let vy = ux;

        let mut u_min = f64::INFINITY;
        let mut u_max = f64::NEG_INFINITY;
        let mut v_min = f64::INFINITY;
        let mut v_max = f64::NEG_INFINITY;
        for &(px, py) in hull {
            let u = px * ux + py * uy;
            let v = px * vx + py * vy;
            u_min = u_min.min(u);
            u_max = u_max.max(u);
            v_min = v_min.min(v);
            v_max = v_max.max(v);
        }
        let area = (u_max - u_min) * (v_max - v_min);
        if area < best_area {
            best_area = area;
            let u_mid = (u_min + u_max) * 0.5;
            let v_mid = (v_min + v_max) * 0.5;
            best_center = (u_mid * ux + v_mid * vx, u_mid * uy + v_mid * vy);
            let hu = (u_max - u_min) * 0.5;
            let hv = (v_max - v_min) * 0.5;
            if hu >= hv {
                best_half = (hu, hv);
                best_axes = [(ux, uy), (vx, vy)];
            } else {
                best_half = (hv, hu);
                best_axes = [(vx, vy), (ux, uy)];
            }
        }
    }
    (best_center, best_half, best_axes)
}

// ---------------------------------------------------------------------------
// Height profile computation per roof style
// ---------------------------------------------------------------------------

/// Given a footprint cell in UV space, compute the roof height (number of layers above the base).
/// Returns 0 if the cell should not receive any voxels for this style.
#[allow(clippy::too_many_arguments)]
fn roof_height_for_style(
    u: f64,
    v: f64,
    style: &str,
    max_height: i32,
    thickness: i32,
    shed_edge_index: i32,
    gable_orientation: i32,
    break_ratio: f32,
    wall_height: i32,
    parapet_height: i32,
    salt_skew: f32,
    // Pre-computed footprint geometry:
    poly_uv: &[(f64, f64)],
    hull: &[(f64, f64)],
    centroid: (f64, f64),
    obb_center: (f64, f64),
    obb_half: (f64, f64),
    obb_axes: [(f64, f64); 2],
    max_boundary_dist: f64,
) -> i32 {
    let bdist = signed_dist_to_polygon_boundary(u, v, poly_uv);
    if bdist < -0.5 {
        return 0;
    }

    let h = max_height.max(1) as f64;
    let t = thickness.max(1);

    match style {
        // -- Flat styles --
        "flat" => t,

        "flat_parapet" => {
            let ph = parapet_height.max(1) as f64;
            if bdist < 1.5 {
                t + (ph as i32)
            } else {
                t
            }
        }

        // -- Pyramid: height proportional to boundary distance --
        "pyramid" => {
            if max_boundary_dist < 1e-6 {
                return t;
            }
            let ratio = (bdist.max(0.0) / max_boundary_dist).clamp(0.0, 1.0);
            t + (ratio * h).round() as i32
        }

        // -- Cone: height proportional to distance from centroid --
        "cone" => {
            let dx = u - centroid.0;
            let dy = v - centroid.1;
            let dist_from_center = (dx * dx + dy * dy).sqrt();
            if max_boundary_dist < 1e-6 {
                return t;
            }
            let ratio = (1.0 - dist_from_center / max_boundary_dist).clamp(0.0, 1.0);
            t + (ratio * h).round() as i32
        }

        // -- Shed: linear ramp from one edge to opposite --
        "shed" => {
            // Use the hull edge at shed_edge_index as the low edge
            let n = hull.len();
            if n < 2 {
                return t;
            }
            let idx = (shed_edge_index.max(0) as usize) % n;
            let (ax, ay) = hull[idx];
            let (bx, by) = hull[(idx + 1) % n];
            // Direction perpendicular to the chosen edge (inward)
            let edge_dx = bx - ax;
            let edge_dy = by - ay;
            let len = (edge_dx * edge_dx + edge_dy * edge_dy).sqrt();
            if len < 1e-12 {
                return t;
            }
            // Perpendicular direction (pointing inward toward centroid)
            let mut nx = -edge_dy / len;
            let mut ny = edge_dx / len;
            // Make sure it points toward centroid
            let to_cent_x = centroid.0 - ax;
            let to_cent_y = centroid.1 - ay;
            if nx * to_cent_x + ny * to_cent_y < 0.0 {
                nx = -nx;
                ny = -ny;
            }
            // Project all hull points onto this perpendicular to find range
            let mut proj_min = f64::INFINITY;
            let mut proj_max = f64::NEG_INFINITY;
            for &(hx, hy) in hull {
                let p = (hx - ax) * nx + (hy - ay) * ny;
                proj_min = proj_min.min(p);
                proj_max = proj_max.max(p);
            }
            let range = proj_max - proj_min;
            if range < 1e-6 {
                return t;
            }
            let proj = (u - ax) * nx + (v - ay) * ny;
            let ratio = ((proj - proj_min) / range).clamp(0.0, 1.0);
            t + (ratio * h).round() as i32
        }

        // -- Gable: V-shaped ridge along longer axis --
        "gable" => {
            // Project point onto shorter OBB axis; height is V-shaped
            let axis_idx = if gable_orientation == 1 { 0 } else { 1 };
            let ax = obb_axes[axis_idx];
            let proj = (u - obb_center.0) * ax.0 + (v - obb_center.1) * ax.1;
            let half_val = if axis_idx == 0 {
                obb_half.0
            } else {
                obb_half.1
            };
            let ratio = (1.0 - (proj.abs() / half_val.max(1.0))).clamp(0.0, 1.0);
            t + (ratio * h).round() as i32
        }

        // -- Saltbox: asymmetric gable with skew --
        "saltbox" => {
            let axis_idx = if gable_orientation == 1 { 0 } else { 1 };
            let ax = obb_axes[axis_idx];
            let proj = (u - obb_center.0) * ax.0 + (v - obb_center.1) * ax.1;
            let half_val = if axis_idx == 0 {
                obb_half.0
            } else {
                obb_half.1
            };
            let half = half_val.max(1.0);
            let skew = salt_skew.clamp(-0.8, 0.8) as f64;
            let ridge_offset = skew * half;
            let ratio = if proj < ridge_offset {
                // Short side
                let dist = (ridge_offset - proj).abs();
                let span = (half + ridge_offset).max(1.0);
                1.0 - dist / span
            } else {
                // Long side
                let dist = (proj - ridge_offset).abs();
                let span = (half - ridge_offset).max(1.0);
                1.0 - dist / span
            };
            t + (ratio.clamp(0.0, 1.0) * h).round() as i32
        }

        // -- Hip: gable clipped at ends --
        "hip" => {
            // Gable along longer axis, but clipped by distance from shorter axis ends
            let long_ax = obb_axes[0];
            let short_ax = obb_axes[1];
            let lu = (u - obb_center.0) * long_ax.0 + (v - obb_center.1) * long_ax.1;
            let su = (u - obb_center.0) * short_ax.0 + (v - obb_center.1) * short_ax.1;
            let long_half = obb_half.0.max(1.0);
            let short_half = obb_half.1.max(1.0);
            // Gable profile along short axis
            let gable_ratio = (1.0 - su.abs() / short_half).clamp(0.0, 1.0);
            // Hip clip along long axis
            let hip_ratio = (1.0 - lu.abs() / long_half).clamp(0.0, 1.0);
            let ratio = gable_ratio.min(hip_ratio);
            t + (ratio * h).round() as i32
        }

        // -- Barrel: semicircular arc profile across shorter axis --
        "barrel" => {
            let axis_idx = if gable_orientation == 1 { 0 } else { 1 };
            let ax = obb_axes[axis_idx];
            let proj = (u - obb_center.0) * ax.0 + (v - obb_center.1) * ax.1;
            let half_val = if axis_idx == 0 {
                obb_half.0
            } else {
                obb_half.1
            };
            let half = half_val.max(1.0);
            let normalized = (proj.abs() / half).clamp(0.0, 1.0);
            // Semicircular profile: sqrt(1 - x^2)
            let arc = (1.0 - normalized * normalized).max(0.0).sqrt();
            t + (arc * h).round() as i32
        }

        // -- Mansard: dual-slope, steep lower + shallow upper --
        "mansard" => {
            let br = break_ratio.clamp(0.1, 0.9) as f64;
            if max_boundary_dist < 1e-6 {
                return t;
            }
            let ratio = (bdist.max(0.0) / max_boundary_dist).clamp(0.0, 1.0);
            let height = if ratio < br {
                // Steep lower section
                let local = ratio / br;
                local * 0.7 * h
            } else {
                // Shallow upper section
                let local = (ratio - br) / (1.0 - br);
                0.7 * h + local * 0.3 * h
            };
            t + height.round() as i32
        }

        // -- Gambrel: dual-slope gable (steep lower + shallow upper along short axis) --
        "gambrel" => {
            let axis_idx = if gable_orientation == 1 { 0 } else { 1 };
            let ax = obb_axes[axis_idx];
            let proj = (u - obb_center.0) * ax.0 + (v - obb_center.1) * ax.1;
            let half_val = if axis_idx == 0 {
                obb_half.0
            } else {
                obb_half.1
            };
            let half = half_val.max(1.0);
            let normalized = (proj.abs() / half).clamp(0.0, 1.0);
            let br = break_ratio.clamp(0.1, 0.9) as f64;
            // From edge (normalized=1) inward (normalized=0)
            let from_edge = 1.0 - normalized;
            let height = if from_edge < br {
                // Steep lower section
                let local = from_edge / br;
                local * 0.7 * h
            } else {
                // Shallow upper section
                let local = (from_edge - br) / (1.0 - br);
                0.7 * h + local * 0.3 * h
            };
            t + height.round() as i32
        }

        // -- Pavilion: dual-slope pyramid (steep lower + shallow upper) --
        "pavilion" => {
            let br = break_ratio.clamp(0.1, 0.9) as f64;
            let dx = u - centroid.0;
            let dy = v - centroid.1;
            let dist_from_center = (dx * dx + dy * dy).sqrt();
            if max_boundary_dist < 1e-6 {
                return t;
            }
            let from_edge = (1.0 - dist_from_center / max_boundary_dist).clamp(0.0, 1.0);
            let height = if from_edge < br {
                let local = from_edge / br;
                local * 0.7 * h
            } else {
                let local = (from_edge - br) / (1.0 - br);
                0.7 * h + local * 0.3 * h
            };
            t + height.round() as i32
        }

        // -- Dutch Gable: wall base + gable cap --
        "dutch_gable" => {
            let wh = wall_height.max(0) as f64;
            // Lower portion is vertical walls (uniform height = wall_height)
            // Upper portion is a gable
            let axis_idx = if gable_orientation == 1 { 0 } else { 1 };
            let ax = obb_axes[axis_idx];
            let proj = (u - obb_center.0) * ax.0 + (v - obb_center.1) * ax.1;
            let half_val = if axis_idx == 0 {
                obb_half.0
            } else {
                obb_half.1
            };
            let half = half_val.max(1.0);
            let gable_ratio = (1.0 - (proj.abs() / half)).clamp(0.0, 1.0);
            let gable_h = (h - wh).max(0.0);
            t + (wh + gable_ratio * gable_h).round() as i32
        }

        // Fallback: treat as flat
        _ => t,
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Preview variant: same logic as `generate_roof_from_pins` but returns raw
/// voxel coordinates without mutating any file state.
pub fn preview_roof_voxels(
    pins: &[[i32; 3]],
    style: &str,
    height: i32,
    thickness: i32,
    shed_edge_index: i32,
    gable_orientation: i32,
    break_ratio: f32,
    wall_height: i32,
    parapet_height: i32,
    salt_skew: f32,
    hollow: bool,
) -> Vec<(i32, i32, i32)> {
    let raw: Vec<[i32; 3]> = pins
        .iter()
        .enumerate()
        .filter(|(i, p)| {
            *i == 0 || pins[i - 1][0] != p[0] || pins[i - 1][1] != p[1] || pins[i - 1][2] != p[2]
        })
        .map(|(_, p)| *p)
        .collect();

    if raw.len() < 3 {
        return Vec::new();
    }

    let o = [raw[0][0] as f64, raw[0][1] as f64, raw[0][2] as f64];
    let e1 = v3_sub([raw[1][0] as f64, raw[1][1] as f64, raw[1][2] as f64], o);
    let mut nvec = v3_cross(
        e1,
        v3_sub([raw[2][0] as f64, raw[2][1] as f64, raw[2][2] as f64], o),
    );
    let mut k = 3usize;
    while v3_len(nvec) < 1e-9 && k < raw.len() {
        nvec = v3_cross(
            e1,
            v3_sub([raw[k][0] as f64, raw[k][1] as f64, raw[k][2] as f64], o),
        );
        k += 1;
    }
    if v3_len(nvec) < 1e-9 {
        return Vec::new();
    }

    let nunit = v3_norm(nvec);
    let mut uaxis = v3_norm(e1);
    if v3_dot(uaxis, nunit).abs() > 0.99 {
        uaxis = v3_norm(v3_cross(nunit, [1.0, 0.0, 0.0]));
        if v3_len(uaxis) < 1e-6 {
            uaxis = v3_norm(v3_cross(nunit, [0.0, 1.0, 0.0]));
        }
    }
    let vaxis = v3_norm(v3_cross(nunit, uaxis));

    let mut uv_poly: Vec<(f64, f64)> = Vec::with_capacity(raw.len());
    for p in &raw {
        let d = v3_sub([p[0] as f64, p[1] as f64, p[2] as f64], o);
        uv_poly.push((v3_dot(d, uaxis), v3_dot(d, vaxis)));
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

    let hull = convex_hull_2d(&uv_poly);

    let mut u_min = f64::INFINITY;
    let mut u_max = f64::NEG_INFINITY;
    let mut v_min = f64::INFINITY;
    let mut v_max = f64::NEG_INFINITY;
    for &(pu, pv) in &uv_poly {
        u_min = u_min.min(pu);
        u_max = u_max.max(pu);
        v_min = v_min.min(pv);
        v_max = v_max.max(pv);
    }

    let centroid = {
        let n = uv_poly.len() as f64;
        let cu = uv_poly.iter().map(|p| p.0).sum::<f64>() / n;
        let cv = uv_poly.iter().map(|p| p.1).sum::<f64>() / n;
        (cu, cv)
    };

    let (obb_center, obb_half, obb_axes) = oriented_bounding_rect(&hull);
    let max_boundary_dist = {
        let d = signed_dist_to_polygon_boundary(centroid.0, centroid.1, &uv_poly);
        d.max(1.0)
    };

    let iu_min = u_min.floor() as i32 - 1;
    let iu_max = u_max.ceil() as i32 + 1;
    let iv_min = v_min.floor() as i32 - 1;
    let iv_max = v_max.ceil() as i32 + 1;

    struct FootprintCell {
        iu: i32,
        iv: i32,
        roof_h: i32,
    }

    let mut footprint: Vec<FootprintCell> = Vec::new();
    for iv in iv_min..=iv_max {
        for iu in iu_min..=iu_max {
            let uf = iu as f64;
            let vf = iv as f64;
            if !point_in_polygon_2d(uf, vf, &uv_poly) {
                let bd = signed_dist_to_polygon_boundary(uf, vf, &uv_poly);
                if bd < -0.4 {
                    continue;
                }
            }
            let rh = roof_height_for_style(
                uf,
                vf,
                style,
                height,
                thickness,
                shed_edge_index,
                gable_orientation,
                break_ratio,
                wall_height,
                parapet_height,
                salt_skew,
                &uv_poly,
                &hull,
                centroid,
                obb_center,
                obb_half,
                obb_axes,
                max_boundary_dist,
            );
            if rh > 0 {
                footprint.push(FootprintCell { iu, iv, roof_h: rh });
            }
        }
    }

    if footprint.is_empty() {
        return Vec::new();
    }

    let mut all_coords: Vec<(i32, i32, i32, i32)> = Vec::new();
    let mut coord_set: HashSet<(i32, i32, i32)> = HashSet::new();

    for cell in &footprint {
        for layer in 0..cell.roof_h {
            let wx = o[0]
                + cell.iu as f64 * uaxis[0]
                + cell.iv as f64 * vaxis[0]
                + layer as f64 * nunit[0];
            let wy = o[1]
                + cell.iu as f64 * uaxis[1]
                + cell.iv as f64 * vaxis[1]
                + layer as f64 * nunit[1];
            let wz = o[2]
                + cell.iu as f64 * uaxis[2]
                + cell.iv as f64 * vaxis[2]
                + layer as f64 * nunit[2];
            let x = wx.round() as i32;
            let y = wy.round() as i32;
            let z = wz.round() as i32;
            if coord_set.insert((x, y, z)) {
                all_coords.push((x, y, z, layer));
            }
        }
    }

    if hollow {
        let occupied: HashSet<(i32, i32, i32)> =
            all_coords.iter().map(|&(x, y, z, _)| (x, y, z)).collect();
        let neighbors: [(i32, i32, i32); 6] = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ];
        all_coords
            .iter()
            .filter(|&&(x, y, z, _)| {
                neighbors
                    .iter()
                    .any(|&(dx, dy, dz)| !occupied.contains(&(x + dx, y + dy, z + dz)))
            })
            .map(|&(x, y, z, _)| (x, y, z))
            .collect()
    } else {
        all_coords.iter().map(|&(x, y, z, _)| (x, y, z)).collect()
    }
}

/// Generate a roof from polygon pin points. Mirrors the web `getRoofFromPinsVoxels`.
///
/// Pins define the footprint polygon. The interior is scanline-filled in the pin plane,
/// then extruded upward (along the plane normal) with a height profile determined by `style`.
#[allow(clippy::too_many_arguments)]
pub fn generate_roof_from_pins(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    pins: &[[i32; 3]],
    style: &str,
    height: i32,
    thickness: i32,
    shed_edge_index: i32,
    gable_orientation: i32,
    break_ratio: f32,
    wall_height: i32,
    parapet_height: i32,
    salt_skew: f32,
    hollow: bool,
    color: u32,
    material: MaterialId,
) -> Vec<VoxelEditDelta> {
    // --- Deduplicate consecutive pins ---
    let raw: Vec<[i32; 3]> = pins
        .iter()
        .enumerate()
        .filter(|(i, p)| {
            *i == 0 || pins[i - 1][0] != p[0] || pins[i - 1][1] != p[1] || pins[i - 1][2] != p[2]
        })
        .map(|(_, p)| *p)
        .collect();

    if raw.len() < 3 {
        return Vec::new();
    }

    // --- Determine plane normal from first 3+ non-collinear points ---
    let o = [raw[0][0] as f64, raw[0][1] as f64, raw[0][2] as f64];
    let e1 = v3_sub([raw[1][0] as f64, raw[1][1] as f64, raw[1][2] as f64], o);
    let mut nvec = v3_cross(
        e1,
        v3_sub([raw[2][0] as f64, raw[2][1] as f64, raw[2][2] as f64], o),
    );
    let mut k = 3usize;
    while v3_len(nvec) < 1e-9 && k < raw.len() {
        nvec = v3_cross(
            e1,
            v3_sub([raw[k][0] as f64, raw[k][1] as f64, raw[k][2] as f64], o),
        );
        k += 1;
    }

    if v3_len(nvec) < 1e-9 {
        // All pins are collinear; cannot form a polygon.
        return Vec::new();
    }

    let nunit = v3_norm(nvec);

    // --- Build UV coordinate system on the pin plane ---
    let mut uaxis = v3_norm(e1);
    if v3_dot(uaxis, nunit).abs() > 0.99 {
        uaxis = v3_norm(v3_cross(nunit, [1.0, 0.0, 0.0]));
        if v3_len(uaxis) < 1e-6 {
            uaxis = v3_norm(v3_cross(nunit, [0.0, 1.0, 0.0]));
        }
    }
    let vaxis = v3_norm(v3_cross(nunit, uaxis));

    // --- Project pins to 2D UV ---
    let mut uv_poly: Vec<(f64, f64)> = Vec::with_capacity(raw.len());
    for p in &raw {
        let d = v3_sub([p[0] as f64, p[1] as f64, p[2] as f64], o);
        uv_poly.push((v3_dot(d, uaxis), v3_dot(d, vaxis)));
    }

    // Ensure CCW winding
    let mut area2 = 0.0;
    for i in 0..uv_poly.len() {
        let p = uv_poly[i];
        let q = uv_poly[(i + 1) % uv_poly.len()];
        area2 += p.0 * q.1 - q.0 * p.1;
    }
    if area2 < 0.0 {
        uv_poly.reverse();
    }

    // --- Compute convex hull ---
    let hull = convex_hull_2d(&uv_poly);

    // --- Compute bounding box in UV for scanline fill ---
    let mut u_min = f64::INFINITY;
    let mut u_max = f64::NEG_INFINITY;
    let mut v_min = f64::INFINITY;
    let mut v_max = f64::NEG_INFINITY;
    for &(pu, pv) in &uv_poly {
        u_min = u_min.min(pu);
        u_max = u_max.max(pu);
        v_min = v_min.min(pv);
        v_max = v_max.max(pv);
    }

    // --- Centroid of the polygon in UV ---
    let centroid = {
        let n = uv_poly.len() as f64;
        let cu = uv_poly.iter().map(|p| p.0).sum::<f64>() / n;
        let cv = uv_poly.iter().map(|p| p.1).sum::<f64>() / n;
        (cu, cv)
    };

    // --- Oriented bounding rectangle from hull ---
    let (obb_center, obb_half, obb_axes) = oriented_bounding_rect(&hull);

    // --- Max boundary distance (for normalization in pyramid/mansard/etc.) ---
    let max_boundary_dist = {
        let d = signed_dist_to_polygon_boundary(centroid.0, centroid.1, &uv_poly);
        d.max(1.0)
    };

    // --- Scanline fill: collect all integer UV points inside the polygon ---
    // We use the original polygon (not hull) for the fill so concave shapes work.
    let iu_min = u_min.floor() as i32 - 1;
    let iu_max = u_max.ceil() as i32 + 1;
    let iv_min = v_min.floor() as i32 - 1;
    let iv_max = v_max.ceil() as i32 + 1;

    struct FootprintCell {
        iu: i32,
        iv: i32,
        roof_h: i32,
    }

    let mut footprint: Vec<FootprintCell> = Vec::new();

    for iv in iv_min..=iv_max {
        for iu in iu_min..=iu_max {
            let uf = iu as f64;
            let vf = iv as f64;
            if !point_in_polygon_2d(uf, vf, &uv_poly) {
                // Also accept points very close to the boundary for clean edges
                let bd = signed_dist_to_polygon_boundary(uf, vf, &uv_poly);
                if bd < -0.4 {
                    continue;
                }
            }

            let rh = roof_height_for_style(
                uf,
                vf,
                style,
                height,
                thickness,
                shed_edge_index,
                gable_orientation,
                break_ratio,
                wall_height,
                parapet_height,
                salt_skew,
                &uv_poly,
                &hull,
                centroid,
                obb_center,
                obb_half,
                obb_axes,
                max_boundary_dist,
            );

            if rh > 0 {
                footprint.push(FootprintCell { iu, iv, roof_h: rh });
            }
        }
    }

    if footprint.is_empty() {
        return Vec::new();
    }

    // --- Extrude footprint cells along the plane normal ---
    // The normal direction is "up" for the roof. We extrude from layer 0 to roof_h.
    // Map each (iu, iv, layer) back to 3D world coordinates.

    let mut all_coords: Vec<(i32, i32, i32, i32)> = Vec::new(); // (x, y, z, layer) -- layer for hollow check
    let mut coord_set: HashSet<(i32, i32, i32)> = HashSet::new();

    for cell in &footprint {
        for layer in 0..cell.roof_h {
            let wx = o[0]
                + cell.iu as f64 * uaxis[0]
                + cell.iv as f64 * vaxis[0]
                + layer as f64 * nunit[0];
            let wy = o[1]
                + cell.iu as f64 * uaxis[1]
                + cell.iv as f64 * vaxis[1]
                + layer as f64 * nunit[1];
            let wz = o[2]
                + cell.iu as f64 * uaxis[2]
                + cell.iv as f64 * vaxis[2]
                + layer as f64 * nunit[2];
            let x = wx.round() as i32;
            let y = wy.round() as i32;
            let z = wz.round() as i32;
            if coord_set.insert((x, y, z)) {
                all_coords.push((x, y, z, layer));
            }
        }
    }

    // --- Hollow: keep only surface voxels (at least one empty 6-neighbor) ---
    let final_coords: Vec<(i32, i32, i32)> = if hollow {
        let occupied: HashSet<(i32, i32, i32)> =
            all_coords.iter().map(|&(x, y, z, _)| (x, y, z)).collect();
        let neighbors: [(i32, i32, i32); 6] = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ];
        all_coords
            .iter()
            .filter(|&&(x, y, z, _)| {
                neighbors
                    .iter()
                    .any(|&(dx, dy, dz)| !occupied.contains(&(x + dx, y + dy, z + dz)))
            })
            .map(|&(x, y, z, _)| (x, y, z))
            .collect()
    } else {
        all_coords.iter().map(|&(x, y, z, _)| (x, y, z)).collect()
    };

    // --- Emit voxel deltas (matching cloth_gen / ashlar_gen pattern) ---
    let mut out: Vec<VoxelEditDelta> = Vec::new();
    let mut seen: HashSet<VoxelCoord> = HashSet::new();

    for (x, y, z) in final_coords {
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
    out
}
