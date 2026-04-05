use crate::camera::OrbitCamera;
use crate::greedy_mesh::VoxelCoord;
use crate::voxel_edit::{
    brush_offset_cells_for_size, effective_ray_grid_size, ensure_grid_fits_coord, ray_first_solid,
    screen_to_world_ray, BrushShape, VoxelEditDelta,
};
use crate::voxelle::{MaterialId, Voxel, VoxelleFile};
use ahash::AHashMap;
use std::collections::HashSet;

fn gravity_unit(dir: &str) -> [f32; 3] {
    match dir {
        "up" => [0.0, 1.0, 0.0],
        "left" => [-1.0, 0.0, 0.0],
        "right" => [1.0, 0.0, 0.0],
        "forward" => [0.0, 0.0, -1.0],
        "back" => [0.0, 0.0, 1.0],
        _ => [0.0, -1.0, 0.0], // "down" and default
    }
}

/// Fill all voxels on a straight line between two voxel coords (3D DDA, no gaps).
fn voxel_line_segment(a: VoxelCoord, b: VoxelCoord) -> impl Iterator<Item = VoxelCoord> {
    let (ax, ay, az) = (a.0, a.1, a.2);
    let (bx, by, bz) = (b.0, b.1, b.2);
    let steps = (bx - ax).abs().max((by - ay).abs()).max((bz - az).abs());
    (0..=steps).map(move |i| {
        let t = if steps == 0 {
            0.0
        } else {
            i as f32 / steps as f32
        };
        let x = (ax as f32 + (bx - ax) as f32 * t).round() as i32;
        let y = (ay as f32 + (by - ay) as f32 * t).round() as i32;
        let z = (az as f32 + (bz - az) as f32 * t).round() as i32;
        (x, y, z)
    })
}

/// Discrete voxel samples along a catenary between `a` and `b` in world space (grid coords).
/// Consecutive samples are connected with a 3D DDA so no gaps appear at size 1.
pub fn catenary_voxel_arc(
    a: VoxelCoord,
    b: VoxelCoord,
    sag: f32,
    segments: i32,
    gravity_direction: &str,
) -> Vec<VoxelCoord> {
    let n = segments.clamp(4, 128);
    let ax = a.0 as f32;
    let ay = a.1 as f32;
    let az = a.2 as f32;
    let bx = b.0 as f32;
    let by = b.1 as f32;
    let bz = b.2 as f32;
    let [gx, gy, gz] = gravity_unit(gravity_direction);

    // Sample the curve at `n` points.
    let samples: Vec<VoxelCoord> = (0..=n)
        .map(|i| {
            let t = i as f32 / n as f32;
            let sag_factor = sag * (1.0 - (2.0 * t - 1.0).powi(2)).max(0.0);
            let x = ax + (bx - ax) * t + gx * sag_factor;
            let y = ay + (by - ay) * t + gy * sag_factor;
            let z = az + (bz - az) * t + gz * sag_factor;
            (x.round() as i32, y.round() as i32, z.round() as i32)
        })
        .collect();

    // Connect consecutive samples with 3D DDA lines to close any gaps.
    let mut out: Vec<VoxelCoord> = Vec::new();
    for pair in samples.windows(2) {
        for p in voxel_line_segment(pair[0], pair[1]) {
            if out.last().copied() != Some(p) {
                out.push(p);
            }
        }
    }
    out
}

/// Build the brush offset list for a given size (display value = diameter in voxels).
///
/// Odd sizes (1, 3, 5…): sphere/cube centered on the path voxel.
/// Even sizes (2, 4, 6…): center shifted to (0.5, 0.5, 0.5) so the cross-section
/// is exactly N voxels wide — e.g. size 2 → 2×2×2 block, size 4 → ~4×4 sphere.
/// Expand centerline voxels with a brush (web `applyBrushAlongPath`).
/// `brush_radius_index + 1` is the diameter in voxels (size display value).
/// Shared with [`crate::generators::cloth_gen`] for rope/cloth generators.
pub fn thicken_centerline_voxels(
    path: &[VoxelCoord],
    brush_radius_index: u32,
    shape: BrushShape,
) -> Vec<VoxelCoord> {
    let size = brush_radius_index + 1;
    let offs = brush_offset_cells_for_size(shape, size, None, None);
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for &(cx, cy, cz) in path {
        for &(dx, dy, dz) in &offs {
            let p = (cx + dx, cy + dy, cz + dz);
            if seen.insert(p) {
                out.push(p);
            }
        }
    }
    out
}

/// Sag in voxels, scaled to rope length so the rope stays above terrain
/// at both extreme tensions regardless of how far apart the endpoints are.
fn rope_sag_from_hits(h1: VoxelCoord, h2: VoxelCoord, tension: f32) -> f32 {
    let dx = (h2.0 - h1.0) as f32;
    let dy = (h2.1 - h1.1) as f32;
    let dz = (h2.2 - h1.2) as f32;
    let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(1.0);
    // ~29% of rope length at tension=-1, 15% at tension=0, shrinks to 0.75% at tension=1.
    dist * 0.15 * (1.0 - tension * 0.95).max(0.05)
}

/// Rope voxel footprint for hover preview (no file mutation).
/// `h1` is the pre-resolved world-space anchor voxel (from the first click).
/// `sx2/sy2` is the current hover screen position used to resolve `h2`.
pub fn preview_rope_voxels_between_screens(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    h1: VoxelCoord,
    sx2: f32,
    sy2: f32,
    tension: f32,
    brush_radius_index: u32,
    brush_shape: BrushShape,
    gravity_direction: &str,
) -> Vec<VoxelCoord> {
    let grid_size = effective_ray_grid_size(file);
    let (o2, d2) = screen_to_world_ray(camera, width, height, sx2, sy2);
    let Some((h2, _)) = ray_first_solid(o2, d2, voxel_map, grid_size) else {
        return Vec::new();
    };
    let t = tension.clamp(-1.0, 1.0);
    let sag_eff = rope_sag_from_hits(h1, h2, t);
    let path = catenary_voxel_arc(h1, h2, sag_eff, 48, gravity_direction);
    thicken_centerline_voxels(&path, brush_radius_index, brush_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── gravity_unit ───────────────────────────────────────────────────

    #[test]
    fn gravity_unit_down() {
        assert_eq!(gravity_unit("down"), [0.0, -1.0, 0.0]);
    }

    #[test]
    fn gravity_unit_up() {
        assert_eq!(gravity_unit("up"), [0.0, 1.0, 0.0]);
    }

    #[test]
    fn gravity_unit_left() {
        assert_eq!(gravity_unit("left"), [-1.0, 0.0, 0.0]);
    }

    #[test]
    fn gravity_unit_right() {
        assert_eq!(gravity_unit("right"), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn gravity_unit_forward() {
        assert_eq!(gravity_unit("forward"), [0.0, 0.0, -1.0]);
    }

    #[test]
    fn gravity_unit_back() {
        assert_eq!(gravity_unit("back"), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn gravity_unit_unknown_defaults_to_down() {
        assert_eq!(gravity_unit("sideways"), [0.0, -1.0, 0.0]);
    }

    // ── catenary_voxel_arc ────────────────────────────────────────────

    #[test]
    fn catenary_same_endpoints_contains_endpoint() {
        let a = (5, 5, 5);
        let arc = catenary_voxel_arc(a, a, 0.0, 4, "down");
        assert!(!arc.is_empty());
        assert!(arc.contains(&a));
    }

    #[test]
    fn catenary_zero_sag_contains_both_endpoints() {
        let a = (0, 5, 0);
        let b = (10, 5, 0);
        let arc = catenary_voxel_arc(a, b, 0.0, 8, "down");
        assert!(arc.contains(&a));
        assert!(arc.contains(&b));
    }

    #[test]
    fn catenary_with_sag_droops_below_line() {
        // With downward gravity sag, the midpoint should be at or below the start/end y
        let a = (0, 10, 0);
        let b = (20, 10, 0);
        let sag = 5.0;
        let arc = catenary_voxel_arc(a, b, sag, 16, "down");
        let min_y = arc.iter().map(|c| c.1).min().unwrap();
        assert!(
            min_y < 10,
            "expected arc to dip below y=10, got min_y={min_y}"
        );
    }

    #[test]
    fn catenary_segment_count_clamped() {
        // segments < 4 should be clamped to 4; result should still be valid
        let a = (0, 0, 0);
        let b = (5, 0, 0);
        let arc_low = catenary_voxel_arc(a, b, 0.0, 1, "down");
        let arc_normal = catenary_voxel_arc(a, b, 0.0, 4, "down");
        assert!(!arc_low.is_empty());
        // Both should start at a and end at b
        assert_eq!(arc_low[0], a);
        assert_eq!(arc_normal[0], a);
    }

    // ── rope_sag_from_hits ────────────────────────────────────────────

    #[test]
    fn rope_sag_high_tension_less_than_low_tension() {
        let h1 = (0, 0, 0);
        let h2 = (20, 0, 0);
        let sag_tight = rope_sag_from_hits(h1, h2, 1.0);
        let sag_loose = rope_sag_from_hits(h1, h2, -1.0);
        assert!(sag_tight < sag_loose, "tight={sag_tight} loose={sag_loose}");
    }

    #[test]
    fn rope_sag_scales_with_distance() {
        let short_sag = rope_sag_from_hits((0, 0, 0), (10, 0, 0), 0.0);
        let long_sag = rope_sag_from_hits((0, 0, 0), (100, 0, 0), 0.0);
        assert!(long_sag > short_sag);
    }

    #[test]
    fn rope_sag_is_positive() {
        let sag = rope_sag_from_hits((0, 0, 0), (5, 3, 2), 0.5);
        assert!(sag > 0.0);
    }
}

pub fn generator_rope_between_screens(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx1: f32,
    sy1: f32,
    sx2: f32,
    sy2: f32,
    // 0 = loose (full sag), 1 = nearly straight (web ropeTension).
    tension: f32,
    brush_radius_index: u32,
    brush_shape: BrushShape,
    color: u32,
    material: MaterialId,
    gravity_direction: &str,
) -> Result<Vec<VoxelEditDelta>, String> {
    let grid_size = effective_ray_grid_size(file);
    let (o1, d1) = screen_to_world_ray(camera, width, height, sx1, sy1);
    let (o2, d2) = screen_to_world_ray(camera, width, height, sx2, sy2);
    let Some((h1, _)) = ray_first_solid(o1, d1, voxel_map, grid_size) else {
        return Ok(Vec::new());
    };
    let Some((h2, _)) = ray_first_solid(o2, d2, voxel_map, grid_size) else {
        return Ok(Vec::new());
    };
    let t = tension.clamp(-1.0, 1.0);
    let sag_eff = rope_sag_from_hits(h1, h2, t);
    let path = catenary_voxel_arc(h1, h2, sag_eff, 48, gravity_direction);
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
