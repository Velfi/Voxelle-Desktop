//! Draw stroke modes (parity with Voxelle web `StrokeMode` / `strokeGeometry.ts`).

mod anchors;
mod polygon;
mod symmetry;

pub use anchors::cuboid_drag_plane_geometry_pub;

use crate::greedy_mesh::VoxelCoord;
use crate::voxel_edit::EditTool;
use crate::voxelle::VoxelleFile;
use ahash::AHashMap;
use glam::Vec3;

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
    /// Solid polygon: extrusion depth in voxel steps along the polygon plane normal.
    #[serde(default)]
    pub polygon_depth: Option<i32>,
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
    /// Frozen depth-phase geometry: anchor voxel `a` (world-space). When all five `cuboid_frozen_*`
    /// fields are present, `cuboid_drag_plane_geometry` is bypassed so camera movement during the
    /// depth phase cannot change the extrusion direction.
    #[serde(default)]
    pub cuboid_frozen_a: Option<[i32; 3]>,
    #[serde(default)]
    pub cuboid_frozen_b: Option<[i32; 3]>,
    #[serde(default)]
    pub cuboid_frozen_plane_ax: Option<u8>,
    #[serde(default)]
    pub cuboid_frozen_hit: Option<[i32; 3]>,
    #[serde(default)]
    pub cuboid_frozen_prev: Option<[i32; 3]>,
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
            polygon_depth: None,
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
            cuboid_frozen_a: None,
            cuboid_frozen_b: None,
            cuboid_frozen_plane_ax: None,
            cuboid_frozen_hit: None,
            cuboid_frozen_prev: None,
        }
    }
}

fn default_stroke_snap_to_surface() -> bool {
    true
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
    anchors::compute_anchors(
        mode,
        plane_axis,
        aux,
        tool,
        file,
        voxel_map,
        camera,
        width,
        height,
        sx,
        sy,
        brush_radius,
        stroke_line_start,
        stroke_segment_prev,
        spray_constraint_plane,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use anchors::{
        axis_aligned_cuboid_from_plane, axis_aligned_cylinder_from_plane,
        disk_in_axis_plane, fill_axis_aligned_plane_rectangle, flip_depth_anchor_if_needed,
    };
    use polygon::fill_non_coplanar_convex_hull_voxels;
    use polygon::{fill_polygon_axis_aligned};

    #[test]
    fn disk_radius_0_single() {
        let d = disk_in_axis_plane((0, 0, 0), 2, 0);
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn axis_aligned_cuboid_depth_zero_matches_plane() {
        let plane = fill_axis_aligned_plane_rectangle((0, 0, 0), (1, 1, 0), 2);
        let cuboid = axis_aligned_cuboid_from_plane((0, 0, 0), (1, 1, 0), 0, 0, 1, 0, false, 1, 2);
        assert_eq!(cuboid.len(), plane.len());
        assert_eq!(cuboid.len(), 4);
    }

    #[test]
    fn axis_aligned_cuboid_depth_one_extends_one_layer() {
        let result = axis_aligned_cuboid_from_plane((0, 0, 0), (0, 0, 0), 0, 0, 1, 1, false, 1, 2);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&(0, 0, 0)));
        assert!(result.contains(&(0, 0, 1)));
    }

    #[test]
    fn axis_aligned_cylinder_depth_one_two_disks_along_normal() {
        // Face +Z: center (0,0,0), edge (2,0,0) → radius 2 in XY at z=0; depth +1 → z=0 and z=1.
        let cyl = axis_aligned_cylinder_from_plane((0, 0, 0), (2, 0, 0), 0, 0, 1, 1, 0, false, 1);
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
        let v = [[0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4]];
        let filled = fill_non_coplanar_convex_hull_voxels(&v).expect("expected 3D hull fill");
        assert!(
            filled.contains(&(1, 1, 1)),
            "interior lattice point should be inside hull"
        );
        assert!(filled.len() > 8);
    }

    #[test]
    fn polygon_fill_non_coplanar_uses_hull() {
        let v = [[0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4]];
        let filled = fill_polygon_axis_aligned(&v);
        assert!(filled.contains(&(1, 1, 1)));
    }
}
