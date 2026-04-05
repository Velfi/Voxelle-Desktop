//! Mirror / symmetry axis helpers and line axis-align snapping.

use crate::greedy_mesh::VoxelCoord;
use super::PlaneAxis;

/// Constrain `b` so the segment from `a` lies on a single axis (X, Y, or Z) through `a`.
pub(super) fn axis_align_line_endpoints(a: VoxelCoord, b: VoxelCoord) -> (VoxelCoord, VoxelCoord) {
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

pub(super) fn axis_from_plane_axis(pa: PlaneAxis, face_axis: Option<usize>) -> Option<usize> {
    match pa {
        // Camera uses the same axis-aligned plane as Auto (face from pick); view plane is fill-only.
        PlaneAxis::Auto | PlaneAxis::Camera => face_axis,
        PlaneAxis::X => Some(0),
        PlaneAxis::Y => Some(1),
        PlaneAxis::Z => Some(2),
    }
}

/// Face normal as axis index 0|1|2 from ray entry (air `prev` → solid `hit`).
pub(super) fn face_normal_axis(prev: VoxelCoord, hit: VoxelCoord) -> Option<usize> {
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
