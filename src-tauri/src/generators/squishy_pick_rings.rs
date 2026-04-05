//! Per-metaball wire pick shells (web `syncSquishyPickMeshes` parity).

use crate::generators::squishy_session::{SquishyMode, SquishySession};
use crate::greedy_mesh::{self, MeshBuffers};

/// Web: `deleteHover ? 0xff3355 : isSelected ? 0xffcc55 : 0x40b8ff`.
pub fn append_squishy_metaball_pick_rings(
    dst: &mut MeshBuffers,
    session: &SquishySession,
    delete_hover_id: Option<u32>,
) {
    for b in &session.balls {
        let cx = b.x as f32 + 0.5;
        let cy = b.y as f32 + 0.5;
        let cz = b.z as f32 + 0.5;
        let r = b.radius.max(0.2);
        let delete_hover = session.mode == SquishyMode::Delete && delete_hover_id == Some(b.id);
        let is_selected = session.selected_id == Some(b.id);
        let color = if delete_hover {
            [1.0, 51.0 / 255.0, 85.0 / 255.0]
        } else if is_selected {
            [1.0, 204.0 / 255.0, 85.0 / 255.0]
        } else {
            [64.0 / 255.0, 184.0 / 255.0, 1.0]
        };
        greedy_mesh::append_sphere_pick_rings(dst, cx, cy, cz, r, color, 2.0, 24);
    }
}
