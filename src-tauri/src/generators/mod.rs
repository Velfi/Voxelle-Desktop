//! Procedural generators (web parity). Face placement uses ray hit + inward normal.

mod grass_gen;
mod rocks;
mod rope_gen;
mod squishy_gen;
mod squishy_gizmo;
mod squishy_pick_rings;
mod squishy_session;

pub use grass_gen::generator_grass_at_screen;
pub use rocks::generator_rocks_at_screen;
pub use rope_gen::generator_rope_between_screens;
pub use squishy_gen::squishy_metaball_at_screen;
pub use squishy_gizmo::{
    append_squishy_gizmo_wire, pick_squishy_gizmo_handle, squishy_gizmo_apply_drag,
    squishy_gizmo_begin_drag, SquishyGizmoDrag,
};
pub use squishy_pick_rings::append_squishy_metaball_pick_rings;
pub use squishy_session::{
    pick_metaball_at_screen, squishy_add_ball_at_screen, squishy_commit_session,
    voxel_coords_for_session_with_limit, Metaball, SquishyMode, SquishySession,
};
